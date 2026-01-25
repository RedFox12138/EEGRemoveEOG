"""
模型包装器，支持消融实验中的条件禁用组件
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 导入原始 DATNet
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
sys.path.insert(0, datnet_dir)

from model import DATNet, DownBlock, UpBlock, TCNBottleneck


class StandardUNetDown(nn.Module):
    """标准UNet的下采样块（无SE注意力，无残差连接，无深度可分离卷积）"""
    def __init__(self, in_channels, out_channels):
        super(StandardUNetDown, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
        
        self.pool = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        """返回: (skip连接, 下采样后的特征)"""
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        skip = x
        out = self.pool(x)
        return skip, out


class StandardUNetUp(nn.Module):
    """标准UNet的上采样块（无SE注意力，无残差连接）"""
    def __init__(self, in_channels, skip_channels, out_channels):
        super(StandardUNetUp, self).__init__()
        
        self.up = nn.ConvTranspose1d(in_channels, in_channels, kernel_size=2, stride=2)
        
        self.conv1 = nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(out_channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.act2 = nn.ReLU(inplace=True)
    
    def forward(self, x, skip):
        """x: 上一层特征, skip: 跳跃连接"""
        x = self.up(x)
        
        # 对齐尺寸：处理由于下采样/上采样导致的尺寸不匹配
        if x.size(2) != skip.size(2):
            # 如果上采样后的尺寸与skip不一致，裁剪或填充
            diff = skip.size(2) - x.size(2)
            if diff > 0:
                # skip更长，裁剪skip
                skip = skip[:, :, :x.size(2)]
            else:
                # x更长，裁剪x
                x = x[:, :, :skip.size(2)]
        
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class SimpleConvBottleneck(nn.Module):
    """
    简单卷积瓶颈层，用于标准UNet（用于消融实验）
    """
    def __init__(self, channels):
        super(SimpleConvBottleneck, self).__init__()
        
        # 标准UNet的瓶颈层：仅2个卷积块
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = nn.BatchNorm1d(channels)
        self.act1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.BatchNorm1d(channels)
        self.act2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.act2(self.norm2(self.conv2(x)))
        return x


class ResidualBottleneck(nn.Module):
    """
    普通残差块瓶颈层（用于对比TCN的消融实验）
    使用标准的残差连接，但没有TCN的因果卷积和扩张卷积
    """
    def __init__(self, channels, num_blocks=10, kernel_size=7, dropout=0.2):
        super(ResidualBottleneck, self).__init__()
        
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            block = nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                         padding=kernel_size//2, bias=False),
                nn.InstanceNorm1d(channels, affine=True),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(channels, channels, kernel_size=kernel_size, 
                         padding=kernel_size//2, bias=False),
                nn.InstanceNorm1d(channels, affine=True),
            )
            self.blocks.append(block)
        
        self.activation = nn.GELU()
    
    def forward(self, x):
        for block in self.blocks:
            residual = x
            out = block(x)
            x = self.activation(out + residual)  # 残差连接
        return x


class DATNet_Ablation(nn.Module):
    """
    支持消融实验的 DATNet 变体
    - use_unet: True=标准UNet, False=DAT-Net
    - use_residual: True=普通残差块, False=使用TCN或UNet
    - use_dual_output: True=双输出头(clean+artifact), False=单输出头(仅去噪结果)
    """
    def __init__(self, in_channels=1, base_channels=32, use_unet=False, 
                 use_residual=False, use_dual_output=True):
        super(DATNet_Ablation, self).__init__()
        
        self.use_unet = use_unet
        self.use_residual = use_residual
        self.use_dual_output = use_dual_output
        
        if use_unet:
            # ========== 标准UNet架构 ==========
            # 编码器 (使用标准UNet块)
            self.encoder1 = StandardUNetDown(in_channels, base_channels)  # 1 -> 32
            self.encoder2 = StandardUNetDown(base_channels, base_channels * 2)  # 32 -> 64
            self.encoder3 = StandardUNetDown(base_channels * 2, base_channels * 4)  # 64 -> 128
            
            # 瓶颈层
            self.bottleneck = SimpleConvBottleneck(channels=base_channels * 4)
            
            # 解码器 (使用标准UNet块)
            self.decoder1 = StandardUNetUp(base_channels * 4, base_channels * 4, base_channels * 2)  # 128+128 -> 64
            self.decoder2 = StandardUNetUp(base_channels * 2, base_channels * 2, base_channels)  # 64+64 -> 32
            self.decoder3 = StandardUNetUp(base_channels, base_channels, base_channels // 2)  # 32+32 -> 16
        else:
            # ========== DAT-Net架构（使用DAT-Net的编解码器） ==========
            # 编码器 (3层)
            self.encoder1 = DownBlock(in_channels, base_channels)  # 1 -> 32
            self.encoder2 = DownBlock(base_channels, base_channels * 2)  # 32 -> 64
            self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)  # 64 -> 128
            
            # 瓶颈层：普通残差块 或 DAT-Net(TCN)
            if use_residual:
                # 普通残差块：没有因果卷积和扩张卷积
                self.bottleneck = ResidualBottleneck(
                    channels=base_channels * 4,  # 128
                    num_blocks=10,
                    kernel_size=7,
                    dropout=0.2
                )
            else:
                # DAT-Net：使用TCN（扩张卷积捕捉长程依赖）
                # 特征长度64 (512/2/2/2)，4个Block的感受野≈91，足够覆盖
                self.bottleneck = TCNBottleneck(
                    channels=base_channels * 4,  # 128
                    num_blocks=4,  # 优化：避免感受野过大造成无效计算
                    kernel_size=7,
                    dropout=0.2
                )
            
            # 解码器 (3层)
            self.decoder1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # 128+128 -> 64
            self.decoder2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)  # 64+64 -> 32
            self.decoder3 = UpBlock(base_channels, base_channels, base_channels // 2)  # 32+32 -> 16
        
        # ========== 输出头 ==========
        if use_dual_output:
            # 双输出头：分别输出clean和artifact
            self.eeg_head = nn.Conv1d(base_channels // 2, 1, kernel_size=1)  # EEG干净信号
            self.eog_head = nn.Conv1d(base_channels // 2, 1, kernel_size=1)  # EOG伪影
        else:
            # 单输出头：仅输出去噪结果
            self.output_head = nn.Conv1d(base_channels // 2, 1, kernel_size=1)
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, L) 输入的受污染EEG信号
        Returns:
            如果use_dual_output=True:
                eeg_clean: (B, 1, L) 干净的EEG信号
                eog_artifact: (B, 1, L) EOG伪影
            如果use_dual_output=False:
                denoised: (B, 1, L) 去噪后的信号
                zeros: (B, 1, L) 全零（为了接口兼容）
        """
        input_length = x.size(2)  # 记录输入长度
        
        # ========== 编码器 (3层) ==========
        skip1, enc1 = self.encoder1(x)
        skip2, enc2 = self.encoder2(enc1)
        skip3, enc3 = self.encoder3(enc2)
        
        # ========== 瓶颈层 ==========
        bottleneck = self.bottleneck(enc3)
        
        # ========== 解码器 (3层) ==========
        dec1 = self.decoder1(bottleneck, skip3)
        dec2 = self.decoder2(dec1, skip2)
        dec3 = self.decoder3(dec2, skip1)
        
        # ========== 输出头 ==========
        if self.use_dual_output:
            eeg_clean = self.eeg_head(dec3)
            eog_artifact = self.eog_head(dec3)
            
            # 确保输出长度与输入一致
            if eeg_clean.size(2) != input_length:
                eeg_clean = F.interpolate(eeg_clean, size=input_length, mode='linear', align_corners=False)
                eog_artifact = F.interpolate(eog_artifact, size=input_length, mode='linear', align_corners=False)
            
            return eeg_clean, eog_artifact
        else:
            denoised = self.output_head(dec3)
            
            # 确保输出长度与输入一致
            if denoised.size(2) != input_length:
                denoised = F.interpolate(denoised, size=input_length, mode='linear', align_corners=False)
            
            # 返回(denoised, zeros)以保持接口兼容性
            zeros = torch.zeros_like(denoised)
            return denoised, zeros


def create_model(ablation_config):
    """
    根据消融配置创建模型
    
    Args:
        ablation_config: dict，包含 use_unet, use_residual, use_dual_output 等配置项
    
    Returns:
        DATNet_Ablation 实例
    """
    use_unet = ablation_config.get('use_unet', False)
    use_residual = ablation_config.get('use_residual', False)
    use_dual_output = ablation_config.get('use_dual_output', True)
    
    model = DATNet_Ablation(
        in_channels=1, 
        base_channels=32, 
        use_unet=use_unet,
        use_residual=use_residual,
        use_dual_output=use_dual_output
    )
    return model


if __name__ == '__main__':
    # 测试不同配置
    print("测试DAT-Net双输出头（使用TCN）:")
    model_dual = create_model({'use_unet': False, 'use_dual_output': True})
    x = torch.randn(2, 1, 512)
    c, a = model_dual(x)
    print(f"  输入: {x.shape}, Clean: {c.shape}, Artifact: {a.shape}")
    
    print("\n测试DAT-Net单输出头:")
    model_single = create_model({'use_unet': False, 'use_dual_output': False})
    denoised, zeros = model_single(x)
    print(f"  输入: {x.shape}, Denoised: {denoised.shape}, Zeros: {zeros.shape}")
    
    print("\n测试标准UNet:")
    model_unet = create_model({'use_unet': True, 'use_dual_output': True})
    c, a = model_unet(x)
    print(f"  输入: {x.shape}, Clean: {c.shape}, Artifact: {a.shape}")
    
    # 参数量对比
    params_dual = sum(p.numel() for p in model_dual.parameters())
    params_single = sum(p.numel() for p in model_single.parameters())
    params_unet = sum(p.numel() for p in model_unet.parameters())
    print(f"\n参数量对比:")
    print(f"  DAT-Net (双输出): {params_dual:,}")
    print(f"  DAT-Net (单输出): {params_single:,}")
    print(f"  Standard UNet: {params_unet:,}")
