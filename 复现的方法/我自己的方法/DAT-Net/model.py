"""
DAT-Net: Disentangling Attention Temporal-Network
用于单通道EEG信号去除EOG伪影的深度学习模型
基于1D U-Net架构，具有解耦的双输出头
使用TCN（时间卷积网络）作为瓶颈层
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation Block for 1D signals
    使用1x1卷积代替全连接层实现通道注意力
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock1D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        # 使用1x1卷积代替全连接层
        self.fc1 = nn.Conv1d(channels, channels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv1d(channels // reduction, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) 输入特征
        Returns:
            (B, C, L) 加权后的特征
        """
        # Squeeze: (B, C, L) -> (B, C, 1)
        y = self.avg_pool(x)
        # Excitation: (B, C, 1) -> (B, C//r, 1) -> (B, C, 1)
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        # Scale: (B, C, L) * (B, C, 1)
        return x * y


class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积: Depthwise Conv + Pointwise Conv
    减少参数量和计算量
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding='same'):
        super(DepthwiseSeparableConv, self).__init__()
        # Depthwise: 每个输入通道单独卷积
        if padding == 'same':
            padding_value = kernel_size // 2
        else:
            padding_value = padding
            
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size,
            padding=padding_value,
            groups=in_channels, bias=False
        )
        # Pointwise: 1x1卷积混合通道
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DownBlock(nn.Module):
    """
    编码器块: 两次深度可分离卷积 + SE注意力 + 残差连接 + 下采样
    """
    def __init__(self, in_channels, out_channels, kernel_size=5):
        super(DownBlock, self).__init__()
        
        # 第一个卷积块
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size, padding='same')
        self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act1 = nn.GELU()
        
        # 第二个卷积块
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act2 = nn.GELU()
        
        # SE注意力
        self.se = SEBlock1D(out_channels)
        
        # 残差连接的维度匹配
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        
        # 下采样
        self.downsample = nn.MaxPool1d(kernel_size=2)
    
    def forward(self, x):
        """
        Args:
            x: (B, in_C, L) 输入特征
        Returns:
            skip: (B, out_C, L) 跳跃连接特征
            out: (B, out_C, L//2) 下采样特征
        """
        residual = self.residual(x)
        
        # 两次卷积
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        
        # SE注意力
        out = self.se(out)
        
        # 残差连接
        out = out + residual
        
        # 保存跳跃连接
        skip = out
        
        # 下采样
        out = self.downsample(out)
        
        return skip, out


class TCNResidualBlock(nn.Module):
    """
    TCN残差块: 使用因果卷积和膨胀卷积捕捉时序依赖
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2):
        super(TCNResidualBlock, self).__init__()
        
        # 计算padding以保持same大小
        padding = (kernel_size - 1) * dilation // 2
        
        # 第一个卷积
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二个卷积
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=False
        )
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(dropout)
        
        # 残差连接
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.final_act = nn.GELU()
    
    def forward(self, x):
        """
        Args:
            x: (B, in_C, L) 输入特征
        Returns:
            (B, out_C, L) 输出特征
        """
        residual = self.residual(x)
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act1(out)
        out = self.dropout1(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.act2(out)
        out = self.dropout2(out)
        
        # 残差连接
        out = self.final_act(out + residual)
        
        return out


class TCNBottleneck(nn.Module):
    """
    TCN瓶颈层: 堆叠多个TCN残差块，膨胀率指数增长
    用于捕捉长程时序依赖
    """
    def __init__(self, channels=128, num_blocks=3, kernel_size=7, dropout=0.2):
        super(TCNBottleneck, self).__init__()
        
        blocks = []
        for i in range(num_blocks):
            dilation = 2 ** i  # 膨胀率: 1, 2, 4, 8, ...
            blocks.append(
                TCNResidualBlock(
                    channels, channels, 
                    kernel_size=kernel_size, 
                    dilation=dilation, 
                    dropout=dropout
                )
            )
        
        self.tcn_blocks = nn.Sequential(*blocks)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, L) 输入特征
        Returns:
            (B, C, L) 处理后的特征
        """
        return self.tcn_blocks(x)


class UpBlock(nn.Module):
    """
    解码器块: 上采样 + 拼接跳跃连接 + 两次深度可分离卷积 + SE注意力
    """
    def __init__(self, in_channels, skip_channels, out_channels, kernel_size=5):
        super(UpBlock, self).__init__()
        
        # 上采样
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        
        # 拼接后的通道数
        concat_channels = in_channels + skip_channels
        
        # 第一个卷积块
        self.conv1 = DepthwiseSeparableConv(concat_channels, out_channels, kernel_size, padding='same')
        self.norm1 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act1 = nn.GELU()
        
        # 第二个卷积块
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size, padding='same')
        self.norm2 = nn.InstanceNorm1d(out_channels, affine=True)
        self.act2 = nn.GELU()
        
        # SE注意力
        self.se = SEBlock1D(out_channels)
    
    def forward(self, x, skip):
        """
        Args:
            x: (B, in_C, L) 来自下层的特征
            skip: (B, skip_C, L*2) 跳跃连接特征
        Returns:
            (B, out_C, L*2) 上采样并融合后的特征
        """
        # 上采样
        x = self.upsample(x)
        
        # 尺寸自适应匹配：如果上采样后的尺寸与skip不匹配，进行裁剪或填充
        if x.size(2) != skip.size(2):
            if x.size(2) > skip.size(2):
                # 裁剪x以匹配skip
                x = x[:, :, :skip.size(2)]
            else:
                # 填充x以匹配skip
                pad_size = skip.size(2) - x.size(2)
                x = torch.nn.functional.pad(x, (0, pad_size), mode='replicate')
        
        # 拼接跳跃连接
        x = torch.cat([x, skip], dim=1)
        
        # 两次卷积
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        
        # SE注意力
        x = self.se(x)
        
        return x


class DATNet(nn.Module):
    """
    DAT-Net: Disentangling Attention Temporal-Network
    
    架构:
        - 输入: (B, 1, 512)
        - Encoder: 3层下采样 (1->32->64->128)
        - Bottleneck: TCN层 (多个膨胀卷积块)
        - Decoder: 3层上采样 (128->64->32->16)
        - 双输出头: EEG干净信号 + EOG伪影
    """
    def __init__(self, in_channels=1, base_channels=32):
        super(DATNet, self).__init__()
        
        # ========== 编码器 ==========
        self.encoder1 = DownBlock(in_channels, base_channels)  # 1 -> 32
        self.encoder2 = DownBlock(base_channels, base_channels * 2)  # 32 -> 64
        self.encoder3 = DownBlock(base_channels * 2, base_channels * 4)  # 64 -> 128
        
        # ========== 瓶颈层 (TCN) ==========
        self.bottleneck = TCNBottleneck(
            channels=base_channels * 4,  # 128
            num_blocks=3,  # 3个TCN块: dilation=1,2,4
            kernel_size=7,
            dropout=0.2
        )
        
        # ========== 解码器 ==========
        # UpBlock(in_channels, skip_channels, out_channels)
        self.decoder1 = UpBlock(base_channels * 4, base_channels * 4, base_channels * 2)  # 128+128 -> 64
        self.decoder2 = UpBlock(base_channels * 2, base_channels * 2, base_channels)  # 64+64 -> 32
        self.decoder3 = UpBlock(base_channels, base_channels, base_channels // 2)  # 32+32 -> 16
        
        # ========== 输出头 ==========
        self.eeg_head = nn.Conv1d(base_channels // 2, 1, kernel_size=1)  # EEG干净信号
        self.eog_head = nn.Conv1d(base_channels // 2, 1, kernel_size=1)  # EOG伪影
        
    def forward(self, x):
        """
        Args:
            x: (B, 1, 512) 输入的受污染EEG信号
        Returns:
            eeg_clean: (B, 1, 512) 干净的EEG信号
            eog_artifact: (B, 1, 512) EOG伪影
        """
        # ========== 编码器 ==========
        skip1, enc1 = self.encoder1(x)  # skip1: (B, 32, 512), enc1: (B, 32, 256)
        skip2, enc2 = self.encoder2(enc1)  # skip2: (B, 64, 256), enc2: (B, 64, 128)
        skip3, enc3 = self.encoder3(enc2)  # skip3: (B, 128, 128), enc3: (B, 128, 64)
        
        # ========== 瓶颈层 (TCN) ==========
        bottleneck = self.bottleneck(enc3)  # (B, 128, 64)
        
        # ========== 解码器 ==========
        dec1 = self.decoder1(bottleneck, skip3)  # (B, 64, 128)
        dec2 = self.decoder2(dec1, skip2)  # (B, 32, 256)
        dec3 = self.decoder3(dec2, skip1)  # (B, 16, 512)
        
        # ========== 输出头 ==========
        eeg_clean = self.eeg_head(dec3)  # (B, 1, 512)
        eog_artifact = self.eog_head(dec3)  # (B, 1, 512)
        
        return eeg_clean, eog_artifact
    
    def count_parameters(self):
        """计算模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def DAT_Loss(eeg_clean_pred, eog_artifact_pred, eeg_true_target, eog_artifact_target, eeg_raw_input, 
             lambda1=1.0, lambda2=1.0, lambda3=1.0):
    """
    DAT-Net损失函数
    
    Args:
        eeg_clean_pred: (B, 1, L) 预测的干净EEG信号
        eog_artifact_pred: (B, 1, L) 预测的EOG伪影
        eeg_true_target: (B, 1, L) 真实的干净EEG信号
        eog_artifact_target: (B, 1, L) 真实的EOG伪影 (eeg_raw_input - eeg_true_target)
        eeg_raw_input: (B, 1, L) 原始的受污染EEG信号
        lambda1: EEG干净信号损失权重
        lambda2: EOG伪影损失权重
        lambda3: 一致性损失权重
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典 (用于监控)
    """
    # 1. EEG干净信号重建损失
    loss_clean = F.mse_loss(eeg_clean_pred, eeg_true_target)
    
    # 2. EOG伪影重建损失
    loss_artifact = F.mse_loss(eog_artifact_pred, eog_artifact_target)
    
    # 3. 一致性损失: 干净EEG + EOG伪影 应该等于 原始信号
    loss_consistency = F.mse_loss(eeg_clean_pred + eog_artifact_pred, eeg_raw_input)
    
    # 4. 总损失（加权）
    total_loss = (lambda1 * loss_clean) + (lambda2 * loss_artifact) + (lambda3 * loss_consistency)
    
    # 返回损失字典用于监控
    loss_dict = {
        'total': total_loss.item(),
        'clean': loss_clean.item(),
        'artifact': loss_artifact.item(),
        'consistency': loss_consistency.item()
    }
    
    return total_loss, loss_dict


if __name__ == '__main__':
    # 测试模型
    print("="*70)
    print("DAT-Net 模型测试")
    print("Disentangling Attention Temporal-Network")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 创建模型
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'\n模型参数量: {model.count_parameters():,}')
    
    # 测试前向传播
    batch_size = 4
    seq_len = 512
    x = torch.randn(batch_size, 1, seq_len).to(device)
    
    print(f'\n输入形状: {x.shape}')
    
    eeg_clean, eog_artifact = model(x)
    
    print(f'EEG干净信号形状: {eeg_clean.shape}')
    print(f'EOG伪影形状: {eog_artifact.shape}')
    
    # 测试损失函数
    eeg_target = torch.randn_like(x)
    eog_target = x - eeg_target  # 伪影 = 原始信号 - 干净信号
    
    loss, loss_dict = DAT_Loss(eeg_clean, eog_artifact, eeg_target, eog_target, x)
    
    print(f'\n损失函数测试:')
    print(f'  总损失: {loss_dict["total"]:.6f}')
    print(f'  EEG损失: {loss_dict["clean"]:.6f}')
    print(f'  EOG损失: {loss_dict["artifact"]:.6f}')
    print(f'  一致性损失: {loss_dict["consistency"]:.6f}')
    
    print("\n✓ 模型测试通过!")
    print("="*70)
