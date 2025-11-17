"""
1D Self2Self UNet Model for EEG Denoising
改编自原始2D Self2Self模型，用于处理1D时序信号

Self2Self核心思想：
- 使用Dropout作为掩蔽机制
- 训练时：对输入应用Dropout掩蔽，预测原始输入
- 推理时：多次前向传播并平均，利用Dropout的随机性
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Conv1DBlock(nn.Module):
    """1D卷积块，包含卷积、LeakyReLU和Dropout"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.3):
        super(Conv1DBlock, self).__init__()
        
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=padding, padding_mode='reflect')
        self.activation = nn.LeakyReLU(0.1, inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """编码器块：卷积 + 下采样"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(EncoderBlock, self).__init__()
        
        self.conv = Conv1DBlock(in_channels, out_channels, kernel_size=3, dropout=dropout)
        self.pool = nn.MaxPool1d(2)
    
    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled  # 返回skip connection和下采样结果


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 拼接 + 卷积"""
    def __init__(self, in_channels, out_channels, dropout=0.3):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        # 拼接后通道数加倍
        self.conv1 = Conv1DBlock(in_channels, out_channels, kernel_size=3, dropout=dropout)
        self.conv2 = Conv1DBlock(out_channels, out_channels, kernel_size=3, dropout=dropout)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        
        # 确保尺寸匹配
        if x.size(-1) != skip.size(-1):
            diff = skip.size(-1) - x.size(-1)
            x = F.pad(x, (diff // 2, diff - diff // 2))
        
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Self2Self_UNet1D(nn.Module):
    """
    1D Self2Self UNet for EEG Denoising
    
    核心特点：
    - 训练时使用Dropout作为掩蔽机制
    - 推理时通过多次前向传播和平均来获得稳定输出
    
    Parameters:
    -----------
    in_channels : int
        输入通道数 (通常为1)
    base_channels : int
        基础通道数
    n_depth : int
        UNet深度
    dropout : float
        Dropout概率（Self2Self的关键参数）
    """
    def __init__(self, in_channels=1, base_channels=48, n_depth=5, dropout=0.3):
        super(Self2Self_UNet1D, self).__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.n_depth = n_depth
        self.dropout = dropout
        
        # 编码器
        self.encoders = nn.ModuleList()
        
        # 第一层编码器
        self.enc_conv0 = Conv1DBlock(in_channels, base_channels, kernel_size=3, dropout=dropout)
        self.enc_conv1 = Conv1DBlock(base_channels, base_channels, kernel_size=3, dropout=dropout)
        self.pool0 = nn.MaxPool1d(2)
        
        # 后续编码器层
        for i in range(n_depth - 1):
            self.encoders.append(EncoderBlock(base_channels, base_channels, dropout=dropout))
        
        # 瓶颈层
        self.bottleneck = Conv1DBlock(base_channels, base_channels, kernel_size=3, dropout=dropout)
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(n_depth):
            # 输入是上一层输出+skip connection
            self.decoders.append(
                DecoderBlock(base_channels * 2, base_channels * 2, dropout=dropout)
            )
        
        # 最后的卷积层
        self.dec_conv_final_a = Conv1DBlock(base_channels * 2, 64, kernel_size=3, dropout=dropout)
        self.dec_conv_final_b = Conv1DBlock(64, 32, kernel_size=3, dropout=dropout)
        
        # 输出层（使用sigmoid激活）
        self.output_conv = nn.Conv1d(32, in_channels, 1)
        self.output_activation = nn.Sigmoid()
    
    def forward(self, x, apply_input_dropout=True):
        """
        前向传播
        
        Parameters:
        -----------
        x : torch.Tensor
            输入信号 (batch, channels, time_steps)
        apply_input_dropout : bool
            是否在输入上应用dropout掩蔽（训练时True，推理时False）
        """
        # Self2Self的关键：在输入上应用dropout掩蔽
        if apply_input_dropout and self.training:
            # 创建dropout掩蔽
            mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout))
            x_masked = x * mask
        else:
            x_masked = x
        
        # 保存skip connections
        skips = []
        
        # 第一层编码
        n = self.enc_conv0(x_masked)
        n = self.enc_conv1(n)
        skips.append(n)
        n = self.pool0(n)
        
        # 后续编码层
        for encoder in self.encoders:
            skip, n = encoder(n)
            skips.append(skip)
        
        # 瓶颈层
        n = self.bottleneck(n)
        
        # 解码路径
        skips = skips[::-1]  # 反转顺序
        for i, decoder in enumerate(self.decoders):
            n = decoder(n, skips[i])
        
        # 最后的卷积层
        n = self.dec_conv_final_a(n)
        n = self.dec_conv_final_b(n)
        n = self.output_conv(n)
        n = self.output_activation(n)
        
        return n
    
    def forward_with_mask(self, x):
        """
        前向传播，返回输出和使用的掩蔽
        用于计算Self2Self损失
        """
        # 创建dropout掩蔽
        mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout))
        x_masked = x * mask
        
        # 前向传播
        output = self.forward(x_masked, apply_input_dropout=False)
        
        # 返回输出和反掩蔽（1-mask表示被掩蔽的位置）
        return output, 1 - mask
    
    def predict_average(self, x, n_predictions=100):
        """
        推理时的平均预测（Self2Self的关键）
        
        Parameters:
        -----------
        x : torch.Tensor
            输入信号
        n_predictions : int
            前向传播的次数
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_predictions):
                # 每次使用不同的dropout掩蔽
                pred = self.forward(x, apply_input_dropout=True)
                predictions.append(pred)
        
        # 平均所有预测
        avg_prediction = torch.stack(predictions).mean(dim=0)
        return avg_prediction
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def self2self_loss(pred, target, mask):
    """
    Self2Self损失函数
    
    只在被掩蔽的位置计算损失（与N2V相反，N2V在掩蔽位置计算）
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测输出 (batch, channels, time_steps)
    target : torch.Tensor
        目标值（原始输入） (batch, channels, time_steps)
    mask : torch.Tensor
        掩蔽矩阵，1表示被掩蔽的位置 (batch, channels, time_steps)
    """
    # 只在被掩蔽的位置计算损失
    diff = (pred - target) ** 2
    masked_loss = diff * mask
    
    # 计算平均损失（只对掩蔽位置）
    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    
    return loss


if __name__ == '__main__':
    # 测试模型
    print('测试Self2Self_UNet1D...')
    
    model = Self2Self_UNet1D(in_channels=1, base_channels=48, n_depth=5, dropout=0.3)
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 测试前向传播
    x = torch.randn(4, 1, 1200)  # batch=4, channels=1, time=1200
    
    # 训练模式
    model.train()
    y, mask = model.forward_with_mask(x)
    print(f'\n训练模式:')
    print(f'  输入形状: {x.shape}')
    print(f'  输出形状: {y.shape}')
    print(f'  掩蔽形状: {mask.shape}')
    print(f'  掩蔽比例: {mask.mean().item():.3f}')
    
    # 测试损失
    loss = self2self_loss(y, x, mask)
    print(f'  损失: {loss.item():.6f}')
    
    # 推理模式（多次预测平均）
    model.eval()
    y_avg = model.predict_average(x, n_predictions=10)
    print(f'\n推理模式（10次平均）:')
    print(f'  输出形状: {y_avg.shape}')
    
    print('\n✓ 模型测试通过')
