"""
1D Noise2Void UNet Model for EEG Denoising
改编自原始2D N2V模型，用于处理1D时序信号
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DBlock(nn.Module):
    """1D卷积块，包含卷积、批归一化和激活"""
    def __init__(self, in_channels, out_channels, kernel_size=3, 
                 activation='relu', batch_norm=True, dropout=0.0):
        super(Conv1DBlock, self).__init__()
        
        padding = kernel_size // 2
        layers = []
        
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, 
                               padding=padding, bias=not batch_norm))
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))
        
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        elif activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(0.1, inplace=True))
        
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """编码器块：两个卷积 + 下采样"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 batch_norm=True, dropout=0.0, pool_size=2):
        super(EncoderBlock, self).__init__()
        
        self.conv1 = Conv1DBlock(in_channels, out_channels, kernel_size,
                                 batch_norm=batch_norm, dropout=dropout)
        self.conv2 = Conv1DBlock(out_channels, out_channels, kernel_size,
                                 batch_norm=batch_norm, dropout=dropout)
        self.pool = nn.MaxPool1d(pool_size)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x_pooled = self.pool(x)
        return x, x_pooled  # 返回skip connection和下采样结果


class DecoderBlock(nn.Module):
    """解码器块：上采样 + 拼接 + 两个卷积"""
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 batch_norm=True, dropout=0.0, scale_factor=2):
        super(DecoderBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='linear', 
                                    align_corners=True)
        # 上采样后拼接，所以输入通道数要翻倍
        self.conv1 = Conv1DBlock(in_channels, out_channels, kernel_size,
                                 batch_norm=batch_norm, dropout=dropout)
        self.conv2 = Conv1DBlock(out_channels, out_channels, kernel_size,
                                 batch_norm=batch_norm, dropout=dropout)
    
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


class N2V_UNet1D(nn.Module):
    """
    1D Noise2Void UNet for EEG Denoising
    
    Parameters:
    -----------
    in_channels : int
        输入通道数 (通常为1)
    n_depth : int
        UNet深度 (编码器和解码器的层数)
    n_first : int
        第一层的滤波器数量，后续层会翻倍
    kernel_size : int
        卷积核大小
    batch_norm : bool
        是否使用批归一化
    dropout : float
        Dropout概率
    residual : bool
        是否使用残差连接
    """
    def __init__(self, in_channels=1, n_depth=3, n_first=32, 
                 kernel_size=5, batch_norm=True, dropout=0.0,
                 residual=True):
        super(N2V_UNet1D, self).__init__()
        
        self.in_channels = in_channels
        self.n_depth = n_depth
        self.residual = residual
        
        # 编码器
        self.encoders = nn.ModuleList()
        channels = [in_channels] + [n_first * (2 ** i) for i in range(n_depth)]
        
        for i in range(n_depth):
            self.encoders.append(
                EncoderBlock(channels[i], channels[i+1], kernel_size,
                           batch_norm, dropout)
            )
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            Conv1DBlock(channels[-1], channels[-1] * 2, kernel_size,
                       batch_norm=batch_norm, dropout=dropout),
            Conv1DBlock(channels[-1] * 2, channels[-1] * 2, kernel_size,
                       batch_norm=batch_norm, dropout=dropout)
        )
        
        # 解码器
        self.decoders = nn.ModuleList()
        for i in range(n_depth - 1, -1, -1):
            # 输入是上一层输出+skip connection，所以是channels[i+1]*2
            self.decoders.append(
                DecoderBlock(channels[i+1] * 2, channels[i+1], kernel_size,
                           batch_norm, dropout)
            )
        
        # 输出层
        self.output_conv = nn.Conv1d(channels[1], in_channels, 1)
    
    def forward(self, x):
        # 保存原始输入用于残差连接
        x_input = x
        
        # 编码路径
        skip_connections = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skip_connections.append(skip)
        
        # 瓶颈层
        x = self.bottleneck(x)
        
        # 解码路径
        skip_connections = skip_connections[::-1]  # 反转顺序
        for i, decoder in enumerate(self.decoders):
            x = decoder(x, skip_connections[i])
        
        # 输出
        x = self.output_conv(x)
        
        # 残差连接
        if self.residual:
            x = x + x_input
        
        return x
    
    def count_parameters(self):
        """统计模型参数量"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == '__main__':
    # 测试模型
    model = N2V_UNet1D(in_channels=1, n_depth=3, n_first=32, kernel_size=5)
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 测试前向传播
    x = torch.randn(4, 1, 1200)  # batch=4, channels=1, time=1200
    y = model(x)
    print(f'输入形状: {x.shape}')
    print(f'输出形状: {y.shape}')
    assert x.shape == y.shape, "输入输出形状不匹配！"
    print('✓ 模型测试通过')
