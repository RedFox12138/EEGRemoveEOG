"""
Self2Self 1D版本 - 用于EEG去噪
核心思想：使用Dropout在训练时创建不同的子网络，从同一个噪声样本中学习去噪
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv1DWithDropout(nn.Module):
    """带Dropout的1D卷积层"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout_rate=0.3):
        super(Conv1DWithDropout, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        return self.dropout(self.conv(x))


class UNet1D_Self2Self(nn.Module):
    """
    Self2Self的1D U-Net架构
    使用Dropout作为核心机制
    """
    def __init__(self, in_channels=1, base_channels=64, dropout_rate=0.3):
        super(UNet1D_Self2Self, self).__init__()
        self.dropout_rate = dropout_rate
        
        # 编码器
        self.enc1 = nn.Sequential(
            Conv1DWithDropout(in_channels, base_channels, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            Conv1DWithDropout(base_channels, base_channels, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True)
        )
        
        self.pool1 = nn.MaxPool1d(2)
        
        self.enc2 = nn.Sequential(
            Conv1DWithDropout(base_channels, base_channels*2, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            Conv1DWithDropout(base_channels*2, base_channels*2, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True)
        )
        
        self.pool2 = nn.MaxPool1d(2)
        
        # 瓶颈层
        self.bottleneck = nn.Sequential(
            Conv1DWithDropout(base_channels*2, base_channels*4, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            Conv1DWithDropout(base_channels*4, base_channels*4, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True)
        )
        
        # 解码器
        self.up1 = nn.ConvTranspose1d(base_channels*4, base_channels*2, 2, stride=2)
        
        self.dec1 = nn.Sequential(
            Conv1DWithDropout(base_channels*4, base_channels*2, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            Conv1DWithDropout(base_channels*2, base_channels*2, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True)
        )
        
        self.up2 = nn.ConvTranspose1d(base_channels*2, base_channels, 2, stride=2)
        
        self.dec2 = nn.Sequential(
            Conv1DWithDropout(base_channels*2, base_channels, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True),
            Conv1DWithDropout(base_channels, base_channels, dropout_rate=dropout_rate),
            nn.ReLU(inplace=True)
        )
        
        # 输出层
        self.out = nn.Conv1d(base_channels, in_channels, 1)
        
    def forward(self, x):
        # 编码
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        
        # 瓶颈
        bottleneck = self.bottleneck(self.pool2(enc2))
        
        # 解码
        dec1 = self.up1(bottleneck)
        dec1 = torch.cat([dec1, enc2], dim=1)
        dec1 = self.dec1(dec1)
        
        dec2 = self.up2(dec1)
        dec2 = torch.cat([dec2, enc1], dim=1)
        dec2 = self.dec2(dec2)
        
        # 输出
        out = self.out(dec2)
        
        return out
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def test_model():
    """测试模型"""
    model = UNet1D_Self2Self(in_channels=1, base_channels=64, dropout_rate=0.3)
    x = torch.randn(4, 1, 1200)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {model.count_parameters():,}")
    

if __name__ == '__main__':
    test_model()
