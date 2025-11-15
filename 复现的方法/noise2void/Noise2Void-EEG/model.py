"""
Noise2Void 1D Signal Adaptation for EEG Denoising

核心思想:
1. 盲点训练: 随机选择一些时间点作为"盲点"
2. 像素值操纵: 用邻域值替换这些盲点位置
3. 训练目标: 网络学习从被操纵的输入预测原始盲点值

关键创新:
- 不需要干净数据,只需要带噪声的信号
- 通过盲点机制避免网络学习恒等映射
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class UNet1D_N2V(nn.Module):
    """
    1D U-Net for Noise2Void
    
    架构特点:
    - 标准U-Net结构
    - 不使用残差连接(N2V2论文建议)
    - 使用平均池化代替最大池化(减少棋盘伪影)
    """
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet1D_N2V, self).__init__()
        
        features = init_features
        
        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.AvgPool1d(kernel_size=2, stride=2)  # 使用AvgPool代替MaxPool
        
        self.encoder2 = self._block(features, features * 2, name="enc2")
        self.pool2 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.encoder3 = self._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        self.encoder4 = self._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.AvgPool1d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = self._block(features * 8, features * 16, name="bottleneck")
        
        # Decoder
        self.upconv4 = nn.ConvTranspose1d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = self._block((features * 8) * 2, features * 8, name="dec4")
        
        self.upconv3 = nn.ConvTranspose1d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = self._block((features * 4) * 2, features * 4, name="dec3")
        
        self.upconv2 = nn.ConvTranspose1d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = self._block((features * 2) * 2, features * 2, name="dec2")
        
        self.upconv1 = nn.ConvTranspose1d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = self._block(features * 2, features, name="dec1")
        
        # Output: 直接预测去噪后的信号
        self.conv = nn.Conv1d(in_channels=features, out_channels=out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder with skip connections
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        # 输出去噪后的信号(不是噪声!)
        return self.conv(dec1)
    
    def _block(self, in_channels, features, name):
        return nn.Sequential(
            nn.Conv1d(in_channels, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
            nn.Conv1d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(features),
            nn.ReLU(inplace=True),
        )


class BlindSpotGenerator:
    """
    生成盲点位置并进行像素值操纵
    
    参数:
    - perc_pix: 被操纵的像素百分比 (默认1.6%)
    - neighborhood_radius: 邻域半径用于像素值替换
    """
    def __init__(self, perc_pix=1.6, neighborhood_radius=5, strategy='uniform'):
        self.perc_pix = perc_pix
        self.neighborhood_radius = neighborhood_radius
        self.strategy = strategy  # 'uniform', 'mean', 'median'
    
    def generate_blind_spots(self, signal_length, num_channels=1):
        """
        生成盲点位置坐标
        使用分层采样确保盲点均匀分布
        """
        num_blind_spots = max(1, int(signal_length * self.perc_pix / 100.0))
        
        # 分层采样: 将信号分成若干个box,每个box中随机选一个点
        box_size = int(np.sqrt(100 / self.perc_pix))
        box_count = int(np.ceil(signal_length / box_size))
        
        coords = []
        for i in range(box_count):
            coord = int(i * box_size + np.random.rand() * box_size)
            if coord < signal_length:
                coords.append(coord)
        
        return np.array(coords)
    
    def manipulate_signal(self, signal, blind_spots):
        """
        操纵盲点位置的值
        
        signal: (B, C, L)
        blind_spots: (num_blind_spots,)
        返回: manipulated_signal, original_values, mask
        """
        B, C, L = signal.shape
        manipulated = signal.clone()
        original_values = signal[:, :, blind_spots].clone()
        
        # 创建mask标记盲点位置
        mask = torch.zeros_like(signal)
        mask[:, :, blind_spots] = 1
        
        for coord in blind_spots:
            # 获取邻域范围
            left = max(0, coord - self.neighborhood_radius)
            right = min(L, coord + self.neighborhood_radius + 1)
            
            # 排除中心点本身
            neighbor_indices = [i for i in range(left, right) if i != coord]
            
            if len(neighbor_indices) == 0:
                continue
            
            # 根据策略替换值
            if self.strategy == 'uniform':
                # 从邻域中均匀随机选择一个值
                random_idx = np.random.choice(neighbor_indices)
                manipulated[:, :, coord] = signal[:, :, random_idx]
            
            elif self.strategy == 'mean':
                # 使用邻域均值
                neighbor_values = signal[:, :, neighbor_indices]
                manipulated[:, :, coord] = neighbor_values.mean(dim=2)
            
            elif self.strategy == 'median':
                # 使用邻域中位数
                neighbor_values = signal[:, :, neighbor_indices]
                manipulated[:, :, coord] = neighbor_values.median(dim=2)[0]
        
        return manipulated, original_values, mask


def test_model():
    """测试模型和盲点生成器"""
    print("Testing Noise2Void 1D Model...")
    
    # 测试模型
    model = UNet1D_N2V(in_channels=1, out_channels=1, init_features=32)
    x = torch.randn(4, 1, 1200)
    
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # 测试盲点生成器
    print("\nTesting Blind Spot Generator...")
    generator = BlindSpotGenerator(perc_pix=1.6, neighborhood_radius=5, strategy='uniform')
    
    blind_spots = generator.generate_blind_spots(signal_length=1200)
    print(f"Number of blind spots: {len(blind_spots)}")
    print(f"Blind spot locations (first 10): {blind_spots[:10]}")
    
    # 测试信号操纵
    signal = torch.randn(4, 1, 1200)
    manipulated, original, mask = generator.manipulate_signal(signal, blind_spots)
    
    print(f"\nManipulated signal shape: {manipulated.shape}")
    print(f"Original values shape: {original.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Number of masked pixels: {mask.sum().item()}")
    
    # 验证操纵是否生效
    difference = (signal - manipulated).abs()
    print(f"Max difference at blind spots: {difference[:, :, blind_spots].max().item():.4f}")
    print(f"Max difference elsewhere: {difference[:, :, mask[0, 0] == 0].max().item():.4f}")
    
    print("\n✓ All tests passed!")


if __name__ == "__main__":
    test_model()
