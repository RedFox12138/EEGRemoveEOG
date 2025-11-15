"""
Noise2Void训练脚本 - 适配用户的EEG数据集

训练流程:
1. 加载带噪声的EEG数据(只需要Contaminated数据)
2. 生成盲点位置并操纵像素值
3. 训练网络预测被操纵位置的原始值
4. 使用MSE损失在盲点位置
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import os
from model import UNet1D_N2V, BlindSpotGenerator


class N2VDataset(Dataset):
    """
    Noise2Void数据集
    只需要带噪声的数据,不需要干净目标
    """
    def __init__(self, data_dir, split='train'):
        # 加载数据
        if split == 'train':
            file_path = os.path.join(data_dir, 'Train_Contaminated.mat')
        elif split == 'val':
            file_path = os.path.join(data_dir, 'Val_Contaminated.mat')
        else:
            raise ValueError(f"Unknown split: {split}")
        
        data = sio.loadmat(file_path)
        self.signals = data['data'].astype(np.float32)
        
        print(f"Loaded {split} data shape: {self.signals.shape}")
        
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = self.signals[idx]
        # 返回: (C, L) 格式
        signal = signal[np.newaxis, :]  # (1, 1200)
        return torch.from_numpy(signal)


def train_epoch(model, train_loader, optimizer, device, blind_spot_generator):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, signals in enumerate(train_loader):
        signals = signals.to(device)  # (B, 1, 1200)
        B, C, L = signals.shape
        
        optimizer.zero_grad()
        
        # 为batch中的每个样本生成不同的盲点
        batch_loss = 0
        for i in range(B):
            signal_i = signals[i:i+1]  # (1, 1, 1200)
            
            # 生成盲点
            blind_spots = blind_spot_generator.generate_blind_spots(L, C)
            
            # 操纵信号
            manipulated, original_values, mask = blind_spot_generator.manipulate_signal(
                signal_i, blind_spots
            )
            
            # 前向传播
            predicted = model(manipulated)  # (1, 1, 1200)
            
            # 计算损失:只在盲点位置
            # 提取盲点位置的预测值和真实值
            predicted_blind = predicted[:, :, blind_spots]  # (1, 1, num_blind_spots)
            original_blind = signal_i[:, :, blind_spots]    # (1, 1, num_blind_spots)
            
            loss = F.mse_loss(predicted_blind, original_blind)
            batch_loss += loss
        
        # 平均batch loss
        batch_loss = batch_loss / B
        batch_loss.backward()
        optimizer.step()
        
        total_loss += batch_loss.item()
        
        if batch_idx % 10 == 0:
            print(f'Batch [{batch_idx}/{len(train_loader)}], Loss: {batch_loss.item():.6f}')
    
    return total_loss / len(train_loader)


def validate(model, val_loader, device, blind_spot_generator):
    """验证"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for signals in val_loader:
            signals = signals.to(device)
            B, C, L = signals.shape
            
            batch_loss = 0
            for i in range(B):
                signal_i = signals[i:i+1]
                
                # 生成盲点
                blind_spots = blind_spot_generator.generate_blind_spots(L, C)
                
                # 操纵信号
                manipulated, original_values, mask = blind_spot_generator.manipulate_signal(
                    signal_i, blind_spots
                )
                
                # 前向传播
                predicted = model(manipulated)
                
                # 计算损失
                predicted_blind = predicted[:, :, blind_spots]
                original_blind = signal_i[:, :, blind_spots]
                
                loss = F.mse_loss(predicted_blind, original_blind)
                batch_loss += loss
            
            batch_loss = batch_loss / B
            total_loss += batch_loss.item()
    
    return total_loss / len(val_loader)


def train_model():
    """主训练函数"""
    # 设置参数
    data_dir = r"D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据"
    save_dir = "./checkpoints"
    os.makedirs(save_dir, exist_ok=True)
    
    # 超参数
    batch_size = 16
    learning_rate = 0.0004
    num_epochs = 100
    
    # Noise2Void参数
    perc_pix = 1.6  # 盲点百分比
    neighborhood_radius = 5  # 邻域半径
    manipulator_strategy = 'uniform'  # 'uniform', 'mean', 'median'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载数据
    print("Loading data...")
    train_dataset = N2VDataset(data_dir, split='train')
    val_dataset = N2VDataset(data_dir, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 创建模型
    print("Creating model...")
    model = UNet1D_N2V(in_channels=1, out_channels=1, init_features=32)
    model = model.to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # 优化器（不使用学习率衰减）
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)  # 已禁用
    
    # 盲点生成器
    blind_spot_generator = BlindSpotGenerator(
        perc_pix=perc_pix,
        neighborhood_radius=neighborhood_radius,
        strategy=manipulator_strategy
    )
    
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Blind spot percentage: {perc_pix}%")
    print(f"  Neighborhood radius: {neighborhood_radius}")
    print(f"  Manipulator strategy: {manipulator_strategy}")
    print(f"  Expected blind spots per sample: ~{int(1200 * perc_pix / 100)}")
    
    # 训练循环
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"{'='*60}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, device, blind_spot_generator)
        print(f"Train Loss: {train_loss:.6f}")
        
        # 验证
        val_loss = validate(model, val_loader, device, blind_spot_generator)
        print(f"Val Loss: {val_loss:.6f}")
        
        # 学习率调整（已禁用衰减，保持恒定学习率）
        # scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f} (固定)")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.6f})")
        
        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print("="*60)


if __name__ == "__main__":
    train_model()
