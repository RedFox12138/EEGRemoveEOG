"""
Self2Self训练脚本 - 应用到EEG去噪
核心思想：使用Dropout机制，从单个噪声样本中学习去噪
每次前向传播时Dropout会产生不同的子网络，实现自监督学习
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from time import time

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import UNet1D_Self2Self



# ========== 超参数配置 ==========
BATCH_SIZE = 64
EPOCHS = 200
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.3  # Self2Self的核心参数
N_ENSEMBLE = 10  # 测试时的集成次数


class Self2SelfDataset(Dataset):
    """
    Self2Self训练数据集
    只需要噪声数据，不需要干净标签
    """
    def __init__(self, noisy_data):
        self.noisy = noisy_data

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # 归一化
        max_val = np.max(np.abs(noisy))
        if max_val == 0:
            max_val = 1.0
        
        noisy_normalized = noisy.astype('float32') / max_val
        
        return torch.tensor(noisy_normalized, dtype=torch.float32), max_val


class SupervisedDataset(Dataset):
    """监督验证数据集"""
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化
        max_val = np.max(np.abs(noisy))
        if max_val == 0:
            max_val = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / max_val, dtype=torch.float32)
        clean_tensor = torch.tensor(clean.astype('float32'), dtype=torch.float32)
        
        return noisy_norm, clean_tensor, max_val


def get_data():
    """加载训练和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 训练集：只加载受污染信号（自监督）
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    
    # 验证集：加载受污染和干净信号（监督评估）
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, val_x, val_y


def train_epoch(model, device, train_loader, optimizer):
    """
    Self2Self训练一个epoch
    核心：通过Dropout创建随机掩码，让模型从部分观测预测完整信号
    这更接近原始Self2Self的实现
    """
    model.train()  # 确保Dropout处于激活状态
    total_loss = 0
    num_batches = 0
    
    for data, _ in train_loader:
        data = data.unsqueeze(1).to(device)  # (B, 1, L)
        optimizer.zero_grad()
        
        # Self2Self核心思想：
        # 1. 通过Dropout，每次前向传播看到不同的子网络
        # 2. 让两个不同子网络的输出保持一致
        # 3. 这迫使网络学习数据的底层结构，而不是拟合噪声
        
        output1 = model(data)
        output2 = model(data)
        
        # J-invariance loss: 两次输出应该一致
        loss = F.mse_loss(output1, output2)
        
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, device, val_loader):
    """
    在验证集上评估
    测试时使用多次前向传播的平均（Monte Carlo Dropout）
    """
    model.train()  # 注意：Self2Self测试时也保持Dropout激活
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, norm in val_loader:
            noisy = noisy.unsqueeze(1).to(device)
            clean = clean.to(device)
            
            # 多次前向传播求平均（集成）
            ensemble_output = torch.zeros_like(noisy)
            for _ in range(N_ENSEMBLE):
                ensemble_output += model(noisy)
            ensemble_output /= N_ENSEMBLE
            
            # 恢复到原始尺度
            output_scaled = ensemble_output.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
            clean_np = clean.cpu().numpy()
            
            all_preds.append(output_scaled)
            all_targets.append(clean_np)
    
    # 拼接所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评价指标
    metrics = compute_all_metrics(all_preds, all_targets, fs=200)
    
    return metrics


def main():
    print("="*70)
    print("Self2Self EEG Denoising 训练")
    print("Self-Supervised Learning with Dropout Ensemble")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 加载数据
    print('\n加载数据...')
    train_x, val_x, val_y = get_data()
    print(f'训练集（仅受污染信号）: {train_x.shape}')
    print(f'验证集: {val_x.shape}')
    
    # 创建数据集
    train_dataset = Self2SelfDataset(train_x)
    val_dataset = SupervisedDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    print('  架构: UNet1D with Dropout')
    print('  训练方式: Self2Self (Dropout Ensemble)')
    print(f'  Dropout率: {DROPOUT_RATE}')
    print(f'  测试集成次数: {N_ENSEMBLE}')
    
    model = UNet1D_Self2Self(
        in_channels=1,
        base_channels=64,
        dropout_rate=DROPOUT_RATE
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器
    print(f'\n训练配置:')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning Rate: {LEARNING_RATE}')
    print(f'  Weight Decay: {WEIGHT_DECAY}')
    print(f'  优化器: Adam')
    print(f'  损失函数: MSE(output1, output2) + 0.5*MSE(output, input)')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 训练循环
    print('\n' + '='*70)
    print('开始训练')
    print('='*70)
    
    best_cc = -1.0
    best_rrmse = float('inf')
    start_time = time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-' * 70)
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        print(f'Train Loss: {train_loss:.6f}')
        
        # 验证（每10个epoch或第1个epoch）
        if epoch % 10 == 0 or epoch == 1:
            print('\n验证（监督指标）:')
            val_metrics = validate(model, device, val_loader)
            print_metrics(val_metrics)
            
            # 保存最佳模型
            if val_metrics['CC'] > best_cc:
                best_cc = val_metrics['CC']
                best_rrmse = val_metrics['RRMSE']
                torch.save(model.state_dict(), 'Self2Self_best.pth')
                print(f'✓ 保存最佳模型 (CC: {best_cc:.4f}, RRMSE: {best_rrmse:.4f})')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'Self2Self_final.pth')
    
    elapsed_time = time() - start_time
    print('\n' + '='*70)
    print(f'训练完成！总用时: {elapsed_time/60:.2f} 分钟')
    print(f'最佳验证指标:')
    print(f'  CC: {best_cc:.4f}')
    print(f'  RRMSE: {best_rrmse:.4f}')
    print('='*70)


if __name__ == '__main__':
    main()
