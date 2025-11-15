"""
Self-Supervised EEG Denoising 训练脚本
使用原始的Self-Supervised模型架构
在我们自己的数据集上训练
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

# 添加路径导入Self-Supervised模型
current_dir = os.path.dirname(os.path.abspath(__file__))
selfsupervised_dir = os.path.join(current_dir, '..', 'Self-Supervised-EEG-Denoising-main')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

sys.path.insert(0, selfsupervised_dir)
sys.path.insert(0, project_root)


from 复现的方法.metrics_utils import compute_all_metrics, print_metrics
from  model import DenoiseEEG

# ========== 超参数配置 ==========
BATCH_SIZE = 128
EPOCHS = 200
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
MASK_RATIO = 0.3  # 掩码比例


class SelfSupervisedDataset(Dataset):
    """
    自监督训练数据集
    只需要受污染的EEG信号，不需要干净标签
    """
    def __init__(self, noisy_data):
        self.noisy = noisy_data

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # 归一化到[-1, 1]
        max_val = np.max(np.abs(noisy))
        if max_val == 0:
            max_val = 1.0
        
        noisy_normalized = noisy.astype('float32') / max_val
        
        return torch.tensor(noisy_normalized, dtype=torch.float32), max_val


class SupervisedDataset(Dataset):
    """
    监督验证数据集
    用于评估模型去伪影效果
    """
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


def mask_input(data, mask_ratio=0.3):
    """
    对输入信号进行随机掩码
    Args:
        data: (B, C, L) 输入信号
        mask_ratio: 掩码比例
    Returns:
        masked_data: 掩码后的信号
        mask: 掩码位置 (B, C, L)
    """
    batch_size, num_channels, seq_len = data.shape
    mask = torch.rand(batch_size, num_channels, seq_len, device=data.device) < mask_ratio
    masked_data = data.clone()
    masked_data[mask] = 0  # 将掩码位置设为0
    return masked_data, mask


def mae_loss(reconstructed, original, mask):
    """MAE损失函数"""
    time_loss = F.mse_loss(reconstructed, original, reduction='mean')
    return time_loss


def get_data():
    """加载训练和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 训练集：只加载受污染信号（自监督）
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    
    # 验证集：加载受污染和干净信号（监督评估）
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, val_x, val_y


def train_epoch(model, device, train_loader, optimizer, mask_ratio):
    """自监督训练一个epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    for data, _ in train_loader:
        data = data.unsqueeze(1).to(device)  # (B, 1, L)
        optimizer.zero_grad()
        
        # 创建掩码输入
        masked_data, mask = mask_input(data, mask_ratio)
        
        # 前向传播
        exp_output = model(data)  # 从原始数据预测
        reconstructed = model(masked_data)  # 从掩码数据重建
        
        # 损失函数
        loss = mae_loss(exp_output, reconstructed, mask) + F.mse_loss(exp_output, data)
        
        total_loss += loss.item()
        
        # 反向传播
        loss.backward()
        optimizer.step()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, device, val_loader):
    """
    在验证集上评估（监督方式）
    计算实际去伪影效果的指标
    """
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, norm in val_loader:
            noisy = noisy.unsqueeze(1).to(device)
            clean = clean.to(device)
            
            # 前向传播
            output = model(noisy)
            
            # 恢复到原始尺度
            output_scaled = output.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
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
    print("Self-Supervised EEG Denoising 训练")
    print("使用原始Self-Supervised模型 + 我们的数据集")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 加载数据
    print('\n加载数据...')
    train_x, val_x, val_y = get_data()
    print(f'训练集（仅受污染信号）: {train_x.shape}')
    print(f'验证集: {val_x.shape}')
    
    # 创建数据集
    train_dataset = SelfSupervisedDataset(train_x)
    val_dataset = SupervisedDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    print('  架构: DenoiseEEG (U-Net + Transformer + Linear Attention)')
    print('  训练方式: 掩码重建（Masked Autoencoding）')
    print(f'  掩码比例: {MASK_RATIO*100:.0f}%')
    
    model = DenoiseEEG(
        in_channels=1,
        length=1200,  # 你的数据集信号长度
        n_feat=128
    ).to(device)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    
    # 优化器
    print(f'\n训练配置:')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning Rate: {LEARNING_RATE}')
    print(f'  Weight Decay: {WEIGHT_DECAY}')
    print(f'  优化器: Adam')
    print(f'  损失函数: MAE(exp_output, reconstructed) + MSE(exp_output, data)')
    
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
        train_loss = train_epoch(model, device, train_loader, optimizer, MASK_RATIO)
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
                torch.save(model.state_dict(), 'SelfSupervised_best.pth')
                print(f'✓ 保存最佳模型 (CC: {best_cc:.4f}, RRMSE: {best_rrmse:.4f})')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'SelfSupervised_final.pth')
    
    elapsed_time = time() - start_time
    print('\n' + '='*70)
    print(f'训练完成！总用时: {elapsed_time/60:.2f} 分钟')
    print(f'最佳验证指标:')
    print(f'  CC: {best_cc:.4f}')
    print(f'  RRMSE: {best_rrmse:.4f}')
    print('='*70)


if __name__ == '__main__':
    main()
