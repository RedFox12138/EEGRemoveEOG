"""
Self-Supervised EEG Denoising 训练脚本
基于 Self-Supervised-EEG-Denoising-main 的模型架构
使用半模拟数据集进行训练
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

# 添加路径以导入metrics
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, os.path.join(root_dir, '复现的方法'))

# 导入模型和工具
from model_selfsupervised import DenoiseEEG
from utils_selfsupervised import get_pearson_correlation, get_snr, trrmse_metric, frrmse_metric

# 导入metrics
try:
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {'RRMSE':0,'CC':0,'RRMSE_PSD':0,'MI':0}
    def print_metrics(m, prefix=""): pass


# ========== 超参数配置（与原始Self-Supervised一致）==========
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
MASK_RATIO = 0.1  # 掩码比例（与原始一致）
SAMPLING_RATE = 200.0

# 我们的数据尺寸
INPUT_CHANNELS = 1  # 单通道EEG
SEQ_LEN = 1200      # 序列长度（6秒 * 200Hz）
HIDDEN_DIM = 128    # 隐藏层维度


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
        
        # 归一化（与DAT-Net-v2一致）
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_normalized = noisy.astype('float32') / norm
        
        return torch.tensor(noisy_normalized, dtype=torch.float32), norm


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
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / norm, dtype=torch.float32)
        clean_norm = torch.tensor(clean.astype('float32') / norm, dtype=torch.float32)
        
        return noisy_norm, clean_norm, norm


def mask_input(data, mask_ratio=0.3):
    """
    对输入信号进行随机掩码（与原始Self-Supervised一致）
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
    """
    掩码自编码器损失（与原始Self-Supervised一致）
    """
    reconstructed = reconstructed.to(mask.device)
    original = original.to(mask.device)
    time_loss = F.mse_loss(reconstructed, original, reduction='mean')
    return time_loss


def train_epoch(model, device, loader, optimizer, mask_ratio):
    """
    训练一个epoch（与原始Self-Supervised一致）
    """
    model.train()
    total_loss = 0
    
    for data, norm in loader:
        data = data.to(device)
        
        # 添加通道维度 (B, L) -> (B, 1, L)
        if len(data.shape) == 2:
            data = data.unsqueeze(1)
        
        # 恢复原始尺度
        norm = norm.float().to(device).view(-1, 1, 1)
        data_scaled = data * norm
        
        optimizer.zero_grad()
        
        # 创建掩码
        masked_data, mask = mask_input(data_scaled, mask_ratio)
        
        # 模型前向传播
        exp_output = model(data_scaled)
        reconstructed = model(masked_data)
        
        # 损失计算（与原始一致）
        loss = mae_loss(exp_output, reconstructed, mask) + F.mse_loss(exp_output, data_scaled)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
    
    return total_loss / len(loader)


def validate(model, device, loader):
    """
    验证模型性能（计算完整的评估指标）
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): 
        for data, clean, norm in loader:
            # 添加通道维度
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            if len(clean.shape) == 2:
                clean = clean.unsqueeze(1)
            
            x, y = data.to(device), clean.to(device)
            
            # 恢复原始尺度
            norm = norm.float().to(device).view(-1, 1, 1)
            x = x * norm
            y = y * norm
            
            # 前向传播
            reconstructed = model(x)

            # 计算损失
            loss = F.mse_loss(reconstructed, y)
            total_loss += loss.item()

            # 收集预测和目标（用于计算完整指标）
            # 预测和目标都已经是原始尺度，直接使用
            all_preds.append(reconstructed.squeeze(1).cpu().numpy())
            all_targets.append(y.squeeze(1).cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算完整的评估指标
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    
    return avg_loss, metrics


def get_data():
    """
    加载数据（与DAT-Net-v2一致）
    """
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 训练集只需要污染数据（自监督）
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    
    # 验证集需要干净标签（用于评估）
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, val_x, val_y


def main():
    print('='*70)
    print('Self-Supervised EEG Denoising 训练')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)
    
    # 加载数据
    print('\n加载数据...')
    train_x, val_x, val_y = get_data()
    print(f'训练集样本数: {len(train_x)}')
    print(f'验证集样本数: {len(val_x)}')
    print(f'数据维度: {train_x.shape}')
    
    # 创建数据集和加载器
    train_dataset = SelfSupervisedDataset(train_x)
    val_dataset = SupervisedDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    model = DenoiseEEG(
        in_channels=INPUT_CHANNELS,
        length=SEQ_LEN,
        n_feat=HIDDEN_DIM
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    best_rrmse = float('inf')
    best_metrics = {}
    patience = 50
    patience_counter = 0
    start_training_time = time()
    
    print('\n开始训练...')
    print(f'训练轮数: {EPOCHS}')
    print(f'批次大小: {BATCH_SIZE}')
    print(f'学习率: {LEARNING_RATE}')
    print(f'掩码比例: {MASK_RATIO}')
    print('='*70)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time()
        
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-'*70)
        
        # 训练
        avg_loss = train_epoch(model, device, train_loader, optimizer, MASK_RATIO)
        print(f'Train Loss: {avg_loss:.6f}')
        
        # 验证
        val_loss, val_metrics = validate(model, device, val_loader)
        
        print(f'Val Loss:   {val_loss:.6f}')
        
        # 打印详细的验证指标
        if val_metrics:
            rrmse = val_metrics.get('RRMSE', 0.0)
            cc = val_metrics.get('CC', 0.0)
            snr = val_metrics.get('SNR', 0.0)
            prd = val_metrics.get('PRD', 0.0)
            
            print(f'  RRMSE: {rrmse:.6f}  CC: {cc:.6f}')
            print(f'  SNR:   {snr:.4f} dB  PRD: {prd:.6f}')
            
            # 更新最佳模型（基于RRMSE）
            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_metrics = val_metrics.copy()
                patience_counter = 0
                torch.save(model.state_dict(), 'Self-Supervised_best.pth')
                print(f'  ✓ 保存最佳模型! (RRMSE: {rrmse:.6f})')
            else:
                patience_counter += 1
        
        epoch_time = time() - epoch_start
        elapsed = time() - start_training_time
        print(f'Epoch Time: {epoch_time:.1f}s  |  Elapsed: {int(elapsed//60)}min {int(elapsed%60)}s')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\n早停触发！{patience} 个epoch内无改善。')
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'Self-Supervised_final.pth')
    
    print('\n' + '='*70)
    print('训练完成!')
    if best_metrics:
        print('\n最佳验证指标:')
        print_metrics(best_metrics, prefix='  ')
    print('='*70)


if __name__ == '__main__':
    main()
