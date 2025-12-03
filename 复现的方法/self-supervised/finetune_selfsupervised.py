"""
Self-Supervised EEG Denoising 微调脚本
使用20%训练数据进行有监督微调

微调流程:
1. 加载自监督训练的最佳模型
2. 使用20%的训练数据(带clean标签)进行有监督微调
3. 使用MSE损失在整个信号上
4. 保存微调后的最佳模型
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
parent_dir = os.path.dirname(current_dir)  # 复现的方法
sys.path.insert(0, parent_dir)

# 导入模型
from model_selfsupervised import DenoiseEEG

# 导入metrics
from metrics_utils import compute_all_metrics, print_metrics

# 导入数据集配置
from data_config import *


# ========== 超参数配置 ==========
BATCH_SIZE = 256
EPOCHS = 300
LEARNING_RATE = 5e-4  # 微调使用较小学习率

# 模型配置
INPUT_CHANNELS = 1
SEQ_LEN = WINDOW_SIZE  # 使用配置中的窗口大小
HIDDEN_DIM = 128


class SupervisedDataset(Dataset):
    """
    有监督数据集
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


def get_data():
    """
    加载20%训练数据和验证数据
    """
    # 加载完整训练集
    full_train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    full_train_y = scipy.io.loadmat(TRAIN_PURE_PATH)[DATA_KEY]
    
    # 取前20%数据
    num_samples = int(len(full_train_x) * 0.1)
    train_x = full_train_x[:num_samples]
    train_y = full_train_y[:num_samples]
    
    # 验证集
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    val_y = scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]
    
    return train_x, train_y, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    """
    有监督训练一个epoch
    注意：模型在归一化尺度上训练（与预训练阶段保持一致）
    """
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for noisy, clean, norm in loader:
        # 添加通道维度
        noisy = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 已归一化
        clean = clean.float().unsqueeze(1).to(device)  # (B, 1, L) - 已归一化
        
        optimizer.zero_grad()
        
        # 前向传播（输入归一化数据）
        output = model(noisy)
        
        # MSE损失（在归一化尺度上计算）
        loss = F.mse_loss(output, clean)
        
        loss.backward()
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def validate(model, device, loader):
    """
    在验证集上评估
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            # 添加通道维度
            noisy = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 已归一化
            clean_norm = clean.float().unsqueeze(1).to(device)  # (B, 1, L) - 已归一化
            
            # 前向传播（输入归一化数据）
            output = model(noisy)
            
            # 计算验证损失（在归一化尺度上）
            loss = F.mse_loss(output, clean_norm)
            total_loss += loss.item()
            num_batches += 1
            
            # 恢复原始尺度（用于计算评估指标）
            norm_t = norm.float().to(device).view(-1, 1, 1)
            output_denorm = output * norm_t
            
            # 收集预测结果（原始尺度）
            all_preds.append(output_denorm.squeeze(1).cpu().numpy())
            all_targets.append(clean.numpy())  # clean 已经是原始尺度（未归一化）
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    return total_loss / max(1, num_batches), metrics


def main():
    print('='*70)
    print('Self-Supervised EEG Denoising 微调')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)
    
    # 加载数据
    print('\n加载数据...')
    train_x, train_y, val_x, val_y = get_data()
    print(f'训练集样本数 (20%): {len(train_x)}')
    print(f'验证集样本数: {len(val_x)}')
    
    # 创建数据集和加载器
    train_dataset = SupervisedDataset(train_x, train_y)
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
    
    # 加载预训练模型
    pretrained_path = 'Self-Supervised_best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f'加载预训练模型: {pretrained_path}')
    else:
        print('⚠️ 未找到预训练模型，从随机初始化开始微调')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 训练循环
    best_rrmse = float('inf')
    patience = 1000
    patience_counter = 0
    
    print('\n开始微调...')
    print(f'微调轮数: {EPOCHS}')
    print(f'批次大小: {BATCH_SIZE}')
    print(f'学习率: {LEARNING_RATE}')
    print('-'*70)
    
    for epoch in range(EPOCHS):
        start_time = time()
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        
        # 验证
        val_loss, metrics = validate(model, device, val_loader)
        
        epoch_time = time() - start_time
        
        # 获取关键指标
        rrmse = metrics.get('RRMSE', float('inf'))
        cc = metrics.get('CC', 0.0)
        snr = metrics.get('SNR', 0.0)
        
        # 更新最佳模型
        if rrmse < best_rrmse:
            best_rrmse = rrmse
            patience_counter = 0
            torch.save(model.state_dict(), 'Self-Supervised_finetuned_best.pth')
            print(f"Epoch {epoch + 1}/{EPOCHS} - 保存最佳模型! (RRMSE: {rrmse:.4f})")
        else:
            patience_counter += 1
        
        print(f"Epoch {epoch + 1}/{EPOCHS} [{epoch_time:.1f}s] "
              f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
              f"RRMSE: {rrmse:.4f}, CC: {cc:.4f}, SNR: {snr:.2f} dB")
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'Self-Supervised_finetuned_final.pth')
    
    print('\n' + '='*70)
    print('微调完成!')
    print(f'最佳 RRMSE: {best_rrmse:.4f}')
    print('='*70)


if __name__ == '__main__':
    main()
