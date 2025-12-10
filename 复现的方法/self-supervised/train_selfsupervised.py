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
parent_dir = os.path.dirname(current_dir) # 复现的方法
sys.path.insert(0, parent_dir)

# 导入模型和工具
from model_selfsupervised import DenoiseEEG
from utils_selfsupervised import get_pearson_correlation, get_snr, trrmse_metric, frrmse_metric

# 导入metrics
from metrics_utils import compute_all_metrics, print_metrics

# 导入数据集配置
from data_config import *


# ========== 超参数配置（与原始Self-Supervised一致）==========
BATCH_SIZE = 128
EPOCHS = 300
LEARNING_RATE = 1e-4
MASK_RATIO = 0.4  # 增大掩码比例，强迫模型更多地进行预测而非复制

# 我们的数据尺寸（从data_config中获取，适配不同数据集）
INPUT_CHANNELS = 1  # 单通道EEG
SEQ_LEN = WINDOW_SIZE  # 序列长度（从配置文件获取，自动适配）
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


def mask_input(data, mask_ratio=0.5, block_size=20):
    """
    块掩码 (Block Masking) 策略
    不只是随机遮挡单个点，而是遮挡连续的时间片段。
    这能防止模型简单地通过相邻点插值来重构低频伪影(EOG)。
    
    Args:
        data: (B, C, L)
        mask_ratio: 总体掩码比例
        block_size: 每个块的长度 (点数)
    """
    batch_size, num_channels, seq_len = data.shape
    
    # 创建一个全0的mask
    mask = torch.zeros(batch_size, num_channels, seq_len, device=data.device, dtype=torch.bool)
    
    # 计算需要掩码的总点数
    num_mask_points = int(seq_len * mask_ratio)
    # 计算需要多少个块
    num_blocks = num_mask_points // block_size
    
    for b in range(batch_size):
        for c in range(num_channels):
            # 随机选择块的起始位置
            start_indices = torch.randint(0, seq_len - block_size, (num_blocks,))
            for start_idx in start_indices:
                mask[b, c, start_idx : start_idx + block_size] = True
                
    masked_data = data.clone()
    
    # 使用 0 填充 (对于块掩码，0填充通常比噪声填充更难，迫使模型生成结构)
    # 或者使用均值填充
    masked_data[mask] = 0
    
    return masked_data, mask


def mae_loss(reconstructed, original, mask):
    """
    掩码自编码器损失（修正版）
    只计算被掩码位置的损失，强迫模型利用上下文信息进行预测
    """
    reconstructed = reconstructed.to(mask.device)
    original = original.to(mask.device)
    
    # 只计算 mask 为 True (被遮挡) 的部分的 loss
    # mask 是 bool 类型，可以直接作为索引
    loss = F.mse_loss(reconstructed[mask], original[mask])
    return loss


def train_epoch(model, device, loader, optimizer, mask_ratio):
    """
    训练一个epoch
    """
    model.train()
    total_loss = 0
    
    for data, norm in loader:
        data = data.to(device)
        
        # 添加通道维度 (B, L) -> (B, 1, L)
        if len(data.shape) == 2:
            data = data.unsqueeze(1)
        
        optimizer.zero_grad()
        
        # 创建掩码 (使用块掩码，block_size=20 对应 100ms @ 200Hz)
        masked_data, mask = mask_input(data, mask_ratio, block_size=20)
        
        # 模型前向传播
        # 按照论文 Figure 2(a) 实现双分支训练
        
        # Branch 1: Masked Input -> Consistency Loss
        pred_masked = model(masked_data)
        
        # Branch 2: Raw Input -> Reconstruction Loss
        pred_raw = model(data)
        
        # 损失计算 (Eq. 9)
        # L_rec: 原始输入的重构误差 (Eq. 7)
        # 由于我们移除了 Skip Connection，模型无法简单复制输入，
        # 必须通过瓶颈层，这会自然过滤掉高频噪声 (Deep Image Prior 效应)
        loss_rec = F.mse_loss(pred_raw, data)
        
        # L_con: 一致性损失 (Eq. 8)
        # 强迫 Masked 输入的输出与 Raw 输入的输出一致
        loss_con = F.mse_loss(pred_masked, pred_raw)
        
        # 总损失
        # lambda 参数论文中说是自适应的，这里我们取一个经验值，或者按论文设为 1.0 左右
        # 论文中 lambda 用于平衡两者，通常取决于 SNR。
        # 增大 lambda_val，强迫模型更关注一致性（去噪），而不是重构（复制）
        lambda_val = 5.0 
        loss = loss_rec + lambda_val * loss_con
        
        total_loss += loss.item()

        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
            
            # 前向传播 (输入归一化数据)
            reconstructed = model(x)

            # 计算验证 Loss
            # 1. 去噪 Loss (与 Clean 信号对比) - 这是我们最终关心的
            loss_denoise = F.mse_loss(reconstructed, y)
            
            # 2. 重构 Loss (与 Noisy 信号对比) - 这反映了模型是否学会了"复制"
            loss_recon = F.mse_loss(reconstructed, x)
            
            total_loss += loss_denoise.item()

            # 恢复原始尺度 (用于计算评估指标)
            norm = norm.float().to(device).view(-1, 1, 1)
            reconstructed_denorm = reconstructed * norm
            y_denorm = y * norm

            # 收集预测和目标（用于计算完整指标）
            all_preds.append(reconstructed_denorm.squeeze(1).cpu().numpy())
            all_targets.append(y_denorm.squeeze(1).cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算完整的评估指标
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    
    # 额外返回一个重构 Loss 的均值，用于诊断
    # 注意：这里我们没有累加 loss_recon，所以只能粗略估计，或者你可以修改代码累加它
    # 为了简单，我们只返回 avg_loss (denoise)
    
    return avg_loss, metrics


def get_data():
    """
    加载数据（与DAT-Net-v2一致）

    """
    # 训练集只需要污染数据（自监督）
    train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    
    # 验证集需要干净标签（用于评估）
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    val_y = scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]
    
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
    # 降低学习率，防止震荡
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    # 训练循环
    best_rrmse = float('inf')
    best_metrics = {}
    patience = 200
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
