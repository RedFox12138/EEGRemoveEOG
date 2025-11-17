"""
DAT-Net 无监督训练脚本
Artifact-aware 自监督方法 Version 1
只需要受污染的EEG信号，不需要干净标签
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time


# 添加路径以导入相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
sys.path.insert(0, datnet_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from model import DATNet
from unsupervised_artifact_v1 import unsupervised_dat_loss_artifact_v1

# 导入评价指标（仅用于验证集评估，训练时不需要）
try:
    from 复现的方法.metrics_utils import compute_all_metrics, print_metrics
except:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass


# ========== 超参数配置 ==========
BATCH_SIZE = 256          # 增加batch size以提高训练稳定性
EPOCHS = 1000
LEARNING_RATE = 1e-3      # 提高初始学习率，配合warmup使用
WEIGHT_DECAY = 1e-5
SAMPLING_RATE = 200.0     # 采样率 Hz

# 学习率调度
USE_LR_SCHEDULER = True   # 是否使用学习率调度
WARMUP_EPOCHS = 50        # warmup轮数
MIN_LR = 1e-6            # 最小学习率

# 梯度裁剪
GRAD_CLIP = 1.0           # 梯度裁剪阈值，防止梯度爆炸

# 早停
PATIENCE = 150            # 早停的耐心值

# 无监督损失权重（优化后的权重）
LAMBDA_N2V = 1.0          # N2V 风格重建损失
LAMBDA_CONS = 0.5         # 降低全局一致性权重，避免过度约束
LAMBDA_TEACHER = 1.0      # 提高伪老师权重，增强频域先验
LAMBDA_BAND = 0.0         # 频带先验（可选）
LAMBDA_LOW = 0.0          # 低频先验（可选）
LAMBDA_DECOR = 0.0        # 解耦约束（可选）

# Artifact-aware 掩蔽参数（优化后）
MASK_BASE = 0.15          # 增加基础掩蔽比例
BOOST_SCALE = 0.4         # 增强伪影区域掩蔽
GAMMA_ART_WEIGHT = 2.0    # 提高伪影区域损失权重


class UnsupervisedEEGDataset(Dataset):
    """
    无监督数据集：只需要受污染信号
    可选地提供干净信号用于验证集评估
    """
    def __init__(self, noisy, clean=None):
        self.noisy = noisy
        self.clean = clean
        self.has_clean = clean is not None

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # 归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        if self.has_clean:
            clean = self.clean[idx]
            return (
                noisy.astype('float32') / norm,
                clean.astype('float32'),
                norm
            )
        else:
            return (
                noisy.astype('float32') / norm,
                norm
            )


def get_data():
    """加载训练和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    
    # 可选：加载干净标签用于验证（不用于训练）
    try:
        val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    except:
        val_y = None
    
    return train_x, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_n2v_sum = 0
    loss_cons_sum = 0
    loss_teacher_sum = 0
    num_batches = 0
    
    for batch_data in loader:
        if len(batch_data) == 3:
            noisy, _, norm = batch_data  # 忽略 clean（如果存在）
        else:
            noisy, norm = batch_data
        
        # 转换为tensor并移到设备
        noisy = noisy.float().unsqueeze(1).to(device)  # (B, 1, L)
        norm = norm.float().to(device).view(-1, 1, 1)
        
        # 恢复到原始尺度
        noisy_scaled = noisy * norm
        
        # 前向传播 + 无监督损失
        optimizer.zero_grad()
        
        total_loss_batch, loss_dict, eeg_clean_pred, eog_artifact_pred = \
            unsupervised_dat_loss_artifact_v1(
                model=model,
                eeg_raw_input=noisy_scaled,
                fs=SAMPLING_RATE,
                mask_base=MASK_BASE,
                boost_scale=BOOST_SCALE,
                lambda_n2v=LAMBDA_N2V,
                lambda_cons=LAMBDA_CONS,
                lambda_teacher=LAMBDA_TEACHER,
                lambda_band=LAMBDA_BAND,
                lambda_low=LAMBDA_LOW,
                lambda_decor=LAMBDA_DECOR,
                gamma_art_weight=GAMMA_ART_WEIGHT
            )
        
        # 反向传播
        total_loss_batch.backward()
        
        # 梯度裁剪，防止梯度爆炸
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        # 累积损失
        total_loss += loss_dict['total']
        loss_n2v_sum += loss_dict['n2v']
        loss_cons_sum += loss_dict['cons']
        loss_teacher_sum += loss_dict['teacher_clean'] + loss_dict['teacher_art']
        num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'n2v': loss_n2v_sum / num_batches,
        'cons': loss_cons_sum / num_batches,
        'teacher': loss_teacher_sum / num_batches
    }


def validate(model, device, loader, has_clean_labels=False):
    """验证模型"""
    model.eval()
    total_loss = 0
    loss_n2v_sum = 0
    loss_cons_sum = 0
    loss_teacher_sum = 0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_data in loader:
            if len(batch_data) == 3:
                noisy, clean, norm = batch_data
            else:
                noisy, norm = batch_data
                clean = None
            
            noisy = noisy.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            noisy_scaled = noisy * norm
            
            # 前向传播 + 无监督损失
            total_loss_batch, loss_dict, eeg_clean_pred, eog_artifact_pred = \
                unsupervised_dat_loss_artifact_v1(
                    model=model,
                    eeg_raw_input=noisy_scaled,
                    fs=SAMPLING_RATE,
                    mask_base=MASK_BASE,
                    boost_scale=BOOST_SCALE,
                    lambda_n2v=LAMBDA_N2V,
                    lambda_cons=LAMBDA_CONS,
                    lambda_teacher=LAMBDA_TEACHER,
                    lambda_band=LAMBDA_BAND,
                    lambda_low=LAMBDA_LOW,
                    lambda_decor=LAMBDA_DECOR,
                    gamma_art_weight=GAMMA_ART_WEIGHT
                )
            
            # 累积损失
            total_loss += loss_dict['total']
            loss_n2v_sum += loss_dict['n2v']
            loss_cons_sum += loss_dict['cons']
            loss_teacher_sum += loss_dict['teacher_clean'] + loss_dict['teacher_art']
            num_batches += 1
            
            # 如果有干净标签，收集用于计算评价指标
            if has_clean_labels and clean is not None:
                all_preds.append(eeg_clean_pred.squeeze(1).cpu().numpy())
                all_targets.append(clean.numpy())
    
    # 计算评价指标（如果有干净标签）
    metrics = None
    if has_clean_labels and len(all_targets) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    # 返回损失和指标
    loss_dict = {
        'total': total_loss / num_batches,
        'n2v': loss_n2v_sum / num_batches,
        'cons': loss_cons_sum / num_batches,
        'teacher': loss_teacher_sum / num_batches
    }
    
    return loss_dict, metrics


def main():
    print("="*70)
    print("DAT-Net 无监督训练")
    print("Artifact-aware Self-Supervised Version 1")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 加载数据
    print('\n加载数据...')
    train_x, val_x, val_y = get_data()
    print(f'训练集: {train_x.shape}')
    print(f'验证集: {val_x.shape}')
    
    train_dataset = UnsupervisedEEGDataset(train_x, clean=None)
    val_dataset = UnsupervisedEEGDataset(val_x, clean=val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    print('  架构: DAT-Net (1D U-Net + TCN + 双输出头)')
    print('  训练方法: Artifact-aware 无监督')
    print('  特点: 伪影概率估计 + 自适应掩蔽 + 伪老师引导')
    
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器和调度器
    print(f'\n训练配置:')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Initial Learning Rate: {LEARNING_RATE}')
    print(f'  Weight Decay: {WEIGHT_DECAY}')
    print(f'  优化器: Adam')
    if USE_LR_SCHEDULER:
        print(f'  LR调度: Warmup({WARMUP_EPOCHS}) + CosineAnnealing')
        print(f'  Min LR: {MIN_LR}')
    print(f'  梯度裁剪: {GRAD_CLIP}')
    print(f'  早停耐心值: {PATIENCE}')
    print(f'\n损失权重:')
    print(f'  λ_N2V: {LAMBDA_N2V} (N2V风格重建损失)')
    print(f'  λ_Consistency: {LAMBDA_CONS} (全局一致性)')
    print(f'  λ_Teacher: {LAMBDA_TEACHER} (伪老师引导)')
    print(f'\n掩蔽参数:')
    print(f'  基础掩蔽率: {MASK_BASE}')
    print(f'  伪影增强: {BOOST_SCALE}')
    print(f'  伪影权重: {GAMMA_ART_WEIGHT}')
    print(f'\n模型选择: 基于验证集无监督损失或CC')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器：Warmup + CosineAnnealingLR
    if USE_LR_SCHEDULER:
        def warmup_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                # Cosine annealing
                progress = (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS)
                return 0.5 * (1.0 + np.cos(np.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = None
    
    # 训练
    print('\n' + '='*70)
    print('开始训练')
    print('='*70)
    
    best_val_loss = float('inf')
    best_cc = -1.0
    patience_counter = 0
    start_time = time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-' * 70)
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        print(f'Train Loss: {train_loss["total"]:.6f} '
              f'(N2V: {train_loss["n2v"]:.6f}, '
              f'Cons: {train_loss["cons"]:.6f}, '
              f'Teacher: {train_loss["teacher"]:.6f})')
        
        # 验证
        val_loss, val_metrics = validate(model, device, val_loader, has_clean_labels=(val_y is not None))
        print(f'Val Loss: {val_loss["total"]:.6f} '
              f'(N2V: {val_loss["n2v"]:.6f}, '
              f'Cons: {val_loss["cons"]:.6f}, '
              f'Teacher: {val_loss["teacher"]:.6f})')
        
        if val_metrics is not None:
            print_metrics(val_metrics, prefix='验证集')
            current_cc = val_metrics.get('CC', -1.0)
        else:
            current_cc = -1.0
        
        # 保存最佳模型（根据验证损失或CC）
        save_model = False
        improved = False
        
        if val_metrics is not None and current_cc > best_cc:
            best_cc = current_cc
            save_model = True
            improved = True
            print(f'✓ 验证CC提升: {best_cc:.4f}')
        elif val_metrics is None and val_loss["total"] < best_val_loss:
            best_val_loss = val_loss["total"]
            save_model = True
            improved = True
            print(f'✓ 验证损失降低: {best_val_loss:.6f}')
        
        if save_model:
            torch.save(model.state_dict(), 'DAT-Net-Unsupervised_best.pth')
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6f}')
        
        elapsed = time() - start_time
        print(f'Elapsed Time: {int(elapsed//60)}min {int(elapsed%60)}s')
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f'\n早停触发！{PATIENCE} 个epoch内无改善。')
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'DAT-Net-Unsupervised_final.pth')
    
    print('\n' + '='*70)
    print('训练完成!')
    if val_metrics is not None:
        print(f'最佳CC: {best_cc:.4f}')
    else:
        print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'最佳模型已保存至: DAT-Net-Unsupervised_best.pth')
    print(f'最终模型已保存至: DAT-Net-Unsupervised_final.pth')
    print('='*70)


if __name__ == '__main__':
    main()
