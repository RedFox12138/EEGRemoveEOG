"""
Self2Self 1D EEG Denoising 训练脚本
基于Self2Self的无监督训练方法（使用Dropout掩蔽）
"""
import os
import sys
from pathlib import Path
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加路径
current_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(current_dir))

# 导入数据配置
from data_config import *

# 确保当前目录存在（处理符号链接等特殊情况）
current_dir.mkdir(parents=True, exist_ok=True)

# 尝试导入metrics
try:
    parent_dir = current_dir.parent
    sys.path.insert(0, str(parent_dir))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {'RRMSE':0,'CC':0,'RRMSE_PSD':0,'MI':0}
    def print_metrics(m, prefix=""): pass

from s2s_model_1d import Self2Self_UNet1D, self2self_loss


# ========== 超参数配置 ==========
BATCH_SIZE = 32
EPOCHS = 2000
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 0.0

# 学习率调度
USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 10
MIN_LR = 1e-6

# 训练配置
GRAD_CLIP = 1.0
PATIENCE = 100

# Self2Self参数
DROPOUT_RATE = 0.3      # Dropout概率（掩蔽比例）
N_PREDICTIONS = 100     # 推理时的预测次数

# 模型参数
BASE_CHANNELS = 48      # 基础通道数
N_DEPTH = 5             # UNet深度


class EEGDataset(Dataset):
    """EEG数据集（无监督训练）"""
    def __init__(self, noisy, clean=None):
        self.noisy = noisy
        self.clean = clean
        self.has_clean = clean is not None
    
    def __len__(self):
        return len(self.noisy)
    
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # 归一化到[0, 1]范围（Self2Self使用sigmoid输出）
        norm_min = np.min(noisy)
        norm_max = np.max(noisy)
        norm_range = norm_max - norm_min
        
        if norm_range == 0:
            norm_range = 1.0
        
        noisy_norm = (noisy - norm_min) / norm_range
        
        if self.has_clean:
            clean = self.clean[idx]
            # clean也使用相同的归一化参数（基于noisy的min/max）
            clean_norm = (clean - norm_min) / norm_range
            return (torch.from_numpy(noisy_norm).float(), 
                   torch.from_numpy(clean_norm).float(),
                   norm_min, norm_range)
        else:
            return torch.from_numpy(noisy_norm).float(), norm_min, norm_range


def get_data():
    """加载训练和验证数据"""
    train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    try:
        val_y = scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]
    except Exception:
        val_y = None
    return train_x, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        if len(batch) == 4:
            noisy, clean, norm_min, norm_range = batch
        else:
            noisy, norm_min, norm_range = batch
        
        noisy = noisy.unsqueeze(1).to(device)  # (B, 1, T)
        
        optimizer.zero_grad()
        
        # Self2Self前向传播（带掩蔽）
        pred, mask = model.forward_with_mask(noisy)
        
        # 计算Self2Self损失（只在掩蔽位置）
        loss = self2self_loss(pred, noisy, mask)
        
        # 反向传播
        loss.backward()
        
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def validate(model, device, loader, has_clean_labels=False, n_predictions=None):
    """
    验证模型
    
    Parameters:
    -----------
    n_predictions : int or None
        推理时使用的预测次数。
        - None: 使用单次前向传播（快速，但精度略低）
        - int: 使用多次预测平均（慢，但精度高）
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    # 根据是否需要计算指标，决定使用快速模式还是完整模式
    use_averaging = (n_predictions is not None) and has_clean_labels
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                noisy, clean, norm_min, norm_range = batch
            else:
                noisy, norm_min, norm_range = batch
                clean = None
            
            noisy = noisy.unsqueeze(1).to(device)
            
            # 计算验证损失（使用单次前向传播）
            pred, mask = model.forward_with_mask(noisy)
            loss = self2self_loss(pred, noisy, mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 如果需要计算完整指标，使用多次预测平均
            if use_averaging and clean is not None:
                # 使用多次预测平均（更准确但更慢）
                pred_avg = model.predict_average(noisy, n_predictions=n_predictions)
                
                # 反归一化到原始尺度
                pred_denorm = pred_avg.squeeze(1).cpu().numpy()
                pred_denorm = pred_denorm * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
                
                # clean也反归一化（它在数据集中已经被归一化了）
                clean_denorm = clean.numpy() * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
                
                all_preds.append(pred_denorm)
                all_targets.append(clean_denorm)
    
    avg_loss = total_loss / max(1, num_batches)
    
    # 计算指标
    metrics = None
    if has_clean_labels and len(all_targets) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    return avg_loss, metrics


def main():
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # 创建 checkpoints 目录（使用相对路径）
    os.makedirs('checkpoints', exist_ok=True)
    
    print('='*70)
    print('Self2Self 1D EEG Denoising 训练')
    print('='*70)
    print(f'工作目录: {os.getcwd()}')
    print(f'模型保存目录: checkpoints/')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    train_x, val_x, val_y = get_data()
    print(f'\n数据集信息:')
    print(f'  训练集: {train_x.shape}')
    print(f'  验证集: {val_x.shape}')
    if val_y is not None:
        print(f'  验证标签: {val_y.shape}')
    
    # 创建数据集
    train_dataset = EEGDataset(train_x, clean=None)
    val_dataset = EEGDataset(val_x, clean=val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # 创建模型
    print(f'\n创建模型...')
    print(f'模型保存路径: {current_dir}')
    print(f'路径是否存在: {current_dir.exists()}')
    
    model = Self2Self_UNet1D(
        in_channels=1,
        base_channels=BASE_CHANNELS,
        n_depth=N_DEPTH,
        dropout=DROPOUT_RATE
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                          weight_decay=WEIGHT_DECAY)
    
    if USE_LR_SCHEDULER:
        def warmup_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
                return max(MIN_LR / LEARNING_RATE, 
                          0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = None
    
    # 训练循环
    print(f'\n开始训练...')
    print(f'超参数配置:')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning Rate: {LEARNING_RATE}')
    print(f'  Dropout Rate: {DROPOUT_RATE}')
    print(f'  N Predictions: {N_PREDICTIONS}')
    print(f'  Base Channels: {BASE_CHANNELS}')
    print(f'  UNet Depth: {N_DEPTH}')
    print('='*70)
    
    best_val_loss = float('inf')
    best_metrics = None
    patience_counter = 0
    start_time = time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-'*70)
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        print(f'Train Loss: {train_loss:.6f}')
        
        # 每轮都验证，但只在特定epoch计算完整指标
        compute_full_metrics = (epoch % 5 == 0 or epoch == 1)
        n_pred = N_PREDICTIONS if compute_full_metrics else None
        
        val_loss, val_metrics = validate(model, device, val_loader, 
                                        has_clean_labels=(val_y is not None),
                                        n_predictions=n_pred)
        print(f'Val Loss:   {val_loss:.6f}')
        
        if val_metrics is not None:
            print_metrics(val_metrics, prefix='验证集')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if val_metrics is not None:
                best_metrics = val_metrics
            print(f'\n✓ 验证损失降低: {best_val_loss:.6f}')
            
            # 使用相对路径保存模型
            save_path = f'checkpoints/Self2Self_{DATASET_NAME}_best.pth'
            torch.save(model.state_dict(), save_path)
            print(f'模型已保存到: {save_path}')
            
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning Rate: {current_lr:.6e}')
        
        # 时间统计
        elapsed = time() - start_time
        print(f'Elapsed: {int(elapsed//60)}min {int(elapsed%60)}s')
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f'\n早停触发！{PATIENCE} 个epoch无改善。')
            break
        
        # 每50个epoch保存一次检查点
        if epoch % 50 == 0:
            checkpoint_path = f'checkpoints/Self2Self_{DATASET_NAME}_epoch_{epoch}.pth'
            torch.save(model.state_dict(), checkpoint_path)
            print(f'检查点已保存: {checkpoint_path}')
    
    # 保存最终模型
    save_path = f'checkpoints/Self2Self_{DATASET_NAME}_final.pth'
    torch.save(model.state_dict(), save_path)
    print(f'\n最终模型已保存到: {save_path}')
    
    print('\n' + '='*70)
    print('训练完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    if best_metrics is not None:
        print('最佳验证指标:')
        print_metrics(best_metrics, prefix='  ')
    print('='*70)


if __name__ == '__main__':
    main()
