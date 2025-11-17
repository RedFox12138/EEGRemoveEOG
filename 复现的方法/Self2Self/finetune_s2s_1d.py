"""
Self2Self 1D EEG Denoising 微调脚本
使用少量有标签数据进行有监督微调
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

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入metrics
try:
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass

from s2s_model_1d import Self2Self_UNet1D


# ========== 超参数配置 ==========
BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 5e-5  # 微调使用更小的学习率
WEIGHT_DECAY = 0.0
SAMPLING_RATE = 200.0

# 训练配置
GRAD_CLIP = 1.0
PATIENCE = 50

# 使用20%的训练数据
FINETUNE_RATIO = 0.2

# Self2Self推理参数
N_PREDICTIONS = 100


class SupervisedDataset(Dataset):
    """有监督数据集"""
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    
    def __len__(self):
        return len(self.noisy)
    
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化到[0, 1]
        norm_min = np.min(noisy)
        norm_max = np.max(noisy)
        norm_range = norm_max - norm_min
        
        if norm_range == 0:
            norm_range = 1.0
        
        noisy_norm = (noisy - norm_min) / norm_range
        clean_norm = (clean - norm_min) / norm_range
        
        return (torch.from_numpy(noisy_norm).float(), 
               torch.from_numpy(clean_norm).float(),
               norm_min, norm_range)


def get_data():
    """加载部分训练数据和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 加载完整训练集
    full_train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    full_train_y = scipy.io.loadmat(f'{data_dir}/Train_Pure.mat')['data']
    
    # 取前20%数据用于微调
    num_samples = int(len(full_train_x) * FINETUNE_RATIO)
    train_x = full_train_x[:num_samples]
    train_y = full_train_y[:num_samples]
    
    # 验证集
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, train_y, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    """有监督训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for noisy_norm, clean_norm, _, _ in loader:
        noisy_norm = noisy_norm.unsqueeze(1).to(device)  # (B, 1, T)
        clean_norm = clean_norm.unsqueeze(1).to(device)  # (B, 1, T)
        
        optimizer.zero_grad()
        
        # 前向传播（不使用输入dropout）
        output = model(noisy_norm, apply_input_dropout=False)
        
        # MSE损失（在整个信号上）
        loss = F.mse_loss(output, clean_norm)
        
        loss.backward()
        
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def validate(model, device, loader):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy_norm, clean_norm, norm_min, norm_range in loader:
            noisy_norm = noisy_norm.unsqueeze(1).to(device)
            clean_norm = clean_norm.unsqueeze(1).to(device)
            
            # 使用多次预测平均（Self2Self的特点）
            output = model.predict_average(noisy_norm, n_predictions=N_PREDICTIONS)
            
            # 计算验证损失（在归一化空间）
            loss = F.mse_loss(output, clean_norm)
            total_loss += loss.item()
            num_batches += 1
            
            # 反归一化用于计算指标
            output_denorm = output.squeeze(1).cpu().numpy()
            output_denorm = output_denorm * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
            
            clean_denorm = clean_norm.squeeze(1).cpu().numpy()
            clean_denorm = clean_denorm * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
            
            all_preds.append(output_denorm)
            all_targets.append(clean_denorm)
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算指标
    avg_loss = total_loss / max(1, num_batches)
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    return avg_loss, metrics


def main():
    print('='*70)
    print('Self2Self 1D EEG Denoising 微调')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    train_x, train_y, val_x, val_y = get_data()
    print(f'\n数据集信息:')
    print(f'  微调训练集: {train_x.shape} ({FINETUNE_RATIO*100:.0f}%)')
    print(f'  验证集: {val_x.shape}')
    
    # 创建数据集
    train_dataset = SupervisedDataset(train_x, train_y)
    val_dataset = SupervisedDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # 创建模型
    print(f'\n创建模型...')
    model = Self2Self_UNet1D(
        in_channels=1,
        base_channels=48,
        n_depth=5,
        dropout=0.3
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 加载预训练的无监督模型
    pretrained_path = os.path.join(current_dir, 'Self2Self_1D_best.pth')
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f'✓ 加载预训练模型: {pretrained_path}')
    else:
        print('⚠️ 找不到预训练模型，从随机初始化开始！')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, 
                          weight_decay=WEIGHT_DECAY)
    
    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, 
                                                     eta_min=1e-7)
    
    # 训练循环
    print(f'\n开始微调...')
    print(f'超参数配置:')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning Rate: {LEARNING_RATE}')
    print(f'  Finetune Ratio: {FINETUNE_RATIO*100:.0f}%')
    print(f'  N Predictions (val): {N_PREDICTIONS}')
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
        
        # 验证（每5个epoch一次）
        if epoch % 5 == 0 or epoch == 1:
            print('验证中（使用多次预测平均）...')
            val_loss, val_metrics = validate(model, device, val_loader)
            print(f'Val Loss:   {val_loss:.6f}')
            print_metrics(val_metrics, prefix='验证集')
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_metrics = val_metrics
                print(f'\n✓ 验证损失降低: {best_val_loss:.6f}')
                torch.save(model.state_dict(), 
                          os.path.join(current_dir, 'Self2Self_1D_finetuned.pth'))
                patience_counter = 0
            else:
                patience_counter += 5
        else:
            patience_counter += 1
        
        # 学习率调度
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
    
    print('\n' + '='*70)
    print('微调完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    if best_metrics is not None:
        print('最佳验证指标:')
        print_metrics(best_metrics, prefix='  ')
    print('='*70)


if __name__ == '__main__':
    main()
