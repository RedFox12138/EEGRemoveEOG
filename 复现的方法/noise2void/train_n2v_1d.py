"""
Noise2Void 1D EEG Denoising 训练脚本
基于Noise2Void的无监督训练方法
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from time import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 尝试导入metrics
try:
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {'RRMSE':0,'CC':0,'RRMSE_PSD':0,'MI':0}
    def print_metrics(m, prefix=""): pass

from n2v_model_1d import N2V_UNet1D
from n2v_dataset_1d import N2V_Dataset1D, N2V_ValidationDataset1D, n2v_loss


# ========== 超参数配置 ==========
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
SAMPLING_RATE = 200.0

# 学习率调度
USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 20
MIN_LR = 1e-6

# 训练配置
GRAD_CLIP = 1.0
PATIENCE = 80

# N2V参数
PERC_PIX = 1.5          # 盲点百分比
NEIGHBORHOOD_RADIUS = 5  # 邻域半径
MANIPULATOR = 'uniform'  # 替换策略: 'uniform', 'median', 'neighbor'

# 模型参数
N_DEPTH = 3             # UNet深度
N_FIRST = 32            # 第一层滤波器数
KERNEL_SIZE = 5         # 卷积核大小
BATCH_NORM = True       # 批归一化
DROPOUT = 0.0           # Dropout
RESIDUAL = True         # 残差连接


def get_data():
    """加载训练和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    try:
        val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    except Exception:
        val_y = None
    return train_x, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        if len(batch) == 5:
            x, y, mask, clean, norm = batch
        else:
            x, y, mask, norm = batch
        
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(x)
        
        # 计算N2V损失（只在盲点位置）
        loss = n2v_loss(pred, y, mask)
        
        # 反向传播
        loss.backward()
        
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def validate(model, device, loader, has_clean_labels=False):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 5:
                x, y, mask, clean, norm = batch
            else:
                x, y, mask, norm = batch
                clean = None
            
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # 前向传播
            pred = model(x)
            
            # 计算损失
            loss = n2v_loss(pred, y, mask)
            
            total_loss += loss.item()
            num_batches += 1
            
            # 如果有干净标签，收集用于计算指标
            if has_clean_labels and clean is not None:
                # 预测结果已经是归一化后的，需要恢复到原始尺度
                pred_scaled = pred.squeeze(1).cpu().numpy() * norm.numpy()[:, None]
                all_preds.append(pred_scaled)
                all_targets.append(clean.numpy())
    
    avg_loss = total_loss / max(1, num_batches)
    
    # 计算指标
    metrics = None
    if has_clean_labels and len(all_targets) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    return avg_loss, metrics


def main():
    print('='*70)
    print('Noise2Void 1D EEG Denoising 训练')
    print('='*70)
    
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
    print(f'\n创建N2V数据集...')
    train_dataset = N2V_Dataset1D(
        train_x, 
        perc_pix=PERC_PIX,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        manipulator=MANIPULATOR,
        clean_data=None
    )
    
    val_dataset = N2V_ValidationDataset1D(
        val_x,
        perc_pix=PERC_PIX,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        manipulator=MANIPULATOR,
        clean_data=val_y
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                           shuffle=False, num_workers=0)
    
    # 创建模型
    print(f'\n创建模型...')
    model = N2V_UNet1D(
        in_channels=1,
        n_depth=N_DEPTH,
        n_first=N_FIRST,
        kernel_size=KERNEL_SIZE,
        batch_norm=BATCH_NORM,
        dropout=DROPOUT,
        residual=RESIDUAL
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
    print(f'  Perc Pix: {PERC_PIX}%')
    print(f'  Neighborhood Radius: {NEIGHBORHOOD_RADIUS}')
    print(f'  Manipulator: {MANIPULATOR}')
    print(f'  UNet Depth: {N_DEPTH}')
    print(f'  Residual: {RESIDUAL}')
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
        
        # 验证
        val_loss, val_metrics = validate(model, device, val_loader, 
                                        has_clean_labels=(val_y is not None))
        print(f'Val Loss:   {val_loss:.6f}')
        
        if val_metrics is not None:
            print_metrics(val_metrics, prefix='验证集')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = val_metrics
            print(f'\n✓ 验证损失降低: {best_val_loss:.6f}')
            torch.save(model.state_dict(), 
                      os.path.join(current_dir, 'N2V_1D_best.pth'))
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
    
    # 保存最终模型
    torch.save(model.state_dict(), 
              os.path.join(current_dir, 'N2V_1D_final.pth'))
    
    print('\n' + '='*70)
    print('训练完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    if best_metrics is not None:
        print('最佳验证指标:')
        print_metrics(best_metrics, prefix='  ')
    print('='*70)


if __name__ == '__main__':
    main()
