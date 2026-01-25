"""
Noise2Void 真实数据集无监督训练脚本
使用90%的真实数据进行无监督训练，10%用于验证
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
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from n2v_model_1d import N2V_UNet1D
from n2v_dataset_1d import N2V_Dataset1D, n2v_loss
from real_data_config import *


def load_and_split_data():
    """
    加载真实数据集并划分为训练集和验证集
    
    返回:
        train_x: 训练集数据 (N_train, L)
        val_x: 验证集数据 (N_val, L)
    """
    print('\n正在加载真实数据集...')
    print(f'数据路径: {REAL_DATA_PATH}')
    
    # 加载数据
    data_dict = scipy.io.loadmat(REAL_DATA_PATH)
    
    # 尝试不同的可能的 key
    possible_keys = [DATA_KEY, 'data', 'eeg_data', 'X', 'signals']
    data = None
    
    for key in possible_keys:
        if key in data_dict:
            data = data_dict[key]
            print(f'  ✓ 使用 key: "{key}"')
            break
    
    if data is None:
        available_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        raise ValueError(f'无法找到数据！可用的 keys: {available_keys}')
    
    print(f'  数据形状: {data.shape}')
    
    n_samples, sample_length = data.shape
    print(f'  样本数量: {n_samples}')
    print(f'  样本长度: {sample_length}')
    
    # 随机打乱并划分数据
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    
    # 计算划分点
    # 比例 7:1:2 (Train:Val:Test)
    train_end = int(n_samples * TRAIN_RATIO)  # 0.7
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO)) # 0.8
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    
    train_x = data[train_indices]
    val_x = data[val_indices]
    
    print(f'\n数据集划分完成 (Train:Val:Test = 7:1:2):')
    print(f'  训练集: {train_x.shape[0]} 样本')
    print(f'  验证集: {val_x.shape[0]} 样本')
    print(f'  测试集: {n_samples - val_end} 样本 (保留)')
    
    return train_x, val_x


def train_epoch(model, device, loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch in loader:
        x, y, mask, norm = batch
        
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(x)
        
        # Noise2Void 损失（只在盲点处计算）
        loss = n2v_loss(pred, y, mask)
        
        loss.backward()
        
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def validate(model, device, loader):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            x, y, mask, norm = batch
            
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)
            
            # 前向传播
            pred = model(x)
            
            # Noise2Void 损失
            loss = n2v_loss(pred, y, mask)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss


def main():
    print("=" * 80)
    print("Noise2Void 真实数据集无监督训练")
    print("=" * 80)
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载并划分数据
    train_x, val_x = load_and_split_data()
    
    # 创建数据集
    print('\n创建数据集...')
    train_dataset = N2V_Dataset1D(
        data=train_x,
        perc_pix=PERC_PIX,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        manipulator=MANIPULATOR
    )
    
    val_dataset = N2V_Dataset1D(
        data=val_x,
        perc_pix=PERC_PIX,
        neighborhood_radius=NEIGHBORHOOD_RADIUS,
        manipulator=MANIPULATOR
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )
    
    print(f'  训练批次数: {len(train_loader)}')
    print(f'  验证批次数: {len(val_loader)}')
    
    # 创建模型
    print('\n创建模型...')
    model = N2V_UNet1D(
        in_channels=1,
        n_depth=N_DEPTH,
        n_first=N_FIRST,
        kernel_size=KERNEL_SIZE,
        batch_norm=BATCH_NORM,
        dropout=DROPOUT,
        residual=RESIDUAL
    ).to(device)
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  总参数量: {total_params:,}')
    print(f'  可训练参数: {trainable_params:,}')
    
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # 学习率调度器
    if USE_LR_SCHEDULER:
        def lr_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                return max(MIN_LR / LEARNING_RATE, 
                          0.5 * (1 + np.cos(np.pi * (epoch - WARMUP_EPOCHS) / (EPOCHS - WARMUP_EPOCHS))))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 训练循环
    print('\n' + '='*80)
    print('开始训练...')
    print('='*80)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time()
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        
        # 验证
        val_loss = validate(model, device, val_loader)
        
        # 更新学习率
        if USE_LR_SCHEDULER:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = LEARNING_RATE
        
        epoch_time = time() - epoch_start
        
        # 打印信息
        print(f'Epoch {epoch}/{EPOCHS} [{epoch_time:.1f}s] '
              f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | '
              f'LR: {current_lr:.2e}')
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            print(f'  ✓ 验证损失改善! 保存最佳模型到: {MODEL_SAVE_PATH}')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            patience_counter += 1
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f'\n早停触发！验证损失已 {PATIENCE} 个 epoch 未改善')
            break
    
    # 保存最终模型
    print(f'\n保存最终模型到: {FINAL_MODEL_PATH}')
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    print('\n' + '='*80)
    print('训练完成！')
    print('='*80)
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'最佳模型: {MODEL_SAVE_PATH}')
    print(f'最终模型: {FINAL_MODEL_PATH}')


if __name__ == '__main__':
    main()
