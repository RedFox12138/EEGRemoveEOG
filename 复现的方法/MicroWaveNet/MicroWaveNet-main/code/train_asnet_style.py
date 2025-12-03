import os
import sys
import time
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import numpy as np

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

sys.path.append(r'D:\Pycharm_Projects\EOG Remove\复现的方法')
from cbamdropout import EEGNetMorletWindowCBAMDropout

# 导入数据集配置
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data_config import *

BATCH_SIZE = 200
LEARNING_RATE = 1e-3
NUM_EPOCHS = 300
MIN_LR = 5e-5  # 最小学习率


class EEGDatasetASNetStyle(Data.Dataset):
    def __init__(self, noisy_signals, clean_signals, is_train=False):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        # 归一化 noisy 信号，保留幅值因子
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0

        noisy_normalized = noisy / norm_factor

        # MicroWaveNet expects channel dimension (1, L)
        noisy_normalized = noisy_normalized[np.newaxis, :].astype(np.float32)
        clean = clean[np.newaxis, :].astype(np.float32)

        return noisy_normalized, clean, np.array([norm_factor], dtype=np.float32)


def load_data(data_dir):
    train_input = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    verify_input = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    test_input = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]

    train_output = scipy.io.loadmat(TRAIN_PURE_PATH)[DATA_KEY]
    verify_output = scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]
    test_output = scipy.io.loadmat(TEST_PURE_PATH)[DATA_KEY]

    train_set = EEGDatasetASNetStyle(train_input, train_output, is_train=True)
    val_set = EEGDatasetASNetStyle(verify_input, verify_output, is_train=False)
    test_set = EEGDatasetASNetStyle(test_input, test_output, is_train=False)

    train_loader = Data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = Data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader


def train_epoch(model, device, loader, optimizer):
    model.train()
    total_loss = 0.0
    count = 0
    start = time.time()
    for noisy, clean, norm in loader:
        noisy = noisy.to(device)
        clean = clean.to(device)
        norm = norm.to(device)

        optimizer.zero_grad()
        out = model(noisy)
        # out: eeg, artefact, eeg_z, artefact_z, wav_tensor, x_inp
        eeg_pred = out[0] * norm.view(-1, 1, 1)
        artefact_pred = out[1] * norm.view(-1, 1, 1)

        # reconstruct tuple for loss
        f_restored = (eeg_pred, artefact_pred, out[2], out[3], out[4], out[5])

        eegrec, artefactrec, mim, wvl, loss = model.loss(f_restored, clean.to(device), clean.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        count += 1

    elapsed = time.time() - start
    return total_loss / max(1, count), elapsed


def validate(model, device, loader):
    model.eval()
    total_loss = 0.0
    count = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            norm = norm.to(device)

            out = model(noisy)
            eeg_pred = out[0] * norm.view(-1, 1, 1)
            artefact_pred = out[1] * norm.view(-1, 1, 1)
            f_restored = (eeg_pred, artefact_pred, out[2], out[3], out[4], out[5])

            eegrec, artefactrec, mim, wvl, loss = model.loss(f_restored, clean.to(device), clean.to(device))
            total_loss += loss.item()
            count += 1
            
            # 收集预测和真实值用于计算指标
            all_predictions.append(eeg_pred.cpu().numpy())
            all_targets.append(clean.cpu().numpy())
    
    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评价指标 (假设采样率200Hz)
    metrics = compute_all_metrics(all_predictions, all_targets, fs=SAMPLING_RATE)
    
    return total_loss / max(1, count), metrics


def main():
    data_dir = DATA_DIR  # 从data_config导入
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = load_data(data_dir)

    model = EEGNetMorletWindowCBAMDropout(device=device)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 使用余弦退火学习率调度器，更温和的衰减
    # T_max: 每个周期的epoch数, eta_min: 最小学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=NUM_EPOCHS,  # 整个训练周期
        eta_min=MIN_LR  # 最小学习率，不会降到0
    )

    best_val_loss = float('inf')  # 使用验证损失作为最佳模型选择标准(越小越好)
    os.makedirs('results', exist_ok=True)
    
    print("="*60)
    print(f"开始训练 MicroWaveNet")
    print(f"训练轮数: {NUM_EPOCHS}")
    print(f"批次大小: {BATCH_SIZE}")
    print(f"初始学习率: {LEARNING_RATE}")
    print(f"最小学习率: {MIN_LR}")
    print(f"学习率策略: CosineAnnealingLR (余弦退火)")
    print(f"设备: {device}")
    print("="*60)

    for epoch in range(NUM_EPOCHS):
        train_loss, elapsed = train_epoch(model, device, train_loader, optimizer)
        val_loss, val_metrics = validate(model, device, val_loader)
        
        # 获取当前学习率
        current_lr = optimizer.param_groups[0]['lr']

        print(f'\nEpoch {epoch+1}/{NUM_EPOCHS}')
        print('-'*60)
        print(f'Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s')
        
        # 打印验证集评价指标
        print_metrics(val_metrics, prefix="验证集")
        
        # 只在有改进时保存最佳模型(基于验证损失)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'MicroWaveNet_best.pt')
            print(f'✓ 保存最佳模型 (Val Loss: {best_val_loss:.6f})')
        
        # 更新学习率
        scheduler.step()

    print("\n" + "="*60)
    print('训练完成!')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'最佳模型已保存至: MicroWaveNet_best.pt')
    print("="*60)


if __name__ == '__main__':
    main()
