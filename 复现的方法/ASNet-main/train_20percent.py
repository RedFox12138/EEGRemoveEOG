"""
ASNet 20%数据训练脚本
使用20%训练数据进行有监督训练，用于公平比较

训练流程:
1. 使用20%的训练数据(带clean标签)进行有监督训练
2. 使用MSE损失
3. 保存最佳模型（基于验证损失）
"""
import scipy
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
from torch.utils.data import Dataset
import sys

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

sys.path.append(r'D:\Pycharm_Projects\EOG Remove\复现的方法')

BATCH_SIZE = 200


class EEGDataset(Dataset):
    def __init__(self, noisy_signals, clean_signals, is_train=False):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        # 归一化 noisy 信号
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0

        noisy_normalized = noisy / norm_factor

        return noisy_normalized, clean, norm_factor


def get_data():
    """加载20%训练数据和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 加载完整训练集
    full_train_input = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    full_train_output = scipy.io.loadmat(f'{data_dir}/Train_Pure.mat')['data']
    
    # 取前20%数据
    num_samples = int(len(full_train_input) * 0.2)
    train_input = full_train_input[:num_samples]
    train_output = full_train_output[:num_samples]
    
    # 验证集
    verify_input = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    verify_output = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']

    print(f'训练数据（20%）: {train_input.shape}')
    print(f'验证数据: {verify_input.shape}')

    train_dataset = EEGDataset(train_input, train_output, is_train=True)
    verify_dataset = EEGDataset(verify_input, verify_output, is_train=False)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    verify_loader = Data.DataLoader(
        dataset=verify_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return train_loader, verify_loader


def train(model, device, train_loader, optimizer):
    """训练函数,返回平均损失和单样本训练时间"""
    model.train()
    step_num = 0
    loss_epoch = 0
    sample_count = 0
    
    batch_start_time = time()
    
    for batch_idx, (train_input, train_output, norm_factors) in enumerate(train_loader):
        step_num += 1
        sample_count += train_input.size(0)
        
        train_input = train_input.float().to(device)
        train_output = train_output.float().to(device)
        norm_factors = norm_factors.float().to(device).view(-1, 1)

        optimizer.zero_grad()
        output = model(train_input)
        output_restored = output * norm_factors

        loss = loss_f(output_restored, train_output)
        loss_epoch += loss.item()
        loss.backward()
        optimizer.step()
    
    total_time = time() - batch_start_time
    avg_loss = loss_epoch / step_num
    time_per_sample = total_time / sample_count
    
    return avg_loss, time_per_sample


def verify(model, device, verify_loader):
    """验证函数,计算验证集上的MSE损失和所有评价指标"""
    model.eval()
    step_num = 0
    loss_epoch = 0
    
    # 用于收集所有预测和真实值
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (verify_input, verify_output, norm_factors) in enumerate(verify_loader):
            step_num += 1
            verify_input = verify_input.float().to(device)
            verify_output = verify_output.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1)

            output = model(verify_input)
            output_restored = output * norm_factors

            loss = loss_f(output_restored, verify_output)
            loss_epoch += loss.item()
            
            # 收集预测和真实值用于计算指标
            all_predictions.append(output_restored.cpu().numpy())
            all_targets.append(verify_output.cpu().numpy())
    
    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 使用统一的评价指标计算
    metrics = compute_all_metrics(all_predictions, all_targets, fs=200)
    
    avg_loss = loss_epoch / step_num
    return avg_loss, metrics


# 加载数据
train_loader, verify_loader = get_data()

from ASNet import ASNet
model = ASNet()
model_name = 'ASNet_20percent'
learning_rate = 5e-4
loss_f = nn.MSELoss(reduction='mean')
print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练配置
NUM_EPOCHS = 100
best_val_loss = float('inf')  # 使用验证损失作为最佳模型选择标准
best_model_path = f'{model_name}_best.pkl'

print("="*60)
print(f"开始训练 {model_name}")
print("使用20%训练数据")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"批次大小: {BATCH_SIZE}")
print(f"学习率: {learning_rate}")
print(f"设备: {device}")
print("="*60)

begin_time = time()

for epoch in range(NUM_EPOCHS):
    print(f"\nEpoch [{epoch+1}/{NUM_EPOCHS}]")
    print("-" * 60)
    
    # 训练
    train_loss, time_per_sample = train(model, device, train_loader, optimizer)
    print(f"Train Loss: {train_loss:.6f} | Time/Sample: {time_per_sample*1000:.3f}ms")
    
    # 验证
    val_loss, val_metrics = verify(model, device, verify_loader)
    print(f"Val Loss: {val_loss:.6f}")
    
    # 打印验证集评价指标
    print_metrics(val_metrics, prefix="验证集")
    
    # 保存最佳模型(基于验证损失)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ 保存最佳模型 (Val Loss: {best_val_loss:.6f})")
    
    # 计算累计时间
    elapsed_time = time() - begin_time
    minute = int(elapsed_time // 60)
    second = int(elapsed_time % 60)
    print(f"Elapsed Time: {minute}min {second}s")

print("\n" + "="*60)
print("训练完成!")
print(f"最佳验证损失: {best_val_loss:.6f}")
print(f"最佳模型已保存至: {best_model_path}")
print("="*60)
