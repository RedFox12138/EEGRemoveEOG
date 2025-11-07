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
        # 找到绝对值的最大值作为归一化因子
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0 # 避免除以零

        noisy_normalized = noisy / norm_factor

        # 不需要在这里添加通道维度，ASNet的forward会自动处理
        # noisy_normalized = noisy_normalized[np.newaxis, :]
        # clean = clean[np.newaxis, :]

        return noisy_normalized, clean, norm_factor

def get_data():
    # 加载已经分割好的数据集（80% 训练, 10% 验证, 10% 测试）
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    train_input = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    verify_input = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    
    train_output = scipy.io.loadmat(f'{data_dir}/Train_Pure.mat')['data']
    verify_output = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']

    train_dataset = EEGDataset(train_input, train_output, is_train=True)
    verify_dataset = EEGDataset(verify_input, verify_output, is_train=False)
    test_dataset = EEGDataset(test_input, test_output, is_train=False)

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

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, verify_loader, test_loader


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


def calculate_mean_std_metrics(param, param1, fs):
    pass


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
    
    # 计算评价指标
    metrics = calculate_mean_std_metrics(
        [all_targets[i] for i in range(len(all_targets))],
        [all_predictions[i] for i in range(len(all_predictions))],
        fs=200
    )
    
    avg_loss = loss_epoch / step_num
    return avg_loss, metrics


train_loader, verify_loader, test_loader = get_data()
from ASNet import ASNet
model = ASNet()
model_name = 'ASNet'
learning_rate = 5e-4
loss_f = nn.MSELoss(reduction='mean')
print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练配置
NUM_EPOCHS = 50
best_cc = -1.0  # 使用CC作为最佳模型选择标准(越大越好)
best_model_path = f'{model_name}_best.pkl'

print("="*60)
print(f"开始训练 {model_name}")
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
    print(f"Val Metrics - RRMSE: {val_metrics['RRMSE']['mean']:.4f}±{val_metrics['RRMSE']['std']:.4f}, "
          f"CC: {val_metrics['CC']['mean']:.4f}±{val_metrics['CC']['std']:.4f}, "
          f"RRMSE_PSD: {val_metrics['RRMSE_PSD']['mean']:.4f}±{val_metrics['RRMSE_PSD']['std']:.4f}, "
          f"MI: {val_metrics['MI']['mean']:.4f}±{val_metrics['MI']['std']:.4f}")
    
    # 保存最佳模型(基于CC指标)
    current_cc = val_metrics['CC']['mean']
    if current_cc > best_cc:
        best_cc = current_cc
        torch.save(model.state_dict(), best_model_path)
        print(f"✓ 保存最佳模型 (CC: {best_cc:.4f})")
    
    # 计算累计时间
    elapsed_time = time() - begin_time
    minute = int(elapsed_time // 60)
    second = int(elapsed_time % 60)
    print(f"Elapsed Time: {minute}min {second}s")

print("\n" + "="*60)
print("训练完成!")
print(f"最佳CC: {best_cc:.4f}")
print(f"最佳模型已保存至: {best_model_path}")
print("="*60)
