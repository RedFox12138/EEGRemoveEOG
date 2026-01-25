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
# 导入配置
from config import *

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
    train_input = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    verify_input = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    
    train_output = scipy.io.loadmat(TRAIN_PURE_PATH)[PURE_KEY]
    verify_output = scipy.io.loadmat(VAL_PURE_PATH)[PURE_KEY]

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

    # 测试集仅在可用时加载（多SNR配置下不需要）
    if TEST_CONTAMINATED_PATH is not None:
        test_input = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]
        test_output = scipy.io.loadmat(TEST_PURE_PATH)[PURE_KEY]
        test_dataset = EEGDataset(test_input, test_output, is_train=False)
        test_loader = Data.DataLoader(
            dataset=test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False
        )
    else:
        test_loader = None
    
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
    metrics = compute_all_metrics(all_predictions, all_targets, fs=SAMPLING_RATE)
    
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
NUM_EPOCHS = 1000
best_val_loss = float('inf')  # 使用验证损失作为最佳模型选择标准(越小越好)
best_model_path = f'{model_name}_best.pkl'

# 早停配置
PATIENCE = 30  # 验证损失不改善时的最大等待轮数
patience_counter = 0  # 当前等待计数器

# 自动加载已有的best模型继续训练
if os.path.exists(best_model_path):
    print(f"\n发现已有模型: {best_model_path}")
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        print(f"✓ 成功加载模型，将从已有最佳模型继续训练")
    except Exception as e:
        print(f"⚠ 加载模型失败: {e}")
        print("将从头开始训练")
else:
    print(f"\n未找到已有模型: {best_model_path}")
    print("将从头开始训练")

print("="*60)
print(f"开始训练 {model_name}")
print(f"训练轮数: {NUM_EPOCHS}")
print(f"早停patience: {PATIENCE}")
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
        patience_counter = 0  # 重置patience计数器
    else:
        patience_counter += 1
        print(f"⚠ 验证损失未改善 (patience: {patience_counter}/{PATIENCE})")
        
        # 早停检查
        if patience_counter >= PATIENCE:
            print(f"\n{'='*60}")
            print(f"早停触发! 验证损失已经 {PATIENCE} 轮未改善")
            print(f"最佳验证损失: {best_val_loss:.6f}")
            print(f"最佳模型已保存至: {best_model_path}")
            print(f"{'='*60}")
            break
    
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