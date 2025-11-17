"""
EEGIFNet 20%数据训练脚本
使用20%训练数据进行有监督训练，用于公平比较

训练流程:
1. 使用20%的训练数据(带clean标签)进行有监督训练
2. 保存最佳模型（基于验证损失）
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from time import time
import sys
import scipy.io
from torch.utils.data import Dataset, DataLoader
from EEGIFNet_1200 import MA_INet, MA_MNet, weights_init
from config import cal_ACC_tensor, cal_RRMSE_tensor, cal_SNR
import os
import argparse

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

# 添加父目录到路径以导入metrics_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BATCH_SIZE = 256
LEARNING_RATE = 5e-5
EPOCHS = 200


class EEGDataset(Dataset):
    """
    与ASNet一致的数据集类,包含标准化逻辑
    """
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


def get_data(data_path, batch_size):
    """
    加载20%训练数据和验证数据
    """
    # 加载完整训练集
    full_train_input = scipy.io.loadmat(os.path.join(data_path, 'Train_Contaminated.mat'))['data']
    full_train_output = scipy.io.loadmat(os.path.join(data_path, 'Train_Pure.mat'))['data']
    
    # 取前20%数据
    num_samples = int(len(full_train_input) * 0.2)
    train_input = full_train_input[:num_samples]
    train_output = full_train_output[:num_samples]
    
    # 验证集
    verify_input = scipy.io.loadmat(os.path.join(data_path, 'Val_Contaminated.mat'))['data']
    verify_output = scipy.io.loadmat(os.path.join(data_path, 'Val_Pure.mat'))['data']
    
    print(f"加载数据: 训练集（20%）={train_input.shape}, 验证集={verify_input.shape}")
    print(f"时间点数量: {train_input.shape[1]}")

    train_dataset = EEGDataset(train_input, train_output, is_train=True)
    verify_dataset = EEGDataset(verify_input, verify_output, is_train=False)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    verify_loader = DataLoader(
        dataset=verify_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    input_length = train_input.shape[1]
    return train_loader, verify_loader, input_length


def train_epoch(I_model, M_model, device, train_loader, optimizer_I, optimizer_M, criterion, epoch, epochs):
    """
    训练一个epoch
    """
    I_model.train()
    M_model.train()

    total_train_loss_e_per_epoch = 0
    total_train_loss_n_per_epoch = 0
    total_train_loss_per_epoch = 0
    train_step_num = 0

    for batch_idx, (x, y, norm_factors) in enumerate(train_loader):
        train_step_num += 1
        
        x = x.float().to(device)
        y = y.float().to(device)
        norm_factors = norm_factors.float().to(device).view(-1, 1)
        
        # 添加通道维度
        x_with_channel = x.unsqueeze(1)

        optimizer_I.zero_grad()
        optimizer_M.zero_grad()

        # INet预测clean EEG和noise
        e_outputs, n_outputs = I_model(x_with_channel)
        # MNet融合预测
        outputs = M_model(x_with_channel, e_outputs, n_outputs)

        # 恢复到原始尺度后计算loss
        e_outputs_restored = e_outputs * norm_factors
        n_outputs_restored = n_outputs * norm_factors
        outputs_restored = outputs * norm_factors
        
        # 计算噪声目标
        z = x - y
        
        # 在原始尺度计算loss
        loss_e = criterion(e_outputs_restored, y)
        loss_n = criterion(n_outputs_restored, z)
        loss_all = criterion(outputs_restored, y)

        total_train_loss_e_per_epoch += loss_e.item()
        total_train_loss_n_per_epoch += loss_n.item()
        total_train_loss_per_epoch += loss_all.item()

        # 总loss
        loss = loss_e + loss_n + loss_all
        loss.backward()
        
        optimizer_I.step()
        optimizer_M.step()

    # 计算平均loss
    average_train_loss_e = total_train_loss_e_per_epoch / train_step_num
    average_train_loss_n = total_train_loss_n_per_epoch / train_step_num
    average_train_loss_all = total_train_loss_per_epoch / train_step_num

    print(f"Epoch [{epoch+1}/{epochs}] Train - Loss_e: {average_train_loss_e:.6f}, "
          f"Loss_n: {average_train_loss_n:.6f}, Loss_all: {average_train_loss_all:.6f}")

    return average_train_loss_all


def validate_epoch(I_model, M_model, device, val_loader, criterion, epoch, epochs):
    """
    验证一个epoch
    """
    I_model.eval()
    M_model.eval()

    total_val_loss = 0
    val_step_num = 0
    
    # 收集所有预测和真实值
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for batch_idx, (x, y, norm_factors) in enumerate(val_loader):
            val_step_num += 1

            x = x.float().to(device)
            y = y.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1)

            # 添加通道维度
            x_with_channel = x.unsqueeze(1)
            
            # 模型预测
            e_outputs, n_outputs = I_model(x_with_channel)
            outputs = M_model(x_with_channel, e_outputs, n_outputs)

            # 恢复到原始尺度
            outputs_restored = outputs * norm_factors
            
            # 在原始尺度计算loss
            loss = criterion(outputs_restored, y)
            total_val_loss += loss.item()
            
            # 收集数据用于统一评价指标计算
            all_predictions.append(outputs_restored.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 计算平均值
    average_val_loss = total_val_loss / val_step_num
    
    # 计算统一评价指标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    unified_metrics = compute_all_metrics(all_predictions, all_targets, fs=200)

    print(f"Epoch [{epoch+1}/{epochs}] Val - Loss: {average_val_loss:.6f}")
    print_metrics(unified_metrics, prefix="验证集")

    return average_val_loss


def main():
    parser = argparse.ArgumentParser(description='EEGIFNet 20% Training')
    parser.add_argument('--data_path', type=str, 
                        default=r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据',
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE, help='批大小')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, help='学习率')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='训练轮数')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备')
    parser.add_argument('--save_dir', type=str, default='./checkpoint', help='模型保存目录')
    args = parser.parse_args()

    # 设置随机种子
    np.random.seed(1)
    torch.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载20%训练数据...")
    train_loader, val_loader, input_length = get_data(args.data_path, args.batch_size)
    print(f"数据时间点: {input_length}")

    # 初始化模型
    print("初始化模型...")
    I_model = MA_INet(input_length=input_length).apply(weights_init).to(device)
    M_model = MA_MNet().apply(weights_init).to(device)

    # 优化器
    optimizer_I = torch.optim.RMSprop(I_model.parameters(), lr=args.lr, alpha=0.9)
    optimizer_M = torch.optim.RMSprop(M_model.parameters(), lr=args.lr, alpha=0.9)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练
    print("="*60)
    print("开始训练 EEGIFNet")
    print("使用20%训练数据")
    print(f"训练轮数: {args.epochs}")
    print(f"批次大小: {args.batch_size}")
    print(f"学习率: {args.lr}")
    print("="*60)
    
    best_val_loss = float('inf')
    begin_time = time()

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(I_model, M_model, device, train_loader, 
                                 optimizer_I, optimizer_M, criterion, epoch, args.epochs)
        
        # 验证
        val_loss = validate_epoch(I_model, M_model, device, val_loader, criterion, epoch, args.epochs)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  >> 保存最佳模型 (val_loss: {val_loss:.6f})")
            torch.save(I_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_20percent_INet_best.pkl'))
            torch.save(M_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_20percent_MNet_best.pkl'))

        # 计算已用时间
        elapsed_time = time() - begin_time
        minute = int(elapsed_time // 60)
        second = int(elapsed_time % 60)
        print(f"  >> 用时: {minute}m {second}s, 最佳验证Loss: {best_val_loss:.6f}")
        print('-' * 80)

    print("\n" + "="*60)
    print("训练完成!")
    print(f"总用时: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
    print(f"最佳验证Loss: {best_val_loss:.6f}")
    print("="*60)


if __name__ == '__main__':
    main()
