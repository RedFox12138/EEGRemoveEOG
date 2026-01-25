"""
EEGIFNet训练脚本 - 使用ASNet数据集格式
保持原有网络结构和参数,但使用与ASNet相同的数据加载和标准化方式
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from time import time
import sys

from sympy.physics.quantum.identitysearch import scipy
from torch.utils.data import Dataset, DataLoader
from EEGIFNet_1200 import MA_INet, MA_MNet, weights_init
from config import cal_ACC_tensor, cal_RRMSE_tensor, cal_SNR
import os
import argparse

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

# 添加父目录到路径以导入metrics_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入数据集配置
from data_config import *

BATCH_SIZE = 256  # EEGIFNet原始batch size
LEARNING_RATE = 5e-5  # EEGIFNet原始学习率 (⚠️ 注意：原始代码使用5e-5而不是1e-3)
EPOCHS = 2000  # EEGIFNet原始训练轮数

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
        # 找到绝对值的最大值作为归一化因子
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0  # 避免除以零

        noisy_normalized = noisy / norm_factor

        # ⚠️ 注意：clean信号不归一化，保持原始值（与ASNet一致）
        # ⚠️ 注意：不添加通道维度（与ASNet一致）
        
        return noisy_normalized, clean, norm_factor


def get_data(data_path, batch_size):
    """
    加载数据并创建DataLoader
    """
    # 加载已经分割好的数据集（80% 训练, 10% 验证, 10% 测试）
    # 使用配置文件中的路径
    train_input = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    verify_input = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    
    train_output = scipy.io.loadmat(TRAIN_PURE_PATH)[PURE_KEY]
    verify_output = scipy.io.loadmat(VAL_PURE_PATH)[PURE_KEY]
    
    print(f"加载数据: 训练集={train_input.shape}, 验证集={verify_input.shape}")
    print(f"时间点数量: {train_input.shape[1]}")
    print(f"训练集: {len(train_input)}, 验证集: {len(verify_input)}")

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

    # 测试集仅在可用时加载（多SNR配置下不需要）
    if TEST_CONTAMINATED_PATH is not None:
        test_input = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]
        test_output = scipy.io.loadmat(TEST_PURE_PATH)[DATA_KEY]
        print(f"测试集: {len(test_input)}")
        test_dataset = EEGDataset(test_input, test_output, is_train=False)
        test_loader = DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=False
        )
    else:
        test_loader = None
        print("多SNR测试集配置，训练时不加载测试集")
    
    # 返回数据加载器和时间点数量
    input_length = train_input.shape[1]
    return train_loader, verify_loader, test_loader, input_length


def train_epoch(I_model, M_model, device, train_loader, optimizer_I, optimizer_M, criterion, epoch, epochs):
    """
    训练一个epoch - 保持EEGIFNet原有的训练逻辑
    """
    I_model.train()
    M_model.train()

    total_train_loss_e_per_epoch = 0
    total_train_loss_n_per_epoch = 0
    total_train_loss_per_epoch = 0
    train_step_num = 0

    for batch_idx, (x, y, norm_factors) in enumerate(train_loader):
        train_step_num += 1
        
        # 数据移到设备 (与ASNet一致)
        x = x.float().to(device)
        y = y.float().to(device)
        norm_factors = norm_factors.float().to(device).view(-1, 1)
        
        # 添加通道维度用于EEGIFNet (batch, 1, time)
        x_with_channel = x.unsqueeze(1)

        optimizer_I.zero_grad()
        optimizer_M.zero_grad()

        # INet预测clean EEG和noise (输出形状: batch, time)
        e_outputs, n_outputs = I_model(x_with_channel)
        # MNet融合预测
        outputs = M_model(x_with_channel, e_outputs, n_outputs)

        # ⚠️ 关键：恢复到原始尺度后计算loss（与ASNet一致）
        e_outputs_restored = e_outputs * norm_factors
        n_outputs_restored = n_outputs * norm_factors
        outputs_restored = outputs * norm_factors
        
        # 计算噪声目标 (原始尺度) - ⚠️ 与原始EEGIFNet一致：使用x.squeeze()-y
        z = x - y  # x已经是(batch, time)，不需要squeeze
        
        # 在原始尺度计算loss
        loss_e = criterion(e_outputs_restored, y)
        loss_n = criterion(n_outputs_restored, z)
        loss_all = criterion(outputs_restored, y)

        total_train_loss_e_per_epoch += loss_e.item()
        total_train_loss_n_per_epoch += loss_n.item()
        total_train_loss_per_epoch += loss_all.item()

        # 总loss - 按照EEGIFNet原始设置
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
    验证一个epoch - 保持EEGIFNet原有的验证逻辑
    同时计算反归一化后的指标和统一评价指标
    """
    I_model.eval()
    M_model.eval()

    total_val_loss = 0
    val_step_num = 0
    
    # 归一化空间的指标
    sum_acc, sum_acc_n, sum_acc_e = 0, 0, 0
    sum_rrmse, sum_rrmse_n, sum_rrmse_e = 0, 0, 0
    
    # 反归一化后的指标
    sum_acc_denorm, sum_rrmse_denorm = 0, 0
    sum_snr = 0
    
    # 收集所有预测和真实值用于统一评价指标计算
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
            
            # 计算噪声目标 (原始尺度) - ⚠️ 与原始EEGIFNet一致
            z = x - y  # x已经是(batch, time)，不需要squeeze

            # 模型预测
            e_outputs, n_outputs = I_model(x_with_channel)
            outputs = M_model(x_with_channel, e_outputs, n_outputs)

            # ⚠️ 恢复到原始尺度（与ASNet一致）
            outputs_restored = outputs * norm_factors
            
            # 在原始尺度计算loss
            loss = criterion(outputs_restored, y)
            total_val_loss += loss.item()

            # 在原始尺度计算指标（与ASNet一致）
            e_outputs_restored = e_outputs * norm_factors
            n_outputs_restored = n_outputs * norm_factors
            
            acc_e = cal_ACC_tensor(e_outputs_restored.detach(), y.detach())
            sum_acc_e += acc_e
            rrmse_e = cal_RRMSE_tensor(e_outputs_restored.detach(), y.detach())
            sum_rrmse_e += rrmse_e

            acc_n = cal_ACC_tensor(n_outputs_restored.detach(), z.detach())
            sum_acc_n += acc_n
            rrmse_n = cal_RRMSE_tensor(n_outputs_restored.detach(), z.detach())
            sum_rrmse_n += rrmse_n

            acc = cal_ACC_tensor(outputs_restored.detach(), y.detach())
            sum_acc += acc
            rrmse = cal_RRMSE_tensor(outputs_restored.detach(), y.detach())
            sum_rrmse += rrmse

            # 最终指标（已经在原始尺度）
            acc_denorm = acc
            sum_acc_denorm += acc_denorm
            rrmse_denorm = rrmse
            sum_rrmse_denorm += rrmse_denorm
            
            snr = cal_SNR(outputs_restored, y)
            sum_snr += snr
            
            # 收集数据用于统一评价指标计算
            all_predictions.append(outputs_restored.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    # 计算平均值
    average_val_loss = total_val_loss / val_step_num
    
    acc_e = sum_acc_e.item() / val_step_num
    rrmse_e = sum_rrmse_e.item() / val_step_num
    acc_n = sum_acc_n.item() / val_step_num
    rrmse_n = sum_rrmse_n.item() / val_step_num
    acc = sum_acc.item() / val_step_num
    rrmse = sum_rrmse.item() / val_step_num
    
    acc_denorm = sum_acc_denorm.item() / val_step_num
    rrmse_denorm = sum_rrmse_denorm.item() / val_step_num
    snr_avg = sum_snr / val_step_num
    
    # 计算统一评价指标
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    unified_metrics = compute_all_metrics(all_predictions, all_targets, fs=SAMPLING_RATE)

    print(f"Epoch [{epoch+1}/{epochs}] Val - Loss: {average_val_loss:.6f}")
    print(f"  [Normalized] ACC_e: {acc_e:.4f}, RRMSE_e: {rrmse_e:.4f}")
    print(f"  [Normalized] ACC_n: {acc_n:.4f}, RRMSE_n: {rrmse_n:.4f}")
    print(f"  [Normalized] ACC: {acc:.4f}, RRMSE: {rrmse:.4f}")
    print(f"  [Denormalized] ACC: {acc_denorm:.4f}, RRMSE: {rrmse_denorm:.4f}, SNR: {snr_avg:.2f} dB")
    
    # 打印统一评价指标
    print_metrics(unified_metrics, prefix="验证集统一指标")

    return average_val_loss, acc_denorm, rrmse_denorm


def main():
    parser = argparse.ArgumentParser(description='EEGIFNet Training with ASNet Dataset')
    parser.add_argument('--data_path', type=str, 
                        default=DATA_DIR,  # 从data_config导入
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
    print("加载数据集...")
    train_loader, val_loader, test_loader, input_length = get_data(args.data_path, args.batch_size)
    print(f"数据时间点: {input_length}")

    # 初始化模型 - 使用EEGIFNet的MA_INet和MA_MNet (适配input_length)
    print("初始化模型...")
    I_model = MA_INet(input_length=input_length).apply(weights_init).to(device)
    M_model = MA_MNet().apply(weights_init).to(device)

    # 优化器 - 使用RMSprop,与EEGIFNet原始设置一致
    optimizer_I = torch.optim.RMSprop(I_model.parameters(), lr=args.lr, alpha=0.9)
    optimizer_M = torch.optim.RMSprop(M_model.parameters(), lr=args.lr, alpha=0.9)

    # 学习率调度器 - ReduceLROnPlateau，最多衰减到原始学习率的1/10
    scheduler_I = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_I, mode='min', factor=0.5, patience=100,
        min_lr=args.lr * 0.1, verbose=True
    )
    scheduler_M = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_M, mode='min', factor=0.5, patience=100,
        min_lr=args.lr * 0.1, verbose=True
    )

    # 损失函数
    criterion = nn.MSELoss()

    # 训练
    print("开始训练...")
    print(f"初始学习率: {args.lr}, 最小学习率: {args.lr * 0.1}")
    print(f"Early Stopping Patience: 50轮")
    best_val_loss = float('inf')
    best_acc = 0
    epochs_no_improve = 0  # Early stopping计数器
    PATIENCE = 100  # Early stopping patience
    
    # 自动加载已有的best模型继续训练
    inet_best_path = os.path.join(args.save_dir, 'EEGIFNet_INet_best.pkl')
    mnet_best_path = os.path.join(args.save_dir, 'EEGIFNet_MNet_best.pkl')
    
    if os.path.exists(inet_best_path) and os.path.exists(mnet_best_path):
        print(f"\n发现已有模型: {inet_best_path} 和 {mnet_best_path}")
        try:
            I_model.load_state_dict(torch.load(inet_best_path, map_location=device))
            M_model.load_state_dict(torch.load(mnet_best_path, map_location=device))
            print(f"✓ 成功加载INet和MNet模型，将从已有最佳模型继续训练")
        except Exception as e:
            print(f"⚠ 加载模型失败: {e}")
            print("将从头开始训练")
    else:
        print(f"\n未找到已有模型，将从头开始训练")
    
    begin_time = time()

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(I_model, M_model, device, train_loader, 
                                 optimizer_I, optimizer_M, criterion, epoch, args.epochs)
        
        # 验证
        val_loss, val_acc, val_rrmse = validate_epoch(I_model, M_model, device, 
                                                       val_loader, criterion, epoch, args.epochs)

        # 学习率调度 - 基于验证loss调整
        scheduler_I.step(val_loss)
        scheduler_M.step(val_loss)
        
        # 获取当前学习率
        current_lr_I = optimizer_I.param_groups[0]['lr']
        current_lr_M = optimizer_M.param_groups[0]['lr']

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0  # 重置计数器
            print(f"  >> 保存最佳模型 (val_loss: {val_loss:.6f})")
            torch.save(I_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_INet_best.pkl'))
            torch.save(M_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_MNet_best.pkl'))
        else:
            epochs_no_improve += 1

        if val_acc > best_acc:
            best_acc = val_acc
        
        # Early stopping检查
        if epochs_no_improve >= PATIENCE:
            print(f"\n早停触发！已连续{PATIENCE}轮验证损失未改善")
            print(f"在第{epoch+1}轮停止训练")
            break

        # 定期保存
        if (epoch + 1) % 10 == 0:
            torch.save(I_model.state_dict(), 
                      os.path.join(args.save_dir, f'EEGIFNet_INet_epoch{epoch+1}.pkl'))
            torch.save(M_model.state_dict(), 
                      os.path.join(args.save_dir, f'EEGIFNet_MNet_epoch{epoch+1}.pkl'))

        # 计算已用时间
        elapsed_time = time() - begin_time
        minute = int(elapsed_time // 60)
        second = int(elapsed_time % 60)
        print(f"  >> 用时: {minute}m {second}s, 最佳验证Loss: {best_val_loss:.6f}, 最佳ACC: {best_acc:.4f}")
        print(f"  >> 当前学习率: I_Net={current_lr_I:.2e}, M_Net={current_lr_M:.2e}")
        print('-' * 80)

    print("训练完成!")
    print(f"总用时: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
    print(f"最佳验证Loss: {best_val_loss:.6f}")
    print(f"最佳ACC: {best_acc:.4f}")


if __name__ == '__main__':
    main()