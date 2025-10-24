"""
EEGIFNet训练脚本 - 使用ASNet数据集格式
保持原有网络结构和参数,但使用与ASNet相同的数据加载和标准化方式
"""
import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from time import time
from torch.utils.data import Dataset, DataLoader
from EEGIFNet_1200 import MA_INet, MA_MNet, weights_init
from config import cal_ACC_tensor, cal_RRMSE_tensor, cal_SNR
import os
import argparse

BATCH_SIZE = 256  # EEGIFNet原始batch size
LEARNING_RATE = 5e-5  # EEGIFNet原始学习率
EPOCHS = 80  # EEGIFNet原始训练轮数

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
        clean_normalized = clean / norm_factor  # clean信号也用同样的因子归一化

        # 增加一个通道维度
        noisy_normalized = noisy_normalized[np.newaxis, :]
        clean_normalized = clean_normalized[np.newaxis, :]

        return noisy_normalized, clean_normalized, norm_factor


def get_data(data_path, batch_size):
    """
    加载数据并创建DataLoader
    """
    # 加载已经分割好的数据
    raw_eeg_segments = np.load(os.path.join(data_path, 'Contaminated.npy'), allow_pickle=True)
    clean_eeg_segments = np.load(os.path.join(data_path, 'Pure_Data.npy'), allow_pickle=True)
    
    print(f"加载数据: {raw_eeg_segments.shape}, {clean_eeg_segments.shape}")
    print(f"时间点数量: {raw_eeg_segments.shape[1]}")

    # 数据集拆分 (80% 训练, 10% 验证, 10% 测试)
    num_samples = len(raw_eeg_segments)
    train_end = int(num_samples * 0.8)
    verify_end = int(num_samples * 0.9)

    train_input = raw_eeg_segments[:train_end]
    verify_input = raw_eeg_segments[train_end:verify_end]
    test_input = raw_eeg_segments[verify_end:]

    train_output = clean_eeg_segments[:train_end]
    verify_output = clean_eeg_segments[train_end:verify_end]
    test_output = clean_eeg_segments[verify_end:]

    print(f"训练集: {len(train_input)}, 验证集: {len(verify_input)}, 测试集: {len(test_input)}")

    train_dataset = EEGDataset(train_input, train_output, is_train=True)
    verify_dataset = EEGDataset(verify_input, verify_output, is_train=False)
    test_dataset = EEGDataset(test_input, test_output, is_train=False)

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

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, verify_loader, test_loader


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
        
        # 数据移到设备
        x = x.float().to(device)
        y = y.float().to(device)
        norm_factors = norm_factors.float().to(device).view(-1, 1, 1)
        
        # 计算噪声信号 (contaminated - clean = noise)
        z = x.squeeze() - y.squeeze()
        z = z.detach()

        optimizer_I.zero_grad()
        optimizer_M.zero_grad()

        # INet预测clean EEG和noise
        e_outputs, n_outputs = I_model(x)
        # MNet融合预测
        outputs = M_model(x, e_outputs, n_outputs)

        # 计算loss (在归一化空间中)
        loss_e = criterion(e_outputs, y.squeeze())
        loss_n = criterion(n_outputs, z)
        loss_all = criterion(outputs, y.squeeze())

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
    同时计算反归一化后的指标
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

    with torch.no_grad():
        for batch_idx, (x, y, norm_factors) in enumerate(val_loader):
            val_step_num += 1

            x = x.float().to(device)
            y = y.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1, 1)

            # 计算噪声
            z = x.squeeze() - y.squeeze()

            # 模型预测
            e_outputs, n_outputs = I_model(x)
            outputs = M_model(x, e_outputs, n_outputs)

            # 归一化空间的loss和指标
            loss = criterion(outputs, y.squeeze())
            total_val_loss += loss.item()

            # 归一化空间的指标
            acc_e = cal_ACC_tensor(e_outputs.detach(), y.squeeze().detach())
            sum_acc_e += acc_e
            rrmse_e = cal_RRMSE_tensor(e_outputs.detach(), y.squeeze().detach())
            sum_rrmse_e += rrmse_e

            acc_n = cal_ACC_tensor(n_outputs.detach(), z.detach())
            sum_acc_n += acc_n
            rrmse_n = cal_RRMSE_tensor(n_outputs.detach(), z.detach())
            sum_rrmse_n += rrmse_n

            acc = cal_ACC_tensor(outputs.detach(), y.squeeze().detach())
            sum_acc += acc
            rrmse = cal_RRMSE_tensor(outputs.detach(), y.squeeze().detach())
            sum_rrmse += rrmse

            # 反归一化后的指标
            # norm_factors: (batch, 1, 1), outputs: (batch, time)
            norm_factors_2d = norm_factors.squeeze(-1)  # (batch, 1)
            outputs_denorm = outputs * norm_factors_2d
            y_denorm = y.squeeze() * norm_factors_2d
            
            acc_denorm = cal_ACC_tensor(outputs_denorm.detach(), y_denorm.detach())
            sum_acc_denorm += acc_denorm
            rrmse_denorm = cal_RRMSE_tensor(outputs_denorm.detach(), y_denorm.detach())
            sum_rrmse_denorm += rrmse_denorm
            
            snr = cal_SNR(outputs_denorm, y_denorm)
            sum_snr += snr

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

    print(f"Epoch [{epoch+1}/{epochs}] Val - Loss: {average_val_loss:.6f}")
    print(f"  [Normalized] ACC_e: {acc_e:.4f}, RRMSE_e: {rrmse_e:.4f}")
    print(f"  [Normalized] ACC_n: {acc_n:.4f}, RRMSE_n: {rrmse_n:.4f}")
    print(f"  [Normalized] ACC: {acc:.4f}, RRMSE: {rrmse:.4f}")
    print(f"  [Denormalized] ACC: {acc_denorm:.4f}, RRMSE: {rrmse_denorm:.4f}, SNR: {snr_avg:.2f} dB")

    return average_val_loss, acc_denorm, rrmse_denorm


def main():
    parser = argparse.ArgumentParser(description='EEGIFNet Training with ASNet Dataset')
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
    print("加载数据集...")
    train_loader, val_loader, test_loader = get_data(args.data_path, args.batch_size)
    
    # 获取数据的时间点数量
    sample_data = np.load(os.path.join(args.data_path, 'Contaminated.npy'), allow_pickle=True)
    input_length = sample_data.shape[1]
    print(f"数据时间点: {input_length}")

    # 初始化模型 - 使用EEGIFNet的MA_INet和MA_MNet (适配input_length)
    print("初始化模型...")
    I_model = MA_INet(input_length=input_length).apply(weights_init).to(device)
    M_model = MA_MNet().apply(weights_init).to(device)

    # 优化器 - 使用RMSprop,与EEGIFNet原始设置一致
    optimizer_I = torch.optim.RMSprop(I_model.parameters(), lr=args.lr, alpha=0.9)
    optimizer_M = torch.optim.RMSprop(M_model.parameters(), lr=args.lr, alpha=0.9)

    # 损失函数
    criterion = nn.MSELoss()

    # 训练
    print("开始训练...")
    best_val_loss = float('inf')
    best_acc = 0
    begin_time = time()

    for epoch in range(args.epochs):
        # 训练
        train_loss = train_epoch(I_model, M_model, device, train_loader, 
                                 optimizer_I, optimizer_M, criterion, epoch, args.epochs)
        
        # 验证
        val_loss, val_acc, val_rrmse = validate_epoch(I_model, M_model, device, 
                                                       val_loader, criterion, epoch, args.epochs)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  >> 保存最佳模型 (val_loss: {val_loss:.6f})")
            torch.save(I_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_INet_best.pkl'))
            torch.save(M_model.state_dict(), os.path.join(args.save_dir, 'EEGIFNet_MNet_best.pkl'))

        if val_acc > best_acc:
            best_acc = val_acc

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
        print('-' * 80)

    print("训练完成!")
    print(f"总用时: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
    print(f"最佳验证Loss: {best_val_loss:.6f}")
    print(f"最佳ACC: {best_acc:.4f}")


if __name__ == '__main__':
    main()
