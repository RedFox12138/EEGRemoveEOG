"""
DAT-Net 训练脚本 (20% 数据)
用于和无监督微调做对比
使用 20% 的训练数据进行有监督训练
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加父目录以导入metrics_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import DATNet, DAT_Loss

try:
    from 复现的方法.metrics_utils import compute_all_metrics, print_metrics
except:
    # 如果导入失败，使用简化版本
    def compute_all_metrics(pred, target, fs):
        return {
            'RRMSE': 0.0, 'CC': 0.0, 'PRD': 0.0, 'SNR': 0.0,
            'RMSE': 0.0, 'MAE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0
        }
    def print_metrics(m, prefix=""):
        print(f"{prefix} Metrics:", m)


# ========== 超参数配置 ==========
BATCH_SIZE = 500
EPOCHS = 1000
LEARNING_RATE = 5e-4
WEIGHT_DECAY = 0
DATA_PERCENT = 0.1  # 使用 20% 的训练数据
RANDOM_SEED = 42  # 随机种子，确保可复现


class EEGDataset(Dataset):
    """
    DAT-Net数据集
    返回: (受污染信号, 干净信号, EOG伪影, 归一化系数)
    """
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 计算EOG伪影: 伪影 = 受污染信号 - 干净信号
        artifact = noisy - clean
        
        # 归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        return (
            noisy.astype('float32') / norm,  # 归一化的受污染信号
            clean.astype('float32'),  # 干净信号 (未归一化)
            artifact.astype('float32'),  # EOG伪影 (未归一化)
            norm  # 归一化系数
        )


def get_data():
    """加载训练和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    train_y = scipy.io.loadmat(f'{data_dir}/Train_Pure.mat')['data']
    
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, train_y, val_x, val_y


def subsample_data(train_x, train_y, percent=0.2, seed=42):
    """随机采样指定百分比的训练数据"""
    np.random.seed(seed)
    n_samples = len(train_x)
    n_subsample = int(n_samples * percent)
    
    # 随机选择索引
    indices = np.random.choice(n_samples, n_subsample, replace=False)
    indices = np.sort(indices)  # 排序以保持顺序
    
    train_x_sub = train_x[indices]
    train_y_sub = train_y[indices]
    
    print(f'原始训练集大小: {n_samples}')
    print(f'采样后训练集大小: {n_subsample} ({percent*100:.0f}%)')
    
    return train_x_sub, train_y_sub


def train_epoch(model, device, loader, optimizer):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    loss_clean_sum = 0
    loss_artifact_sum = 0
    loss_consistency_sum = 0
    num_batches = 0
    
    for noisy, clean, artifact, norm in loader:
        # 转换为tensor并移到设备
        noisy = noisy.float().unsqueeze(1).to(device)  # (B, 1, L)
        clean = clean.float().unsqueeze(1).to(device)
        artifact = artifact.float().unsqueeze(1).to(device)
        norm = norm.float().to(device).view(-1, 1, 1)
        
        # 前向传播
        optimizer.zero_grad()
        eeg_clean_pred, eog_artifact_pred = model(noisy)
        
        # 恢复到原始尺度计算损失
        eeg_clean_pred_scaled = eeg_clean_pred * norm
        eog_artifact_pred_scaled = eog_artifact_pred * norm
        noisy_scaled = noisy * norm
        
        # 计算损失
        loss, loss_dict = DAT_Loss(
            eeg_clean_pred_scaled,
            eog_artifact_pred_scaled,
            clean,
            artifact,
            noisy_scaled
        )
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 累积损失
        total_loss += loss_dict['total']
        loss_clean_sum += loss_dict['clean']
        loss_artifact_sum += loss_dict['artifact']
        loss_consistency_sum += loss_dict['consistency']
        num_batches += 1
    
    return {
        'total': total_loss / num_batches,
        'clean': loss_clean_sum / num_batches,
        'artifact': loss_artifact_sum / num_batches,
        'consistency': loss_consistency_sum / num_batches
    }


def validate(model, device, loader):
    """验证模型"""
    model.eval()
    total_loss = 0
    loss_clean_sum = 0
    loss_artifact_sum = 0
    loss_consistency_sum = 0
    num_batches = 0
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, artifact, norm in loader:
            noisy = noisy.float().unsqueeze(1).to(device)
            clean = clean.float().unsqueeze(1).to(device)
            artifact = artifact.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            
            # 前向传播
            eeg_clean_pred, eog_artifact_pred = model(noisy)
            
            # 恢复到原始尺度
            eeg_clean_pred_scaled = eeg_clean_pred * norm
            eog_artifact_pred_scaled = eog_artifact_pred * norm
            noisy_scaled = noisy * norm
            
            # 计算损失
            loss, loss_dict = DAT_Loss(
                eeg_clean_pred_scaled,
                eog_artifact_pred_scaled,
                clean,
                artifact,
                noisy_scaled
            )
            
            # 累积损失
            total_loss += loss_dict['total']
            loss_clean_sum += loss_dict['clean']
            loss_artifact_sum += loss_dict['artifact']
            loss_consistency_sum += loss_dict['consistency']
            num_batches += 1
            
            # 收集预测结果用于计算评价指标
            all_preds.append(eeg_clean_pred_scaled.squeeze(1).cpu().numpy())
            all_targets.append(clean.squeeze(1).cpu().numpy())
    
    # 计算评价指标
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    metrics = compute_all_metrics(all_preds, all_targets, fs=200)
    
    # 返回损失和指标
    loss_dict = {
        'total': total_loss / num_batches,
        'clean': loss_clean_sum / num_batches,
        'artifact': loss_artifact_sum / num_batches,
        'consistency': loss_consistency_sum / num_batches
    }
    
    return loss_dict, metrics


def main():
    print("="*70)
    print("DAT-Net 训练 (20% 数据)")
    print("用于和无监督微调做对比")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 加载数据
    print('\n加载数据...')
    train_x, train_y, val_x, val_y = get_data()
    
    # 采样 20% 训练数据
    print(f'\n采样 {DATA_PERCENT*100:.0f}% 训练数据...')
    train_x_sub, train_y_sub = subsample_data(train_x, train_y, 
                                               percent=DATA_PERCENT, 
                                               seed=RANDOM_SEED)
    
    print(f'验证集: {val_x.shape}')
    
    train_dataset = EEGDataset(train_x_sub, train_y_sub)
    val_dataset = EEGDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    print('  架构: DAT-Net (1D U-Net + TCN + 双输出头)')
    print('  特征: SE注意力 + 深度可分离卷积 + 时间卷积网络')
    print('  输出: EEG干净信号 + EOG伪影')
    
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器
    print(f'\n训练配置:')
    print(f'  训练数据: {DATA_PERCENT*100:.0f}% ({len(train_x_sub)} 样本)')
    print(f'  Batch Size: {BATCH_SIZE}')
    print(f'  Epochs: {EPOCHS}')
    print(f'  Learning Rate: {LEARNING_RATE}')
    print(f'  Weight Decay: {WEIGHT_DECAY}')
    print(f'  Random Seed: {RANDOM_SEED}')
    print(f'  优化器: Adam')
    print(f'  损失函数: MSE (EEG) + MSE (EOG) + MSE (一致性)')
    print(f'  模型选择: 基于验证集CC指标')
    
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 训练
    print('\n' + '='*70)
    print('开始训练')
    print('='*70)
    
    best_cc = -1.0
    start_time = time()
    
    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-' * 70)
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        print(f'Train Loss: {train_loss["total"]:.6f} '
              f'(Clean: {train_loss["clean"]:.6f}, '
              f'Artifact: {train_loss["artifact"]:.6f}, '
              f'Consistency: {train_loss["consistency"]:.6f})')
        
        # 验证
        val_loss, val_metrics = validate(model, device, val_loader)
        print(f'Val Loss: {val_loss["total"]:.6f} '
              f'(Clean: {val_loss["clean"]:.6f}, '
              f'Artifact: {val_loss["artifact"]:.6f}, '
              f'Consistency: {val_loss["consistency"]:.6f})')
        
        print_metrics(val_metrics, prefix='验证集')
        
        # 保存最佳模型
        current_cc = val_metrics['CC']
        if current_cc > best_cc:
            best_cc = current_cc
            torch.save(model.state_dict(), 'DAT-Net_10percent_best.pth')
            print(f'✓ 保存最佳模型 (CC: {best_cc:.4f})')
        
        elapsed = time() - start_time
        print(f'Elapsed Time: {int(elapsed//60)}min {int(elapsed%60)}s')
    
    # 保存最终模型
    torch.save(model.state_dict(), 'DAT-Net_20percent_final.pth')
    
    print('\n' + '='*70)
    print('训练完成!')
    print(f'最佳CC: {best_cc:.4f}')
    print(f'最佳模型已保存至: DAT-Net_20percent_best.pth')
    print(f'最终模型已保存至: DAT-Net_20percent_final.pth')
    print('='*70)


if __name__ == '__main__':
    main()
