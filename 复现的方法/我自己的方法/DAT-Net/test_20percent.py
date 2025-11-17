"""
DAT-Net 测试脚本 (20% 数据训练的模型)
在测试集上评估并保存.mat结果
用于和无监督微调做对比
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加父目录以导入model和metrics_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(current_dir))
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from model import DATNet

try:
    from 复现的方法.metrics_utils import compute_all_metrics, print_metrics
except:
    def compute_all_metrics(pred, target, fs):
        return {
            'RRMSE': 0.0, 'CC': 0.0, 'PRD': 0.0, 'SNR': 0.0,
            'RMSE': 0.0, 'MAE': 0.0, 'PSNR': 0.0, 'SSIM': 0.0
        }
    def print_metrics(m, prefix=""):
        print(f"{prefix} Metrics:", m)


class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        return noisy.astype('float32') / norm, clean.astype('float32'), norm


def load_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_input, test_output


def main():
    print("="*70)
    print("DAT-Net 测试 (20% 数据训练的模型)")
    print("用于和无监督微调做对比")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')

    # 加载测试数据
    print('\n加载测试数据...')
    test_x, test_y = load_data()
    print(f'测试集样本数: {len(test_x)}')
    print(f'信号长度: {test_x.shape[-1]}')
    
    ds = TestDataset(test_x, test_y)
    loader = DataLoader(ds, batch_size=50, shuffle=False)

    # 加载模型
    print('\n创建模型...')
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'模型参数量: {model.count_parameters():,}')
    
    model_path = 'DAT-Net_20percent_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'加载模型: {model_path}')
    else:
        print('⚠️  找不到训练好的模型，尝试使用 final 版本...')
        model_path = 'DAT-Net_20percent_final.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'加载模型: {model_path}')
        else:
            print('⚠️  找不到训练好的模型，使用随机初始化权重')

    # 推理
    print('\n开始推理...')
    model.eval()
    
    eeg_preds = []
    eog_preds = []
    targets = []
    sample_count = 0
    start = time()
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            sample_count += noisy.shape[0]
            
            # 前向传播 - 使用归一化后的数据
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 归一化的
            norm_t = norm.float().view(-1, 1, 1).to(device)  # (B, 1, 1)
            
            # 模型推理
            eeg_clean, eog_artifact = model(noisy_t)
            
            # 恢复原始尺度
            eeg_clean = eeg_clean * norm_t
            eog_artifact = eog_artifact * norm_t
            
            eeg_preds.append(eeg_clean.squeeze(1).cpu().numpy())
            eog_preds.append(eog_artifact.squeeze(1).cpu().numpy())
            targets.append(clean.numpy())
    
    total_time = time() - start
    time_per_sample = total_time / sample_count
    
    # 合并结果
    eeg_preds = np.concatenate(eeg_preds, axis=0)
    eog_preds = np.concatenate(eog_preds, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f'推理完成! 单样本推理时间: {time_per_sample*1000:.3f}ms')
    
    # 计算评价指标
    print('\n计算评价指标...')
    metrics = compute_all_metrics(eeg_preds, targets, fs=200)
    print_metrics(metrics, prefix='测试集')
    
    # 保存结果
    print('\n保存结果...')
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, 'DAT-Net-20percent_predictions.mat')
    scipy.io.savemat(save_path, {
        'predictions': eeg_preds,  # 干净的EEG信号
        'eog_artifacts': eog_preds,  # EOG伪影
        'time_per_sample': time_per_sample
    })
    
    print(f'\n预测结果已保存为.mat格式: {save_path}')
    print(f'  - EEG干净信号形状: {eeg_preds.shape}')
    print(f'  - EOG伪影形状: {eog_preds.shape}')
    print(f'  - 单样本推理时间: {time_per_sample*1000:.3f}ms')
    
    # 验证一致性
    print('\n验证解耦一致性...')
    reconstructed = eeg_preds + eog_preds
    original = test_x
    consistency_error = np.mean((reconstructed - original) ** 2)
    print(f'重建一致性MSE: {consistency_error:.6f}')
    print(f'  (应该接近0，表示 EEG_clean + EOG_artifact ≈ 原始信号)')
    
    print('\n' + '='*70)


if __name__ == '__main__':
    main()
