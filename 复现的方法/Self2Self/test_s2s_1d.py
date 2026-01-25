"""
Self2Self 1D EEG Denoising 测试脚本
评估训练好的模型并保存预测结果
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入数据配置
from data_config import *

# 导入metrics
try:
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs):
        return {'RRMSE':0,'CC':0,'PRD':0,'SNR':0,'RMSE':0,'MAE':0,'PSNR':0,'SSIM':0}
    def print_metrics(m, prefix=""):
        print(prefix, 'Metrics:', m)

from s2s_model_1d import Self2Self_UNet1D

# ========== 数据集选择 ==========
# 数据集由 data_config.py 中的 DATASET_NAME 变量控制
# 可选值: 'semi_simulated' 或 'fully_simulated'
# 请修改 data_config.py 中的 DATASET_NAME 来切换数据集
# ================================

# Self2Self推理参数
N_PREDICTIONS = 100  # 多次预测平均的次数


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
        
        # 归一化到[0, 1]
        norm_min = np.min(noisy)
        norm_max = np.max(noisy)
        norm_range = norm_max - norm_min
        
        if norm_range == 0:
            norm_range = 1.0
        
        noisy_norm = (noisy - norm_min) / norm_range
        
        return (torch.from_numpy(noisy_norm).float(), 
               torch.from_numpy(clean).float(),
               norm_min, norm_range)


def load_test_data_by_snr(snr_db):
    """
    根据SNR加载测试数据
    """
    test_snr_paths = dataset_config['test_snr_paths']
    contaminated_path = test_snr_paths[snr_db]['contaminated']
    pure_path = test_snr_paths[snr_db]['pure']
    
    test_input = scipy.io.loadmat(contaminated_path)[DATA_KEY]
    test_output = scipy.io.loadmat(pure_path)[PURE_KEY]
    return test_input, test_output


def main():
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    print('='*70)
    print('Self2Self 1D EEG Denoising 测试')
    print('='*70)
    print(f'工作目录: {os.getcwd()}')
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建模型（参数要与训练时一致）
    model = Self2Self_UNet1D(
        in_channels=1,
        base_channels=48,
        n_depth=5,
        dropout=0.3
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 加载训练好的权重（使用相对路径）
    model_path = f'checkpoints/Self2Self_{DATASET_NAME}_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'加载模型: {model_path}')
    else:
        print('⚠️ 找不到训练好的模型，尝试 final 版本...')
        model_path = f'checkpoints/Self2Self_{DATASET_NAME}_final.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'加载模型: {model_path}')
        else:
            print('⚠️ 找不到训练好的模型，使用随机初始化权重')
    
    # 获取SNR级别
    snr_levels = dataset_config['test_snr_levels']
    print(f"\n多SNR测试模式，SNR级别: {snr_levels}")
    
    results_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(results_dir, exist_ok=True)
    
    # 对每个SNR级别进行测试
    for snr_db in snr_levels:
        print(f"\n========== 测试 SNR = {snr_db} dB ==========")
        
        # 加载数据
        test_x, test_y = load_test_data_by_snr(snr_db)
        print(f'测试集样本数: {len(test_x)}')
        print(f'信号长度: {test_x.shape[1]}')
        
        # 创建数据集和加载器
        ds = TestDataset(test_x, test_y)
        loader = DataLoader(ds, batch_size=10, shuffle=False)  # 较小batch size因为要做多次预测
        
        # 推理
        model.eval()
        predictions = []
        targets = []
        sample_count = 0
        
        print(f'开始推理（每个样本{N_PREDICTIONS}次预测平均）...')
        start = time()
        
        with torch.no_grad():
            for noisy_norm, clean, norm_min, norm_range in loader:
                sample_count += noisy_norm.shape[0]
                
                # 转换为tensor
                noisy_norm = noisy_norm.unsqueeze(1).to(device)  # (batch, 1, time)
                
                # Self2Self推理：多次预测并平均
                pred = model.predict_average(noisy_norm, n_predictions=N_PREDICTIONS)
                
                # 反归一化
                pred = pred.squeeze(1).cpu().numpy()  # (batch, time)
                pred = pred * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
                
                predictions.append(pred)
                targets.append(clean.numpy())
        
        total_time = time() - start
        time_per_sample = total_time / max(1, sample_count)
        
        # 合并结果
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        print(f'推理完成!')
        print(f'  总时间: {int(total_time//60)}min {int(total_time%60)}s')
        print(f'  单样本时间: {time_per_sample*1000:.3f} ms')
        print(f'  (包含{N_PREDICTIONS}次预测平均)')
        
        # 计算评价指标
        print('计算评价指标...')
        metrics = compute_all_metrics(predictions, targets, fs=SAMPLING_RATE)
        print_metrics(metrics, prefix='测试集')
        
        # 保存带SNR标识的结果
        save_path = os.path.join(results_dir, f'Self2Self_predictions_SNR{snr_db}dB.mat')
        
        scipy.io.savemat(save_path, {
            'predictions': predictions,
            'time_per_sample': time_per_sample,
            'n_predictions': N_PREDICTIONS,
        })
        
        print(f'预测结果已保存: {save_path}')
    
    print('\n全部SNR测试完成！')
    print('='*70)


if __name__ == '__main__':
    main()