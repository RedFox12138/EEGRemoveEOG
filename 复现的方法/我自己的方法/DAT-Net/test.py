"""
DAT-Net 测试脚本
在测试集上评估并保存.mat结果
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time

from 复现的方法.metrics_utils import compute_all_metrics, print_metrics

# 添加父目录以导入metrics_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from model import DATNet

# 导入数据集配置
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)
from dataset_config import get_dataset_config

# 获取全模拟数据集配置
dataset_config = get_dataset_config('fully_simulated')


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
    test_input = scipy.io.loadmat(dataset_config['test_contaminated_path'])[dataset_config['data_key']]
    test_output = scipy.io.loadmat(dataset_config['test_pure_path'])[dataset_config['data_key']]
    return test_input, test_output


def main():
    print("="*70)
    print("DAT-Net 测试")
    print("Disentangling Attention Temporal-Network")
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
    
    model_path = 'DAT-Net_best.pth'
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
            
            # 前向传播
            noisy_t = noisy.unsqueeze(1).to(device)  # (B, 1, L)
            eeg_clean, eog_artifact = model(noisy_t)
            
            # 恢复尺度
            norm_np = norm.numpy().reshape(-1, 1)
            eeg_clean_scaled = eeg_clean.squeeze(1).cpu().numpy() * norm_np
            eog_artifact_scaled = eog_artifact.squeeze(1).cpu().numpy() * norm_np
            
            eeg_preds.append(eeg_clean_scaled)
            eog_preds.append(eog_artifact_scaled)
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
    metrics = compute_all_metrics(eeg_preds, targets, fs=dataset_config['sampling_rate'])
    print_metrics(metrics, prefix='测试集')
    
    # 保存结果
    print('\n保存结果...')
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    
    save_path = os.path.join(out_dir, 'DAT-Net_predictions.mat')
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
