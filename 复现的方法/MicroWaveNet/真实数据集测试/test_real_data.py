"""
MicroWaveNet 真实数据集测试脚本
使用训练好的模型对真实数据集进行测试
"""
import os
import sys
import scipy.io
import torch
import torch.utils.data as Data
import numpy as np
from time import time
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)  # 复现的方法目录
code_dir = os.path.join(parent_dir, 'MicroWaveNet-main', 'code')
sys.path.insert(0, code_dir)
sys.path.insert(0, grandparent_dir)  # 添加以访问load_real_dataset_split

from cbamdropout import EEGNetMorletWindowCBAMDropout
from real_data_config import *
from load_real_dataset_split import load_real_dataset_split

BATCH_SIZE = 50


class RealDataset(Data.Dataset):
    """真实数据集（用于测试）"""
    def __init__(self, noisy_signals):
        self.noisy_signals = noisy_signals

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        
        # 归一化
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0
        
        noisy_normalized = noisy / norm_factor
        noisy_normalized = noisy_normalized[np.newaxis, :].astype(np.float32)
        
        return noisy_normalized, np.array([norm_factor], dtype=np.float32)


def load_real_data():
    """加载真实数据（只加载测试集）"""
    # 使用统一的数据划分函数，只返回测试集
    data = load_real_dataset_split(
        data_path=REAL_DATA_PATH,
        data_key=DATA_KEY,
        return_train=False  # 只需要测试集
    )
    
    # 创建数据集
    dataset = RealDataset(data)
    loader = Data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    return loader, data


def test_model(model, device, test_loader):
    """
    在真实数据集上进行推理
    
    Returns:
        predictions: 预测的去噪结果 (N, L)
        time_per_sample: 单样本推理时间
    """
    model.eval()
    
    all_predictions = []
    sample_count = 0
    
    start_time = time()
    
    with torch.no_grad():
        for noisy, norm in test_loader:
            noisy = noisy.to(device)
            norm = norm.to(device)
            
            # 模型推理
            out = model(noisy)
            eeg_pred = (out[0] * norm.view(-1, 1, 1)).cpu().numpy()
            
            all_predictions.append(eeg_pred)
            sample_count += eeg_pred.shape[0]
    
    total_time = time() - start_time
    time_per_sample = total_time / max(1, sample_count)
    
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # 去掉通道维度，从 (N, 1, L) 变为 (N, L)
    all_predictions = np.squeeze(all_predictions, axis=1)
    
    return all_predictions, time_per_sample


def main():
    print("=" * 80)
    print("MicroWaveNet 真实数据集测试")
    print("=" * 80)
    
    # 设备
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    test_loader, original_data = load_real_data()
    
    # 创建模型
    print('\n创建模型...')
    model = EEGNetMorletWindowCBAMDropout(device=device)
    model.to(device)
    
    # 加载预训练模型
    print(f'\n加载预训练模型...')
    print(f'模型路径: {MODEL_PATH}')
    
    if not os.path.exists(MODEL_PATH):
        print(f'✗ 找不到模型文件: {MODEL_PATH}')
        print('请确保模型文件存在！')
        return
    
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print('  ✓ 模型加载成功')
    
    # 测试
    print('\n' + '='*80)
    print('开始测试...')
    print('='*80)
    
    predictions, time_per_sample = test_model(model, device, test_loader)
    
    print(f'\n测试完成！')
    print(f'  预测结果形状: {predictions.shape}')
    print(f'  平均推理时间: {time_per_sample*1000:.2f} ms/样本')
    print(f'  总推理时间: {time_per_sample*len(predictions):.2f} 秒')
    
    # 计算伪影（原始信号 - 去噪信号）
    artifacts = original_data - predictions
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = predictions + artifacts
    consistency_error = np.mean((reconstructed - original_data) ** 2)
    print(f'  重建一致性 MSE: {consistency_error:.6f}')
    
    # 统计信息
    print('\n统计信息:')
    original_std = np.std(original_data)
    cleaned_std = np.std(predictions)
    artifact_std = np.std(artifacts)
    print(f'  原始信号标准差: {original_std:.4f}')
    print(f'  去噪信号标准差: {cleaned_std:.4f}')
    print(f'  伪影标准差: {artifact_std:.4f}')
    
    power_reduction = (np.mean(original_data ** 2) - np.mean(predictions ** 2)) / np.mean(original_data ** 2)
    print(f'  平均功率降低: {power_reduction * 100:.2f}%')
    
    # 保存结果到本地目录
    print('\n保存结果...')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scipy.io.savemat(PREDICTION_SAVE_PATH, {
        'cleaned_eeg': predictions,
        'extracted_eog': artifacts,
        'original': original_data,
        'time_per_sample': time_per_sample,
        'consistency_error': consistency_error,
        'power_reduction': power_reduction,
        'sampling_rate': SAMPLING_RATE,
        'window_size': WINDOW_SIZE,
    })
    print(f'  ✓ 本地结果已保存: {PREDICTION_SAVE_PATH}')
    
    # 同时保存到总结果目录
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    shutil.copy(PREDICTION_SAVE_PATH, FINAL_PREDICTION_PATH)
    print(f'  ✓ 总结果已保存: {FINAL_PREDICTION_PATH}')
    print(f'    - 去噪 EEG 形状: {predictions.shape}')
    print(f'    - 提取 EOG 形状: {artifacts.shape}')
    
    print('\n' + '='*80)
    print('测试完成！')
    print('='*80)


if __name__ == '__main__':
    main()
