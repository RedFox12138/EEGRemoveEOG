"""
EEGIFNet 真实数据集测试脚本
使用训练好的模型对真实数据集进行测试
"""
import os
import sys
import scipy.io
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from time import time
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))  # 复现的方法目录
sys.path.insert(0, parent_dir)
sys.path.insert(0, grandparent_dir)  # 添加以访问load_real_dataset_split

from EEGIFNet_1200 import MA_INet, MA_MNet
from real_data_config import *
from load_real_dataset_split import load_real_dataset_split

BATCH_SIZE = 50


class RealDataset(Dataset):
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
        
        return noisy_normalized, norm_factor


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
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    return loader, data


def test_model(I_model, M_model, device, test_loader):
    """
    在真实数据集上进行推理
    
    Returns:
        predictions: 预测的去噪结果 (N, L)
        artifacts: 提取的伪影 (N, L)
        time_per_sample: 单样本推理时间
    """
    I_model.eval()
    M_model.eval()
    
    all_predictions = []
    all_artifacts = []
    sample_count = 0
    
    start_time = time()
    
    with torch.no_grad():
        for x, norm_factors in test_loader:
            sample_count += x.size(0)
            
            x = x.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1, 1)
            
            # 第一步：使用INet提取clean和noise
            # INet返回 (clean_eeg, noise)
            y_inet, z_inet = I_model(x.unsqueeze(1))  # 两个输出都是 (B, L)
            
            # 第二步：使用MNet融合
            # MNet需要3个输入: 原始信号, INet的clean, INet的noise
            y_mnet = M_model(x.unsqueeze(1), y_inet, z_inet)  # 返回 (B, L) 而不是 (B, 1, L)
            
            # 计算最终的artifact
            z_hat = x - y_mnet  # (B, L)
            
            # 反归一化
            # norm_factors 是 (B, 1, 1)，需要转为 (B, 1) 以匹配 (B, L)
            norm_factor_2d = norm_factors.view(-1, 1)  # (B, 1)
            y_hat_restored = y_mnet * norm_factor_2d  # (B, L)
            z_hat_restored = z_hat * norm_factor_2d  # (B, L)
            
            # 调试：打印第一个batch的形状
            if len(all_predictions) == 0:
                print(f'  调试信息:')
                print(f'    x shape: {x.shape}')
                print(f'    y_inet shape: {y_inet.shape}')
                print(f'    z_inet shape: {z_inet.shape}')
                print(f'    y_mnet shape: {y_mnet.shape}')
                print(f'    z_hat shape: {z_hat.shape}')
                print(f'    y_hat_restored shape: {y_hat_restored.shape}')
                print(f'    z_hat_restored shape: {z_hat_restored.shape}')
            
            all_predictions.append(y_hat_restored.cpu().numpy())
            all_artifacts.append(z_hat_restored.cpu().numpy())
    
    total_time = time() - start_time
    time_per_sample = total_time / sample_count
    
    predictions = np.concatenate(all_predictions, axis=0)
    artifacts = np.concatenate(all_artifacts, axis=0)
    
    return predictions, artifacts, time_per_sample


def main():
    print("=" * 80)
    print("EEGIFNet 真实数据集测试")
    print("=" * 80)
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载数据
    test_loader, original_data = load_real_data()
    
    # 创建模型
    print('\n创建模型...')
    I_model = MA_INet().to(device)
    M_model = MA_MNet().to(device)
    
    # 加载预训练模型
    print(f'\n加载预训练模型...')
    print(f'INet 模型路径: {INET_MODEL_PATH}')
    print(f'MNet 模型路径: {MNET_MODEL_PATH}')
    
    if not os.path.exists(INET_MODEL_PATH):
        print(f'✗ 找不到 INet 模型文件: {INET_MODEL_PATH}')
        return
    
    if not os.path.exists(MNET_MODEL_PATH):
        print(f'✗ 找不到 MNet 模型文件: {MNET_MODEL_PATH}')
        return
    
    # 加载模型（使用torch.load）
    I_state_dict = torch.load(INET_MODEL_PATH, map_location=device)
    M_state_dict = torch.load(MNET_MODEL_PATH, map_location=device)
    
    I_model.load_state_dict(I_state_dict)
    M_model.load_state_dict(M_state_dict)
    
    print('  ✓ 模型加载成功')
    
    # 测试
    print('\n' + '='*80)
    print('开始测试...')
    print('='*80)
    
    predictions, artifacts, time_per_sample = test_model(I_model, M_model, device, test_loader)
    
    print(f'\n测试完成！')
    print(f'  预测结果形状: {predictions.shape}')
    print(f'  伪影形状: {artifacts.shape}')
    print(f'  平均推理时间: {time_per_sample*1000:.2f} ms/样本')
    print(f'  总推理时间: {time_per_sample*len(predictions):.2f} 秒')
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = predictions + artifacts
    consistency_error = np.mean((reconstructed - original_data) ** 2)
    print(f'  重建一致性 MSE: {consistency_error:.6f}')
    print(f'  (应该接近0，表示 EEG_clean + EOG_artifact ≈ 原始信号)')
    
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
