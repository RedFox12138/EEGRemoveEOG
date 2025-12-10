"""
基于阈值方法 - 真实数据集测试脚本
对真实数据集进行去噪处理并保存结果
"""
import os
import sys
import scipy.io
import numpy as np
from time import time
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(os.path.dirname(current_dir))  # 复现的方法目录
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)
sys.path.insert(0, grandparent_dir)  # 添加以访问load_real_dataset_split

from SingleDenoise_CORRECTED import eog_removal_corrected
from real_data_config import *
from load_real_dataset_split import load_real_dataset_split


def load_real_data():
    """加载真实数据（只加载测试集）"""
    # 使用统一的数据划分函数，只返回测试集
    test_data = load_real_dataset_split(
        data_path=REAL_DATA_PATH,
        data_key=DATA_KEY,
        return_train=False  # 只需要测试集
    )
    return test_data


def process_all_samples(contaminated_data, fs=250):
    """
    处理所有样本
    
    Args:
        contaminated_data: [n_samples, n_timepoints]
        fs: 采样率
    
    Returns:
        predictions: 去噪后的信号
        time_per_sample: 单样本平均处理时间
    """
    n_samples, n_timepoints = contaminated_data.shape
    print(f"\n开始处理 {n_samples} 个样本...")
    
    predictions = np.zeros_like(contaminated_data)
    
    start_time = time()
    
    for i in range(n_samples):
        signal = contaminated_data[i, :]
        clean_signal = eog_removal_corrected(signal, fs=fs, visualize=False)
        predictions[i, :] = clean_signal
        
        # 进度显示
        if (i + 1) % 10 == 0:
            elapsed = time() - start_time
            avg_time = elapsed / (i + 1)
            remaining = avg_time * (n_samples - i - 1)
            print(f"  已处理 {i+1}/{n_samples} 样本 "
                  f"(平均 {avg_time:.3f}s/样本, 预计剩余 {remaining/60:.1f}分钟)")
    
    total_time = time() - start_time
    time_per_sample = total_time / n_samples
    
    print(f"\n处理完成!")
    print(f"  总耗时: {total_time:.1f}秒")
    print(f"  单样本平均时间: {time_per_sample:.3f}秒")
    
    return predictions, time_per_sample


def main():
    print("=" * 80)
    print("基于阈值方法 - 真实数据集测试")
    print("=" * 80)
    
    # 加载数据
    test_x = load_real_data()
    
    # 处理所有样本
    predictions, time_per_sample = process_all_samples(test_x, fs=SAMPLING_RATE)
    
    print(f'\n测试完成！')
    print(f'  预测结果形状: {predictions.shape}')
    print(f'  平均处理时间: {time_per_sample*1000:.2f} ms/样本')
    
    # 计算伪影（原始信号 - 去噪信号）
    artifacts = test_x - predictions
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = predictions + artifacts
    consistency_error = np.mean((reconstructed - test_x) ** 2)
    print(f'  重建一致性 MSE: {consistency_error:.6f}')
    
    # 统计信息
    print('\n统计信息:')
    original_std = np.std(test_x)
    cleaned_std = np.std(predictions)
    artifact_std = np.std(artifacts)
    print(f'  原始信号标准差: {original_std:.4f}')
    print(f'  去噪信号标准差: {cleaned_std:.4f}')
    print(f'  伪影标准差: {artifact_std:.4f}')
    
    power_reduction = (np.mean(test_x ** 2) - np.mean(predictions ** 2)) / np.mean(test_x ** 2)
    print(f'  平均功率降低: {power_reduction * 100:.2f}%')
    
    # 保存结果到本地目录
    print('\n保存结果...')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scipy.io.savemat(PREDICTION_SAVE_PATH, {
        'cleaned_eeg': predictions,
        'extracted_eog': artifacts,
        'original': test_x,
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
