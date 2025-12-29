"""
基于阈值方法测试脚本 - 支持多SNR测试集
"""
import numpy as np
import scipy.io as sio
import os
from time import time
from SingleDenoise_CORRECTED import eog_removal_corrected

# 导入数据集配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset_config import get_dataset_config


def load_test_data_by_snr(snr_level):
    """加载指定SNR级别的测试数据"""
    config = get_dataset_config('semi_simulated')
    
    if 'test_snr_paths' in config:
        # 多SNR测试集
        contaminated_path = config['test_snr_paths'][snr_level]['contaminated']
        pure_path = config['test_snr_paths'][snr_level]['pure']
    else:
        # 向后兼容：单一测试集
        contaminated_path = config['test_contaminated_path']
        pure_path = config['test_pure_path']
    
    test_contaminated = sio.loadmat(contaminated_path)['data']
    test_pure = sio.loadmat(pure_path)['data']
    
    print(f"SNR={snr_level}dB 测试集形状: {test_contaminated.shape}")
    
    return test_contaminated, test_pure


def process_all_samples(contaminated_data, fs=200):
    """
    处理所有测试样本
    
    Args:
        contaminated_data: [n_samples, n_timepoints] 或 [n_samples, n_channels, n_timepoints]
        fs: 采样率
    
    Returns:
        predictions: 与输入相同形状
        time_per_sample: 单样本平均处理时间
    """
    # 处理2D或3D数据
    if contaminated_data.ndim == 2:
        # 2D: [n_samples, n_timepoints] - 单通道
        n_samples, n_timepoints = contaminated_data.shape
        n_channels = 1
        print(f"\n数据格式: 2D [n_samples={n_samples}, n_timepoints={n_timepoints}]")
    else:
        # 3D: [n_samples, n_channels, n_timepoints]
        n_samples, n_channels, n_timepoints = contaminated_data.shape
        print(f"\n数据格式: 3D [n_samples={n_samples}, n_channels={n_channels}, n_timepoints={n_timepoints}]")
    
    predictions = np.zeros_like(contaminated_data)
    
    print(f"开始处理 {n_samples} 个样本...")
    
    start_time = time()
    
    for i in range(n_samples):
        if contaminated_data.ndim == 2:
            # 单通道数据,直接处理
            signal = contaminated_data[i, :]
            clean_signal = eog_removal_corrected(signal, fs=fs, visualize=False)
            predictions[i, :] = clean_signal
        else:
            # 多通道数据,逐通道处理
            for ch in range(n_channels):
                signal = contaminated_data[i, ch, :]
                clean_signal = eog_removal_corrected(signal, fs=fs, visualize=False)
                predictions[i, ch, :] = clean_signal
        
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
    print(f"总耗时: {total_time:.1f}秒")
    print(f"单样本平均时间: {time_per_sample:.3f}秒")
    
    return predictions, time_per_sample


def main():
    print("="*80)
    print("基于阈值的EOG去除方法测试 - 多SNR测试集")
    print("="*80)
    
    # 获取配置
    config = get_dataset_config('semi_simulated')
    output_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有SNR级别
    if 'test_snr_levels' in config:
        snr_levels = config['test_snr_levels']
        print(f"\n检测到多SNR测试集，SNR级别: {snr_levels}")
    else:
        snr_levels = [None]  # 单一测试集
        print("\n使用单一测试集")
    
    # 对每个SNR级别进行测试
    for snr in snr_levels:
        if snr is not None:
            print(f"\n{'='*80}")
            print(f"测试 SNR = {snr} dB")
            print(f"{'='*80}")
        
        # 加载测试数据
        print("\n加载测试数据...")
        test_contaminated, test_pure = load_test_data_by_snr(snr)
        
        # 处理所有样本
        predictions, time_per_sample = process_all_samples(test_contaminated, fs=200)
        
        # 保存结果为.mat格式
        if snr is not None:
            pred_save_path = os.path.join(output_dir, f'Threshold_predictions_SNR{snr}dB.mat')
        else:
            pred_save_path = os.path.join(output_dir, 'Threshold_predictions.mat')
        
        sio.savemat(pred_save_path, {
            'predictions': predictions,
            'time_per_sample': time_per_sample
        })
        
        print(f"\n✓ 结果已保存到: {pred_save_path}")
    
    print(f"\n{'='*80}")
    print("所有SNR级别测试完成!")
    print(f"{'='*80}")
    
    sio.savemat(pred_save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample
    })
    
    print(f"\n预测结果已保存为.mat格式:")
    print(f"  {pred_save_path}")
    print(f"  形状: {predictions.shape}")
    print(f"  单样本处理时间: {time_per_sample*1000:.1f}ms")
    
    print("\n✓ 完成！请运行统一指标计算脚本来评估所有方法。")
    print("="*60)


if __name__ == '__main__':
    main()
