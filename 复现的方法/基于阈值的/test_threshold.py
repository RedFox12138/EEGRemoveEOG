"""
基于阈值方法测试脚本 - 输出.mat格式
"""
import numpy as np
import scipy.io as sio
import os
from time import time
from SingleDenoise_CORRECTED import eog_removal_corrected


def load_test_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    import scipy.io
    test_contaminated = scipy.io.loadmat(os.path.join(data_dir, 'Test_Contaminated.mat'))['data']
    test_pure = scipy.io.loadmat(os.path.join(data_dir, 'Test_Pure.mat'))['data']
    
    print(f"测试集形状: {test_contaminated.shape}")
    
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
    print("="*60)
    print("基于阈值的EOG去除方法测试")
    print("="*60)
    
    # 加载测试数据
    print("\n加载测试数据...")
    test_contaminated, test_pure = load_test_data()
    
    # 处理所有样本
    predictions, time_per_sample = process_all_samples(test_contaminated, fs=200)
    
    # 保存结果为.mat格式
    output_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(output_dir, exist_ok=True)
    
    pred_save_path = os.path.join(output_dir, 'Threshold_predictions.mat')
    
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
