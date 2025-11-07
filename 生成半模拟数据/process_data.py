### 该脚本用于处理EEG数据集中的.mat文件，提取指定通道的数据，并按照80%、10%、10%的比例划分训练集、验证集、测试集，保存为.mat格式文件。

import scipy.io
import numpy as np
import re

def process_and_split_data(contaminated_path, pure_path, output_dir, prefix):
    """
    处理并划分数据集
    
    参数:
        contaminated_path: 受污染数据的路径
        pure_path: 纯净数据的路径
        output_dir: 输出目录
        prefix: 数据key的前缀
    """
    # 加载数据
    try:
        contaminated_data = scipy.io.loadmat(contaminated_path)
        pure_data = scipy.io.loadmat(pure_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 数据参数
    sample_rate = 200  # Hz
    window_duration = 6  # seconds
    window_size = sample_rate * window_duration
    step = window_size  # 无重叠

    # 处理受污染的数据
    contaminated_segments = extract_segments(contaminated_data, prefix, window_size, step)
    # 处理纯净的数据
    pure_segments = extract_segments(pure_data, prefix, window_size, step)
    
    print(f"Total contaminated segments: {len(contaminated_segments)}")
    print(f"Total pure segments: {len(pure_segments)}")
    
    # 确保两个数据集长度一致
    min_len = min(len(contaminated_segments), len(pure_segments))
    contaminated_segments = contaminated_segments[:min_len]
    pure_segments = pure_segments[:min_len]
    
    # 随机打乱数据（使用固定的随机种子以保证可重复性）
    np.random.seed(42)
    indices = np.random.permutation(min_len)
    contaminated_segments = [contaminated_segments[i] for i in indices]
    pure_segments = [pure_segments[i] for i in indices]
    
    # 计算划分点
    train_end = int(0.8 * min_len)
    val_end = int(0.9 * min_len)
    
    # 划分数据集
    train_contaminated = np.array(contaminated_segments[:train_end])
    val_contaminated = np.array(contaminated_segments[train_end:val_end])
    test_contaminated = np.array(contaminated_segments[val_end:])
    
    train_pure = np.array(pure_segments[:train_end])
    val_pure = np.array(pure_segments[train_end:val_end])
    test_pure = np.array(pure_segments[val_end:])
    
    # 保存为.mat格式
    scipy.io.savemat(f'{output_dir}/Train_Contaminated.mat', {'data': train_contaminated})
    scipy.io.savemat(f'{output_dir}/Train_Pure.mat', {'data': train_pure})
    scipy.io.savemat(f'{output_dir}/Val_Contaminated.mat', {'data': val_contaminated})
    scipy.io.savemat(f'{output_dir}/Val_Pure.mat', {'data': val_pure})
    scipy.io.savemat(f'{output_dir}/Test_Contaminated.mat', {'data': test_contaminated})
    scipy.io.savemat(f'{output_dir}/Test_Pure.mat', {'data': test_pure})
    
    print(f"\n数据集划分完成:")
    print(f"训练集: {train_contaminated.shape} (80%)")
    print(f"验证集: {val_contaminated.shape} (10%)")
    print(f"测试集: {test_contaminated.shape} (10%)")
    print(f"\n所有数据已保存到 {output_dir}/ 目录")

def extract_segments(mat_data, prefix, window_size, step):
    """
    从.mat数据中提取分段数据
    
    参数:
        mat_data: 加载的.mat数据
        prefix: key前缀
        window_size: 窗口大小
        step: 步长
    
    返回:
        all_segments: 所有分段的列表
    """
    # 提取并排序相关的key
    keys = [k for k in mat_data.keys() if k.startswith(prefix)]
    keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    all_segments = []

    for key in keys:
        # 每个数组的尺寸是19*n
        sample_data = mat_data[key]
        # 取前4个通道
        sample_data_4ch = sample_data[:4, :]

        num_channels, signal_len = sample_data_4ch.shape
        for i in range(num_channels):
            signal = sample_data_4ch[i, :]

            num_windows = (signal_len - window_size) // step + 1
            for j in range(num_windows):
                start = j * step
                end = start + window_size
                if end <= signal_len:
                    segment = signal[start:end]
                    all_segments.append(segment)

    return all_segments

if __name__ == '__main__':
    # 处理并划分数据集
    process_and_split_data(
        contaminated_path='Contaminated_Data.mat',
        pure_path='Pure_Data.mat',
        output_dir='已经生成好的数据',
        prefix='sim'
    )
