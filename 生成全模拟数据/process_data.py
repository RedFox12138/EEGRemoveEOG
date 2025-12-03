### 该脚本用于处理全模拟EEG数据集中的.mat文件，按照80%、10%、10%的比例划分训练集、验证集、测试集
### 注意：全模拟数据包含4种数据类型，每1/4的数据（4000条）代表一种类型
### 划分时需要确保每种类型的数据都按相同比例划分到训练集、验证集、测试集中

import scipy.io
import numpy as np
import os

def process_and_split_data(contaminated_path, pure_path, output_dir):
    """
    处理并划分全模拟数据集
    
    参数:
        contaminated_path: 受污染数据的路径
        pure_path: 纯净数据的路径
        output_dir: 输出目录
    """
    # 加载数据
    try:
        contaminated_data = scipy.io.loadmat(contaminated_path)
        pure_data = scipy.io.loadmat(pure_path)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # 提取数据
    contaminated_eeg = contaminated_data['contaminatedEEG']  # shape: (16000, 1500)
    pure_eeg = pure_data['pureEEG']  # shape: (16000, 1500)
    
    print(f"Contaminated data shape: {contaminated_eeg.shape}")
    print(f"Pure data shape: {pure_eeg.shape}")
    
    total_samples = contaminated_eeg.shape[0]
    samples_per_type = total_samples // 4  # 每种类型的样本数量
    
    print(f"\n总样本数: {total_samples}")
    print(f"每种类型样本数: {samples_per_type}")
    print(f"数据类型数量: 4")
    
    # 初始化列表存储分割后的数据
    train_contaminated_list = []
    val_contaminated_list = []
    test_contaminated_list = []
    
    train_pure_list = []
    val_pure_list = []
    test_pure_list = []
    
    # 使用固定随机种子以保证可重复性
    np.random.seed(42)
    
    # 对每种类型的数据分别进行划分
    for type_idx in range(4):
        start_idx = type_idx * samples_per_type
        end_idx = (type_idx + 1) * samples_per_type
        
        # 提取当前类型的数据
        type_contaminated = contaminated_eeg[start_idx:end_idx]
        type_pure = pure_eeg[start_idx:end_idx]
        
        # 随机打乱当前类型的数据
        indices = np.random.permutation(samples_per_type)
        type_contaminated = type_contaminated[indices]
        type_pure = type_pure[indices]
        
        # 计算划分点 (80%, 10%, 10%)
        train_end = int(0.8 * samples_per_type)
        val_end = int(0.9 * samples_per_type)
        
        # 划分当前类型的数据
        train_contaminated_list.append(type_contaminated[:train_end])
        val_contaminated_list.append(type_contaminated[train_end:val_end])
        test_contaminated_list.append(type_contaminated[val_end:])
        
        train_pure_list.append(type_pure[:train_end])
        val_pure_list.append(type_pure[train_end:val_end])
        test_pure_list.append(type_pure[val_end:])
        
        print(f"\n类型 {type_idx + 1}:")
        print(f"  训练集: {train_contaminated_list[-1].shape[0]} 样本")
        print(f"  验证集: {val_contaminated_list[-1].shape[0]} 样本")
        print(f"  测试集: {test_contaminated_list[-1].shape[0]} 样本")
    
    # 合并所有类型的数据
    train_contaminated = np.vstack(train_contaminated_list)
    val_contaminated = np.vstack(val_contaminated_list)
    test_contaminated = np.vstack(test_contaminated_list)
    
    train_pure = np.vstack(train_pure_list)
    val_pure = np.vstack(val_pure_list)
    test_pure = np.vstack(test_pure_list)
    
    # 对合并后的数据再次随机打乱
    train_indices = np.random.permutation(train_contaminated.shape[0])
    train_contaminated = train_contaminated[train_indices]
    train_pure = train_pure[train_indices]
    
    val_indices = np.random.permutation(val_contaminated.shape[0])
    val_contaminated = val_contaminated[val_indices]
    val_pure = val_pure[val_indices]
    
    test_indices = np.random.permutation(test_contaminated.shape[0])
    test_contaminated = test_contaminated[test_indices]
    test_pure = test_pure[test_indices]
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为.mat格式
    scipy.io.savemat(f'{output_dir}/Train_Contaminated.mat', {'data': train_contaminated})
    scipy.io.savemat(f'{output_dir}/Train_Pure.mat', {'data': train_pure})
    scipy.io.savemat(f'{output_dir}/Val_Contaminated.mat', {'data': val_contaminated})
    scipy.io.savemat(f'{output_dir}/Val_Pure.mat', {'data': val_pure})
    scipy.io.savemat(f'{output_dir}/Test_Contaminated.mat', {'data': test_contaminated})
    scipy.io.savemat(f'{output_dir}/Test_Pure.mat', {'data': test_pure})
    
    print(f"\n{'='*60}")
    print(f"数据集划分完成:")
    print(f"{'='*60}")
    print(f"训练集: {train_contaminated.shape} (80%)")
    print(f"验证集: {val_contaminated.shape} (10%)")
    print(f"测试集: {test_contaminated.shape} (10%)")
    print(f"\n每个数据集都包含来自4种类型的数据，每种类型按相同比例划分")
    print(f"所有数据已保存到 {output_dir}/ 目录")

if __name__ == '__main__':
    # 处理并划分数据集
    process_and_split_data(
        contaminated_path='已经生成好的数据/Contaminated.mat',
        pure_path='已经生成好的数据/Pure_Data.mat',
        output_dir='已经生成好的数据'
    )
