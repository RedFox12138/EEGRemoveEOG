"""
真实数据集统一加载和划分工具

提供统一的数据划分方案，确保所有方法使用相同的训练集/测试集划分
- 训练集: 80% (用于无监督方法训练)
- 测试集: 20% (用于所有方法测试评估)
- 随机种子: 42 (确保可复现)

作者: GitHub Copilot
日期: 2025-12-10
"""

import numpy as np
import scipy.io
import os


def load_real_dataset_split(
    data_path=None,
    data_key='eog_dataset',
    train_ratio=0.8,
    random_seed=42,
    return_train=False
):
    """
    加载真实数据集并划分为训练集和测试集
    
    参数:
        data_path: str, 数据文件路径，默认为真实数据集路径
        data_key: str, .mat文件中的数据键名，默认 'eog_dataset'
        train_ratio: float, 训练集+验证集比例，默认 0.8 (80%用于训练和验证，20%用于测试)
        random_seed: int, 随机种子，默认 42 (确保所有方法数据划分一致)
        return_train: bool, 是否返回训练集数据，默认 False
                      - True: 返回 (test_data, train_data)，用于无监督方法
                      - False: 只返回 test_data，用于有监督方法和传统方法
    
    返回:
        test_data: ndarray, 测试集数据 (20% 的样本)
        train_data: ndarray, 训练集数据 (80% 的样本)，仅当 return_train=True 时返回
    
    使用示例:
        # 只需要测试集数据 (有监督方法和传统方法)
        test_data = load_real_dataset_split()
        
        # 需要训练集和测试集数据 (无监督方法)
        test_data, train_data = load_real_dataset_split(return_train=True)
    """
    # 默认数据路径
    if data_path is None:
        data_path = r'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat'
    
    print('\n正在加载真实数据集...')
    print(f'数据路径: {data_path}')
    
    # 加载数据
    data_dict = scipy.io.loadmat(data_path)
    
    # 尝试不同的可能的 key
    possible_keys = [data_key, 'data', 'eeg_data', 'X', 'signals']
    data = None
    
    for key in possible_keys:
        if key in data_dict:
            data = data_dict[key]
            print(f'  ✓ 使用 key: "{key}"')
            break
    
    if data is None:
        available_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        raise ValueError(f'无法找到数据！可用的 keys: {available_keys}')
    
    print(f'  数据形状: {data.shape}')
    print(f'  总样本数量: {data.shape[0]}')
    print(f'  样本长度: {data.shape[1]}')
    
    # 随机划分数据集
    n_samples = data.shape[0]
    
    # 设置随机种子以确保可复现性
    np.random.seed(random_seed)
    
    # 生成随机打乱的索引
    indices = np.random.permutation(n_samples)
    
    # 计算划分点
    train_size = int(n_samples * train_ratio)
    
    # 划分数据
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = data[train_indices]
    test_data = data[test_indices]
    
    # 输出划分信息
    print(f'\n数据集划分完成 (随机种子={random_seed}):')
    print(f'  训练集: {train_data.shape[0]} 样本 ({train_ratio*100:.0f}%)')
    print(f'  测试集: {test_data.shape[0]} 样本 ({(1-train_ratio)*100:.0f}%)')
    print(f'  ⚠️  所有方法应只在测试集上进行评估！\n')
    
    if return_train:
        return test_data, train_data
    else:
        return test_data


# 用于向后兼容的别名
loadRealDatasetSplit = load_real_dataset_split


if __name__ == '__main__':
    # 测试函数
    print("=" * 80)
    print("测试数据划分函数")
    print("=" * 80)
    
    # 测试只返回测试集
    print("\n1. 只返回测试集:")
    test_data = load_real_dataset_split()
    print(f"测试集形状: {test_data.shape}")
    
    # 测试返回训练集和测试集
    print("\n2. 返回训练集和测试集:")
    test_data, train_data = load_real_dataset_split(return_train=True)
    print(f"训练集形状: {train_data.shape}")
    print(f"测试集形状: {test_data.shape}")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)
