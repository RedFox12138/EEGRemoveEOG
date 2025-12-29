"""
数据集配置文件
用于统一管理不同数据集的配置,方便切换和扩展

使用方法:
    from dataset_config import get_dataset_config
    config = get_dataset_config('semi_simulated')  # 或 'fully_simulated'
    print(config['sampling_rate'])
    print(config['train_contaminated_path'])
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集配置字典
DATASET_CONFIGS = {
    # 半模拟数据集 (多SNR版本)
    'semi_simulated': {
        'name': '半模拟数据集',
        'sampling_rate': 200.0,  # Hz
        'window_duration': 6,  # seconds
        'window_size': 1200,  # 200 * 6
        'data_dir': os.path.join(PROJECT_ROOT, '生成半模拟数据', '已经生成好的数据', 'multi_snr'),
        'train_contaminated': 'Train_Contaminated.mat',
        'train_pure': 'Train_Pure.mat',
        'val_contaminated': 'Val_Contaminated.mat',
        'val_pure': 'Val_Pure.mat',
        'test_snr_levels': [-8,-6, -4, -2, 0, 2,4],  # 多个SNR级别的测试集
        'test_contaminated_template': 'Test_Contaminated_SNR{}dB.mat',  # 模板，{}会被替换为SNR值
        'test_pure_template': 'Test_Pure_SNR{}dB.mat',
        'data_key': 'data',  # .mat文件中的key
        'description': '基于真实EEG数据生成的半模拟数据集,采样率200Hz,包含5种SNR级别的测试集'
    },
    
    # 全模拟数据集 (新数据集)
    'fully_simulated': {
        'name': '全模拟数据集',
        'sampling_rate': 250.0,  # Hz
        'window_duration': 6,  # seconds
        'window_size': 1500,  # 250 * 6
        'data_dir': os.path.join(PROJECT_ROOT, '生成全模拟数据', '已经生成好的数据'),
        'train_contaminated': 'Train_Contaminated.mat',
        'train_pure': 'Train_Pure.mat',
        'val_contaminated': 'Val_Contaminated.mat',
        'val_pure': 'Val_Pure.mat',
        'test_contaminated': 'Test_Contaminated.mat',
        'test_pure': 'Test_Pure.mat',
        'data_key': 'data',  # .mat文件中的key
        'description': '完全模拟生成的数据集,包含4种类型,采样率250Hz'
    }
}

# 默认使用的数据集 (可以在这里切换)
DEFAULT_DATASET = 'semi_simulated'  # 改为 'semi_simulated' 可切换回半模拟数据


def get_dataset_config(dataset_name=None):
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称 ('semi_simulated' 或 'fully_simulated')
                     如果为None,则使用DEFAULT_DATASET
    
    Returns:
        dict: 数据集配置字典
    """
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    
    # 添加完整路径
    config['train_contaminated_path'] = os.path.join(config['data_dir'], config['train_contaminated'])
    config['train_pure_path'] = os.path.join(config['data_dir'], config['train_pure'])
    config['val_contaminated_path'] = os.path.join(config['data_dir'], config['val_contaminated'])
    config['val_pure_path'] = os.path.join(config['data_dir'], config['val_pure'])
    
    # 处理测试集路径 - 支持多SNR级别
    if 'test_snr_levels' in config:
        # 多SNR测试集
        config['test_snr_paths'] = {}
        for snr in config['test_snr_levels']:
            config['test_snr_paths'][snr] = {
                'contaminated': os.path.join(config['data_dir'], 
                                            config['test_contaminated_template'].format(snr)),
                'pure': os.path.join(config['data_dir'], 
                                    config['test_pure_template'].format(snr))
            }
    else:
        # 单一测试集（向后兼容）
        config['test_contaminated_path'] = os.path.join(config['data_dir'], config['test_contaminated'])
        config['test_pure_path'] = os.path.join(config['data_dir'], config['test_pure'])
    
    return config


def list_available_datasets():
    """列出所有可用的数据集"""
    print("可用数据集:")
    print("=" * 80)
    for name, config in DATASET_CONFIGS.items():
        is_default = " [当前默认]" if name == DEFAULT_DATASET else ""
        print(f"\n数据集名称: {name}{is_default}")
        print(f"  描述: {config['description']}")
        print(f"  采样率: {config['sampling_rate']} Hz")
        print(f"  窗口大小: {config['window_size']} 样本 ({config['window_duration']}秒)")
        print(f"  数据目录: {config['data_dir']}")


def print_dataset_info(dataset_name=None):
    """打印数据集详细信息"""
    config = get_dataset_config(dataset_name)
    print(f"\n数据集信息: {config['name']}")
    print("=" * 80)
    print(f"描述: {config['description']}")
    print(f"采样率: {config['sampling_rate']} Hz")
    print(f"窗口大小: {config['window_size']} 样本 ({config['window_duration']}秒)")
    print(f"\n数据文件:")
    print(f"  训练集 (污染): {config['train_contaminated_path']}")
    print(f"  训练集 (纯净): {config['train_pure_path']}")
    print(f"  验证集 (污染): {config['val_contaminated_path']}")
    print(f"  验证集 (纯净): {config['val_pure_path']}")
    print(f"  测试集 (污染): {config['test_contaminated_path']}")
    print(f"  测试集 (纯净): {config['test_pure_path']}")
    print(f"\n数据键名: {config['data_key']}")


if __name__ == '__main__':
    # 测试代码
    print("数据集配置系统测试")
    print("=" * 80)
    
    # 列出所有可用数据集
    list_available_datasets()
    
    # 打印默认数据集信息
    print("\n" + "=" * 80)
    print_dataset_info()
    
    # 测试加载不同数据集
    print("\n" + "=" * 80)
    print("测试加载半模拟数据集:")
    semi_config = get_dataset_config('semi_simulated')
    print(f"  采样率: {semi_config['sampling_rate']} Hz")
    
    print("\n测试加载全模拟数据集:")
    fully_config = get_dataset_config('fully_simulated')
    print(f"  采样率: {fully_config['sampling_rate']} Hz")
