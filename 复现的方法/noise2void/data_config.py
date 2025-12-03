"""
N2V_1D 数据配置文件
用于管理数据集路径和参数
"""
import os
import sys

# 添加父目录以导入数据集配置
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from dataset_config import get_dataset_config

# ========== 数据集选择 ==========
DATASET_NAME = 'fully_simulated'  # 可选: 'semi_simulated', 'fully_simulated'

# 获取数据集配置
dataset_config = get_dataset_config(DATASET_NAME)

# ========== 数据参数 ==========
SAMPLING_RATE = dataset_config['sampling_rate']
WINDOW_SIZE = dataset_config['window_size']
DATA_DIR = dataset_config['data_dir']
DATA_KEY = dataset_config['data_key']

# 数据文件路径
TRAIN_CONTAMINATED_PATH = dataset_config['train_contaminated_path']
TRAIN_PURE_PATH = dataset_config['train_pure_path']
VAL_CONTAMINATED_PATH = dataset_config['val_contaminated_path']
VAL_PURE_PATH = dataset_config['val_pure_path']
TEST_CONTAMINATED_PATH = dataset_config['test_contaminated_path']
TEST_PURE_PATH = dataset_config['test_pure_path']

# ========== 训练超参数 ==========
BATCH_SIZE = 256
EPOCHS = 1000
LEARNING_RATE = 0.001
PATIENCE = 100

# ========== 模型保存路径 ==========
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, f'N2V_1D_{}_best.pth'.format(DATASET_NAME))
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'N2V_1D_{}_final.pth'.format(DATASET_NAME))

# ========== 结果保存路径 ==========
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, f'N2V_1D_{}_predictions.mat'.format(DATASET_NAME))


def print_config():
    """打印当前配置信息"""
    print("=" * 80)
    print(f"N2V_1D 数据配置信息")
    print("=" * 80)
    print(f"\n[数据集配置]")
    print(f"  数据集名称: {dataset_config['name']} ({DATASET_NAME})")
    print(f"  采样率: {SAMPLING_RATE} Hz")
    print(f"  窗口大小: {WINDOW_SIZE} 样本")
    print(f"  数据目录: {DATA_DIR}")
    
    print(f"\n[训练参数]")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    
    print(f"\n[保存路径]")
    print(f"  最佳模型: {MODEL_SAVE_PATH}")
    print(f"  预测结果: {PREDICTION_SAVE_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
