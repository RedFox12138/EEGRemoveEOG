"""
DAT-Net-Unsupervised-v2 配置文件
用于管理数据集、模型和训练参数
"""
import os
import sys

# 添加父目录以导入数据集配置
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from dataset_config import get_dataset_config

# ========== 数据集选择 ==========
# 可选值: 'semi_simulated', 'fully_simulated'
DATASET_NAME = 'semi_simulated'  # 修改这里可以切换数据集

# 获取数据集配置
dataset_config = get_dataset_config(DATASET_NAME)

# ========== 数据参数 ==========
SAMPLING_RATE = dataset_config['sampling_rate']  # 采样率 (Hz)
WINDOW_SIZE = dataset_config['window_size']  # 窗口大小 (样本数)
DATA_DIR = dataset_config['data_dir']  # 数据目录
DATA_KEY = dataset_config['data_key']  # .mat文件中的key

# 数据文件路径
TRAIN_CONTAMINATED_PATH = dataset_config['train_contaminated_path']
TRAIN_PURE_PATH = dataset_config['train_pure_path']
VAL_CONTAMINATED_PATH = dataset_config['val_contaminated_path']
VAL_PURE_PATH = dataset_config['val_pure_path']
TEST_CONTAMINATED_PATH = dataset_config['test_contaminated_path']
TEST_PURE_PATH = dataset_config['test_pure_path']

# ========== 训练超参数 ==========
BATCH_SIZE = 256
EPOCHS = 1500
LEARNING_RATE = 0.0090  # 调优后的学习率
WEIGHT_DECAY = 1e-5

USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 50
MIN_LR = 1e-3  # 最小学习率

GRAD_CLIP = 1.0
PATIENCE = 150

# ========== 损失函数权重 (v2调优后的最优参数) ==========
LAMBDA_REC = 0.7864
LAMBDA_CON = 1.5586
LAMBDA_TEACHER = 0.1804
LAMBDA_N2V = 0.3515
LAMBDA_BAND = 0.4997
LAMBDA_LOW = 0.0290
LAMBDA_DECOR = 0.2217
LAMBDA_CONTENT = 0.1662

# ========== Artifact-aware 掩蔽参数 ==========
MASK_BASE = 0.0633
BOOST_SCALE = 0.1334
GAMMA_ART_WEIGHT = 0.6339

# ========== 内部算法参数 ==========
ARTIFACT_WIN_SIZE = 82  # compute_artifact_prob的窗口大小
MASK_NEIGHBORHOOD = 5  # N2V掩蔽的邻域半径
TEACHER_CUTOFF = 4.6188  # teacher信号分离的高通截止频率
LOWPASS_CUTOFF = 2.7216  # 伪影检测的低频截止频率
TEACHER_THRESHOLD = 0.7651  # teacher损失应用的阈值

# ========== 模型保存路径 ==========
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, f'datnet_unsupervised_v2_{DATASET_NAME}_best.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'datnet_unsupervised_v2_{DATASET_NAME}_final.pth')

# ========== 结果保存路径 ==========
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, f'DAT-Net-Unsupervised-v2_{DATASET_NAME}_predictions.mat')

# ========== 微调参数 ==========
FINETUNE_RATIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30%的数据用于微调
FINETUNE_EPOCHS = 300
FINETUNE_LR = 0.001
FINETUNE_PATIENCE = 50


def print_config():
    """打印当前配置信息"""
    print("=" * 80)
    print(f"DAT-Net-Unsupervised-v2 配置信息")
    print("=" * 80)
    print(f"\n[数据集配置]")
    print(f"  数据集名称: {dataset_config['name']} ({DATASET_NAME})")
    print(f"  数据集描述: {dataset_config['description']}")
    print(f"  采样率: {SAMPLING_RATE} Hz")
    print(f"  窗口大小: {WINDOW_SIZE} 样本")
    print(f"  数据目录: {DATA_DIR}")
    
    print(f"\n[训练参数]")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  使用学习率调度: {USE_LR_SCHEDULER}")
    print(f"  Warmup Epochs: {WARMUP_EPOCHS}")
    print(f"  最小学习率: {MIN_LR}")
    
    print(f"\n[损失权重]")
    print(f"  λ_rec: {LAMBDA_REC}")
    print(f"  λ_con: {LAMBDA_CON}")
    print(f"  λ_teacher: {LAMBDA_TEACHER}")
    print(f"  λ_n2v: {LAMBDA_N2V}")
    print(f"  λ_band: {LAMBDA_BAND}")
    print(f"  λ_low: {LAMBDA_LOW}")
    print(f"  λ_decor: {LAMBDA_DECOR}")
    print(f"  λ_content: {LAMBDA_CONTENT}")
    
    print(f"\n[Artifact-aware参数]")
    print(f"  Mask Base: {MASK_BASE}")
    print(f"  Boost Scale: {BOOST_SCALE}")
    print(f"  γ_art_weight: {GAMMA_ART_WEIGHT}")
    
    print(f"\n[保存路径]")
    print(f"  最佳模型: {MODEL_SAVE_PATH}")
    print(f"  最终模型: {FINAL_MODEL_PATH}")
    print(f"  预测结果: {PREDICTION_SAVE_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
