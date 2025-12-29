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

# 测试集路径（向后兼容多SNR和单一测试集）
if 'test_contaminated_path' in dataset_config:
    TEST_CONTAMINATED_PATH = dataset_config['test_contaminated_path']
    TEST_PURE_PATH = dataset_config['test_pure_path']
else:
    # 多SNR测试集，训练/调优时不需要
    TEST_CONTAMINATED_PATH = None
    TEST_PURE_PATH = None

# 多SNR测试配置
if 'test_snr_levels' in dataset_config:
    TEST_SNR_LEVELS = dataset_config['test_snr_levels']
    TEST_SNR_PATHS = dataset_config['test_snr_paths']
else:
    TEST_SNR_LEVELS = []
    TEST_SNR_PATHS = {}



# BATCH_SIZE = 256
# EPOCHS = 2000
# LEARNING_RATE = 0.02  # 调优后的学习率
# WEIGHT_DECAY = 1.0380237289610372e-06
#
# USE_LR_SCHEDULER = True
# WARMUP_EPOCHS = 20
# MIN_LR = 0.00001  # 最小学习率（激进衰减）
# # 学习率策略：Warmup(100epoch) + 线性下降(1900epoch)
# # 从LEARNING_RATE线性下降到MIN_LR，更快收敛
#
# GRAD_CLIP = 1.0
# PATIENCE = 500  # 给模型更多时间充分收敛
#
# # ========== 损失函数权重 (v2调优后的最优参数) ==========
# LAMBDA_REC = 0.7864
# LAMBDA_CON = 1.5586
# LAMBDA_TEACHER = 0.1804
# LAMBDA_N2V = 0.3515
# LAMBDA_BAND = 0.4997
# LAMBDA_LOW = 0.0290
# LAMBDA_DECOR = 0.2217
# LAMBDA_CONTENT = 0.1662
#
# # ========== Artifact-aware 掩蔽参数 ==========
# MASK_BASE = 0.0633
# BOOST_SCALE = 0.1334
# GAMMA_ART_WEIGHT = 0.6339
#
# # ========== 内部算法参数 ==========
# ARTIFACT_WIN_SIZE = 82  # compute_artifact_prob的窗口大小
# MASK_NEIGHBORHOOD = 5  # N2V掩蔽的邻域半径
# TEACHER_CUTOFF = 4.6188  # teacher信号分离的高通截止频率
# LOWPASS_CUTOFF = 2.7216  # 伪影检测的低频截止频率
# TEACHER_THRESHOLD = 0.7651  # teacher损失应用的阈值


BATCH_SIZE = 256
EPOCHS = 4000
LEARNING_RATE =0.00005  # 调优后的学习率
# LEARNING_RATE = 0.02914632759933304  # 调优后的学习率
WEIGHT_DECAY = 3.0162691408032546e-06

USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 10
MIN_LR = 0.00001  # 最小学习率（激进衰减）
# 学习率策略：Warmup(100epoch) + 线性下降(1900epoch)
# 从LEARNING_RATE线性下降到MIN_LR，更快收敛

GRAD_CLIP = 1.0
PATIENCE = 500  # 给模型更多时间充分收敛

# ========== 损失函数权重 (v2调优后的最优参数) ==========
LAMBDA_REC =10000
LAMBDA_CON = 0
LAMBDA_TEACHER =0
LAMBDA_N2V = 0
LAMBDA_BAND = 0.29790974972068235
LAMBDA_LOW = 0.06630350426620805
LAMBDA_DECOR = 0.3393320679545625
LAMBDA_CONTENT = 1.4131338175273647

# ========== Artifact-aware 掩蔽参数 ==========
MASK_BASE = 0.05000882099902099
BOOST_SCALE = 0.11437730106023324
GAMMA_ART_WEIGHT = 2.8845684550289423

# ========== 内部算法参数 ==========
ARTIFACT_WIN_SIZE = 100  # compute_artifact_prob的窗口大小
MASK_NEIGHBORHOOD = 7  # N2V掩蔽的邻域半径
TEACHER_CUTOFF = 5.197457373000224  # teacher信号分离的高通截止频率
LOWPASS_CUTOFF = 1.849810301795507  # 伪影检测的低频截止频率
TEACHER_THRESHOLD = 0.8968825392989637  # teacher损失应用的阈值


# ========== 模型保存路径 ==========
CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# 两种最佳模型路径
MODEL_SAVE_PATH_RRMSE = os.path.join(CHECKPOINT_DIR, f'datnet_unsupervised_v2_{DATASET_NAME}_best_rrmse.pth')
MODEL_SAVE_PATH_LOSS = os.path.join(CHECKPOINT_DIR, f'datnet_unsupervised_v2_{DATASET_NAME}_best_loss.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, f'datnet_unsupervised_v2_{DATASET_NAME}_final.pth')

# 恢复训练策略：'rrmse'(基于RRMSE的最佳模型) 'loss'(基于Loss的最佳模型) 'auto'(自动选择)
RESUME_FROM = 'rrmse'  # 可选: 'rrmse', 'loss', 'auto'

# ========== 结果保存路径 ==========
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, f'DAT-Net-Unsupervised-v2_{DATASET_NAME}_predictions.mat')

# ========== 微调参数 ==========
FINETUNE_RATIOS = [0.1, 0.2, 0.3]  # 10%, 20%, 30%的数据用于微调
FINETUNE_RATIO = 0.2  # 默认使用20%数据进行微调
FINETUNE_EPOCHS = 300
FINETUNE_LR = 0.001
FINETUNE_PATIENCE = 50

# 微调数据集路径
FINETUNE_CONTAMINATED_PATH = os.path.join(DATA_DIR, f'Finetune_{int(FINETUNE_RATIO*100)}percent_Contaminated.mat')
FINETUNE_PURE_PATH = os.path.join(DATA_DIR, f'Finetune_{int(FINETUNE_RATIO*100)}percent_Pure.mat')


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
    
    print(f"\n[微调参数]")
    print(f"  微调比例: {FINETUNE_RATIO*100:.0f}%")
    print(f"  微调数据集(污染): {FINETUNE_CONTAMINATED_PATH}")
    print(f"  微调数据集(纯净): {FINETUNE_PURE_PATH}")
    print(f"  微调轮数: {FINETUNE_EPOCHS}")
    print(f"  微调学习率: {FINETUNE_LR}")
    
    print(f"\n[保存路径]")
    print(f"  RRMSE最佳模型: {MODEL_SAVE_PATH_RRMSE}")
    print(f"  Loss最佳模型: {MODEL_SAVE_PATH_LOSS}")
    print(f"  最终模型: {FINAL_MODEL_PATH}")
    print(f"  预测结果: {PREDICTION_SAVE_PATH}")
    print(f"\n[恢复训练策略]")
    print(f"  RESUME_FROM: {RESUME_FROM}")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
