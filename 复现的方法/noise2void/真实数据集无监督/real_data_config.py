"""
真实数据集配置文件 - Noise2Void
用于无监督训练和测试
"""
import os

# ========== 数据集路径 ==========
REAL_DATA_PATH = r'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat'
DATA_KEY = 'eog_dataset'

# ========== 数据参数 ==========
SAMPLING_RATE = 250.0
WINDOW_SIZE = 1500

# ========== 数据集划分比例 ==========
TRAIN_RATIO = 0.7  # 70% 用于训练
VAL_RATIO = 0.1    # 10% 用于验证
TEST_RATIO = 0.2   # 20% 用于测试
RANDOM_SEED = 42

# ========== 训练超参数 ==========
BATCH_SIZE = 64
EPOCHS = 500
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5

USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 20
MIN_LR = 1e-6

GRAD_CLIP = 1.0
PATIENCE = 80

# ========== N2V参数 ==========
PERC_PIX = 1.5
NEIGHBORHOOD_RADIUS = 5
MANIPULATOR = 'uniform_withCP'

# ========== 模型参数 ==========
N_DEPTH = 3
N_FIRST = 32
KERNEL_SIZE = 5
BATCH_NORM = True
DROPOUT = 0.0
RESIDUAL = True

# ========== 模型保存路径 ==========
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'n2v_real_data_best.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'n2v_real_data_final.pth')

# ========== 结果保存路径 ==========
RESULTS_DIR = 'results'
PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, 'N2V_real_data_predictions.mat')

# ========== 总结果保存路径 ==========
FINAL_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'results'
)
FINAL_PREDICTION_PATH = os.path.join(FINAL_RESULTS_DIR, 'N2V_real_data_predictions.mat')
