"""
真实数据集配置文件
用于 DAT-Net-Unsupervised-v2 在真实数据集上的无监督训练
"""
import os

# ========== 数据集路径 ==========
# 真实数据集路径（无标签）
REAL_DATA_PATH = r'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat'
DATA_KEY = 'eog_dataset'  # .mat 文件中的 key

# ========== 数据参数 ==========
SAMPLING_RATE = 250.0  # 采样率 (Hz) - 根据您的数据调整
WINDOW_SIZE = 1500  # 窗口大小 (样本数) - 与您的数据一致
WINDOW_DURATION = WINDOW_SIZE / SAMPLING_RATE  # 6 秒

# ========== 训练超参数 ==========
BATCH_SIZE = 128
EPOCHS = 500



GRAD_CLIP = 1.0
PATIENCE = 200

# ========== 数据集划分比例 ==========
# 统一比例：训练集 70%, 验证集 10%, 测试集 20%
TRAIN_RATIO = 0.7  # 70% 用于训练
VAL_RATIO = 0.1    # 10% 用于验证
TEST_RATIO = 0.2   # 20% 用于测试 (其余)
RANDOM_SEED = 42   # 随机种子，确保可复现

# ========== 损失函数权重 (针对眨眼伪影优化) ==========
# 针对眨眼干扰去除力度不够的问题，调整以下参数：
# 1. 增强低频伪影检测（LAMBDA_LOW）- 眨眼是低频信号
# 2. 提高伪影分离能力（LAMBDA_TEACHER、LAMBDA_DECOR）
# 3. 增加伪影检测权重（GAMMA_ART_WEIGHT、BOOST_SCALE）
# 4. 降低内容保持约束，允许更激进的去噪

# 以下为调优后的最优参数（全模拟数据集）
LEARNING_RATE =0.002
WEIGHT_DECAY = 0.00031075165040875654

# ========== 损失函数权重 (v2调优后的最优参数) ==========
LAMBDA_REC =35.223032835629375
LAMBDA_CON = 0
LAMBDA_TEACHER =87.45715260559828
LAMBDA_N2V = 44.603344466197726
LAMBDA_BAND =92.0152221815943
LAMBDA_LOW = 0
LAMBDA_DECOR =  0
LAMBDA_CONTENT =0

# ========== Artifact-aware 掩蔽参数 ==========
MASK_BASE = 1.6934384512498837
BOOST_SCALE = 0.34744565396184746
GAMMA_ART_WEIGHT = 4.798090110168316

# ========== 内部算法参数 ==========
ARTIFACT_WIN_SIZE = 72 # compute_artifact_prob的窗口大小
MASK_NEIGHBORHOOD = 52# N2V掩蔽的邻域半径
TEACHER_CUTOFF = 4.5  # teacher信号分离的高通截止频率
LOWPASS_CUTOFF = 4 # 伪影检测的低频截止频率
TEACHER_THRESHOLD = 0.6838728045795288  # teacher损失应用的阈值

USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 20
MIN_LR = LEARNING_RATE*0.01  # 最小学习率

# ========== 模型保存路径 ==========
# 使用相对路径避免中文路径问题
CHECKPOINT_DIR = 'checkpoints'
MODEL_SAVE_PATH = os.path.join(CHECKPOINT_DIR, 'datnet_unsupervised_real_best.pth')
FINAL_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'datnet_unsupervised_real_final.pth')

# ========== 结果保存路径 ==========
RESULTS_DIR = 'results'
PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, 'real_data_predictions.mat')


def print_config():
    """打印当前配置信息"""
    print("=" * 80)
    print(f"DAT-Net-Unsupervised-v2 真实数据集配置")
    print("=" * 80)
    print(f"\n[数据集配置]")
    print(f"  数据路径: {REAL_DATA_PATH}")
    print(f"  采样率: {SAMPLING_RATE} Hz")
    print(f"  窗口大小: {WINDOW_SIZE} 样本 ({WINDOW_DURATION:.1f}秒)")
    print(f"  训练/验证比例: {TRAIN_RATIO*100:.0f}% / {VAL_RATIO*100:.0f}%")
    
    print(f"\n[训练配置]")
    print(f"  Batch Size: {BATCH_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Weight Decay: {WEIGHT_DECAY}")
    print(f"  使用学习率调度器: {USE_LR_SCHEDULER}")
    if USE_LR_SCHEDULER:
        print(f"  预热轮数: {WARMUP_EPOCHS}")
        print(f"  最小学习率: {MIN_LR}")
    print(f"  梯度裁剪: {GRAD_CLIP}")
    print(f"  早停耐心值: {PATIENCE}")
    
    print(f"\n[损失函数权重]")
    print(f"  λ_rec: {LAMBDA_REC:.4f}  λ_con: {LAMBDA_CON:.4f}")
    print(f"  λ_teacher: {LAMBDA_TEACHER:.4f}  λ_n2v: {LAMBDA_N2V:.4f}")
    print(f"  λ_band: {LAMBDA_BAND:.4f}  λ_low: {LAMBDA_LOW:.4f}")
    print(f"  λ_decor: {LAMBDA_DECOR:.4f}  λ_content: {LAMBDA_CONTENT:.4f}")
    
    print(f"\n[Artifact-aware 参数]")
    print(f"  mask_base: {MASK_BASE:.4f}  boost_scale: {BOOST_SCALE:.4f}")
    print(f"  gamma_art_weight: {GAMMA_ART_WEIGHT:.4f}")
    
    print(f"\n[模型路径]")
    print(f"  检查点目录: {CHECKPOINT_DIR}")
    print(f"  最佳模型: {MODEL_SAVE_PATH}")
    print(f"  最终模型: {FINAL_MODEL_PATH}")
    print(f"  预测结果: {PREDICTION_SAVE_PATH}")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
