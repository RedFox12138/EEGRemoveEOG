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
BATCH_SIZE = 256
EPOCHS = 1500
LEARNING_RATE = 0.0090  # 调优后的学习率
WEIGHT_DECAY = 1e-5

USE_LR_SCHEDULER = True
WARMUP_EPOCHS = 50
MIN_LR = 5e-4  # 最小学习率

GRAD_CLIP = 1.0
PATIENCE = 150

# ========== 数据集划分比例 ==========
# 由于没有验证集，我们从真实数据中随机抽取一部分作为验证
TRAIN_RATIO = 0.9  # 90% 用于训练
VAL_RATIO = 0.1    # 10% 用于验证
RANDOM_SEED = 42   # 随机种子，确保可复现

# ========== 损失函数权重 (针对眨眼伪影优化) ==========
# 针对眨眼干扰去除力度不够的问题，调整以下参数：
# 1. 增强低频伪影检测（LAMBDA_LOW）- 眨眼是低频信号
# 2. 提高伪影分离能力（LAMBDA_TEACHER、LAMBDA_DECOR）
# 3. 增加伪影检测权重（GAMMA_ART_WEIGHT、BOOST_SCALE）
# 4. 降低内容保持约束，允许更激进的去噪

LAMBDA_REC = 0.7864      # 重建损失（保持不变）
LAMBDA_CON = 1.5586      # 一致性损失（保持不变）
LAMBDA_TEACHER = 0.35    # ↑ 提高 Teacher 损失，增强伪影分离（原 0.1804）
LAMBDA_N2V = 0.3515      # Noise2Void 损失（保持不变）
LAMBDA_BAND = 0.6        # ↑ 提高频带损失，增强频域约束（原 0.4997）
LAMBDA_LOW = 0.15        # ↑↑ 大幅提高低频损失，针对眨眼（原 0.0290）
LAMBDA_DECOR = 0.4       # ↑ 提高去相关损失，增强EEG/EOG分离（原 0.2217）
LAMBDA_CONTENT = 0.08    # ↓ 降低内容损失，允许更激进去噪（原 0.1662）

# ========== Artifact-aware 掩蔽参数 (增强伪影检测) ==========
MASK_BASE = 0.1          # ↑ 提高基础掩蔽率，更多关注伪影区域（原 0.0633）
BOOST_SCALE = 0.25       # ↑ 提高伪影区域增强系数（原 0.1334）
GAMMA_ART_WEIGHT = 0.85  # ↑ 提高伪影加权因子（原 0.6339）

# ========== 内部算法参数 (优化眨眼检测) ==========
ARTIFACT_WIN_SIZE = 100  # ↑ 增大窗口捕获更长的眨眼伪影（原 82）
MASK_NEIGHBORHOOD = 5    # N2V掩蔽的邻域半径（保持不变）
TEACHER_CUTOFF = 5     # ↓ 降低高通截止，更好分离低频眨眼（原 4.6188）
LOWPASS_CUTOFF = 5     # ↑ 提高低通截止，增强低频伪影检测（原 2.7216）
TEACHER_THRESHOLD = 0.65 # ↓ 降低阈值，更宽松应用 teacher 损失（原 0.7651）

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
