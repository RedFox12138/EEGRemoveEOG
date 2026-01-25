"""
消融实验配置文件
定义了各种消融实验的配置
所有非消融参数与 DAT-Net-Unsupervised-v2/config.py 完全一致
"""
import os
import sys

# 添加父目录以导入数据集配置和原始config
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# 导入原始配置作为基准
v2_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'DAT-Net-Unsupervised-v2')
sys.path.insert(0, v2_dir)
from config import *

# ========== 消融实验配置字典 ==========
# 根据损失函数分类设计的消融实验：
# - 重建类损失: loss_rec, loss_n2v
# - 频域先验类损失: loss_teacher, loss_band
# - 正则化类损失: loss_low, loss_decor, loss_content
# 注: loss_con (一致性损失) 在无监督v2中已移除

ABLATION_CONFIGS = {
    # 基线实验: 完整模型（所有组件和损失都启用）
    'full_model': {
        'use_dual_output': True,
        'use_unet': False,  # 使用完整架构（TCN + SE注意力）
        'use_residual': False,
        'use_random_masking': False,  # 伪影感知加权掩蔽
        'use_n2v': True,
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,  # low + decor + content
        'description': '完整模型（所有组件和损失函数均启用，作为基线对照）'
    },
    
    # 消融实验1: 标准UNet架构
    'unet_baseline': {
        'use_dual_output': True,
        'use_unet': True,  # 使用标准UNet（无TCN，简单瓶颈）
        'use_residual': False,
        'use_random_masking': False,
        'use_n2v': True,
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,  # low + decor + content
        'description': '标准UNet架构（无TCN时序建模和SE注意力）'
    },
    
    # 消融实验2: 常规随机掩蔽
    'random_masking': {
        'use_dual_output': True,
        'use_unet': False,
        'use_residual': False,
        'use_random_masking': True,  # 使用常规随机块掩蔽
        'use_n2v': True,
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,
        'description': '常规随机块掩蔽（而非伪影感知加权掩蔽）'
    },
    
    # 消融实验3: 移除"重建类"损失
    'no_reconstruction': {
        'use_dual_output': True,
        'use_unet': False,
        'use_residual': False,
        'use_random_masking': False,
        'use_n2v': False,  # 禁用N2V重建损失
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,
        'description': '移除重建类损失（loss_rec, loss_n2v）'
    },
    
    # 消融实验4: 移除"频域先验类"损失
    'no_frequency': {
        'use_dual_output': True,
        'use_unet': False,
        'use_residual': False,
        'use_random_masking': False,
        'use_n2v': True,
        'use_teacher': False,  # 禁用Teacher损失
        'use_band': False,  # 禁用频带先验损失
        'use_regularization': True,
        'description': '移除频域先验类损失（loss_teacher, loss_band）'
    },
    
    # 消融实验5: 普通残差块（对比TCN）
    'residual_baseline': {
        'use_dual_output': True,
        'use_unet': False,
        'use_residual': True,  # 使用普通残差块替代TCN
        'use_random_masking': False,
        'use_n2v': True,
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,
        'description': '普通残差块瓶颈层（无TCN因果卷积和扩张卷积）'
    },
}

# ========== 消融实验顺序 ==========
# 定义实验执行的顺序（首先是完整基线，然后按消融程度从轻到重）
ABLATION_ORDER = [
    'full_model',            # 0. 完整模型（基线对照）
    'unet_baseline',         # 1. 架构变体：标准UNet
    'residual_baseline',     # 2. 架构变体：普通残差块
    'random_masking',        # 3. 掩蔽策略变体
    'no_reconstruction',     # 4. 移除重建类损失
    'no_frequency',          # 5. 移除频域先验类损失
]

# ========== 随机种子（确保可重复性） ==========
RANDOM_SEED = 42

# ========== 消融实验目录结构 ==========
ABLATION_ROOT = os.path.dirname(os.path.abspath(__file__))
ABLATION_CHECKPOINT_DIR = os.path.join(ABLATION_ROOT, 'checkpoints')

# 结果保存到项目根目录的results文件夹（与其他方法统一，便于compute_all_metrics.m处理）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(ABLATION_ROOT)))
ABLATION_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

os.makedirs(ABLATION_CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ABLATION_RESULTS_DIR, exist_ok=True)


def get_ablation_config(experiment_name):
    """
    获取指定实验的配置
    
    Args:
        experiment_name: 实验名称
    
    Returns:
        dict: 实验配置
    """
    if experiment_name not in ABLATION_CONFIGS:
        raise ValueError(f"未知的实验名称: {experiment_name}")
    return ABLATION_CONFIGS[experiment_name]


def get_checkpoint_path(experiment_name):
    """获取实验的模型保存路径"""
    return os.path.join(ABLATION_CHECKPOINT_DIR, f'model_{experiment_name}.pth')


def get_result_dir(experiment_name):
    """获取实验的结果保存目录"""
    result_dir = os.path.join(ABLATION_RESULTS_DIR, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    return result_dir


def print_ablation_summary():
    """打印消融实验概览"""
    print("=" * 80)
    print("DAT-Net 消融实验配置概览")
    print("=" * 80)
    print(f"\n总实验数: {len(ABLATION_CONFIGS)}")
    print(f"执行顺序: {' -> '.join(ABLATION_ORDER)}")
    print(f"\n实验详情:")
    for exp_name in ABLATION_ORDER:
        config = ABLATION_CONFIGS[exp_name]
        print(f"\n  [{exp_name}]")
        print(f"    描述: {config['description']}")
        # 统计禁用的组件
        disabled = [k for k, v in config.items() if k.startswith('use_') and not v]
        if disabled:
            print(f"    禁用组件: {', '.join([d.replace('use_', '') for d in disabled])}")
        else:
            print(f"    禁用组件: 无（完整配置）")
    print("\n" + "=" * 80)


if __name__ == '__main__':
    print_ablation_summary()
