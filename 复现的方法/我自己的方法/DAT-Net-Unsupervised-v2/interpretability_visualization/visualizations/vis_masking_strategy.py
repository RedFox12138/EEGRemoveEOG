"""
任务3: Masked策略可视化

展示Masked生成过程：
- OriginalSignal
- Artifact概率
- WeightedMasked概率
- 实际Masked位置
- Masked的Signal

Comparison展示普通随机Masked vs Artifact感知Masked的区别
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

from utils import *
from config import *

# 导入无监督Loss函数
try:
    from unsupervised_artifact_v2 import compute_artifact_prob_v2
    from unsupervised_artifact_v1 import generate_masked_input_artifact_aware
except:
    print("警告: 无法导入部分工具函数")


def generate_random_mask(x, mask_ratio=0.15, neighborhood=5):
    """
    生成随机Masked(用于Comparison)
    
    Args:
        x: (B, 1, L) InputSignal
        mask_ratio: Masked比例
        neighborhood: 邻域大小
    
    Returns:
        x_masked: Masked的Signal
        mask: Masked位置 (B, 1, L)
    """
    B, C, L = x.shape
    device = x.device
    
    # 随机选择Masked位置
    num_mask = int(L * mask_ratio)
    mask = torch.zeros_like(x)
    
    for b in range(B):
        # 随机选择Masked中心点
        mask_indices = np.random.choice(L, num_mask, replace=False)
        
        # 扩展邻域
        for idx in mask_indices:
            start = max(0, idx - neighborhood // 2)
            end = min(L, idx + neighborhood // 2 + 1)
            mask[b, :, start:end] = 1.0
    
    # 应用Masked
    x_masked = x.clone()
    x_masked[mask.bool()] = 0.0
    
    return x_masked, mask


def visualize_masking_strategy(model, data_loader, device, sample_idx=0, **kwargs):
    """
    可视化Masked策略
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        sample_idx: sample索引
        **kwargs: 其他参数
    """
    print(f"正在可视化sample {sample_idx} 的Masked策略...")
    
    # 从data_loader获取样本
    from utils import get_sample_by_index
    sample = get_sample_by_index(data_loader, sample_idx)
    if sample is None:
        print(f"错误: 无法获取样本 {sample_idx}")
        return
    
    sample_input = sample['contaminated'].numpy()[0, 0]
    sample_clean = sample['target'].numpy()[0, 0]
    sample_eog = sample_input - sample_clean  # 真实的EOG伪影
    
    # 准备Input
    input_tensor = sample['contaminated'].to(device)
    norm_factor = torch.abs(input_tensor).max()
    if norm_factor > 0:
        input_scaled = input_tensor / norm_factor
    else:
        input_scaled = input_tensor
    
    # 参数
    fs = DATA_CONFIG['fs']
    mask_base = LOSS_CONFIG['mask_base']
    boost_scale = LOSS_CONFIG['boost_scale']
    neighborhood = LOSS_CONFIG['mask_neighborhood']
    
    # 1. 计算Artifact概率
    p_art = compute_artifact_prob_v2(input_scaled, fs)
    
    # 2. 生成Artifact感知Masked
    x_masked_artifact, mask_artifact = generate_masked_input_artifact_aware(
        input_scaled, fs, mask_base=mask_base, 
        boost_scale=boost_scale, neighborhood=neighborhood
    )
    
    # 3. 生成随机Masked(用于Comparison)
    x_masked_random, mask_random = generate_random_mask(
        input_scaled, mask_ratio=mask_base, neighborhood=neighborhood
    )
    
    # 转换为numpy
    time = create_time_axis(len(sample_input), fs)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().squeeze()
    
    original = to_numpy(input_scaled)
    p_art_np = to_numpy(p_art)
    mask_artifact_np = to_numpy(mask_artifact)
    mask_random_np = to_numpy(mask_random)
    x_masked_artifact_np = to_numpy(x_masked_artifact)
    x_masked_random_np = to_numpy(x_masked_random)
    
    # 计算WeightedMasked概率
    p_mask = mask_base + boost_scale * p_art_np
    p_mask = np.clip(p_mask, 0, 1)
    
    # 创建图像
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    colors = VIS_CONFIG['colors']
    
    # 第1行：Original Signal和Artifact概率 + 真实EOG
    ax0 = fig.add_subplot(gs[0, :])
    ax0_twin = ax0.twinx()
    
    line1 = ax0.plot(time, original, color=colors['original'], linewidth=1.5, 
                     label='Contaminated Signal', alpha=0.8)
    line2 = ax0_twin.fill_between(time, 0, p_art_np, color=colors['artifact'], 
                                   alpha=0.3, label='Artifact Probability')
    ax0_twin.plot(time, p_art_np, color=colors['artifact'], linewidth=1.5, alpha=0.8)
    line3 = ax0.plot(time, sample_eog * np.max(np.abs(original)) / np.max(np.abs(sample_eog)), 
                     color='#FF6B35', linewidth=1.2, label='True EOG (scaled)', alpha=0.6, linestyle='--')
    
    ax0.set_ylabel('Amplitude', fontsize=10)
    ax0_twin.set_ylabel('Artifact Probability', fontsize=10)
    ax0_twin.set_ylim([0, 1])
    ax0.set_title('Step 1: Original Signal vs Artifact Probability & True EOG', fontsize=12, fontweight='bold')
    
    lines = line1 + line3 + [line2]
    labels = [l.get_label() for l in line1] + [l.get_label() for l in line3] + ['Artifact Probability']
    ax0.legend(lines, labels, loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # 第2行：Weighted Masking概率
    ax1 = fig.add_subplot(gs[1, :])
    ax1.fill_between(time, 0, p_mask, color='#9B59B6', alpha=0.5, label='Weighted Mask Probability')
    ax1.plot(time, p_mask, color='#9B59B6', linewidth=1.5)
    ax1.axhline(y=mask_base, color='gray', linestyle='--', alpha=0.6, 
                label=f'Base Probability = {mask_base:.3f}')
    ax1.axhline(y=mask_base + boost_scale, color='red', linestyle='--', alpha=0.6,
                label=f'Max Probability = {mask_base + boost_scale:.3f}')
    ax1.set_ylabel('Mask Probability', fontsize=10)
    ax1.set_ylim([0, 1])
    ax1.set_title(f'Step 2: Weighted Mask Probability (base={mask_base:.3f}, boost={boost_scale:.3f})', 
                 fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 第3行左：Artifact-Aware Masking
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(time, original, color='lightgray', linewidth=1, alpha=0.6, label='Original Signal')
    ax2.plot(time, x_masked_artifact_np, color=colors['masked'], linewidth=1.5, 
             label='Masked Signal')
    ax2.fill_between(time, original.min(), original.max(), where=(mask_artifact_np > 0), 
                     alpha=0.3, color='yellow', label='Masked Region')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title('Step 3a: Artifact-Aware Masking', fontsize=11, fontweight='bold')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    # 第3行右：Random Masking
    ax3 = fig.add_subplot(gs[2, 1])
    ax3.plot(time, original, color='lightgray', linewidth=1, alpha=0.6, label='Original Signal')
    ax3.plot(time, x_masked_random_np, color='#FF6B6B', linewidth=1.5, 
             label='Masked Signal')
    ax3.fill_between(time, original.min(), original.max(), where=(mask_random_np > 0), 
                     alpha=0.3, color='yellow', label='Masked Region')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title('Step 3b: Random Masking', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right')
    ax3.grid(True, alpha=0.3)
    
    # 第4行左：Artifact-Aware Masked部分 vs 真实EOG
    ax4_left = fig.add_subplot(gs[3, 0])
    masked_signal_artifact = original.copy()
    masked_signal_artifact[mask_artifact_np > 0] = 0
    extracted_artifact = original - masked_signal_artifact  # 被掩蔽的部分
    ax4_left.plot(time, extracted_artifact, color=colors['masked'], linewidth=1.5, 
                  label='Masked Components', alpha=0.8)
    ax4_left.plot(time, sample_eog * np.max(np.abs(extracted_artifact)) / (np.max(np.abs(sample_eog)) + 1e-8), 
                  color='#FF6B35', linewidth=1.2, label='True EOG (scaled)', alpha=0.6, linestyle='--')
    ax4_left.set_xlabel('Time (s)', fontsize=10)
    ax4_left.set_ylabel('Amplitude', fontsize=10)
    ax4_left.set_title('Step 4a: Artifact-Aware Masked Components vs True EOG', fontsize=11, fontweight='bold')
    ax4_left.legend(loc='upper right')
    ax4_left.grid(True, alpha=0.3)
    
    # 第4行右：Random Masked部分 vs 真实EOG
    ax4_right = fig.add_subplot(gs[3, 1])
    masked_signal_random = original.copy()
    masked_signal_random[mask_random_np > 0] = 0
    extracted_random = original - masked_signal_random  # 被掩蔽的部分
    ax4_right.plot(time, extracted_random, color='#FF6B6B', linewidth=1.5, 
                   label='Masked Components', alpha=0.8)
    ax4_right.plot(time, sample_eog * np.max(np.abs(extracted_random)) / (np.max(np.abs(sample_eog)) + 1e-8), 
                   color='#FF6B35', linewidth=1.2, label='True EOG (scaled)', alpha=0.6, linestyle='--')
    ax4_right.set_xlabel('Time (s)', fontsize=10)
    ax4_right.set_ylabel('Amplitude', fontsize=10)
    ax4_right.set_title('Step 4b: Random Masked Components vs True EOG', fontsize=11, fontweight='bold')
    ax4_right.legend(loc='upper right')
    ax4_right.grid(True, alpha=0.3)
    
    # 第5行：Mask位置对比
    ax5 = fig.add_subplot(gs[4, :])
    
    # 绘制Artifact概率作为背景
    ax5.fill_between(time, 0, 0.5, where=(p_art_np > 0.5), alpha=0.2, 
                     color=colors['artifact'], label='High Artifact Region (p>0.5)')
    
    # 绘制两种Mask位置
    artifact_mask_positions = np.where(mask_artifact_np > 0)[0]
    random_mask_positions = np.where(mask_random_np > 0)[0]
    
    ax5.scatter(time[artifact_mask_positions], 
               np.ones(len(artifact_mask_positions)) * 0.3,
               c=colors['masked'], s=1, alpha=0.6, label='Artifact-Aware Mask Positions')
    ax5.scatter(time[random_mask_positions], 
               np.ones(len(random_mask_positions)) * 0.1,
               c='#FF6B6B', s=1, alpha=0.6, label='Random Mask Positions')
    
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Mask Type', fontsize=10)
    ax5.set_ylim([0, 0.5])
    ax5.set_yticks([0.1, 0.3])
    ax5.set_yticklabels(['Random', 'Artifact-Aware'])
    ax5.set_title('Step 5: Mask Position Comparison', fontsize=12, fontweight='bold')
    ax5.legend(loc='upper right')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # 总标题
    fig.suptitle(f'Artifact-Aware Masking Strategy - Sample {sample_idx}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 保存
    save_figure(fig, f'masking_strategy_sample_{sample_idx}', subdir='task3')
    
    # 统计信息
    artifact_mask_ratio = mask_artifact_np.sum() / len(mask_artifact_np)
    random_mask_ratio = mask_random_np.sum() / len(mask_random_np)
    
    # 计算Masked与高Artifact区域的重叠
    high_artifact_mask = (p_art_np > 0.5).astype(float)
    artifact_overlap = (mask_artifact_np * high_artifact_mask).sum() / max(1, mask_artifact_np.sum())
    random_overlap = (mask_random_np * high_artifact_mask).sum() / max(1, mask_random_np.sum())
    
    print(f"  Artifact感知Masked比例: {artifact_mask_ratio*100:.2f}%")
    print(f"  随机Masked比例: {random_mask_ratio*100:.2f}%")
    print(f"  Artifact感知Masked在高Artifact区域的Ratio: {artifact_overlap*100:.2f}%")
    print(f"  随机Masked在高Artifact区域的Ratio: {random_overlap*100:.2f}%")
    
    # 保存数据
    save_data({
        'time': time,
        'original': original,
        'p_art': p_art_np,
        'p_mask': p_mask,
        'mask_artifact': mask_artifact_np,
        'mask_random': mask_random_np,
        'x_masked_artifact': x_masked_artifact_np,
        'x_masked_random': x_masked_random_np,
        'artifact_mask_ratio': artifact_mask_ratio,
        'random_mask_ratio': random_mask_ratio,
        'artifact_overlap': artifact_overlap,
        'random_overlap': random_overlap,
        'sample_idx': sample_idx,
    }, f'masking_strategy_sample_{sample_idx}', subdir='task3')
    
    plt.close(fig)


if __name__ == '__main__':
    # 测试本模块
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils import initialize
    initialize()
    visualize_masking_strategy(sample_idx=0)
