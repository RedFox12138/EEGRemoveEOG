"""
任务2: 伪影概率计算可视化

展示 compute_artifact_prob_v2 函数中的各个步骤：
- 局部幅度
- 局部变化速度
- 低频能量占比
- 归一化后的特征
- 最终的伪影概率分布

通过多子图展示每个中间特征的时序变化
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

# 导入无监督损失函数中的工具
try:
    from unsupervised_artifact_v2 import _fft_lowpass
    from unsupervised_artifact_v1 import _moving_average
except:
    print("警告: 无法导入部分工具函数")


def compute_artifact_prob_with_intermediates(x, fs, win_size=64, lowpass_cutoff=4.0):
    """
    计算伪影概率并返回所有中间步骤
    
    Args:
        x: (B, 1, L) 或 (1, L) 输入信号
        fs: 采样率
        win_size: 窗口大小
        lowpass_cutoff: 低通截止频率
    
    Returns:
        dict: 包含所有中间步骤的字典
    """
    eps = 1e-8
    
    # 确保输入是 (B, 1, L) 格式
    if x.dim() == 2:
        x = x.unsqueeze(1)
    
    B, C, L = x.shape
    device = x.device
    
    # 1) 局部幅度
    amp = _moving_average(torch.abs(x), win_size)
    
    # 2) 局部变化速度
    diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    diff = F.pad(diff, (0, 1))
    diff = _moving_average(diff, win_size)
    
    # 3) 低频能量占比
    x_low = _fft_lowpass(x, fs, cutoff=lowpass_cutoff)
    power_low = _moving_average(x_low ** 2, win_size)
    power_total = _moving_average(x ** 2, win_size)
    r = power_low / (power_total + eps)
    
    # 4) MAD归一化
    def mad_normalize(a):
        med = a.median(dim=-1, keepdim=True).values
        mad = (a - med).abs().median(dim=-1, keepdim=True).values
        mad = mad.clamp(min=eps)
        return (a - med) / mad
    
    amp_n = mad_normalize(amp)
    diff_n = mad_normalize(diff)
    r_n = mad_normalize(r)
    
    # 5) 综合分数
    s = amp_n + diff_n + r_n
    
    # 6) 阈值和sigmoid映射
    try:
        tau = torch.quantile(s, 0.7, dim=-1, keepdim=True)
    except:
        s_np = s.detach().cpu().numpy()
        tau_vals = [np.quantile(s_np[i, 0, :], 0.7) for i in range(s_np.shape[0])]
        tau = torch.tensor(np.array(tau_vals), device=device, dtype=s.dtype).view(B, 1, 1)
    
    alpha = 10.0
    p = torch.sigmoid(alpha * (s - tau))
    p = p.clamp(0.0, 1.0)
    
    return {
        'original': x,
        'amp': amp,
        'diff': diff,
        'r': r,
        'x_low': x_low,
        'power_low': power_low,
        'power_total': power_total,
        'amp_n': amp_n,
        'diff_n': diff_n,
        'r_n': r_n,
        's': s,
        'tau': tau,
        'p': p,
    }


def visualize_artifact_probability(model, data_loader, device, sample_idx=0, **kwargs):
    """
    可视化伪影概率计算过程
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        sample_idx: 样本索引
        **kwargs: 其他参数
    """
    print(f"正在可视化样本 {sample_idx} 的伪影概率计算过程...")
    
    # 从data_loader获取样本
    from utils import get_sample_by_index
    sample = get_sample_by_index(data_loader, sample_idx)
    if sample is None:
        print(f"错误: 无法获取样本 {sample_idx}")
        return
    
    sample_input = sample['contaminated'].numpy()[0, 0]
    sample_clean = sample['target'].numpy()[0, 0]
    sample_eog = sample_input - sample_clean  # 真实的EOG伪影  # (L,)
    
    # 准备输入
    input_tensor = sample['contaminated'].to(device)
    norm_factor = torch.abs(input_tensor).max()
    if norm_factor > 0:
        input_scaled = input_tensor / norm_factor
    else:
        input_scaled = input_tensor
    
    # 计算伪影概率及中间步骤
    fs = DATA_CONFIG['fs']
    win_size = ARTIFACT_PROB_CONFIG['win_size']
    lowpass_cutoff = ARTIFACT_PROB_CONFIG['lowpass_cutoff']
    
    intermediates = compute_artifact_prob_with_intermediates(
        input_scaled, fs, win_size, lowpass_cutoff
    )
    
    # 转换为numpy
    time = create_time_axis(len(sample_input), fs)
    
    def to_numpy(tensor):
        return tensor.detach().cpu().numpy().squeeze()
    
    original = to_numpy(intermediates['original'])
    amp = to_numpy(intermediates['amp'])
    diff = to_numpy(intermediates['diff'])
    r = to_numpy(intermediates['r'])
    x_low = to_numpy(intermediates['x_low'])
    amp_n = to_numpy(intermediates['amp_n'])
    diff_n = to_numpy(intermediates['diff_n'])
    r_n = to_numpy(intermediates['r_n'])
    s = to_numpy(intermediates['s'])
    p = to_numpy(intermediates['p'])
    tau_value = float(intermediates['tau'].item())
    
    # 创建图像 - 分为两列：左列原始特征，右列归一化特征
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    colors = VIS_CONFIG['colors']
    
    # 第1行：原始信号（跨两列）- 添加真实EOG对比
    ax0 = fig.add_subplot(gs[0, :])
    ax0_twin = ax0.twinx()
    ax0.plot(time, original, color=colors['original'], linewidth=1.5, label='Contaminated EEG', alpha=0.8)
    ax0.fill_between(time, original.min(), original, where=(p > 0.5), 
                      alpha=0.2, color=colors['artifact'], label='Detected Artifact (p>0.5)')
    ax0_twin.plot(time, sample_eog, color=colors['artifact'], linewidth=1.2, 
                   label='True EOG Artifact', alpha=0.6, linestyle='--')
    ax0.set_ylabel('Amplitude (uV)', fontsize=10)
    ax0_twin.set_ylabel('EOG Amplitude (uV)', fontsize=10, color=colors['artifact'])
    ax0_twin.tick_params(axis='y', labelcolor=colors['artifact'])
    ax0.set_title('Step 0: Original EEG Signal with EOG Artifacts', fontsize=12, fontweight='bold')
    lines1, labels1 = ax0.get_legend_handles_labels()
    lines2, labels2 = ax0_twin.get_legend_handles_labels()
    ax0.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax0.grid(True, alpha=0.3)
    
    # 第2行左：局部幅度
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.plot(time, amp, color='#E63946', linewidth=1.2, label='Local Amplitude')
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title(f'Step 1a: Local Amplitude (window={win_size})', fontsize=11)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 第2行右：归一化后的局部幅度
    ax1_n = fig.add_subplot(gs[1, 1])
    ax1_n.plot(time, amp_n, color='#E63946', linewidth=1.2, label='Normalized Amplitude')
    ax1_n.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1_n.set_ylabel('Normalized Value', fontsize=10)
    ax1_n.set_title('Step 1b: MAD-Normalized Amplitude', fontsize=11)
    ax1_n.legend()
    ax1_n.grid(True, alpha=0.3)
    
    # 第3行左：局部变化速度
    ax2 = fig.add_subplot(gs[2, 0])
    ax2.plot(time, diff, color='#F77F00', linewidth=1.2, label='Local Variation')
    ax2.set_ylabel('Variation Rate', fontsize=10)
    ax2.set_title('Step 2a: Local Variation Rate', fontsize=11)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 第3行右：归一化后的变化速度
    ax2_n = fig.add_subplot(gs[2, 1])
    ax2_n.plot(time, diff_n, color='#F77F00', linewidth=1.2, label='Normalized Variation')
    ax2_n.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax2_n.set_ylabel('Normalized Value', fontsize=10)
    ax2_n.set_title('Step 2b: MAD-Normalized Variation', fontsize=11)
    ax2_n.legend()
    ax2_n.grid(True, alpha=0.3)
    
    # 第4行左：低频能量占比
    ax3 = fig.add_subplot(gs[3, 0])
    ax3.plot(time, r, color='#06A77D', linewidth=1.2, label='Low-Freq Ratio')
    ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='50% Reference')
    ax3.set_ylabel('Ratio', fontsize=10)
    ax3.set_ylim([0, 1])
    ax3.set_title(f'Step 3a: Low-Frequency Energy Ratio (cutoff={lowpass_cutoff}Hz)', fontsize=11)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 第4行右：归一化后的低频占比
    ax3_n = fig.add_subplot(gs[3, 1])
    ax3_n.plot(time, r_n, color='#06A77D', linewidth=1.2, label='Normalized Ratio')
    ax3_n.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax3_n.set_ylabel('Normalized Value', fontsize=10)
    ax3_n.set_title('Step 3b: MAD-Normalized Ratio', fontsize=11)
    ax3_n.legend()
    ax3_n.grid(True, alpha=0.3)
    
    # 第5行左：综合分数
    ax4 = fig.add_subplot(gs[4, 0])
    ax4.plot(time, s, color='#457B9D', linewidth=1.5, label='Composite Score s')
    ax4.axhline(y=tau_value, color='red', linestyle='--', linewidth=2, 
                label=f'Threshold tau (70th percentile) = {tau_value:.2f}')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('Score', fontsize=10)
    ax4.set_title('Step 4: Composite Score (amp_n + diff_n + r_n)', fontsize=11)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 第5行右：最终伪影概率
    ax5 = fig.add_subplot(gs[4, 1])
    ax5.fill_between(time, 0, p, color=colors['artifact'], alpha=0.6, label='Artifact Probability')
    ax5.axhline(y=0.5, color='black', linestyle='--', linewidth=1.5, label='Decision Threshold (0.5)')
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Probability', fontsize=10)
    ax5.set_ylim([0, 1])
    ax5.set_title('Step 5: Final Artifact Probability p = sigmoid(alpha*(s-tau))', fontsize=11)
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 总标题
    fig.suptitle(f'Artifact Probability Computation - Sample {sample_idx}', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 保存
    save_figure(fig, f'artifact_probability_sample_{sample_idx}', subdir='task2')
    
    # 保存中间数据
    save_data({
        'time': time,
        'original': original,
        'amp': amp,
        'diff': diff,
        'r': r,
        'amp_n': amp_n,
        'diff_n': diff_n,
        'r_n': r_n,
        's': s,
        'p': p,
        'tau': tau_value,
        'sample_idx': sample_idx,
    }, f'artifact_probability_sample_{sample_idx}', subdir='task2')
    
    # 统计信息
    artifact_ratio = (p > 0.5).sum() / len(p)
    print(f"  检测到的伪影比例: {artifact_ratio*100:.2f}%")
    print(f"  平均伪影概率: {p.mean():.4f}")
    print(f"  最大伪影概率: {p.max():.4f}")
    
    plt.close(fig)
    
    return intermediates


if __name__ == '__main__':
    # 测试本模块
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils import initialize
    initialize()
    visualize_artifact_probability(sample_idx=0)
