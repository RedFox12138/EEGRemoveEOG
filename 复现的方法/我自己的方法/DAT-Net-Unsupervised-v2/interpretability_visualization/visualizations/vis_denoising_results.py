"""
任务12: Denoising Performance可视化

展示典型sample的处理Result：
- OriginalContaminated Signal
- Denoising的CleanSignal
- ExtractedArtifactSignal
- 与True标签的Comparison
- 多个评价指标(RRMSE, CC, SNR等)
- 不同sample的Denoising PerformanceComparison

直观展示模型的Denoising能力和效果
"""

import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import pearsonr

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.dirname(parent_dir))

from utils import *
from config import *


def compute_denoising_metrics(pred, target):
    """
    计算Denoising评价指标
    
    Args:
        pred: Prediction的CleanSignal (L,)
        target: True的CleanSignal (L,)
    
    Returns:
        dict: 评价指标
    """
    # 1. RRMSE (Relative Root Mean Square Error)
    mse = np.mean((pred - target) ** 2)
    rmse = np.sqrt(mse)
    target_rms = np.sqrt(np.mean(target ** 2))
    rrmse = rmse / (target_rms + 1e-8)
    
    # 2. CC (Correlation Coefficient)
    cc, _ = pearsonr(pred, target)
    
    # 3. SNR (Signal-to-Noise Ratio)
    signal_power = np.mean(target ** 2)
    noise_power = np.mean((pred - target) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-8))
    
    # 4. MAE (Mean Absolute Error)
    mae = np.mean(np.abs(pred - target))
    
    # 5. PSNR (Peak Signal-to-Noise Ratio)
    max_val = max(np.abs(target).max(), np.abs(pred).max())
    psnr = 10 * np.log10((max_val ** 2) / (mse + 1e-8))
    
    # 6. SSIM-like measure (1D adapted)
    # 简化版：基于Local相似度
    window_size = 50
    local_sim = []
    for i in range(0, len(pred) - window_size, window_size // 2):
        p_seg = pred[i:i+window_size]
        t_seg = target[i:i+window_size]
        if np.std(p_seg) > 1e-8 and np.std(t_seg) > 1e-8:
            sim, _ = pearsonr(p_seg, t_seg)
            local_sim.append(sim)
    ssim_score = np.mean(local_sim) if len(local_sim) > 0 else 0.0
    
    return {
        'rrmse': rrmse,
        'cc': cc,
        'snr_db': snr,
        'mae': mae,
        'rmse': rmse,
        'psnr_db': psnr,
        'ssim': ssim_score
    }


def compute_improvement_metrics(contaminated, pred, target):
    """
    计算改进指标(相对于Input)
    
    Args:
        contaminated: Contaminated Signal (L,)
        pred: Prediction的CleanSignal (L,)
        target: True的CleanSignal (L,)
    
    Returns:
        dict: 改进指标
    """
    # Input的Error
    input_mse = np.mean((contaminated - target) ** 2)
    input_rmse = np.sqrt(input_mse)
    input_cc, _ = pearsonr(contaminated, target)
    
    # Output的Error
    output_mse = np.mean((pred - target) ** 2)
    output_rmse = np.sqrt(output_mse)
    output_cc, _ = pearsonr(pred, target)
    
    # 改进率
    rmse_improvement = (input_rmse - output_rmse) / (input_rmse + 1e-8) * 100
    cc_improvement = (output_cc - input_cc) / (1 - input_cc + 1e-8) * 100
    
    # SNR改进
    input_snr = 10 * np.log10(np.mean(target ** 2) / (input_mse + 1e-8))
    output_snr = 10 * np.log10(np.mean(target ** 2) / (output_mse + 1e-8))
    snr_improvement = output_snr - input_snr
    
    return {
        'input_rmse': input_rmse,
        'output_rmse': output_rmse,
        'rmse_improvement': rmse_improvement,
        'input_cc': input_cc,
        'output_cc': output_cc,
        'cc_improvement': cc_improvement,
        'input_snr': input_snr,
        'output_snr': output_snr,
        'snr_improvement': snr_improvement
    }


def visualize_denoising_results(model, data_loader, device, sample_idx=0,
                                output_dir='outputs/figures', **kwargs):
    """
    可视化Denoising Performance
    
    Args:
        model: DAT-Net模型
        data_loader: 数据加载器
        device: 设备
        sample_idx: sample索引
        output_dir: Output目录
    """
    print(f"\n{'='*60}")
    print(f"任务12: Denoising Performance可视化")
    print(f"{'='*60}\n")
    
    # 获取sample
    sample = get_sample_by_index(data_loader, sample_idx)
    if sample is None:
        print(f"错误: 无法获取sample {sample_idx}")
        return
    
    contaminated = sample['contaminated'].to(device)  # (1, 1, L)
    target = sample['target'].to(device) if 'target' in sample else None
    
    if target is None:
        print("错误: 没有True标签，无法评估Denoising Performance")
        return
    
    # First向传播
    model.eval()
    with torch.no_grad():
        outputs = model(contaminated)
        clean_pred = outputs['clean_B']  # 使用BranchB (1, 1, L)
        artifact_pred = outputs['artifact_B']  # (1, 1, L)
    
    # 转为numpy
    contaminated_np = contaminated.cpu().numpy()[0, 0]  # (L,)
    clean_np = clean_pred.cpu().numpy()[0, 0]
    artifact_np = artifact_pred.cpu().numpy()[0, 0]
    target_np = target.cpu().numpy()[0, 0]
    
    # 采样率
    fs = VIS_CONFIG.get('sampling_rate', 200.0)
    time = np.arange(len(contaminated_np)) / fs
    
    # ==================== 计算评价指标 ====================
    print("计算Denoising评价指标...")
    
    # DenoisingQuality
    metrics = compute_denoising_metrics(clean_np, target_np)
    print(f"\nDenoisingQuality:")
    print(f"  RRMSE: {metrics['rrmse']:.6f}")
    print(f"  CC: {metrics['cc']:.6f}")
    print(f"  SNR: {metrics['snr_db']:.2f} dB")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  PSNR: {metrics['psnr_db']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.6f}")
    
    # 改进指标
    improvement = compute_improvement_metrics(contaminated_np, clean_np, target_np)
    print(f"\n改进指标:")
    print(f"  RMSE: {improvement['input_rmse']:.6f} → {improvement['output_rmse']:.6f} (改进 {improvement['rmse_improvement']:.2f}%)")
    print(f"  CC: {improvement['input_cc']:.6f} → {improvement['output_cc']:.6f} (改进 {improvement['cc_improvement']:.2f}%)")
    print(f"  SNR: {improvement['input_snr']:.2f} dB → {improvement['output_snr']:.2f} dB (改进 {improvement['snr_improvement']:.2f} dB)")
    
    # ==================== 可视化 ====================
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(5, 2, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1, 1, 1, 1, 0.8])
    
    # 设置中文字体
    setup_chinese_font()
    
    # ========== (a) Contaminated Signal ==========
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(time, contaminated_np, 'k-', linewidth=0.8, label='Contaminated')
    ax1.set_xlabel('Time (s)', fontsize=10)
    ax1.set_ylabel('Amplitude', fontsize=10)
    ax1.set_title('(a) Contaminated Signal (Input)', fontsize=11, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='upper right')
    
    # ========== (b) DenoisingResultComparison ==========
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(time, target_np, 'g-', linewidth=1.0, alpha=0.7, label='True Clean')
    ax2.plot(time, clean_np, 'b--', linewidth=0.8, alpha=0.8, label='Predicted Clean')
    ax2.set_xlabel('Time (s)', fontsize=10)
    ax2.set_ylabel('Amplitude', fontsize=10)
    ax2.set_title(f'(b) DenoisingResultComparison (CC={metrics["cc"]:.4f})', 
                  fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='upper right')
    
    # ========== (c) 三Signal叠加Comparison ==========
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(time, contaminated_np, 'k-', linewidth=0.8, alpha=0.5, label='Contaminated')
    ax3.plot(time, target_np, 'g-', linewidth=1.0, alpha=0.7, label='True Clean')
    ax3.plot(time, clean_np, 'b--', linewidth=0.8, alpha=0.8, label='Predicted Clean')
    ax3.set_xlabel('Time (s)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title('(c) SignalComparison(Overlay)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='upper right')
    
    # ========== (d) DenoisingError ==========
    ax4 = fig.add_subplot(gs[2, 0])
    error = clean_np - target_np
    ax4.plot(time, error, 'r-', linewidth=0.8, alpha=0.7, label='DenoisingError')
    ax4.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.5)
    ax4.fill_between(time, 0, error, alpha=0.3, color='red')
    ax4.set_xlabel('Time (s)', fontsize=10)
    ax4.set_ylabel('ErrorAmplitude', fontsize=10)
    ax4.set_title(f'(d) DenoisingError (RMSE={metrics["rmse"]:.4f})', 
                  fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='upper right')
    
    # ========== (e) ExtractedArtifact ==========
    ax5 = fig.add_subplot(gs[2, 1])
    true_artifact = contaminated_np - target_np
    ax5.plot(time, true_artifact, 'orange', linewidth=1.0, alpha=0.6, label='True Artifact')
    ax5.plot(time, artifact_np, 'r--', linewidth=0.8, alpha=0.8, label='Predicted Artifact')
    ax5.set_xlabel('Time (s)', fontsize=10)
    ax5.set_ylabel('Amplitude', fontsize=10)
    ax5.set_title('(e) ExtractedArtifactComparison', fontsize=11, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9, loc='upper right')
    
    # ========== (f) Scatter Plot：Prediction vs True ==========
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.scatter(target_np, clean_np, c='blue', s=10, alpha=0.4)
    
    # Ideal线
    min_val = min(target_np.min(), clean_np.min())
    max_val = max(target_np.max(), clean_np.max())
    ax6.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=1.5, 
            alpha=0.6, label='y=x (Ideal)')
    
    ax6.set_xlabel('True Clean', fontsize=10)
    ax6.set_ylabel('Predicted Clean', fontsize=10)
    ax6.set_title(f'(f) Prediction vs True (CC={metrics["cc"]:.4f})', 
                  fontsize=11, fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=9)
    ax6.set_aspect('equal', adjustable='box')
    
    # ========== (g) 评价指标雷达图 ==========
    ax7 = fig.add_subplot(gs[3, 1], projection='polar')
    
    # 准备数据(归一化到0-1)
    metrics_labels = ['CC', 'SSIM', 'SNR\n改进', 'RMSE\n改进', 'RRMSE\n(逆)']
    metrics_values = [
        max(0, metrics['cc']),  # 0-1
        max(0, metrics['ssim']),  # 0-1
        min(1.0, max(0, improvement['snr_improvement'] / 20)),  # 归一化 (假设20dB为满分)
        min(1.0, max(0, improvement['rmse_improvement'] / 100)),  # 0-100% -> 0-1
        max(0, 1.0 - metrics['rrmse'])  # RRMSE越小越好
    ]
    
    # 闭合雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics_labels), endpoint=False).tolist()
    metrics_values += metrics_values[:1]
    angles += angles[:1]
    
    ax7.plot(angles, metrics_values, 'o-', linewidth=2, color='green', alpha=0.7)
    ax7.fill(angles, metrics_values, alpha=0.25, color='green')
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(metrics_labels, fontsize=9)
    ax7.set_ylim(0, 1)
    ax7.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax7.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=8)
    ax7.set_title('(g) DenoisingPerformanceComprehensiveScore', fontsize=11, fontweight='bold', pad=20)
    ax7.grid(True)
    
    # ========== (h) 指标Total结与改进Comparison ==========
    ax8 = fig.add_subplot(gs[4, :])
    ax8.axis('off')
    
    # 左侧：指标表
    table_data = [
        ['评价指标', 'Value', '评价'],
        ['-'*15, '-'*15, '-'*30],
        ['RRMSE', f'{metrics["rrmse"]:.6f}', '越小越好' + ('✓优秀' if metrics["rrmse"] < 0.1 else '✓良好' if metrics["rrmse"] < 0.3 else '')],
        ['CC', f'{metrics["cc"]:.6f}', '越大越好' + ('✓优秀' if metrics["cc"] > 0.95 else '✓良好' if metrics["cc"] > 0.85 else '')],
        ['SNR (dB)', f'{metrics["snr_db"]:.2f}', '越大越好' + ('✓优秀' if metrics["snr_db"] > 20 else '✓良好' if metrics["snr_db"] > 10 else '')],
        ['MAE', f'{metrics["mae"]:.6f}', '越小越好'],
        ['PSNR (dB)', f'{metrics["psnr_db"]:.2f}', '越大越好'],
        ['SSIM', f'{metrics["ssim"]:.6f}', '越大越好' + ('✓优秀' if metrics["ssim"] > 0.9 else '✓良好' if metrics["ssim"] > 0.7 else '')],
        ['-'*15, '-'*15, '-'*30],
        ['改进指标', 'Input → Output', '改进率'],
        ['-'*15, '-'*15, '-'*30],
        ['RMSE', f'{improvement["input_rmse"]:.4f} → {improvement["output_rmse"]:.4f}', 
         f'{improvement["rmse_improvement"]:.2f}%' + ('✓' if improvement["rmse_improvement"] > 0 else '✗')],
        ['CC', f'{improvement["input_cc"]:.4f} → {improvement["output_cc"]:.4f}',
         f'{improvement["cc_improvement"]:.2f}%' + ('✓' if improvement["cc_improvement"] > 0 else '✗')],
        ['SNR (dB)', f'{improvement["input_snr"]:.2f} → {improvement["output_snr"]:.2f}',
         f'+{improvement["snr_improvement"]:.2f} dB' + ('✓' if improvement["snr_improvement"] > 0 else '✗')],
    ]
    
    table = ax8.table(cellText=table_data, loc='center', cellLoc='left',
                     colWidths=[0.2, 0.3, 0.5])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # 设置样式
    for i in range(3):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
        table[(9, i)].set_facecolor('#2196F3')
        table[(9, i)].set_text_props(weight='bold', color='white')
    
    # Total标题
    fig.suptitle(f'Task 12: Denoising Performance Visualization (Sample {sample_idx})', 
                 fontsize=14, fontweight='bold', y=0.995)
    
    # 保存 - 使用详细的描述性文件名
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    save_path = os.path.join(output_dir, 
        f'Task12_DenoisingPerformance_sample{sample_idx:03d}_RRMSE{metrics["rrmse"]:.3f}_CC{metrics["cc"]:.3f}_{timestamp}.png')
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ 图像已保存到: {save_path}")
    
    if VIS_CONFIG.get('show_plots', False):
        plt.show()
    plt.close()
    
    # ==================== Comprehensive评价 ====================
    print(f"\n{'='*60}")
    print("Denoising PerformanceComprehensive评价:")
    print(f"{'='*60}")
    
    score = 0
    max_score = 4
    
    if metrics['cc'] > 0.95:
        print("✓ 相关性: 优秀 (得分: 1/1)")
        score += 1
    elif metrics['cc'] > 0.85:
        print("✓ 相关性: 良好 (得分: 0.7/1)")
        score += 0.7
    else:
        print("⚠ 相关性: 需改进 (得分: 0.4/1)")
        score += 0.4
    
    if metrics['rrmse'] < 0.1:
        print("✓ RRMSE: 优秀 (得分: 1/1)")
        score += 1
    elif metrics['rrmse'] < 0.3:
        print("✓ RRMSE: 良好 (得分: 0.7/1)")
        score += 0.7
    else:
        print("⚠ RRMSE: 需改进 (得分: 0.4/1)")
        score += 0.4
    
    if improvement['snr_improvement'] > 10:
        print("✓ SNR改进: 显著 (得分: 1/1)")
        score += 1
    elif improvement['snr_improvement'] > 5:
        print("✓ SNR改进: 明显 (得分: 0.7/1)")
        score += 0.7
    else:
        print("⚠ SNR改进: 有限 (得分: 0.4/1)")
        score += 0.4
    
    if improvement['rmse_improvement'] > 70:
        print("✓ RMSE改进: 显著 (得分: 1/1)")
        score += 1
    elif improvement['rmse_improvement'] > 50:
        print("✓ RMSE改进: 明显 (得分: 0.7/1)")
        score += 0.7
    else:
        print("⚠ RMSE改进: 有限 (得分: 0.4/1)")
        score += 0.4
    
    print(f"\nTotal体得分: {score:.2f}/{max_score} ({score/max_score*100:.1f}%)")
    
    if score / max_score >= 0.9:
        print("🌟 Denoising Performance: 优秀")
    elif score / max_score >= 0.7:
        print("✓ Denoising Performance: 良好")
    else:
        print("⚠ Denoising Performance: 需要改进")
    
    print(f"{'='*60}\n")


# ==================== 测试代码 ====================
if __name__ == '__main__':
    print("测试任务12: Denoising Performance可视化\n")
    
    # 初始化
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和数据
    model = load_model(device)
    data_loader = load_test_data()
    
    # 运行可视化
    visualize_denoising_results(
        model=model,
        data_loader=data_loader,
        device=device,
        sample_idx=0
    )
    
    print("测试完成!")
