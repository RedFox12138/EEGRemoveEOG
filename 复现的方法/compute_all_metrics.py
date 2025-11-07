"""
统一指标计算脚本 - 为所有方法计算评估指标

该脚本会：
1. 加载测试集的真实纯净信号（Test_Pure.mat）
2. 加载所有方法的预测结果（*_predictions.mat）
3. 使用相同的指标计算函数计算RRMSE, CC, RRMSE_PSD, MI
4. 保存所有指标到results/all_metrics.csv
5. 生成对比图表

作者: GitHub Copilot
日期: 2025-11-07
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from scipy.stats import entropy
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_rrmse(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """
    计算相对均方根误差 (Relative Root Mean Square Error)
    
    文献公式 (33): RRMSE(s, ŝ) = RMS(s - ŝ) / RMS(s)
    其中: RMS(x) = sqrt(mean(x²))
    
    等价于: RRMSE = sqrt(mean((s - ŝ)²) / mean(s²))
    
    参数:
        true_signal: 真实纯净信号 s [n_timepoints]
        pred_signal: 预测去噪信号 ŝ [n_timepoints]
    
    返回:
        rrmse: 相对均方根误差 (越小越好)
    """
    # RMS(s - ŝ)² = mean((s - ŝ)²)
    mse = np.mean((true_signal - pred_signal) ** 2)
    # RMS(s)² = mean(s²)
    true_power = np.mean(true_signal ** 2)
    
    if true_power == 0:
        return np.inf
    
    # RRMSE = sqrt(mean((s - ŝ)²) / mean(s²))
    return np.sqrt(mse / true_power)


def compute_cc(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """
    计算相关系数 (Correlation Coefficient)
    
    文献公式 (34): CC(s, ŝ) = Cov(s, ŝ) / sqrt(Var(s) * Var(ŝ))
    
    参数:
        true_signal: 真实纯净信号 s [n_timepoints]
        pred_signal: 预测去噪信号 ŝ [n_timepoints]
    
    返回:
        cc: 皮尔逊相关系数 [-1, 1] (越大越好)
    """
    # 展平为1D
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    
    # 计算相关系数矩阵 (numpy实现了文献公式)
    corr_matrix = np.corrcoef(true_flat, pred_flat)
    
    # 返回非对角元素
    return corr_matrix[0, 1]


def compute_rrmse_psd(true_signal: np.ndarray, pred_signal: np.ndarray, fs: int = 200) -> float:
    """
    计算基于功率谱密度的相对均方根误差 (RRMSE_PSD)
    
    文献公式 (35): RRMSE_PSD = RMS(PSD(ŝ) - PSD(s)) / RMS(PSD(s))
    
    等价于: RRMSE_PSD = sqrt(mean((PSD(ŝ) - PSD(s))²) / mean(PSD(s)²))
    
    参数:
        true_signal: 真实纯净信号 s [n_timepoints]
        pred_signal: 预测去噪信号 ŝ [n_timepoints]
        fs: 采样率 (Hz)
    
    返回:
        rrmse_psd: 频域的相对均方根误差 (越小越好)
    """
    nperseg = min(256, len(true_signal))
    noverlap = nperseg // 2
    
    # 使用Welch方法计算功率谱密度
    _, psd_true = welch(true_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, psd_pred = welch(pred_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    # 计算功率谱密度的RRMSE
    mse_psd = np.mean((psd_true - psd_pred) ** 2)
    true_psd_power = np.mean(psd_true ** 2)
    
    if true_psd_power == 0:
        return np.inf
    
    return np.sqrt(mse_psd / true_psd_power)


def compute_mi(true_signal: np.ndarray, pred_signal: np.ndarray, bins: int = 50) -> float:
    """
    计算互信息 (Mutual Information)
    
    文献公式 (36): MI = ∫∫ p(ŝ, s) log(p(ŝ, s) / (p(ŝ)p(s))) dŝds
    
    使用离散化方法计算两个信号之间的互信息
    
    参数:
        true_signal: 真实纯净信号 s [n_timepoints]
        pred_signal: 预测去噪信号 ŝ [n_timepoints]
        bins: 直方图分箱数量
    
    返回:
        mi: 互信息 (越大越好)
    """
    # 展平信号
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    
    # 计算2D直方图
    hist_2d, _, _ = np.histogram2d(true_flat, pred_flat, bins=bins)
    
    # 归一化为联合概率分布 p(ŝ, s)
    pxy = hist_2d / np.sum(hist_2d)
    
    # 计算边缘概率 p(s) 和 p(ŝ)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    # 计算独立概率 p(ŝ) * p(s)
    px_py = px[:, np.newaxis] * py[np.newaxis, :]
    
    # 避免log(0)
    nonzero_mask = (pxy > 0) & (px_py > 0)
    
    if np.sum(nonzero_mask) == 0:
        return 0.0
    
    # 计算互信息: MI = Σ p(ŝ,s) * log(p(ŝ,s) / (p(ŝ)*p(s)))
    mi = np.sum(pxy[nonzero_mask] * np.log(pxy[nonzero_mask] / px_py[nonzero_mask]))
    
    return mi


def compute_metrics_for_method(predictions: np.ndarray, true_signals: np.ndarray, 
                                fs: int = 200) -> Dict[str, Tuple[float, float]]:
    """
    为单个方法计算所有指标
    
    参数:
        predictions: 预测信号 [n_samples, n_timepoints]
        true_signals: 真实信号 [n_samples, n_timepoints]
        fs: 采样率
    
    返回:
        metrics: 字典，包含每个指标的均值和标准差
    """
    n_samples = predictions.shape[0]
    
    rrmse_list = []
    cc_list = []
    rrmse_psd_list = []
    mi_list = []
    
    print(f"  计算 {n_samples} 个样本的指标...")
    
    for i in range(n_samples):
        true_sig = true_signals[i]
        pred_sig = predictions[i]
        
        rrmse_list.append(compute_rrmse(true_sig, pred_sig))
        cc_list.append(compute_cc(true_sig, pred_sig))
        rrmse_psd_list.append(compute_rrmse_psd(true_sig, pred_sig, fs))
        mi_list.append(compute_mi(true_sig, pred_sig))
        
        if (i + 1) % 50 == 0:
            print(f"    进度: {i+1}/{n_samples}")
    
    metrics = {
        'RRMSE': (np.mean(rrmse_list), np.std(rrmse_list)),
        'CC': (np.mean(cc_list), np.std(cc_list)),
        'RRMSE_PSD': (np.mean(rrmse_psd_list), np.std(rrmse_psd_list)),
        'MI': (np.mean(mi_list), np.std(mi_list))
    }
    
    return metrics


def load_test_data(data_dir: str) -> np.ndarray:
    """加载测试集的真实纯净信号"""
    pure_path = os.path.join(data_dir, 'Test_Pure.mat')
    
    if not os.path.exists(pure_path):
        raise FileNotFoundError(f"找不到测试集文件: {pure_path}")
    
    data = sio.loadmat(pure_path)
    pure_signals = data['data']
    
    print(f"✓ 加载测试集: {pure_signals.shape}")
    return pure_signals


def load_method_predictions(results_dir: str, method_name: str) -> Tuple[np.ndarray, float]:
    """
    加载指定方法的预测结果
    
    返回:
        predictions: 预测信号
        time_per_sample: 单样本处理时间
    """
    pred_path = os.path.join(results_dir, f'{method_name}_predictions.mat')
    
    if not os.path.exists(pred_path):
        print(f"  ⚠ 警告: 找不到 {method_name} 的预测结果")
        return None, None
    
    data = sio.loadmat(pred_path)
    predictions = data['predictions']
    
    # 获取时间信息
    time_per_sample = data.get('time_per_sample', np.array([[0]]))[0, 0]
    
    print(f"  ✓ {method_name:15s} - 形状: {predictions.shape}, 时间: {time_per_sample*1000:.3f}ms/样本")
    
    return predictions, time_per_sample


def save_results(results_dict: Dict, output_path: str):
    """保存结果到CSV文件"""
    # 准备DataFrame
    data = []
    
    for method, metrics in results_dict.items():
        row = {
            'Method': method,
            'RRMSE_mean': metrics['RRMSE'][0],
            'RRMSE_std': metrics['RRMSE'][1],
            'CC_mean': metrics['CC'][0],
            'CC_std': metrics['CC'][1],
            'RRMSE_PSD_mean': metrics['RRMSE_PSD'][0],
            'RRMSE_PSD_std': metrics['RRMSE_PSD'][1],
            'MI_mean': metrics['MI'][0],
            'MI_std': metrics['MI'][1],
            'Time_per_sample_ms': metrics.get('time_per_sample', 0) * 1000
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 按照RRMSE排序（越小越好）
    df = df.sort_values('RRMSE_mean')
    
    df.to_csv(output_path, index=False, float_format='%.6f')
    print(f"\n✓ 结果已保存到: {output_path}")
    
    return df


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """生成对比图表"""
    methods = df['Method'].values
    n_methods = len(methods)
    
    # 创建2x2子图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('EOG去除方法性能对比', fontsize=16, fontweight='bold')
    
    metrics_info = [
        ('RRMSE', 'RRMSE (↓越小越好)', axes[0, 0]),
        ('CC', 'CC (↑越大越好)', axes[0, 1]),
        ('RRMSE_PSD', 'RRMSE_PSD (↓越小越好)', axes[1, 0]),
        ('MI', 'MI (↑越大越好)', axes[1, 1])
    ]
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_methods))
    
    for metric, title, ax in metrics_info:
        means = df[f'{metric}_mean'].values
        stds = df[f'{metric}_std'].values
        
        bars = ax.bar(range(n_methods), means, yerr=stds, capsize=5, 
                      color=colors, edgecolor='black', linewidth=1.2, alpha=0.8)
        
        ax.set_xlabel('方法', fontsize=11, fontweight='bold')
        ax.set_ylabel(title, fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # 在柱子上显示数值
        for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{mean:.4f}\n±{std:.4f}',
                   ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存到: {plot_path}")
    
    plt.close()


def plot_time_comparison(df: pd.DataFrame, output_dir: str):
    """生成处理时间对比图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = df['Method'].values
    times = df['Time_per_sample_ms'].values
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
    bars = ax.bar(range(len(methods)), times, color=colors, edgecolor='black', 
                  linewidth=1.2, alpha=0.8)
    
    ax.set_xlabel('方法', fontsize=12, fontweight='bold')
    ax.set_ylabel('单样本处理时间 (ms)', fontsize=12, fontweight='bold')
    ax.set_title('各方法处理速度对比', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    # 在柱子上显示数值
    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_val:.2f}ms',
               ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 时间对比图已保存到: {plot_path}")
    
    plt.close()


def main():
    print("="*80)
    print("统一指标计算脚本".center(80))
    print("="*80)
    
    # 路径配置
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    results_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    
    # 方法列表
    methods = [
        'ACMD',
        'EWTICEEMDAN',
        'SSA',
        'VME_EFD',
        'Threshold',
        'ASNet',
        'EEGIFNet'
    ]
    
    # 1. 加载测试集真实数据
    print("\n1. 加载测试集真实数据")
    print("-"*80)
    true_signals = load_test_data(data_dir)
    
    # 2. 加载所有方法的预测结果
    print("\n2. 加载所有方法的预测结果")
    print("-"*80)
    
    predictions_dict = {}
    time_dict = {}
    
    for method in methods:
        pred, time_val = load_method_predictions(results_dir, method)
        if pred is not None:
            predictions_dict[method] = pred
            time_dict[method] = time_val
    
    # 3. 计算所有方法的指标
    print("\n3. 计算评估指标")
    print("-"*80)
    
    results_dict = {}
    
    for method in predictions_dict.keys():
        print(f"\n计算 {method} 的指标:")
        predictions = predictions_dict[method]
        
        # 确保维度匹配
        if predictions.shape != true_signals.shape:
            print(f"  ⚠ 警告: 形状不匹配 {predictions.shape} vs {true_signals.shape}")
            continue
        
        metrics = compute_metrics_for_method(predictions, true_signals)
        metrics['time_per_sample'] = time_dict[method]
        
        results_dict[method] = metrics
        
        print(f"  完成! RRMSE={metrics['RRMSE'][0]:.4f}, CC={metrics['CC'][0]:.4f}")
    
    # 4. 保存结果
    print("\n4. 保存结果")
    print("-"*80)
    
    output_csv = os.path.join(results_dir, 'all_metrics.csv')
    df = save_results(results_dict, output_csv)
    
    # 5. 打印对比表格
    print("\n5. 结果对比表")
    print("="*80)
    print(df.to_string(index=False))
    
    # 6. 生成图表
    print("\n6. 生成对比图表")
    print("-"*80)
    plot_comparison(df, results_dir)
    plot_time_comparison(df, results_dir)
    
    print("\n" + "="*80)
    print("✓ 所有指标计算完成！".center(80))
    print("="*80)
    print(f"\n结果文件:")
    print(f"  - CSV表格: {output_csv}")
    print(f"  - 指标对比图: {os.path.join(results_dir, 'metrics_comparison.png')}")
    print(f"  - 时间对比图: {os.path.join(results_dir, 'time_comparison.png')}")
    print("\n" + "="*80)


if __name__ == '__main__':
    main()
