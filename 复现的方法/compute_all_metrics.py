"""
统一指标计算脚本 - 自动扫描并对比所有方法 (格式化CSV输出版)

该脚本会：
1. 加载测试集的真实纯净信号（Test_Pure.mat）
2. 自动扫描 results 文件夹下所有 .mat 文件
3. 提取文件名第一个 "_" 之前的内容作为方法名
4. 计算 RRMSE, CC, RRMSE_PSD, MI 指标
5. 保存结果到CSV：使用 "Mean ± Std" 格式，保疙3位小数
6. 生成对比图

修改内容：CSV格式调整 (Mean ± Std)
日期: 2025-11-29
"""

import os
import glob
import numpy as np
import scipy.io as sio
from scipy.signal import welch

# 导入数据集配置
from dataset_config import get_dataset_config
DATA_CONFIG = get_dataset_config()
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def compute_rrmse(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """计算相对均方根误差 (RRMSE)"""
    mse = np.mean((true_signal - pred_signal) ** 2)
    true_power = np.mean(true_signal ** 2)
    if true_power == 0: return np.inf
    return np.sqrt(mse / true_power)


def compute_cc(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """计算相关系数 (CC)"""
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    corr_matrix = np.corrcoef(true_flat, pred_flat)
    return corr_matrix[0, 1]


def compute_rrmse_psd(true_signal: np.ndarray, pred_signal: np.ndarray, fs: int = 200) -> float:
    """计算功率谱相对误差 (RRMSE_PSD)"""
    nperseg = min(256, len(true_signal))
    noverlap = nperseg // 2
    _, psd_true = welch(true_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, psd_pred = welch(pred_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    mse_psd = np.mean((psd_true - psd_pred) ** 2)
    true_psd_power = np.mean(psd_true ** 2)
    if true_psd_power == 0: return np.inf
    return np.sqrt(mse_psd / true_psd_power)


def compute_mi(true_signal: np.ndarray, pred_signal: np.ndarray, bins: int = 50) -> float:
    """计算互信息 (MI)"""
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    hist_2d, _, _ = np.histogram2d(true_flat, pred_flat, bins=bins)
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, np.newaxis] * py[np.newaxis, :]
    nonzero_mask = (pxy > 0) & (px_py > 0)
    if np.sum(nonzero_mask) == 0: return 0.0
    mi = np.sum(pxy[nonzero_mask] * np.log(pxy[nonzero_mask] / px_py[nonzero_mask]))
    return mi


def compute_metrics_for_method(predictions: np.ndarray, true_signals: np.ndarray,
                                fs: int = 200) -> Dict[str, Tuple[float, float]]:
    """为单个方法计算所有指标"""
    n_samples = predictions.shape[0]
    rrmse_list, cc_list, rrmse_psd_list, mi_list = [], [], [], []

    print(f"    正在计算 {n_samples} 个样本...", end='', flush=True)

    for i in range(n_samples):
        true_sig = true_signals[i]
        pred_sig = predictions[i]

        rrmse_list.append(compute_rrmse(true_sig, pred_sig))
        cc_list.append(compute_cc(true_sig, pred_sig))
        rrmse_psd_list.append(compute_rrmse_psd(true_sig, pred_sig, fs))
        mi_list.append(compute_mi(true_sig, pred_sig))

    print(" 完成")

    metrics = {
        'RRMSE': (np.mean(rrmse_list), np.std(rrmse_list)),
        'CC': (np.mean(cc_list), np.std(cc_list)),
        'RRMSE_PSD': (np.mean(rrmse_psd_list), np.std(rrmse_psd_list)),
        'MI': (np.mean(mi_list), np.std(mi_list))
    }
    return metrics


def load_test_data() -> np.ndarray:
    """加载测试集的真实纯净信号"""
    pure_path = DATA_CONFIG['test_pure_path']
    if not os.path.exists(pure_path):
        raise FileNotFoundError(f"找不到测试集文件: {pure_path}")
    data = sio.loadmat(pure_path)
    pure_signals = data[DATA_CONFIG['data_key']]
    print(f"✓ 已加载基准测试集 (Test_Pure.mat): {pure_signals.shape}")
    return pure_signals


def load_prediction_file(file_path: str) -> Tuple[np.ndarray, float]:
    """加载指定路径的预测文件"""
    try:
        data = sio.loadmat(file_path)
    except Exception as e:
        print(f"  ❌ 文件损坏或无法读取: {os.path.basename(file_path)}")
        return None, None

    predictions = None
    if 'predictions' in data:
        predictions = data['predictions']
    elif 'data' in data:
        predictions = data['data']
    elif 'clean_data' in data:
        predictions = data['clean_data']
    else:
        max_size = 0
        for key in data:
            if not key.startswith('__') and isinstance(data[key], np.ndarray):
                if data[key].size > max_size:
                    predictions = data[key]
                    max_size = data[key].size
        if predictions is not None:
            print(f"  ⚠ 警告: 未找到标准变量名，已自动选择最大变量作为数据。")

    if predictions is None:
        print(f"  ❌ 错误: 在 {os.path.basename(file_path)} 中未找到有效数据变量")
        return None, None

    time_per_sample = 0.0
    if 'time_per_sample' in data:
        time_arr = data['time_per_sample']
        if time_arr.size > 0:
            time_per_sample = time_arr.item()

    return predictions, time_per_sample


def save_results(results_dict: Dict, output_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    格式化并保存结果
    返回: (df_numeric, df_formatted)
    - df_numeric: 用于画图的纯数值表格
    - df_formatted: 用于保存CSV和展示的格式化表格 (Mean ± Std)
    """

    # 1. 首先构建数值型 DataFrame（用于排序和画图）
    data_numeric = []
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
        data_numeric.append(row)

    df_numeric = pd.DataFrame(data_numeric)

    # 按照 RRMSE_mean 进行排序
    if not df_numeric.empty:
        df_numeric = df_numeric.sort_values('RRMSE_mean')

    # 2. 构建格式化 DataFrame（用于CSV输出）
    data_formatted = []
    for _, row in df_numeric.iterrows():
        # 格式化函数：保留3位小数
        fmt = lambda m, s: f"{m:.3f} ± {s:.3f}"

        csv_row = {
            'Method': row['Method'],
            'RRMSE': fmt(row['RRMSE_mean'], row['RRMSE_std']),
            'CC': fmt(row['CC_mean'], row['CC_std']),
            'RRMSE_PSD': fmt(row['RRMSE_PSD_mean'], row['RRMSE_PSD_std']),
            'MI': fmt(row['MI_mean'], row['MI_std']),
            'Time (ms)': f"{row['Time_per_sample_ms']:.3f}"
        }
        data_formatted.append(csv_row)

    df_formatted = pd.DataFrame(data_formatted)

    # 保存 CSV (使用 utf-8-sig 编码以防止 Excel 中 ± 显示乱码)
    df_formatted.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n✓ 结果已保存到: {output_path}")

    return df_numeric, df_formatted


def plot_comparison(df: pd.DataFrame, output_dir: str):
    """生成对比图表 (使用数值型 DataFrame)"""
    if df.empty: return

    methods = df['Method'].values
    n_methods = len(methods)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('各方法去噪性能对比', fontsize=18, fontweight='bold')

    metrics_info = [
        ('RRMSE', 'RRMSE (↓越小越好)', axes[0, 0]),
        ('CC', 'CC (↑越大越好)', axes[0, 1]),
        ('RRMSE_PSD', 'RRMSE_PSD (↓越小越好)', axes[1, 0]),
        ('MI', 'MI (↑越大越好)', axes[1, 1])
    ]

    colors = plt.cm.tab20(np.linspace(0, 1, n_methods))

    for metric, title, ax in metrics_info:
        # 注意：这里需要使用 numeric df 中的列名 (xxx_mean, xxx_std)
        means = df[f'{metric}_mean'].values
        stds = df[f'{metric}_std'].values

        bars = ax.bar(range(n_methods), means, yerr=stds, capsize=5,
                      color=colors, edgecolor='black', alpha=0.85)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xticks(range(n_methods))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=10)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        for bar, mean in zip(bars, means):
            height = bar.get_height()
            y_pos = height if height > 0 else 0
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{mean:.3f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(output_dir, 'metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 性能对比图已保存: {plot_path}")
    plt.close()


def plot_time_comparison(df: pd.DataFrame, output_dir: str):
    """生成时间对比图"""
    if df.empty: return

    fig, ax = plt.subplots(figsize=(12, 6))

    methods = df['Method'].values
    times = df['Time_per_sample_ms'].values

    bars = ax.bar(range(len(methods)), times, color='skyblue', edgecolor='black', alpha=0.8)

    ax.set_ylabel('单样本处理时间 (ms)', fontsize=12)
    ax.set_title('各方法运行速度对比', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for bar, time_val in zip(bars, times):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{time_val:.1f}ms',
               ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'time_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ 时间对比图已保存: {plot_path}")
    plt.close()


def main():
    print("="*80)
    print(" 自动化指标对比脚本 ".center(80, '='))
    print("="*80)

    # ---------------- 配置路径 ----------------
    results_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    # ----------------------------------------

    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在 -> {results_dir}")
        return

    # 1. 加载测试集
    try:
        true_signals = load_test_data()
    except FileNotFoundError as e:
        print(e)
        return

    # 2. 扫描文件
    print(f"\n[扫描目录] {results_dir}")
    mat_files = [f for f in os.listdir(results_dir) if f.endswith('.mat')]

    if not mat_files:
        print("该目录下没有找到 .mat 文件！")
        return

    predictions_dict = {}

    print(f"发现 {len(mat_files)} 个数据文件，开始处理...")

    # 3. 遍历计算
    for file_name in mat_files:
        if "Test_Pure" in file_name: continue

        # 提取方法名：取第一个 "_" 之前
        if '_' in file_name:
            method_name = file_name.split('_')[0]
        else:
            method_name = os.path.splitext(file_name)[0]

        file_path = os.path.join(results_dir, file_name)
        print(f"\n>>> 处理方法: [{method_name}] (文件: {file_name})")

        predictions, time_val = load_prediction_file(file_path)
        if predictions is None: continue

        if predictions.shape != true_signals.shape:
            print(f"  ⚠ 警告: 维度不匹配! 跳过")
            continue

        metrics = compute_metrics_for_method(predictions, true_signals)
        metrics['time_per_sample'] = time_val
        predictions_dict[method_name] = metrics
        print(f"  ✓ {method_name} 计算完毕")

    if not predictions_dict:
        print("\n没有成功计算任何方法的指标。")
        return

    # 4. 保存结果 (获取 数值版 和 格式化版 两个DF)
    output_csv = os.path.join(results_dir, 'all_metrics.csv')
    df_numeric, df_formatted = save_results(predictions_dict, output_csv)

    # 5. 打印最终结果 (使用格式化版，好看)
    print("\n" + "="*80)
    print("最终结果排行".center(80))
    print("="*80)
    print(df_formatted.to_string(index=False))

    # 6. 生成图表 (必须使用数值版 df_numeric)
    plot_comparison(df_numeric, results_dir)
    plot_time_comparison(df_numeric, results_dir)

    print("\n" + "="*80)
    print("全部完成！".center(80))

if __name__ == '__main__':
    main()