"""
统一评估脚本 - 读取所有方法的.mat结果并计算指标对比

所有方法(深度学习+传统)都应该保存.mat格式:
- predictions: [n_samples, n_channels, n_timepoints] 预测的干净EEG
- metrics: 包含RRMSE_mean, CC_mean, RRMSE_PSD_mean, MI_mean等字段

使用方法:
    python evaluate_all_methods.py
"""

import os
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from sklearn.metrics import mutual_info_score
import pandas as pd


def compute_rrmse(true_signal, pred_signal):
    """相对均方根误差"""
    mse = np.mean((true_signal - pred_signal) ** 2)
    true_power = np.mean(true_signal ** 2)
    return np.sqrt(mse / true_power)


def compute_cc(true_signal, pred_signal):
    """相关系数"""
    return np.corrcoef(true_signal.flatten(), pred_signal.flatten())[0, 1]


def compute_rrmse_psd(true_signal, pred_signal, fs=200):
    """基于功率谱密度的相对均方根误差"""
    nperseg = 256
    
    # 计算PSD
    _, psd_true = welch(true_signal, fs=fs, nperseg=nperseg, axis=-1)
    _, psd_pred = welch(pred_signal, fs=fs, nperseg=nperseg, axis=-1)
    
    # 计算RRMSE
    mse_psd = np.mean((psd_true - psd_pred) ** 2)
    true_psd_power = np.mean(psd_true ** 2)
    return np.sqrt(mse_psd / true_psd_power)


def compute_mi(true_signal, pred_signal, bins=50):
    """互信息 - 使用离散化方法"""
    # 展平信号
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    
    # 离散化
    true_discrete = np.digitize(true_flat, bins=np.linspace(true_flat.min(), true_flat.max(), bins))
    pred_discrete = np.digitize(pred_flat, bins=np.linspace(pred_flat.min(), pred_flat.max(), bins))
    
    # 计算互信息
    return mutual_info_score(true_discrete, pred_discrete)


def evaluate_predictions(true_signals, pred_signals, fs=200):
    """
    评估预测结果
    
    Args:
        true_signals: [n_samples, n_channels, n_timepoints]
        pred_signals: [n_samples, n_channels, n_timepoints]
        fs: 采样率
    
    Returns:
        dict: 包含mean和std的指标字典
    """
    n_samples = true_signals.shape[0]
    
    rrmse_list = []
    cc_list = []
    rrmse_psd_list = []
    mi_list = []
    
    for i in range(n_samples):
        true_sig = true_signals[i]
        pred_sig = pred_signals[i]
        
        rrmse_list.append(compute_rrmse(true_sig, pred_sig))
        cc_list.append(compute_cc(true_sig, pred_sig))
        rrmse_psd_list.append(compute_rrmse_psd(true_sig, pred_sig, fs))
        mi_list.append(compute_mi(true_sig, pred_sig))
    
    return {
        'RRMSE': {'mean': np.mean(rrmse_list), 'std': np.std(rrmse_list)},
        'CC': {'mean': np.mean(cc_list), 'std': np.std(cc_list)},
        'RRMSE_PSD': {'mean': np.mean(rrmse_psd_list), 'std': np.std(rrmse_psd_list)},
        'MI': {'mean': np.mean(mi_list), 'std': np.std(mi_list)}
    }


def load_mat_result(mat_path):
    """
    加载.mat格式的结果
    
    尝试不同的变量名:
    - predictions/denoised/y_pred/pred
    - metrics (如果已经计算好了指标)
    """
    data = sio.loadmat(mat_path)
    
    # 尝试加载预测结果
    pred_vars = ['predictions', 'denoised', 'y_pred', 'pred', 'clean_eeg']
    predictions = None
    for var in pred_vars:
        if var in data:
            predictions = data[var]
            break
    
    # 尝试加载已计算的指标
    metrics = None
    if 'metrics' in data:
        metrics_struct = data['metrics']
        # MATLAB结构体需要特殊处理
        try:
            metrics = {
                'RRMSE': {
                    'mean': float(metrics_struct['RRMSE_mean'][0][0][0][0]),
                    'std': float(metrics_struct['RRMSE_std'][0][0][0][0])
                },
                'CC': {
                    'mean': float(metrics_struct['CC_mean'][0][0][0][0]),
                    'std': float(metrics_struct['CC_std'][0][0][0][0])
                },
                'RRMSE_PSD': {
                    'mean': float(metrics_struct['RRMSE_PSD_mean'][0][0][0][0]),
                    'std': float(metrics_struct['RRMSE_PSD_std'][0][0][0][0])
                },
                'MI': {
                    'mean': float(metrics_struct['MI_mean'][0][0][0][0]),
                    'std': float(metrics_struct['MI_std'][0][0][0][0])
                }
            }
        except:
            pass
    
    return predictions, metrics


def main():
    # 路径配置
    results_dir = 'results'
    data_dir = '../生成半模拟数据/已经生成好的数据'
    
    # 加载真实数据
    print("加载真实数据...")
    true_data = sio.loadmat(os.path.join(data_dir, 'Test_Pure.mat'))
    
    # 尝试不同的变量名
    possible_names = ['Pure_Data', 'pure', 'clean', 'y_clean', 'test_pure']
    true_signals = None
    for name in possible_names:
        if name in true_data:
            true_signals = true_data[name]
            print(f"找到真实数据变量: '{name}'")
            break
    
    if true_signals is None:
        print(f"错误: 未找到真实数据变量")
        print(f"可用变量: {[k for k in true_data.keys() if not k.startswith('__')]}")
        return
    
    print(f"真实数据形状: {true_signals.shape}")
    
    # 方法列表
    methods = ['Threshold', 'ACMD', 'SSA', 'VME_EFD', 'EWTICEEMDAN', 'ASNet', 'EEGIFNet']
    
    # 存储所有结果
    all_results = {}
    
    print("\n" + "="*60)
    print("评估各方法...")
    print("="*60)
    
    for method in methods:
        print(f"\n处理 {method}...")
        
        # 尝试加载预测结果
        pred_path = os.path.join(results_dir, f'{method}_predictions.mat')
        metrics_path = os.path.join(results_dir, f'{method}_metrics.mat')
        
        if not os.path.exists(pred_path) and not os.path.exists(metrics_path):
            print(f"  ⚠️  未找到结果文件")
            continue
        
        # 优先加载已计算的指标
        if os.path.exists(metrics_path):
            _, metrics = load_mat_result(metrics_path)
            if metrics is not None:
                all_results[method] = metrics
                print(f"  ✓ 从 {method}_metrics.mat 加载指标")
                continue
        
        # 否则加载预测结果并计算指标
        if os.path.exists(pred_path):
            predictions, _ = load_mat_result(pred_path)
            if predictions is not None:
                print(f"  预测结果形状: {predictions.shape}")
                
                # 计算指标
                print(f"  计算指标...")
                metrics = evaluate_predictions(true_signals, predictions)
                all_results[method] = metrics
                
                # 保存指标到.mat文件
                metrics_save = {
                    'RRMSE_mean': metrics['RRMSE']['mean'],
                    'RRMSE_std': metrics['RRMSE']['std'],
                    'CC_mean': metrics['CC']['mean'],
                    'CC_std': metrics['CC']['std'],
                    'RRMSE_PSD_mean': metrics['RRMSE_PSD']['mean'],
                    'RRMSE_PSD_std': metrics['RRMSE_PSD']['std'],
                    'MI_mean': metrics['MI']['mean'],
                    'MI_std': metrics['MI']['std']
                }
                sio.savemat(metrics_path, {'metrics': metrics_save})
                print(f"  ✓ 指标已保存到 {method}_metrics.mat")
            else:
                print(f"  ⚠️  无法从 {method}_predictions.mat 加载预测结果")
    
    # 生成对比表格
    if not all_results:
        print("\n没有找到任何方法的结果!")
        return
    
    print("\n" + "="*60)
    print("生成对比表格...")
    print("="*60)
    
    # 创建DataFrame
    rows = []
    for method, metrics in all_results.items():
        row = {
            'Method': method,
            'RRMSE': f"{metrics['RRMSE']['mean']:.4f}±{metrics['RRMSE']['std']:.4f}",
            'CC': f"{metrics['CC']['mean']:.4f}±{metrics['CC']['std']:.4f}",
            'RRMSE_PSD': f"{metrics['RRMSE_PSD']['mean']:.4f}±{metrics['RRMSE_PSD']['std']:.4f}",
            'MI': f"{metrics['MI']['mean']:.2f}±{metrics['MI']['std']:.2f}"
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('Method')
    
    # 打印表格
    print("\n" + df.to_string())
    
    # 保存为CSV
    csv_path = os.path.join(results_dir, 'comparison_table.csv')
    df.to_csv(csv_path)
    print(f"\n✓ CSV表格已保存: {csv_path}")
    
    # 保存为LaTeX
    latex_path = os.path.join(results_dir, 'comparison_table.tex')
    with open(latex_path, 'w') as f:
        f.write(df.to_latex())
    print(f"✓ LaTeX表格已保存: {latex_path}")
    
    # 找出最佳方法
    print("\n" + "="*60)
    print("最佳方法:")
    print("="*60)
    
    # RRMSE和RRMSE_PSD越小越好
    rrmse_values = {m: all_results[m]['RRMSE']['mean'] for m in all_results}
    best_rrmse = min(rrmse_values, key=rrmse_values.get)
    print(f"RRMSE最小: {best_rrmse} ({rrmse_values[best_rrmse]:.4f})")
    
    rrmse_psd_values = {m: all_results[m]['RRMSE_PSD']['mean'] for m in all_results}
    best_rrmse_psd = min(rrmse_psd_values, key=rrmse_psd_values.get)
    print(f"RRMSE_PSD最小: {best_rrmse_psd} ({rrmse_psd_values[best_rrmse_psd]:.4f})")
    
    # CC和MI越大越好
    cc_values = {m: all_results[m]['CC']['mean'] for m in all_results}
    best_cc = max(cc_values, key=cc_values.get)
    print(f"CC最大: {best_cc} ({cc_values[best_cc]:.4f})")
    
    mi_values = {m: all_results[m]['MI']['mean'] for m in all_results}
    best_mi = max(mi_values, key=mi_values.get)
    print(f"MI最大: {best_mi} ({mi_values[best_mi]:.2f})")


if __name__ == '__main__':
    main()
