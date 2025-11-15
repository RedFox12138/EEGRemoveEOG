"""
统一的评价指标计算模块
供所有深度学习训练脚本使用

从 compute_all_metrics.py 中提取核心评价指标计算函数
不包含文件加载和可视化功能，专注于指标计算
"""

import numpy as np
from scipy.signal import welch
from typing import Dict


def compute_rrmse(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """
    计算相对均方根误差 (Relative Root Mean Square Error)
    
    RRMSE = sqrt(mean((s - ŝ)²) / mean(s²))
    
    参数:
        true_signal: 真实纯净信号
        pred_signal: 预测去噪信号
    
    返回:
        rrmse: 相对均方根误差 (越小越好)
    """
    mse = np.mean((true_signal - pred_signal) ** 2)
    true_power = np.mean(true_signal ** 2)
    
    if true_power == 0:
        return np.inf
    
    return np.sqrt(mse / true_power)


def compute_cc(true_signal: np.ndarray, pred_signal: np.ndarray) -> float:
    """
    计算相关系数 (Correlation Coefficient)
    
    参数:
        true_signal: 真实纯净信号
        pred_signal: 预测去噪信号
    
    返回:
        cc: 皮尔逊相关系数 [-1, 1] (越大越好)
    """
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    
    corr_matrix = np.corrcoef(true_flat, pred_flat)
    
    return corr_matrix[0, 1]


def compute_rrmse_psd(true_signal: np.ndarray, pred_signal: np.ndarray, fs: int = 200) -> float:
    """
    计算基于功率谱密度的相对均方根误差 (RRMSE_PSD)
    
    参数:
        true_signal: 真实纯净信号
        pred_signal: 预测去噪信号
        fs: 采样率 (Hz)
    
    返回:
        rrmse_psd: 频域的相对均方根误差 (越小越好)
    """
    nperseg = min(256, len(true_signal))
    noverlap = nperseg // 2
    
    _, psd_true = welch(true_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    _, psd_pred = welch(pred_signal, fs=fs, nperseg=nperseg, noverlap=noverlap)
    
    mse_psd = np.mean((psd_true - psd_pred) ** 2)
    true_psd_power = np.mean(psd_true ** 2)
    
    if true_psd_power == 0:
        return np.inf
    
    return np.sqrt(mse_psd / true_psd_power)


def compute_mi(true_signal: np.ndarray, pred_signal: np.ndarray, bins: int = 50) -> float:
    """
    计算互信息 (Mutual Information)
    
    参数:
        true_signal: 真实纯净信号
        pred_signal: 预测去噪信号
        bins: 直方图分箱数量
    
    返回:
        mi: 互信息 (越大越好)
    """
    true_flat = true_signal.flatten()
    pred_flat = pred_signal.flatten()
    
    hist_2d, _, _ = np.histogram2d(true_flat, pred_flat, bins=bins)
    
    pxy = hist_2d / np.sum(hist_2d)
    
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    
    px_py = px[:, np.newaxis] * py[np.newaxis, :]
    
    nonzero_mask = (pxy > 0) & (px_py > 0)
    
    if np.sum(nonzero_mask) == 0:
        return 0.0
    
    mi = np.sum(pxy[nonzero_mask] * np.log(pxy[nonzero_mask] / px_py[nonzero_mask]))
    
    return mi


def compute_all_metrics(predictions: np.ndarray, true_signals: np.ndarray, 
                        fs: int = 200) -> Dict[str, float]:
    """
    为一个batch的预测计算所有评价指标
    
    参数:
        predictions: 预测信号 [n_samples, n_timepoints] 或 [n_samples, 1, n_timepoints]
        true_signals: 真实信号 [n_samples, n_timepoints] 或 [n_samples, 1, n_timepoints]
        fs: 采样率
    
    返回:
        metrics: 包含RRMSE, CC, RRMSE_PSD, MI的字典
    """
    # 处理维度
    if predictions.ndim == 3:
        predictions = predictions.squeeze(1)
    if true_signals.ndim == 3:
        true_signals = true_signals.squeeze(1)
    
    n_samples = predictions.shape[0]
    
    rrmse_list = []
    cc_list = []
    rrmse_psd_list = []
    mi_list = []
    
    for i in range(n_samples):
        true_sig = true_signals[i]
        pred_sig = predictions[i]
        
        rrmse_list.append(compute_rrmse(true_sig, pred_sig))
        cc_list.append(compute_cc(true_sig, pred_sig))
        rrmse_psd_list.append(compute_rrmse_psd(true_sig, pred_sig, fs))
        mi_list.append(compute_mi(true_sig, pred_sig))
    
    metrics = {
        'RRMSE': np.mean(rrmse_list),
        'CC': np.mean(cc_list),
        'RRMSE_PSD': np.mean(rrmse_psd_list),
        'MI': np.mean(mi_list)
    }
    
    return metrics


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    格式化打印评价指标
    
    参数:
        metrics: 指标字典
        prefix: 前缀字符串（如 "验证集"）
    """
    if prefix:
        print(f"\n{prefix}评价指标:")
    else:
        print(f"\n评价指标:")
    
    print(f"  RRMSE:     {metrics['RRMSE']:.6f}")
    print(f"  CC:        {metrics['CC']:.6f}")
    print(f"  RRMSE_PSD: {metrics['RRMSE_PSD']:.6f}")
    print(f"  MI:        {metrics['MI']:.6f}")
