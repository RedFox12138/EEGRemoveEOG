"""
DAT-Net 真实数据集测试脚本
对真实数据集进行去噪处理并保存结果
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial']
matplotlib.rcParams['axes.unicode_minus'] = False

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # DAT-Net-Unsupervised-v2
grandparent_dir = os.path.dirname(parent_dir)  # 我自己的方法
great_grandparent_dir = os.path.dirname(grandparent_dir)  # 复现的方法目录
datnet_dir = os.path.join(grandparent_dir, 'DAT-Net')

# 添加必要的路径
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, parent_dir)  # 添加 DAT-Net-Unsupervised-v2 目录
sys.path.insert(0, current_dir)  # 添加当前目录
sys.path.insert(0, great_grandparent_dir)  # 添加复现的方法目录，以访问 load_real_dataset_split

from model import DATNet
from real_data_config import *
from load_real_dataset_split import load_real_dataset_split


class RealDataset(Dataset):
    """真实数据集（用于测试）"""
    def __init__(self, noisy):
        self.noisy = noisy

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        return noisy.astype('float32') / norm, norm


def print_config():
    """打印配置信息"""
    print('\n' + '='*80)
    print('配置信息')
    print('='*80)
    print(f'采样率: {SAMPLING_RATE} Hz')
    print(f'窗口大小: {WINDOW_SIZE} 样本 ({WINDOW_DURATION:.1f} 秒)')
    print(f'数据路径: {REAL_DATA_PATH}')
    print('='*80)


def load_data():
    """加载真实数据（只加载测试集）"""
    # 使用统一的数据划分函数，只返回测试集
    data = load_real_dataset_split(
        data_path=REAL_DATA_PATH,
        data_key=DATA_KEY,
        return_train=False  # 只需要测试集
    )
    return data


def visualize_results(original, cleaned, artifacts, num_samples=5, save_dir='./results'):
    """
    可视化去噪结果
    
    参数:
        original: 原始污染信号 (N, L)
        cleaned: 去噪后的信号 (N, L)
        artifacts: 提取的伪影 (N, L)
        num_samples: 展示的样本数量
        save_dir: 保存图像的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择一些样本进行可视化
    indices = np.random.choice(len(original), min(num_samples, len(original)), replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        
        # 计算时间轴
        time_axis = np.arange(len(original[idx])) / SAMPLING_RATE
        
        # 子图1: 原始信号
        ax1 = axes[0]
        ax1.plot(time_axis, original[idx], 'r-', linewidth=1, label='原始污染信号')
        ax1.set_xlabel('时间 (秒)', fontsize=11)
        ax1.set_ylabel('幅值', fontsize=11)
        ax1.set_title(f'样本 #{idx+1}: 原始信号', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 去噪后的信号
        ax2 = axes[1]
        ax2.plot(time_axis, cleaned[idx], 'b-', linewidth=1, label='去噪后的 EEG 信号')
        ax2.set_xlabel('时间 (秒)', fontsize=11)
        ax2.set_ylabel('幅值', fontsize=11)
        ax2.set_title(f'样本 #{idx+1}: 去噪结果', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 子图3: 提取的伪影
        ax3 = axes[2]
        ax3.plot(time_axis, artifacts[idx], 'g-', linewidth=1, label='提取的 EOG 伪影')
        ax3.set_xlabel('时间 (秒)', fontsize=11)
        ax3.set_ylabel('幅值', fontsize=11)
        ax3.set_title(f'样本 #{idx+1}: 提取的伪影', fontsize=13, fontweight='bold')
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{idx+1:04d}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'  已保存: {save_path}')
        plt.close()
    
    # 创建性能汇总图
    print('\n生成性能汇总图...')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 计算统计信息
    original_power = np.mean(original ** 2, axis=1)
    cleaned_power = np.mean(cleaned ** 2, axis=1)
    artifact_power = np.mean(artifacts ** 2, axis=1)
    
    # 子图1: 信号功率分布
    ax1 = axes[0, 0]
    ax1.hist(original_power, bins=50, alpha=0.5, label='原始信号', color='red')
    ax1.hist(cleaned_power, bins=50, alpha=0.5, label='去噪信号', color='blue')
    ax1.set_xlabel('信号功率', fontsize=11)
    ax1.set_ylabel('样本数量', fontsize=11)
    ax1.set_title('信号功率分布', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 伪影功率分布
    ax2 = axes[0, 1]
    ax2.hist(artifact_power, bins=50, alpha=0.7, label='提取的伪影', color='green')
    ax2.set_xlabel('伪影功率', fontsize=11)
    ax2.set_ylabel('样本数量', fontsize=11)
    ax2.set_title('伪影功率分布', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 功率降低比例
    power_reduction = (original_power - cleaned_power) / (original_power + 1e-10)
    ax3 = axes[1, 0]
    ax3.hist(power_reduction * 100, bins=50, alpha=0.7, color='purple')
    ax3.set_xlabel('功率降低比例 (%)', fontsize=11)
    ax3.set_ylabel('样本数量', fontsize=11)
    ax3.set_title('去噪功率降低比例', fontsize=13, fontweight='bold')
    ax3.axvline(np.mean(power_reduction) * 100, color='red', linestyle='--', 
                linewidth=2, label=f'平均: {np.mean(power_reduction)*100:.1f}%')
    ax3.legend(loc='upper right', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 重建一致性验证
    reconstructed = cleaned + artifacts
    consistency_error = np.mean((reconstructed - original) ** 2, axis=1)
    ax4 = axes[1, 1]
    ax4.hist(consistency_error, bins=50, alpha=0.7, color='orange')
    ax4.set_xlabel('重建误差 (MSE)', fontsize=11)
    ax4.set_ylabel('样本数量', fontsize=11)
    ax4.set_title('重建一致性 (EEG + EOG ≈ 原始)', fontsize=13, fontweight='bold')
    ax4.axvline(np.mean(consistency_error), color='red', linestyle='--', 
                linewidth=2, label=f'平均: {np.mean(consistency_error):.6f}')
    ax4.legend(loc='upper right', fontsize=10)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'performance_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f'  已保存汇总图: {summary_path}')
    plt.close()
    
    print(f'\n可视化完成！共生成 {len(indices)} 个样本对比图和 1 个性能汇总图')
    print(f'所有图像保存在: {save_dir}')


def main():
    print('='*80)
    print('DAT-Net 真实数据集测试')
    print('无监督去除 EOG 伪影')
    print('='*80)
    
    # 打印配置
    print_config()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f'当前工作目录: {os.getcwd()}')
    
    # 创建结果目录
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # 加载数据（只加载测试集）
    test_x = load_data()
    print(f'\n测试集样本数: {len(test_x)}')
    
    # 创建数据集和加载器
    ds = RealDataset(test_x)
    loader = DataLoader(ds, batch_size=50, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'  模型参数量: {model.count_parameters():,}')
    
    # 加载训练好的模型
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f'  ✓ 加载模型: {MODEL_SAVE_PATH}')
    elif os.path.exists(FINAL_MODEL_PATH):
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
        print(f'  ✓ 加载模型: {FINAL_MODEL_PATH}')
    else:
        print(f'  ⚠️ 找不到训练好的模型，使用随机初始化权重')
        print(f'  请先运行 train_real_data.py 训练模型')
    
    # 推理
    print('\n开始推理...')
    model.eval()
    
    eeg_preds = []
    eog_preds = []
    sample_count = 0
    start = time()
    
    with torch.no_grad():
        for noisy, norm in loader:
            sample_count += noisy.shape[0]
            
            # 前向传播
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 归一化的
            norm_t = norm.float().view(-1, 1, 1).to(device)  # (B, 1, 1)
            noisy_scaled = noisy_t * norm_t  # 恢复原始尺度（与训练时一致）
            
            # 模型推理
            eeg_clean, eog_artifact = model(noisy_scaled)
            
            eeg_preds.append(eeg_clean.squeeze(1).cpu().numpy())
            eog_preds.append(eog_artifact.squeeze(1).cpu().numpy())
    
    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)
    
    # 合并结果
    eeg_preds = np.concatenate(eeg_preds, axis=0)
    eog_preds = np.concatenate(eog_preds, axis=0)
    
    print(f'推理完成! 单样本时间: {time_per_sample*1000:.3f} ms')
    print(f'总处理时间: {total_time:.2f} 秒')
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = eeg_preds + eog_preds
    original = test_x
    consistency_error = np.mean((reconstructed - original) ** 2)
    print(f'  重建一致性 MSE: {consistency_error:.6f}')
    print(f'  (应该接近0，表示 EEG_clean + EOG_artifact ≈ 原始信号)')
    
    # 计算统计信息
    print('\n统计信息:')
    original_std = np.std(test_x)
    cleaned_std = np.std(eeg_preds)
    artifact_std = np.std(eog_preds)
    print(f'  原始信号标准差: {original_std:.4f}')
    print(f'  去噪信号标准差: {cleaned_std:.4f}')
    print(f'  伪影标准差: {artifact_std:.4f}')
    
    power_reduction = (np.mean(test_x ** 2) - np.mean(eeg_preds ** 2)) / np.mean(test_x ** 2)
    print(f'  平均功率降低: {power_reduction * 100:.2f}%')
    
    # 保存结果
    print('\n保存结果...')
    scipy.io.savemat(PREDICTION_SAVE_PATH, {
        'cleaned_eeg': eeg_preds,  # 去噪后的 EEG 信号
        'extracted_eog': eog_preds,  # 提取的 EOG 伪影
        'original': test_x,  # 原始信号
        'time_per_sample': time_per_sample,
        'consistency_error': consistency_error,
        'power_reduction': power_reduction,
        'sampling_rate': SAMPLING_RATE,
        'window_size': WINDOW_SIZE,
    })
    print(f'  ✓ 预测结果已保存: {PREDICTION_SAVE_PATH}')
    print(f'    - 去噪 EEG 形状: {eeg_preds.shape}')
    print(f'    - 提取 EOG 形状: {eog_preds.shape}')
    
    # 生成可视化结果
    print('\n' + '='*80)
    print('生成可视化结果...')
    print('='*80)
    visualize_results(
        original=test_x,
        cleaned=eeg_preds,
        artifacts=eog_preds,
        num_samples=5,
        save_dir=RESULTS_DIR
    )
    
    print('\n' + '='*80)
    print('测试完成！')
    print('='*80)


if __name__ == '__main__':
    main()
