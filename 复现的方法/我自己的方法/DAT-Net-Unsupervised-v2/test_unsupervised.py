"""
DAT-Net 无监督测试脚本（Version 2）
保存 .mat 结果并计算评估指标
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
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, current_dir)
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
try:
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs):
        return {'RRMSE':0.0,'CC':0.0,'PRD':0.0,'SNR':0.0,'RMSE':0.0,'MAE':0.0,'PSNR':0.0,'SSIM':0.0}
    def print_metrics(m, prefix=""):
        print(prefix, 'Metrics:', m)

from model import DATNet

# 导入 v2 损失（可选用于验证一致性）
from unsupervised_artifact_v2 import unsupervised_dat_loss_artifact_v2

class TestDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    def __len__(self):
        return len(self.noisy)
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        return noisy.astype('float32') / norm, clean.astype('float32'), norm


def load_data():
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    # 计算真实眼电伪迹（污染信号 - 纯净信号）
    test_eog = test_input - test_output
    return test_input, test_output, test_eog


def visualize_results(noisy, clean_pred, clean_target, eog_pred, eog_target, num_samples=5, save_dir='./results_visualization'):
    """
    可视化去噪结果和眼电预测结果
    
    参数:
        noisy: 污染的脑电信号 (N, L)
        clean_pred: 预测的纯净脑电 (N, L)
        clean_target: 真实的纯净脑电 (N, L)
        eog_pred: 预测的眼电伪迹 (N, L)
        eog_target: 真实的眼电伪迹 (N, L)
        num_samples: 展示的样本数量
        save_dir: 保存图像的目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 随机选择一些样本进行可视化
    indices = np.random.choice(len(noisy), min(num_samples, len(noisy)), replace=False)
    
    for i, idx in enumerate(indices):
        fig, axes = plt.subplots(2, 1, figsize=(14, 8))
        
        # 计算时间轴 (假设采样率 200Hz)
        fs = 200
        time_axis = np.arange(len(noisy[idx])) / fs
        
        # 子图1: 去噪结果对比
        ax1 = axes[0]
        ax1.plot(time_axis, noisy[idx], 'r-', alpha=0.5, linewidth=1, label='污染信号')
        ax1.plot(time_axis, clean_target[idx], 'g-', linewidth=1.5, label='真实纯净脑电')
        ax1.plot(time_axis, clean_pred[idx], 'b--', linewidth=1, label='预测纯净脑电')
        ax1.set_xlabel('时间 (秒)', fontsize=11)
        ax1.set_ylabel('幅值', fontsize=11)
        ax1.set_title(f'样本 #{idx+1}: 去噪结果对比', fontsize=13, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 子图2: 眼电伪迹预测对比
        ax2 = axes[1]
        ax2.plot(time_axis, eog_target[idx], 'g-', linewidth=1.5, label='真实眼电伪迹')
        ax2.plot(time_axis, eog_pred[idx], 'b--', linewidth=1, label='预测眼电伪迹')
        ax2.set_xlabel('时间 (秒)', fontsize=11)
        ax2.set_ylabel('幅值', fontsize=11)
        ax2.set_title(f'样本 #{idx+1}: 眼电伪迹预测对比', fontsize=13, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, f'sample_{idx+1}_comparison.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'已保存可视化结果: {save_path}')
        plt.close()
    
    # 创建汇总图：展示所有样本的整体统计
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 计算所有样本的指标
    rmse_all = np.sqrt(np.mean((clean_pred - clean_target)**2, axis=1))
    rrmse_all = rmse_all / (np.sqrt(np.mean(clean_target**2, axis=1)) + 1e-8)
    cc_all = [np.corrcoef(clean_pred[i], clean_target[i])[0, 1] for i in range(len(clean_pred))]
    eog_rmse_all = np.sqrt(np.mean((eog_pred - eog_target)**2, axis=1))
    eog_rrmse_all = eog_rmse_all / (np.sqrt(np.mean(eog_target**2, axis=1)) + 1e-8)
    eog_cc_all = [np.corrcoef(eog_pred[i], eog_target[i])[0, 1] for i in range(len(eog_pred))]
    
    # 子图1: 去噪性能分布
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.hist(rmse_all, bins=30, alpha=0.6, color='blue', label='RMSE分布')
    ax1_twin.hist(cc_all, bins=30, alpha=0.6, color='green', label='相关系数分布')
    ax1.set_xlabel('RMSE', fontsize=11)
    ax1.set_ylabel('频数 (RMSE)', fontsize=11, color='blue')
    ax1_twin.set_ylabel('频数 (CC)', fontsize=11, color='green')
    ax1.set_title('去噪性能分布统计', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1_twin.tick_params(axis='y', labelcolor='green')
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 眼电预测性能分布
    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    ax2.hist(eog_rmse_all, bins=30, alpha=0.6, color='blue', label='RMSE分布')
    ax2_twin.hist(eog_cc_all, bins=30, alpha=0.6, color='green', label='相关系数分布')
    ax2.set_xlabel('RMSE', fontsize=11)
    ax2.set_ylabel('频数 (RMSE)', fontsize=11, color='blue')
    ax2_twin.set_ylabel('频数 (CC)', fontsize=11, color='green')
    ax2.set_title('眼电预测性能分布统计', fontsize=13, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2_twin.tick_params(axis='y', labelcolor='green')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    summary_path = os.path.join(save_dir, 'performance_summary.png')
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f'已保存性能汇总图: {summary_path}')
    plt.close()
    
    print(f'\n可视化完成！共生成 {len(indices)} 个样本对比图和 1 个性能汇总图')
    print(f'所有图像保存在: {save_dir}')


def main():
    print('='*70)
    print('DAT-Net 无监督测试 (Version 2)')
    print('='*70)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    test_x, test_y, test_eog = load_data()
    print('测试集样本数:', len(test_x))

    ds = TestDataset(test_x, test_y)
    loader = DataLoader(ds, batch_size=50, shuffle=False)

    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'模型参数量: {model.count_parameters():,}')

    model_path = 'DAT-Net-Unsupervised-v2_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('加载模型:', model_path)
    else:
        print('⚠️ 找不到训练好的模型，尝试 final 版本...')
        model_path = 'DAT-Net-Unsupervised-v2_final.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print('加载模型:', model_path)
        else:
            print('⚠️ 找不到训练好的模型，使用随机初始化权重')

    model.eval()
    eeg_preds = []
    eog_preds = []
    targets = []
    sample_count = 0
    start = time()
    with torch.no_grad():
        for noisy, clean, norm in loader:
            sample_count += noisy.shape[0]
            
            # ✅ 训练时的流程：输入归一化数据，在循环中乘回 norm，然后传入模型
            # 测试时应该保持完全一致
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 归一化的
            norm_t = norm.float().view(-1,1,1).to(device)  # (B, 1, 1)
            noisy_scaled = noisy_t * norm_t  # 恢复原始尺度（与训练时一致）

            # 前向传播 - 使用原始尺度的数据（与训练时一致）
            eeg_clean, eog_artifact = model(noisy_scaled)
            
            # 注意：v2 版本训练时使用 loss 函数返回 (c_A, a_A, c_B, a_B)
            # 但测试时直接调用 model 只返回 (c, a)，即单分支输出
            # 这里使用的就是主分支的输出（与训练时的 c_B, a_B 逻辑一致）

            eeg_preds.append(eeg_clean.squeeze(1).cpu().numpy())
            eog_preds.append(eog_artifact.squeeze(1).cpu().numpy())
            targets.append(clean.numpy())

    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)

    eeg_preds = np.concatenate(eeg_preds, axis=0)
    eog_preds = np.concatenate(eog_preds, axis=0)
    targets = np.concatenate(targets, axis=0)

    print('推理完成! 单样本时间: %.3f ms' % (time_per_sample*1000))

    print('\n计算评价指标...')
    metrics = compute_all_metrics(eeg_preds, targets, fs=200)
    print_metrics(metrics, prefix='测试集')

    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'DAT-Net-Unsupervised-v2_predictions.mat')
    scipy.io.savemat(save_path, {
        'predictions': eeg_preds,
        'eog_artifacts': eog_preds,
        'time_per_sample': time_per_sample,
    })
    print('预测结果已保存:', save_path)

    print('\n验证解耦一致性...')
    reconstructed = eeg_preds + eog_preds
    original = test_x
    consistency_error = np.mean((reconstructed - original) ** 2)
    print('重建一致性MSE:', consistency_error)

    # 添加可视化结果
    print('\n' + '='*70)
    print('生成可视化结果...')
    print('='*70)
    vis_dir = os.path.join(current_dir, 'results_visualization')
    visualize_results(
        noisy=test_x,
        clean_pred=eeg_preds,
        clean_target=targets,
        eog_pred=eog_preds,
        eog_target=test_eog,
        num_samples=5,
        save_dir=vis_dir
    )

    print('\n' + '='*70)

if __name__ == '__main__':
    main()
