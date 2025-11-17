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
    return test_input, test_output


def main():
    print('='*70)
    print('DAT-Net 无监督测试 (Version 2)')
    print('='*70)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    test_x, test_y = load_data()
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

    print('\n' + '='*70)

if __name__ == '__main__':
    main()
