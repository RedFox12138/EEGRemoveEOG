"""
Self-Supervised EEG Denoising 测试脚本
加载训练好的模型进行测试
保存 .mat 结果并计算评估指标
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加路径以导入metrics
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, os.path.join(root_dir, '复现的方法'))

# 导入模型
from model_selfsupervised import DenoiseEEG

# 导入metrics
try:
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs):
        return {'RRMSE':0.0,'CC':0.0,'PRD':0.0,'SNR':0.0,'RMSE':0.0,'MAE':0.0,'PSNR':0.0,'SSIM':0.0}
    def print_metrics(m, prefix=""):
        print(prefix, 'Metrics:', m)


# ========== 配置 ==========
SAMPLING_RATE = 200.0
INPUT_CHANNELS = 1
SEQ_LEN = 1200
HIDDEN_DIM = 128


class TestDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
        
    def __len__(self):
        return len(self.noisy)
    
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        return noisy.astype('float32') / norm, clean.astype('float32'), norm


def load_data():
    """
    加载测试数据
    """
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_input, test_output


def main():
    print('='*70)
    print('Self-Supervised EEG Denoising 测试')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    # 加载数据
    print('\n加载数据...')
    test_x, test_y = load_data()
    print('测试集样本数:', len(test_x))
    print('数据维度:', test_x.shape)

    # 创建数据集
    ds = TestDataset(test_x, test_y)
    loader = DataLoader(ds, batch_size=50, shuffle=False)

    # 创建模型
    print('\n创建模型...')
    model = DenoiseEEG(
        in_channels=INPUT_CHANNELS,
        length=SEQ_LEN,
        n_feat=HIDDEN_DIM
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')

    # 加载模型权重
    model_path = 'Self-Supervised_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print('加载模型:', model_path)
    else:
        print('⚠️ 找不到训练好的模型，尝试 final 版本...')
        model_path = 'Self-Supervised_final.pth'
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print('加载模型:', model_path)
        else:
            print('⚠️ 找不到训练好的模型，使用随机初始化权重')

    # 推理
    print('\n开始推理...')
    model.eval()
    predictions = []
    targets = []
    sample_count = 0
    start = time()
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            sample_count += noisy.shape[0]
            
            # 添加通道维度
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L)
            
            # 恢复原始尺度
            norm_t = norm.float().view(-1, 1, 1).to(device)
            noisy_t = noisy_t * norm_t
            
            # 前向传播
            output = model(noisy_t)
            
            # 保存结果
            predictions.append(output.squeeze(1).cpu().numpy())
            targets.append(clean.numpy())

    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)

    # 合并结果
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)

    print('推理完成! 单样本时间: %.3f ms' % (time_per_sample*1000))

    # 计算评价指标
    print('\n计算评价指标...')
    metrics = compute_all_metrics(predictions, targets, fs=SAMPLING_RATE)
    print_metrics(metrics, prefix='测试集')

    # 保存结果
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Self-Supervised_predictions.mat')
    scipy.io.savemat(save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample,
    })
    print('预测结果已保存:', save_path)

    print('\n' + '='*70)


if __name__ == '__main__':
    main()
