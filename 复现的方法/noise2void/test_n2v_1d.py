"""
Noise2Void 1D EEG Denoising 测试脚本
评估训练好的模型并保存预测结果
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
sys.path.insert(0, current_dir)

# 导入metrics
try:
    parent_dir = os.path.dirname(current_dir)
    sys.path.insert(0, parent_dir)
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs):
        return {'RRMSE':0,'CC':0,'PRD':0,'SNR':0,'RMSE':0,'MAE':0,'PSNR':0,'SSIM':0}
    def print_metrics(m, prefix=""):
        print(prefix, 'Metrics:', m)

from n2v_model_1d import N2V_UNet1D


class TestDataset(Dataset):
    """测试数据集"""
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
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_input, test_output


def main():
    print('='*70)
    print('Noise2Void 1D EEG Denoising 测试')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    test_x, test_y = load_data()
    print(f'\n测试集样本数: {len(test_x)}')
    print(f'信号长度: {test_x.shape[1]}')
    
    # 创建数据集和加载器
    ds = TestDataset(test_x, test_y)
    loader = DataLoader(ds, batch_size=50, shuffle=False)
    
    # 创建模型（参数要与训练时一致）
    model = N2V_UNet1D(
        in_channels=1,
        n_depth=3,
        n_first=32,
        kernel_size=5,
        batch_norm=True,
        dropout=0.0,
        residual=True
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 加载训练好的权重
    model_path = os.path.join(current_dir, 'N2V_1D_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'加载模型: {model_path}')
    else:
        print('⚠️ 找不到训练好的模型，尝试 final 版本...')
        model_path = os.path.join(current_dir, 'N2V_1D_final.pth')
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f'加载模型: {model_path}')
        else:
            print('⚠️ 找不到训练好的模型，使用随机初始化权重')
    
    # 推理
    model.eval()
    predictions = []
    targets = []
    sample_count = 0
    
    print('\n开始推理...')
    start = time()
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            sample_count += noisy.shape[0]
            
            # 转换为tensor
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (batch, 1, time)
            norm_t = norm.float().view(-1, 1, 1).to(device)
            
            # 归一化
            noisy_t = noisy_t * norm_t
            
            # 模型推理
            pred = model(noisy_t)
            
            # 反归一化
            pred = pred.squeeze(1).cpu().numpy()  # (batch, time)
            
            predictions.append(pred)
            targets.append(clean.numpy())
    
    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)
    
    # 合并结果
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f'推理完成! 单样本时间: {time_per_sample*1000:.3f} ms')
    
    # 计算评价指标
    print('\n计算评价指标...')
    metrics = compute_all_metrics(predictions, targets, fs=200)
    print_metrics(metrics, prefix='测试集')
    
    # 保存结果
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'N2V_1D_predictions.mat')
    
    scipy.io.savemat(save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample,
    })
    
    print(f'\n预测结果已保存: {save_path}')
    print('='*70)


if __name__ == '__main__':
    main()
