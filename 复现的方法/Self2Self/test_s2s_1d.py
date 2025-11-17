"""
Self2Self 1D EEG Denoising 测试脚本
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

from s2s_model_1d import Self2Self_UNet1D


# Self2Self推理参数
N_PREDICTIONS = 100  # 多次预测平均的次数


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
        
        # 归一化到[0, 1]
        norm_min = np.min(noisy)
        norm_max = np.max(noisy)
        norm_range = norm_max - norm_min
        
        if norm_range == 0:
            norm_range = 1.0
        
        noisy_norm = (noisy - norm_min) / norm_range
        
        return (torch.from_numpy(noisy_norm).float(), 
               torch.from_numpy(clean).float(),
               norm_min, norm_range)


def load_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_input, test_output


def main():
    print('='*70)
    print('Self2Self 1D EEG Denoising 测试')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 加载数据
    test_x, test_y = load_data()
    print(f'\n测试集样本数: {len(test_x)}')
    print(f'信号长度: {test_x.shape[1]}')
    
    # 创建数据集和加载器
    ds = TestDataset(test_x, test_y)
    loader = DataLoader(ds, batch_size=10, shuffle=False)  # 较小batch size因为要做多次预测
    
    # 创建模型（参数要与训练时一致）
    model = Self2Self_UNet1D(
        in_channels=1,
        base_channels=48,
        n_depth=5,
        dropout=0.3
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 加载训练好的权重
    model_path = os.path.join(current_dir, 'Self2Self_1D_best.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'加载模型: {model_path}')
    else:
        print('⚠️ 找不到训练好的模型，尝试 final 版本...')
        model_path = os.path.join(current_dir, 'Self2Self_1D_final.pth')
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
    
    print(f'\n开始推理（每个样本{N_PREDICTIONS}次预测平均）...')
    print('这可能需要一些时间...')
    start = time()
    
    with torch.no_grad():
        for noisy_norm, clean, norm_min, norm_range in loader:
            sample_count += noisy_norm.shape[0]
            
            # 转换为tensor
            noisy_norm = noisy_norm.unsqueeze(1).to(device)  # (batch, 1, time)
            
            # Self2Self推理：多次预测并平均
            pred = model.predict_average(noisy_norm, n_predictions=N_PREDICTIONS)
            
            # 反归一化
            pred = pred.squeeze(1).cpu().numpy()  # (batch, time)
            pred = pred * norm_range.numpy()[:, None] + norm_min.numpy()[:, None]
            
            predictions.append(pred)
            targets.append(clean.numpy())
            
            # 进度显示
            if sample_count % 100 == 0:
                elapsed = time() - start
                print(f'  已处理 {sample_count}/{len(test_x)} 样本, '
                      f'用时 {int(elapsed)}s')
    
    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)
    
    # 合并结果
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    
    print(f'\n推理完成!')
    print(f'  总时间: {int(total_time//60)}min {int(total_time%60)}s')
    print(f'  单样本时间: {time_per_sample*1000:.3f} ms')
    print(f'  (包含{N_PREDICTIONS}次预测平均)')
    
    # 计算评价指标
    print('\n计算评价指标...')
    metrics = compute_all_metrics(predictions, targets, fs=200)
    print_metrics(metrics, prefix='测试集')
    
    # 保存结果
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, 'Self2Self_1D_predictions.mat')
    
    scipy.io.savemat(save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample,
        'n_predictions': N_PREDICTIONS,
    })
    
    print(f'\n预测结果已保存: {save_path}')
    print('='*70)


if __name__ == '__main__':
    main()
