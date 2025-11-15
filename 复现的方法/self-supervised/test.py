"""
Self-Supervised EEG Denoising 测试脚本
使用训练好的Self-Supervised模型在测试集上评估
保存结果为.mat文件，与其他方法统一比较
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
selfsupervised_dir = os.path.join(current_dir, '..', 'Self-Supervised-EEG-Denoising-main')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

sys.path.insert(0, selfsupervised_dir)
sys.path.insert(0, project_root)

from model import DenoiseEEG
from 复现的方法.metrics_utils import compute_all_metrics, print_metrics


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
        
        # 归一化
        max_val = np.max(np.abs(noisy))
        if max_val == 0:
            max_val = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / max_val, dtype=torch.float32)
        clean_tensor = torch.tensor(clean.astype('float32'), dtype=torch.float32)
        
        return noisy_norm, clean_tensor, max_val


def load_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_input, test_output


def main():
    print("="*70)
    print("Self-Supervised EEG Denoising 测试")
    print("="*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')

    # 加载测试数据
    print('\n加载测试数据...')
    test_x, test_y = load_data()
    print(f'测试集样本数: {len(test_x)}')
    print(f'信号长度: {test_x.shape[-1]}')
    
    test_dataset = TestDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)

    # 创建模型
    print('\n创建模型...')
    model = DenoiseEEG(
        in_channels=1,
        length=1200,
        n_feat=128
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {total_params:,}')
    
    # 加载训练好的模型
    model_path = 'SelfSupervised_best.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f'加载模型: {model_path}')
    else:
        print('⚠️  找不到训练好的模型，使用随机初始化权重')

    # 推理
    print('\n开始推理...')
    model.eval()
    
    all_preds = []
    all_targets = []
    sample_count = 0
    start = time()
    
    with torch.no_grad():
        for noisy, clean, norm in test_loader:
            sample_count += noisy.shape[0]
            
            # 前向传播
            noisy = noisy.unsqueeze(1).to(device)  # (B, 1, L)
            output = model(noisy)
            
            # 恢复到原始尺度
            output_scaled = output.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
            clean_np = clean.cpu().numpy()
            
            all_preds.append(output_scaled)
            all_targets.append(clean_np)
            
            if sample_count % 500 == 0:
                print(f'  已处理: {sample_count} 个样本')
    
    elapsed = time() - start
    print(f'推理完成，总用时: {elapsed:.2f}秒')
    print(f'平均每个样本: {elapsed/sample_count*1000:.2f}毫秒')
    
    # 拼接结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f'\n预测结果形状: {all_preds.shape}')
    
    # 计算评价指标
    print('\n' + '='*70)
    print('评价指标')
    print('='*70)
    
    metrics = compute_all_metrics(all_preds, all_targets, fs=200)
    print_metrics(metrics)
    
    # 保存结果为.mat格式（与其他方法统一）
    save_path = '../../../results/SelfSupervised_predictions.mat'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    scipy.io.savemat(save_path, {
        'predictions': all_preds,
        'targets': all_targets,
        'metrics': {
            'CC': metrics['CC'],
            'RMSE': metrics['RMSE'],
            'SNR': metrics['SNR_dB'],
            'RRMSE_time': metrics['RRMSE_time'],
            'RRMSE_freq': metrics['RRMSE_freq']
        }
    })
    
    print(f'\n结果已保存到: {save_path}')
    print('='*70)


if __name__ == '__main__':
    main()
