"""
DAT-Net-Unsupervised-v2 测试脚本
用于测试微调后的模型，支持切换不同的模型文件
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
import argparse

# 添加路径以导入相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from model import DATNet

# 导入配置
from config import *

# 导入metrics
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass


# ========== 配置 ==========
BATCH_SIZE = 256
SAMPLING_RATE = 250


class SupervisedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # Max-Abs归一化（与原始版本一致）
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = (noisy / norm).astype('float32')
        clean_norm = (clean / norm).astype('float32')
        
        return torch.tensor(noisy_norm), torch.tensor(clean_norm), norm


def load_test_data_by_snr(snr_db):
    """
    根据SNR加载测试数据
    """
    contaminated_path = TEST_SNR_PATHS[snr_db]['contaminated']
    pure_path = TEST_SNR_PATHS[snr_db]['pure']
    
    test_x = scipy.io.loadmat(contaminated_path)[DATA_KEY]
    test_y = scipy.io.loadmat(pure_path)[PURE_KEY]
    return test_x, test_y


def get_test_data():
    """加载测试数据（兼容单一测试集和多SNR）"""
    if TEST_CONTAMINATED_PATH is not None:
        # 单一测试集模式
        test_x = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]
        test_y = scipy.io.loadmat(TEST_PURE_PATH)[DATA_KEY]
        return test_x, test_y
    else:
        # 多SNR模式，返回None（由调用者循环处理）
        return None, None


def test_model(model_path, output_suffix, device):
    """
    在测试集上评估模型并保存结果
    
    Args:
        model_path: 模型权重文件路径
        output_suffix: 输出文件后缀名
        device: 计算设备
    """
    print('\n' + '='*70)
    print(f'测试模型: {model_path}')
    print('='*70)
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f'❌ 模型文件不存在: {model_path}')
        return
    
    # 创建模型并加载权重
    model = DATNet(in_channels=1, base_channels=32).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print(f'✓ 成功加载模型: {model_path}')
    
    # 检查测试模式（多SNR或单一）
    if TEST_SNR_LEVELS:
        print(f'\n多SNR测试模式，SNR级别: {TEST_SNR_LEVELS}')
        snr_levels = TEST_SNR_LEVELS
    else:
        print('\n单一测试集模式')
        snr_levels = [None]
    
    # 对每个SNR级别进行测试
    for snr_db in snr_levels:
        if snr_db is not None:
            print(f'\n{"="*70}')
            print(f'测试 SNR = {snr_db} dB')
            print('='*70)
            test_x, test_y = load_test_data_by_snr(snr_db)
            save_suffix = f'_SNR{snr_db}dB'
        else:
            test_x, test_y = get_test_data()
            save_suffix = ''
        
        print(f'测试集样本数: {len(test_x)}')
        
        test_dataset = SupervisedDataset(test_x, test_y)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        
        # 评估
        model.eval()
        all_preds = []
        all_eog_preds = []
        all_targets = []
        sample_count = 0
        start = time()
        
        with torch.no_grad():
            for noisy, clean, norm in test_loader:
                sample_count += noisy.shape[0]
                
                noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L)
                clean_t = clean.float().unsqueeze(1).to(device)  # (B, 1, L)
                norm_t = norm.float().to(device).view(-1, 1, 1)  # (B, 1, 1)
                
                # 恢复原始幅度: max-abs反归一化
                noisy_scaled = noisy_t * norm_t
                clean_scaled = clean_t * norm_t
                
                # 前向传播 - 使用原始尺度的数据（与训练时一致）
                eeg_clean, eog_artifact = model(noisy_scaled)
                
                # 模型输出已经是原始尺度
                all_preds.append(eeg_clean.squeeze(1).cpu().numpy())
                all_eog_preds.append(eog_artifact.squeeze(1).cpu().numpy())
                all_targets.append(clean_scaled.squeeze(1).cpu().numpy())
        
        total_time = time() - start
        time_per_sample = total_time / max(1, sample_count)
        
        # 合并结果
        all_preds = np.concatenate(all_preds, axis=0)
        all_eog_preds = np.concatenate(all_eog_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        print(f'推理完成! 单样本时间: {time_per_sample*1000:.3f} ms')
        
        # 计算评价指标
        print('\n计算评价指标...')
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
        print_metrics(metrics, prefix='测试集')
        
        # 验证解耦一致性
        print('\n验证解耦一致性...')
        reconstructed = all_preds + all_eog_preds
        original = test_x
        consistency_error = np.mean((reconstructed - original) ** 2)
        print(f'重建一致性MSE: {consistency_error:.6f}')
        
        # 保存结果（带SNR标识）
        out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
        os.makedirs(out_dir, exist_ok=True)
        save_path = os.path.join(out_dir, f'DAT-Net-Unsupervised-improve-20%数据_{output_suffix}_predictions{save_suffix}.mat')
        scipy.io.savemat(save_path, {
            'predictions': all_preds,
            'eog_artifacts': all_eog_preds,
            'time_per_sample': time_per_sample,
        })
        print(f'\n预测结果已保存: {save_path}')
    
    print('\n' + '='*70)
    print('全部SNR测试完成！')
    print('='*70)


def main():
    parser = argparse.ArgumentParser(description='DAT-Net-Unsupervised-v2 测试脚本')
    parser.add_argument('--model', type=str, default='DAT-Net-Unsupervised-v2_finetuned_best_20%鏁版嵁.pth',
                        help='模型权重文件路径')
    parser.add_argument('--suffix', type=str, default='finetuned_best',
                        help='输出文件后缀名')
    
    args = parser.parse_args()
    
    print('='*70)
    print('DAT-Net-Unsupervised-v2 测试脚本')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)
    
    # 测试模型
    test_model(args.model, args.suffix, device)


if __name__ == '__main__':
    main()
