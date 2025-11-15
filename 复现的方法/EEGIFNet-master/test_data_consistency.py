"""
测试数据加载的一致性 - 对比ASNet和EEGIFNet的数据处理
"""
import sys
import numpy as np
import scipy.io
sys.path.append(r'D:\Pycharm_Projects\EOG Remove\复现的方法')

# 测试数据集类
from torch.utils.data import Dataset

class ASNetDataset(Dataset):
    """ASNet的数据集类"""
    def __init__(self, noisy_signals, clean_signals):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]
        
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0
        
        noisy_normalized = noisy / norm_factor
        # clean不归一化
        return noisy_normalized, clean, norm_factor


class EEGIFNetDataset(Dataset):
    """修正后的EEGIFNet数据集类"""
    def __init__(self, noisy_signals, clean_signals):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]
        
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0
        
        noisy_normalized = noisy / norm_factor
        # clean不归一化（与ASNet一致）
        return noisy_normalized, clean, norm_factor


print("="*80)
print("测试数据加载一致性")
print("="*80)

# 加载测试数据
data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']

print(f"\n原始数据形状:")
print(f"  Contaminated: {test_input.shape}")
print(f"  Pure: {test_output.shape}")

# 创建数据集
asnet_dataset = ASNetDataset(test_input, test_output)
eegifnet_dataset = EEGIFNetDataset(test_input, test_output)

# 测试第一个样本
idx = 0
asnet_noisy, asnet_clean, asnet_norm = asnet_dataset[idx]
eegif_noisy, eegif_clean, eegif_norm = eegifnet_dataset[idx]

print(f"\n测试样本 {idx}:")
print(f"\nASNet数据集:")
print(f"  noisy形状: {asnet_noisy.shape}, 范围: [{asnet_noisy.min():.4f}, {asnet_noisy.max():.4f}]")
print(f"  clean形状: {asnet_clean.shape}, 范围: [{asnet_clean.min():.4f}, {asnet_clean.max():.4f}]")
print(f"  norm_factor: {asnet_norm:.4f}")

print(f"\nEEGIFNet数据集:")
print(f"  noisy形状: {eegif_noisy.shape}, 范围: [{eegif_noisy.min():.4f}, {eegif_noisy.max():.4f}]")
print(f"  clean形状: {eegif_clean.shape}, 范围: [{eegif_clean.min():.4f}, {eegif_clean.max():.4f}]")
print(f"  norm_factor: {eegif_norm:.4f}")

# 检查是否一致
print(f"\n一致性检查:")
print(f"  noisy一致: {np.allclose(asnet_noisy, eegif_noisy)}")
print(f"  clean一致: {np.allclose(asnet_clean, eegif_clean)}")
print(f"  norm_factor一致: {np.allclose(asnet_norm, eegif_norm)}")

# 验证反归一化
print(f"\n反归一化验证:")
print(f"  ASNet: noisy * norm_factor 是否等于原始? {np.allclose(asnet_noisy * asnet_norm, test_input[idx])}")
print(f"  EEGIFNet: noisy * norm_factor 是否等于原始? {np.allclose(eegif_noisy * eegif_norm, test_input[idx])}")
print(f"  ASNet: clean 是否等于原始? {np.allclose(asnet_clean, test_output[idx])}")
print(f"  EEGIFNet: clean 是否等于原始? {np.allclose(eegif_clean, test_output[idx])}")

print("\n" + "="*80)
print("✓ 数据处理一致性测试完成")
print("="*80)
