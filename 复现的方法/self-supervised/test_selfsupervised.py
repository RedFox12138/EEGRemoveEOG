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
parent_dir = os.path.dirname(current_dir) # 复现的方法
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# 导入模型
from model_selfsupervised import DenoiseEEG

# 导入metrics
from metrics_utils import compute_all_metrics, print_metrics

# 导入数据配置
from data_config import *

# ========== 数据集选择 ==========
# 数据集由 data_config.py 中的 DATASET_NAME 变量控制
# 可选值: 'semi_simulated' 或 'fully_simulated'
# 请修改 data_config.py 中的 DATASET_NAME 来切换数据集
# ================================

# ========== 配置 ==========
INPUT_CHANNELS = 1
SEQ_LEN = WINDOW_SIZE  # 使用配置中的窗口大小
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


def load_test_data_by_snr(snr_db):
    """
    根据SNR加载测试数据
    """
    test_snr_paths = dataset_config['test_snr_paths']
    contaminated_path = test_snr_paths[snr_db]['contaminated']
    pure_path = test_snr_paths[snr_db]['pure']
    
    test_input = scipy.io.loadmat(contaminated_path)[DATA_KEY]
    test_output = scipy.io.loadmat(pure_path)[PURE_KEY]
    return test_input, test_output


def main():
    print('='*70)
    print('Self-Supervised EEG Denoising 测试')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)
    
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
    
    # 获取SNR级别
    snr_levels = dataset_config['test_snr_levels']
    print(f"\n多SNR测试模式，SNR级别: {snr_levels}")
    
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    
    # 对每个SNR级别进行测试
    for snr_db in snr_levels:
        print(f"\n========== 测试 SNR = {snr_db} dB ==========")
        
        # 加载数据
        test_x, test_y = load_test_data_by_snr(snr_db)
        print('测试集样本数:', len(test_x))
        print('数据维度:', test_x.shape)

        # 创建数据集
        ds = TestDataset(test_x, test_y)
        loader = DataLoader(ds, batch_size=50, shuffle=False)

        # 推理
        print('开始推理...')
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
                
                # 前向传播 (输入归一化数据)
                output = model(noisy_t)
                
                # 恢复原始尺度
                norm_t = norm.float().view(-1, 1, 1).to(device)
                output_denorm = output * norm_t
                
                # 保存结果
                predictions.append(output_denorm.squeeze(1).cpu().numpy())
                targets.append(clean.numpy()) # clean 已经是原始尺度

        total_time = time() - start
        time_per_sample = total_time / max(1, sample_count)

        # 合并结果
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)

        print('推理完成! 单样本时间: %.3f ms' % (time_per_sample*1000))

        # 计算评价指标
        print('计算评价指标...')
        metrics = compute_all_metrics(predictions, targets, fs=SAMPLING_RATE)
        print_metrics(metrics, prefix='测试集')

        # 保存带SNR标识的结果
        save_path = os.path.join(out_dir, f'SelfSupervised_predictions_SNR{snr_db}dB.mat')
        scipy.io.savemat(save_path, {
            'predictions': predictions,
            'time_per_sample': time_per_sample,
        })
        print('预测结果已保存:', save_path)
    
    print('\n全部SNR测试完成！')

    print('\n' + '='*70)


if __name__ == '__main__':
    main()