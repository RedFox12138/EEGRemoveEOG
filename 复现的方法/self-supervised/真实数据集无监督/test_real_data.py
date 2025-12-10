"""
Self-Supervised 真实数据集测试脚本
使用训练好的模型对验证集进行测试
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from time import time
import shutil

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from model_selfsupervised import DenoiseEEG
from real_data_config import *


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


def load_data():
    """加载真实数据并划分出验证集"""
    print('\n正在加载真实数据集...')
    print(f'数据路径: {REAL_DATA_PATH}')
    
    # 加载数据
    data_dict = scipy.io.loadmat(REAL_DATA_PATH)
    
    # 尝试不同的可能的 key
    possible_keys = [DATA_KEY, 'data', 'eeg_data', 'X', 'signals']
    data = None
    
    for key in possible_keys:
        if key in data_dict:
            data = data_dict[key]
            print(f'  ✓ 使用 key: "{key}"')
            break
    
    if data is None:
        available_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        raise ValueError(f'无法找到数据！可用的 keys: {available_keys}')
    
    print(f'  数据形状: {data.shape}')
    
    n_samples = data.shape[0]
    
    # 使用相同的随机种子划分验证集
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * TRAIN_RATIO)
    val_indices = indices[train_size:]
    
    test_x = data[val_indices]
    
    print(f'  测试集样本数: {test_x.shape[0]} ({VAL_RATIO*100:.0f}%)')
    
    return test_x


def test_model(model, device, test_loader):
    """
    在验证集上进行测试
    
    Returns:
        predictions: 预测的去噪结果 (N, L)
        time_per_sample: 单样本推理时间
    """
    model.eval()
    
    all_predictions = []
    sample_count = 0
    
    start_time = time()
    
    with torch.no_grad():
        for noisy, norm in test_loader:
            noisy = noisy.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            noisy_scaled = noisy * norm
            
            # 推理（不使用掩码）
            pred = model(noisy_scaled)
            
            # 收集结果
            all_predictions.append(pred.squeeze(1).cpu().numpy())
            sample_count += noisy.size(0)
    
    total_time = time() - start_time
    time_per_sample = total_time / sample_count
    
    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    return all_predictions, time_per_sample


def main():
    print("=" * 80)
    print("Self-Supervised 真实数据集测试")
    print("=" * 80)
    
    # 设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n使用设备: {device}")
    
    # 加载测试数据
    test_x = load_data()
    
    # 创建数据集
    test_dataset = RealDataset(test_x)
    test_loader = DataLoader(
        test_dataset,
        batch_size=50,
        shuffle=False
    )
    
    # 创建模型
    print('\n创建模型...')
    model = DenoiseEEG(
        in_channels=INPUT_CHANNELS,
        length=int(WINDOW_SIZE),
        n_feat=HIDDEN_DIM
    ).to(device)
    
    # 加载预训练模型
    print(f'\n加载预训练模型...')
    
    if os.path.exists(MODEL_SAVE_PATH):
        model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
        print(f'  ✓ 加载模型: {MODEL_SAVE_PATH}')
    elif os.path.exists(FINAL_MODEL_PATH):
        model.load_state_dict(torch.load(FINAL_MODEL_PATH, map_location=device))
        print(f'  ✓ 加载模型: {FINAL_MODEL_PATH}')
    else:
        print(f'  ✗ 找不到训练好的模型!')
        print(f'  请先运行 train_real_data.py 进行训练')
        return
    
    # 测试
    print('\n' + '='*80)
    print('开始测试...')
    print('='*80)
    
    predictions, time_per_sample = test_model(model, device, test_loader)
    
    print(f'\n测试完成！')
    print(f'  预测结果形状: {predictions.shape}')
    print(f'  平均推理时间: {time_per_sample*1000:.2f} ms/样本')
    print(f'  总推理时间: {time_per_sample*len(predictions):.2f} 秒')
    
    # 计算伪影（原始信号 - 去噪信号）
    artifacts = test_x - predictions
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = predictions + artifacts
    consistency_error = np.mean((reconstructed - test_x) ** 2)
    print(f'  重建一致性 MSE: {consistency_error:.6f}')
    
    # 统计信息
    print('\n统计信息:')
    original_std = np.std(test_x)
    cleaned_std = np.std(predictions)
    artifact_std = np.std(artifacts)
    print(f'  原始信号标准差: {original_std:.4f}')
    print(f'  去噪信号标准差: {cleaned_std:.4f}')
    print(f'  伪影标准差: {artifact_std:.4f}')
    
    power_reduction = (np.mean(test_x ** 2) - np.mean(predictions ** 2)) / np.mean(test_x ** 2)
    print(f'  平均功率降低: {power_reduction * 100:.2f}%')
    
    # 保存结果到本地目录
    print('\n保存结果...')
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    scipy.io.savemat(PREDICTION_SAVE_PATH, {
        'cleaned_eeg': predictions,
        'extracted_eog': artifacts,
        'original': test_x,
        'time_per_sample': time_per_sample,
        'consistency_error': consistency_error,
        'power_reduction': power_reduction,
        'sampling_rate': SAMPLING_RATE,
        'window_size': WINDOW_SIZE,
    })
    print(f'  ✓ 本地结果已保存: {PREDICTION_SAVE_PATH}')
    
    # 同时保存到总结果目录
    os.makedirs(FINAL_RESULTS_DIR, exist_ok=True)
    shutil.copy(PREDICTION_SAVE_PATH, FINAL_PREDICTION_PATH)
    print(f'  ✓ 总结果已保存: {FINAL_PREDICTION_PATH}')
    print(f'    - 去噪 EEG 形状: {predictions.shape}')
    print(f'    - 提取 EOG 形状: {artifacts.shape}')
    
    print('\n' + '='*80)
    print('测试完成！')
    print('='*80)


if __name__ == '__main__':
    main()
