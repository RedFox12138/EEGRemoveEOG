"""
EEGIFNet测试脚本 - 输出.mat格式
"""
import torch
import numpy as np
import scipy.io as sio
from torch.utils.data import DataLoader, Dataset
from EEGIFNet_1200 import MA_INet, MA_MNet
import os
from time import time


class EEGDataset(Dataset):
    def __init__(self, noisy_signals, clean_signals):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]
        
        # 归一化
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0
        
        noisy_normalized = noisy / norm_factor
        
        return noisy_normalized, clean, norm_factor


def load_test_data():
    """加载测试数据"""
    raw_eeg = np.load(r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Contaminated.npy')
    clean_eeg = np.load(r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Pure_Data.npy')
    
    # 测试集 (最后10%)
    num_samples = len(raw_eeg)
    test_start = int(num_samples * 0.9)
    
    test_input = raw_eeg[test_start:]
    test_output = clean_eeg[test_start:]
    
    test_dataset = EEGDataset(test_input, test_output)
    test_loader = DataLoader(test_dataset, batch_size=50, shuffle=False)
    
    return test_loader, test_output


def test_model(I_model, M_model, device, test_loader):
    """测试模型"""
    I_model.eval()
    M_model.eval()
    
    all_predictions = []
    sample_count = 0
    
    start_time = time()
    
    with torch.no_grad():
        for x, y, norm_factors in test_loader:
            sample_count += x.size(0)
            
            x = x.float().to(device)
            y = y.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1, 1)
            
            # 计算噪声
            z = x - y
            
            # 模型推理
            y_hat = I_model(x.unsqueeze(1))
            z_hat = M_model(x.unsqueeze(1))
            
            # 反归一化
            y_hat_restored = y_hat.squeeze(1) * norm_factors
            
            all_predictions.append(y_hat_restored.cpu().numpy())
    
    total_time = time() - start_time
    time_per_sample = total_time / sample_count
    
    predictions = np.concatenate(all_predictions, axis=0)
    
    return predictions, time_per_sample


def main():
    print("="*60)
    print("EEGIFNet 测试脚本")
    print("="*60)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    I_model = MA_INet().to(device)
    M_model = MA_MNet().to(device)
    
    I_model_path = 'checkpoint/I_model_best.pth'
    M_model_path = 'checkpoint/M_model_best.pth'
    
    if not os.path.exists(I_model_path) or not os.path.exists(M_model_path):
        print("错误: 找不到模型文件")
        print("请先运行 train.py 训练模型")
        return
    
    print(f"加载模型:")
    print(f"  - {I_model_path}")
    print(f"  - {M_model_path}")
    
    I_model.load_state_dict(torch.load(I_model_path, map_location=device))
    M_model.load_state_dict(torch.load(M_model_path, map_location=device))
    
    # 加载测试数据
    print("加载测试数据...")
    test_loader, test_targets = load_test_data()
    print(f"测试集样本数: {len(test_targets)}")
    
    # 推理
    print("\n开始推理...")
    predictions, time_per_sample = test_model(I_model, M_model, device, test_loader)
    
    print(f"推理完成! 单样本推理时间: {time_per_sample*1000:.3f}ms")
    
    # 保存为.mat格式
    output_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(output_dir, exist_ok=True)
    
    pred_save_path = os.path.join(output_dir, 'EEGIFNet_predictions.mat')
    
    sio.savemat(pred_save_path, {
        'predictions': predictions,
        'method': 'EEGIFNet',
        'inference_time_per_sample': time_per_sample
    })
    
    print(f"\n预测结果已保存为.mat格式: {pred_save_path}")
    print(f"预测结果形状: {predictions.shape}")
    print(f"单样本推理时间: {time_per_sample*1000:.3f}ms")
    print("\n请运行 evaluate_all_methods.py 来计算指标并进行对比")
    print("="*60)


if __name__ == "__main__":
    main()
