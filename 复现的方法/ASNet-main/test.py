"""
ASNet测试脚本
加载最佳模型,在测试集上进行推理,保存预测结果为.mat格式
"""

import scipy.io
import torch
import torch.utils.data as Data
import os
import numpy as np
from time import time
from torch.utils.data import Dataset
from ASNet import ASNet

BATCH_SIZE = 50

class EEGDataset(Dataset):
    def __init__(self, noisy_signals, clean_signals, is_train=False):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        # 归一化 noisy 信号
        # 找到绝对值的最大值作为归一化因子
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0 # 避免除以零

        noisy_normalized = noisy / norm_factor

        # 不需要在这里添加通道维度，ASNet的forward会自动处理
        # noisy_normalized = noisy_normalized[np.newaxis, :]
        # clean = clean[np.newaxis, :]

        return noisy_normalized, clean, norm_factor

def load_test_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    test_input = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_output = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    
    test_dataset = EEGDataset(test_input, test_output, is_train=False)
    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    
    return test_loader, test_output


def test_model(model, device, test_loader):
    """
    在测试集上进行推理
    
    Returns:
        predictions: 预测结果列表
        targets: 真实标签列表
        time_per_sample: 单样本推理时间
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    sample_count = 0
    
    start_time = time()
    
    with torch.no_grad():
        for batch_idx, (test_input, test_output, norm_factors) in enumerate(test_loader):
            sample_count += test_input.size(0)
            
            test_input = test_input.float().to(device)
            test_output = test_output.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1)
            
            # 推理
            output = model(test_input)
            output_restored = output * norm_factors
            
            # 收集结果
            all_predictions.append(output_restored.cpu().numpy())
            all_targets.append(test_output.cpu().numpy())
    
    total_time = time() - start_time
    time_per_sample = total_time / sample_count
    
    # 合并所有batch
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    return all_predictions, all_targets, time_per_sample


def main():
    print("="*60)
    print("ASNet 测试脚本")
    print("="*60)
    
    # 设置设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载模型
    model = ASNet()
    model_path = 'ASNet_best.pkl'
    
    if not os.path.exists(model_path):
        print(f"错误: 找不到模型文件 {model_path}")
        print("请先运行 train.py 训练模型")
        return
    
    print(f"加载模型: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    
    # 加载测试数据
    print("加载测试数据...")
    test_loader, test_targets = load_test_data()
    print(f"测试集样本数: {len(test_targets)}")
    
    # 进行推理
    print("\n开始推理...")
    predictions, targets, time_per_sample = test_model(model, device, test_loader)
    
    print(f"推理完成! 单样本推理时间: {time_per_sample*1000:.3f}ms")
    
    # 保存预测结果为.mat格式
    output_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(output_dir, exist_ok=True)
    
    pred_save_path = os.path.join(output_dir, 'ASNet_predictions.mat')
    
    scipy.io.savemat(pred_save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample
    })
    
    print(f"\n预测结果已保存为.mat格式: {pred_save_path}")
    print(f"预测结果形状: {predictions.shape}")
    print(f"单样本推理时间: {time_per_sample*1000:.3f}ms")
    print("\n✓ 完成！请运行统一指标计算脚本来评估所有方法。")
    print("="*60)




if __name__ == "__main__":
    main()



