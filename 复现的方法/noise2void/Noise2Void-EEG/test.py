"""
Noise2Void测试脚本

测试流程:
1. 加载测试数据(带噪声和干净数据)
2. 使用训练好的模型直接预测
3. 计算评估指标并保存结果
"""

import torch
import torch.nn as nn
import numpy as np
import scipy.io as sio
import os
import sys

# 添加上级目录到路径以导入metrics
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from metrics_utils import compute_all_metrics

from model import UNet1D_N2V


def load_test_data(data_path):
    """加载测试数据"""
    data = sio.loadmat(data_path)
    test_contaminated = data['Test_Contaminated'].astype(np.float32)
    test_pure = data['Test_Pure'].astype(np.float32)
    
    print(f"Test Contaminated shape: {test_contaminated.shape}")
    print(f"Test Pure shape: {test_pure.shape}")
    
    return test_contaminated, test_pure


def test_model(model_path, data_path, device):
    """测试模型"""
    
    # 加载测试数据
    print("Loading test data...")
    test_contaminated, test_pure = load_test_data(data_path)
    
    # 加载模型
    print("Loading model...")
    model = UNet1D_N2V(in_channels=1, out_channels=1, init_features=32)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Training loss: {checkpoint['train_loss']:.6f}")
    print(f"Validation loss: {checkpoint['val_loss']:.6f}")
    
    # 预测
    print("\nRunning inference...")
    predictions = []
    
    with torch.no_grad():
        for i in range(len(test_contaminated)):
            # 准备输入
            signal = test_contaminated[i]
            signal_tensor = torch.from_numpy(signal[np.newaxis, np.newaxis, :])  # (1, 1, 1200)
            signal_tensor = signal_tensor.to(device)
            
            # 预测 - Noise2Void直接输出去噪信号
            predicted = model(signal_tensor)
            
            # 转回numpy
            predicted_np = predicted.cpu().numpy().squeeze()  # (1200,)
            predictions.append(predicted_np)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i+1}/{len(test_contaminated)} samples")
    
    predictions = np.array(predictions)
    print(f"Predictions shape: {predictions.shape}")
    
    # 计算指标
    print("\nComputing metrics...")
    metrics = compute_all_metrics(predictions, test_pure, test_contaminated)
    
    print("\n" + "="*60)
    print("Test Results:")
    print("="*60)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print("="*60)
    
    return predictions, metrics


def save_results(predictions, metrics, save_dir):
    """保存结果"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存预测结果
    pred_path = os.path.join(save_dir, "Noise2Void_predictions.mat")
    sio.savemat(pred_path, {'predictions': predictions})
    print(f"\n✓ Predictions saved to: {pred_path}")
    
    # 保存指标
    metrics_path = os.path.join(save_dir, "Noise2Void_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write("Noise2Void Test Metrics\n")
        f.write("="*60 + "\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    print(f"✓ Metrics saved to: {metrics_path}")


def main():
    """主函数"""
    # 路径配置
    model_path = "./checkpoints/best_model.pth"
    data_path = r"D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Train_Val_Test_Data.mat"
    results_dir = r"D:\Pycharm_Projects\EOG Remove\复现的方法\results"
    
    # 检查模型文件是否存在
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running train.py")
        return
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 测试模型
    predictions, metrics = test_model(model_path, data_path, device)
    
    # 保存结果
    save_results(predictions, metrics, results_dir)
    
    print("\n✓ Testing completed successfully!")


if __name__ == "__main__":
    main()
