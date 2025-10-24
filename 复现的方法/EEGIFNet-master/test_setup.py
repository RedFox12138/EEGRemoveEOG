"""
快速测试脚本 - 验证网络和数据加载是否正常
"""
import torch
import numpy as np
import sys
import os

# 添加路径
sys.path.insert(0, os.path.dirname(__file__))

from EEGIFNet_1200 import MA_INet, MA_MNet, weights_init

def test_network():
    print("="*80)
    print("测试EEGIFNet网络 (1200时间点)")
    print("="*80)
    
    # 测试不同的batch size
    batch_sizes = [1, 4, 100, 256]
    time_points = 1200
    
    for batch_size in batch_sizes:
        print(f"\n测试 batch_size={batch_size}")
        
        # 创建测试数据
        x = torch.randn(batch_size, 1, time_points)
        norm_factors = torch.randn(batch_size, 1, 1).abs() + 0.1
        
        # 初始化网络
        inet = MA_INet(input_length=time_points)
        mnet = MA_MNet()
        
        # 前向传播
        try:
            e_out, n_out = inet(x)
            final_out = mnet(x, e_out, n_out)
            
            print(f"  ✓ 输入: {x.shape}")
            print(f"  ✓ INet EEG输出: {e_out.shape}")
            print(f"  ✓ INet Noise输出: {n_out.shape}")
            print(f"  ✓ MNet最终输出: {final_out.shape}")
            
            # 测试反归一化
            norm_factors_2d = norm_factors.squeeze(-1)
            out_denorm = final_out * norm_factors_2d
            print(f"  ✓ 反归一化输出: {out_denorm.shape}")
            
            assert e_out.shape == (batch_size, time_points), f"EEG输出形状错误"
            assert n_out.shape == (batch_size, time_points), f"Noise输出形状错误"
            assert final_out.shape == (batch_size, time_points), f"最终输出形状错误"
            assert out_denorm.shape == (batch_size, time_points), f"反归一化形状错误"
            
        except Exception as e:
            print(f"  ✗ 错误: {e}")
            return False
    
    print("\n" + "="*80)
    print("✓ 所有测试通过!")
    print("="*80)
    return True

def test_data_loading():
    print("\n" + "="*80)
    print("测试数据加载")
    print("="*80)
    
    data_path = r"D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据"
    
    try:
        contaminated = np.load(os.path.join(data_path, 'Contaminated.npy'), allow_pickle=True)
        pure = np.load(os.path.join(data_path, 'Pure_Data.npy'), allow_pickle=True)
        
        print(f"✓ Contaminated shape: {contaminated.shape}")
        print(f"✓ Pure shape: {pure.shape}")
        print(f"✓ Time points: {contaminated.shape[1]}")
        print(f"✓ Samples: {contaminated.shape[0]}")
        
        # 测试标准化
        sample = contaminated[0]
        norm_factor = np.max(np.abs(sample))
        normalized = sample / norm_factor
        denormalized = normalized * norm_factor
        
        print(f"✓ 原始范围: [{sample.min():.2f}, {sample.max():.2f}]")
        print(f"✓ 归一化范围: [{normalized.min():.2f}, {normalized.max():.2f}]")
        print(f"✓ 反归一化范围: [{denormalized.min():.2f}, {denormalized.max():.2f}]")
        print(f"✓ 归一化因子: {norm_factor:.2f}")
        
        assert np.allclose(sample, denormalized), "反归一化失败"
        
        print("\n✓ 数据加载和标准化测试通过!")
        return True
        
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False

if __name__ == "__main__":
    print("\n" + "="*80)
    print("EEGIFNet快速验证测试")
    print("="*80)
    
    # 测试网络
    network_ok = test_network()
    
    # 测试数据加载
    data_ok = test_data_loading()
    
    print("\n" + "="*80)
    if network_ok and data_ok:
        print("✓✓✓ 所有测试通过！可以开始训练 ✓✓✓")
        print("\n运行以下命令开始训练:")
        print("  python train_new.py --epochs 5")
    else:
        print("✗✗✗ 测试失败，请检查错误信息 ✗✗✗")
    print("="*80)
