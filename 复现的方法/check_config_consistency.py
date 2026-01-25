"""
快速检查Python和MATLAB配置是否一致
用于诊断compute_all_metrics.m和test_unsupervised.py结果不一致的问题
"""
import os
import sys
import scipy.io

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 导入Python配置
sys.path.insert(0, os.path.join(current_dir, '我自己的方法', 'DAT-Net-Unsupervised-v2'))
from config import *
from dataset_config import get_dataset_config

print("="*80)
print("Python配置检查")
print("="*80)

print(f"\n数据集名称: {DATASET_NAME}")
print(f"采样率: {SAMPLING_RATE} Hz")
print(f"窗口大小: {WINDOW_SIZE}")
print(f"数据键名 (污染): {DATA_KEY}")
print(f"数据键名 (纯净): {PURE_KEY}")

print(f"\n测试集SNR级别: {TEST_SNR_LEVELS}")

if TEST_SNR_LEVELS:
    print(f"\n多SNR测试集配置:")
    for snr_db in TEST_SNR_LEVELS:
        print(f"\n  SNR = {snr_db} dB:")
        paths = TEST_SNR_PATHS[snr_db]
        print(f"    污染数据: {paths['contaminated']}")
        print(f"      存在: {os.path.exists(paths['contaminated'])}")
        if os.path.exists(paths['contaminated']):
            data = scipy.io.loadmat(paths['contaminated'])
            if DATA_KEY in data:
                print(f"      维度: {data[DATA_KEY].shape}")
            else:
                print(f"      ⚠ 键名 '{DATA_KEY}' 不存在，文件中的键: {[k for k in data.keys() if not k.startswith('__')]}")
        
        print(f"    纯净数据: {paths['pure']}")
        print(f"      存在: {os.path.exists(paths['pure'])}")
        if os.path.exists(paths['pure']):
            data = scipy.io.loadmat(paths['pure'])
            if PURE_KEY in data:
                print(f"      维度: {data[PURE_KEY].shape}")
                print(f"      数据范围: [{data[PURE_KEY].min():.3f}, {data[PURE_KEY].max():.3f}]")
            else:
                print(f"      ⚠ 键名 '{PURE_KEY}' 不存在，文件中的键: {[k for k in data.keys() if not k.startswith('__')]}")

print("\n"+"="*80)
print("预测结果保存路径配置")
print("="*80)
print(f"RESULTS_DIR: {RESULTS_DIR}")
print(f"基础文件名: {os.path.basename(PREDICTION_SAVE_PATH)}")
print(f"\n实际保存的文件名格式（以SNR=0dB为例）:")
pred_save_path = PREDICTION_SAVE_PATH.replace('.mat', '_SNR0dB.mat')
print(f"  {os.path.basename(pred_save_path)}")

print("\n"+"="*80)
print("对比MATLAB配置")
print("="*80)
print("MATLAB脚本应该使用以下配置:")
print(f"  getDatasetConfig('fully_simulated')  # 数据集名称")
print(f"  config.fs = 250  # 采样率")
print(f"  config.pureKey = 'pureEEG'  # 纯净数据键名")
print(f"  config.testSnrLevels = [4, 0, -6, -10, -14, -18, -20, -22]")
print("\n如果MATLAB结果不一致，请检查:")
print("  1. getDatasetConfig.m 中的 fully_simulated 配置是否与上述一致")
print("  2. compute_all_metrics.m 第22行是否使用 getDatasetConfig('fully_simulated')")
print("  3. MATLAB读取的测试集文件是否与Python相同")
print("="*80)
