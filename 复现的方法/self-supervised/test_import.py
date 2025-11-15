"""
测试导入和模型创建
"""
import os
import sys

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
selfsupervised_dir = os.path.join(current_dir, '..', 'Self-Supervised-EEG-Denoising-main')
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))

sys.path.insert(0, selfsupervised_dir)
sys.path.insert(0, project_root)

print("测试导入...")
print(f"Self-Supervised目录: {selfsupervised_dir}")
print(f"项目根目录: {project_root}")
print()

try:
    from model import DenoiseEEG
    print("✓ 成功导入DenoiseEEG模型")
except Exception as e:
    print(f"✗ 导入DenoiseEEG失败: {e}")

try:
    from 复现的方法.metrics_utils import compute_all_metrics, print_metrics
    print("✓ 成功导入metrics_utils")
except Exception as e:
    print(f"✗ 导入metrics_utils失败: {e}")

try:
    import torch
    model = DenoiseEEG(in_channels=1, length=512, n_feat=128)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ 成功创建模型，参数量: {total_params:,}")
except Exception as e:
    print(f"✗ 创建模型失败: {e}")

print("\n测试完成！")
