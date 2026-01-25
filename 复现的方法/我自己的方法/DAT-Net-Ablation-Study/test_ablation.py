"""
自动化测试脚本 - 消融实验
测试训练好的模型并生成.mat结果文件
"""
import os
import sys
import scipy.io
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import json
from time import time

# 导入配置和工具
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ablation_config import *
from model_wrapper import create_model

# 导入原始config
v2_dir = os.path.join(os.path.dirname(current_dir), 'DAT-Net-Unsupervised-v2')
sys.path.insert(0, v2_dir)
import config as base_config

# 导入metrics（可选）
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): 
        return {}
    def print_metrics(m, prefix=""): 
        pass


class TestDataset(Dataset):
    """测试数据集"""
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean
    
    def __len__(self):
        return len(self.noisy)
    
    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # Max-Abs归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = (noisy / norm).astype('float32')
        
        return noisy_norm, clean.astype('float32'), norm


def load_test_data_by_snr(snr_db):
    """根据SNR加载测试数据"""
    contaminated_path = base_config.TEST_SNR_PATHS[snr_db]['contaminated']
    pure_path = base_config.TEST_SNR_PATHS[snr_db]['pure']
    
    test_input = scipy.io.loadmat(contaminated_path)[base_config.DATA_KEY]
    test_output = scipy.io.loadmat(pure_path)[base_config.PURE_KEY]
    test_eog = test_input - test_output
    return test_input, test_output, test_eog


def test_single_experiment(experiment_name, ablation_config, device, snr_db=None):
    """
    测试单个消融实验
    
    Args:
        experiment_name: 实验名称
        ablation_config: 消融配置
        device: 测试设备
        snr_db: 信噪比（dB），如果为None则使用单一测试集
    
    Returns:
        dict: 测试结果（包含去噪数据、指标等）
    """
    print(f"\n{'='*80}")
    print(f"测试实验: {experiment_name}")
    if snr_db is not None:
        print(f"测试SNR: {snr_db} dB")
    print(f"{'='*80}")
    
    # 加载模型 - 始终使用Loss最佳模型（用于消融实验评估）
    dataset_name = base_config.DATASET_NAME
    checkpoint_path = os.path.join('checkpoints', f'ablation_{experiment_name}_{dataset_name}_best_loss.pth')
    
    # 检查模型文件是否存在且不为空
    if not os.path.exists(checkpoint_path) or os.path.getsize(checkpoint_path) == 0:
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) == 0:
            print(f"错误: Loss最佳模型文件为空: {checkpoint_path}")
        else:
            print(f"错误: Loss最佳模型文件不存在: {checkpoint_path}")
        print("       将跳过此实验的测试。")
        return None
    
    print(f"✓ 使用Loss最佳模型（消融实验标准）: {checkpoint_path}")
    
    model = create_model(ablation_config).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # 兼容两种保存格式
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"已加载模型（检查点格式）: {checkpoint_path}")
        if 'save_reason' in checkpoint:
            print(f"  保存原因: {checkpoint['save_reason']}")
    else:
        model.load_state_dict(checkpoint)
        print(f"已加载模型（state_dict格式）: {checkpoint_path}")
    
    model.eval()
    
    # 加载测试数据
    if snr_db is not None:
        test_input, test_output, test_eog = load_test_data_by_snr(snr_db)
    else:
        # 使用单一测试集
        test_input = scipy.io.loadmat(base_config.TEST_CONTAMINATED_PATH)[base_config.DATA_KEY]
        test_output = scipy.io.loadmat(base_config.TEST_PURE_PATH)[base_config.PURE_KEY]
        test_eog = test_input - test_output
    
    print(f"测试数据: {test_input.shape}")
    
    # 创建数据集和加载器
    test_dataset = TestDataset(test_input, test_output)
    test_loader = DataLoader(test_dataset, batch_size=base_config.BATCH_SIZE, shuffle=False)
    
    # 推理
    all_clean_pred = []
    all_artifact_pred = []
    all_clean_target = []
    all_noisy = []
    
    # 推理时间统计
    total_samples = 0
    start_time = time()
    
    with torch.no_grad():
        for noisy_norm, clean_target, norm in test_loader:
            noisy_norm = noisy_norm.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            
            # 恢复原始幅度
            noisy_scaled = noisy_norm * norm
            
            # 模型推理
            clean_pred, artifact_pred = model(noisy_scaled)
            
            # 收集结果
            all_clean_pred.append(clean_pred.squeeze(1).cpu().numpy())
            all_artifact_pred.append(artifact_pred.squeeze(1).cpu().numpy())
            all_clean_target.append(clean_target.numpy())
            all_noisy.append(noisy_scaled.squeeze(1).cpu().numpy())
            
            total_samples += noisy_norm.size(0)
    
    total_inference_time = time() - start_time
    avg_time_per_sample = total_inference_time / total_samples if total_samples > 0 else 0
    
    # 拼接结果
    all_clean_pred = np.concatenate(all_clean_pred, axis=0)
    all_artifact_pred = np.concatenate(all_artifact_pred, axis=0)
    all_clean_target = np.concatenate(all_clean_target, axis=0)
    all_noisy = np.concatenate(all_noisy, axis=0)
    
    print(f"推理完成 - 总耗时: {total_inference_time:.2f}s | 样本数: {total_samples} | 单样本平均: {avg_time_per_sample*1000:.2f}ms")
    
    # 计算评估指标
    metrics = compute_all_metrics(all_clean_pred, all_clean_target, fs=base_config.SAMPLING_RATE)
    
    if metrics:
        print("\n去噪性能指标:")
        print_metrics(metrics, prefix="  ")
    
    # 保存结果为.mat文件（统一目录，兼容compute_all_metrics.m脚本）
    # 文件名格式: {实验名}_SNR{snr}dB.mat (使用下划线，第一个下划线前是方法名)
    result_dir = ABLATION_RESULTS_DIR
    
    # 根据要求，将实验名中的下划线替换为横线
    method_name = experiment_name.replace('_', '-')
    
    if snr_db is not None:
        result_filename = f'{method_name}_SNR{snr_db}dB.mat'
    else:
        result_filename = f'{method_name}_predictions.mat'
    
    result_path = os.path.join(result_dir, result_filename)
    
    # 准备保存的数据
    mat_data = {
        'noisy_data': all_noisy,
        'clean_data': all_clean_target,
        'denoised_data': all_clean_pred,
        'artifact_data': all_artifact_pred,
        'eog_target': test_eog,
        'sampling_rate': base_config.SAMPLING_RATE,
        'experiment_name': experiment_name,
        'total_inference_time': total_inference_time,
        'avg_time_per_sample': avg_time_per_sample,
        'time_per_sample': avg_time_per_sample,  # MATLAB脚本期望的键名
        'total_samples': total_samples
    }
    
    # 添加指标到.mat文件
    if metrics:
        for key, value in metrics.items():
            mat_data[f'metric_{key}'] = value
    
    scipy.io.savemat(result_path, mat_data)
    print(f"\n结果已保存至: {result_path}")
    print(f"  - 单样本平均推理时间: {avg_time_per_sample*1000:.2f}ms")
    
    return {
        'metrics': metrics,
        'result_path': result_path,
        'inference_time': avg_time_per_sample,  # 单样本平均推理时间（秒）
        'total_inference_time': total_inference_time,
        'total_samples': total_samples
    }


def main(selected_experiments=None):
    """主测试流程
    
    Args:
        selected_experiments: 要执行的实验名称列表，None表示全部执行
    """
    print("="*80)
    print("DAT-Net 消融实验 - 自动化测试")
    print("="*80)
    
    # 确定要执行的实验列表
    if selected_experiments is not None:
        experiments_to_test = [exp for exp in selected_experiments if exp in ABLATION_ORDER]
        print(f"\n✅ 指定测试的实验: {experiments_to_test}")
    else:
        experiments_to_test = ABLATION_ORDER
        print(f"\n测试所有实验: {len(experiments_to_test)} 个")
    
    # 设备选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 检查是否使用多SNR测试
    use_multi_snr = len(base_config.TEST_SNR_LEVELS) > 0
    
    if use_multi_snr:
        print(f"\n检测到多SNR测试集，将测试以下SNR级别: {base_config.TEST_SNR_LEVELS}")
        
        # 测试所有实验 × 所有SNR级别
        all_results = {}
        
        for exp_name in experiments_to_test:
            exp_config = get_ablation_config(exp_name)
            exp_results = {}
            
            for snr_db in base_config.TEST_SNR_LEVELS:
                result = test_single_experiment(
                    experiment_name=exp_name,
                    ablation_config=exp_config,
                    device=device,
                    snr_db=snr_db
                )
                if result is not None:
                    exp_results[f'SNR{snr_db}dB'] = result
            
            all_results[exp_name] = exp_results
    else:
        print(f"\n使用单一测试集")
        
        # 测试所有实验
        all_results = {}
        
        for exp_name in experiments_to_test:
            exp_config = get_ablation_config(exp_name)
            result = test_single_experiment(
                experiment_name=exp_name,
                ablation_config=exp_config,
                device=device,
                snr_db=None
            )
            if result is not None:
                all_results[exp_name] = result
    
    # 保存测试结果汇总
    summary_path = os.path.join(ABLATION_ROOT, 'test_results_summary.json')
    
    # 转换为可JSON序列化的格式
    serializable_results = {}
    for exp_name, exp_result in all_results.items():
        if use_multi_snr:
            serializable_results[exp_name] = {}
            for snr_key, snr_result in exp_result.items():
                serializable_results[exp_name][snr_key] = {
                    'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                               for k, v in snr_result['metrics'].items()} if snr_result['metrics'] else {},
                    'result_path': snr_result['result_path'],
                    'inference_time_ms': float(snr_result['inference_time'] * 1000) if 'inference_time' in snr_result else 0,
                    'total_inference_time_s': float(snr_result.get('total_inference_time', 0)),
                    'total_samples': int(snr_result.get('total_samples', 0))
                }
        else:
            serializable_results[exp_name] = {
                'metrics': {k: float(v) if isinstance(v, (int, float, np.number)) else v 
                           for k, v in exp_result['metrics'].items()} if exp_result['metrics'] else {},
                'result_path': exp_result['result_path'],
                'inference_time_ms': float(exp_result['inference_time'] * 1000) if 'inference_time' in exp_result else 0,
                'total_inference_time_s': float(exp_result.get('total_inference_time', 0)),
                'total_samples': int(exp_result.get('total_samples', 0))
            }
    
    with open(summary_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"\n测试结果汇总已保存至: {summary_path}")
    
    # 打印汇总表格
    print("\n" + "="*80)
    print("所有实验测试完成！性能汇总：")
    print("="*80)
    
    if use_multi_snr:
        # 多SNR汇总
        print(f"{'实验名称':<25}", end='')
        for snr_db in base_config.TEST_SNR_LEVELS[:3]:  # 只显示前3个SNR
            print(f"SNR{snr_db}dB RRMSE  ", end='')
        print()
        print("-"*80)
        
        for exp_name in ABLATION_ORDER:
            print(f"{exp_name:<25}", end='')
            if exp_name in all_results:
                for snr_db in base_config.TEST_SNR_LEVELS[:3]:
                    snr_key = f'SNR{snr_db}dB'
                    if snr_key in all_results[exp_name]:
                        metrics = all_results[exp_name][snr_key]['metrics']
                        rrmse = metrics.get('RRMSE', 0.0)
                        print(f"{rrmse:<15.6f}", end='')
                    else:
                        print(f"{'N/A':<15}", end='')
            print()
    else:
        # 单测试集汇总
        print(f"{'实验名称':<30} {'RRMSE':<12} {'CC':<12} {'SNR(dB)':<12}")
        print("-"*80)
        
        for exp_name in ABLATION_ORDER:
            if exp_name in all_results and all_results[exp_name]['metrics']:
                metrics = all_results[exp_name]['metrics']
                rrmse = metrics.get('RRMSE', 0.0)
                cc = metrics.get('CC', 0.0)
                snr = metrics.get('SNR', 0.0)
                print(f"{exp_name:<30} {rrmse:<12.6f} {cc:<12.6f} {snr:<12.2f}")
            else:
                print(f"{exp_name:<30} {'N/A':<12} {'N/A':<12} {'N/A':<12}")
    
    print("="*80)


if __name__ == '__main__':
    main()
