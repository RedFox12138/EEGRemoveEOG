"""
DAT-Net 无监督训练自动调参脚本
目标：最小化真实数据集上的标准差，提高模型稳定性

策略：
1. 使用随机搜索/网格搜索优化关键超参数
2. 针对每组参数进行短时训练（快速评估）
3. 在验证集上评估性能的均值和标准差
4. 选择标准差最小且性能较好的参数组合

参考：finetune_adaptive.py 的调参经验
"""

import os
import sys
import json
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)
datnet_dir = os.path.join(grandparent_dir, 'DAT-Net')

if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

from model import DATNet
from unsupervised_artifact_v2 import unsupervised_dat_loss_artifact_v2
from real_data_config import *


class UnsupervisedEEGDataset(Dataset):
    """无监督 EEG 数据集"""
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


def load_and_split_data():
    """加载真实数据集并划分为训练集和验证集"""
    print('\n正在加载真实数据集...')
    data_dict = scipy.io.loadmat(REAL_DATA_PATH)
    
    possible_keys = [DATA_KEY, 'data', 'eeg_data', 'X', 'signals']
    data = None
    for key in possible_keys:
        if key in data_dict:
            data = data_dict[key]
            break
    
    if data is None:
        available_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        raise ValueError(f'无法找到数据！可用的 keys: {available_keys}')
    
    n_samples = len(data)
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    
    train_size = int(n_samples * TRAIN_RATIO)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    train_x = data[train_indices]
    val_x = data[val_indices]
    
    print(f'  训练集: {len(train_x)} 样本')
    print(f'  验证集: {len(val_x)} 样本')
    
    return train_x, val_x


def quick_train_and_evaluate(params_dict, train_loader, val_loader, device, epochs=40):
    """
    快速训练并评估一组参数
    
    Args:
        params_dict: 参数字典
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器  
        device: 设备
        epochs: 快速训练的轮数（用于初步评估）
    
    Returns:
        val_loss_mean: 验证集损失均值
        val_loss_std: 验证集损失标准差（关键指标）
        output_std: 输出标准差
    """
    # 创建模型
    model = DATNet(in_channels=1, base_channels=32).to(device)
    
    # 创建优化器
    optimizer = optim.AdamW(
        model.parameters(),
        lr=params_dict['learning_rate'],
        weight_decay=params_dict['weight_decay']
    )
    
    # 学习率调度器
    if params_dict.get('use_scheduler', True):
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=epochs,
            eta_min=params_dict.get('min_lr', 1e-5)
        )
    
    # 训练
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0
        
        for noisy, norm in train_loader:
            noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1).to(device)
            norm = torch.tensor(norm, dtype=torch.float32).view(-1, 1, 1).to(device)
            noisy_scaled = noisy * norm
            
            optimizer.zero_grad()
            
            # 计算无监督损失（损失函数会自己调用模型）
            loss, _, _ = unsupervised_dat_loss_artifact_v2(
                model,
                noisy_scaled,
                SAMPLING_RATE,
                mask_base=params_dict['mask_base'],
                boost_scale=params_dict['boost_scale'],
                lambda_rec=params_dict['lambda_rec'],
                lambda_con=params_dict['lambda_con'],
                lambda_teacher=params_dict['lambda_teacher'],
                lambda_n2v=params_dict['lambda_n2v'],
                lambda_band=params_dict['lambda_band'],
                lambda_low=params_dict['lambda_low'],
                lambda_decor=params_dict['lambda_decor'],
                lambda_content=params_dict['lambda_content'],
                gamma_art_weight=params_dict['gamma_art_weight'],
                artifact_win_size=params_dict['artifact_win_size'],
                mask_neighborhood=MASK_NEIGHBORHOOD,
                teacher_cutoff=params_dict['teacher_cutoff'],
                lowpass_cutoff=params_dict['lowpass_cutoff'],
                teacher_threshold=params_dict['teacher_threshold']
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), params_dict['grad_clip'])
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        if params_dict.get('use_scheduler', True):
            scheduler.step()
    
    # 评估：计算验证集上的性能和标准差
    model.eval()
    val_losses = []
    val_outputs = []
    
    with torch.no_grad():
        for noisy, norm in val_loader:
            noisy = torch.tensor(noisy, dtype=torch.float32).unsqueeze(1).to(device)
            norm = torch.tensor(norm, dtype=torch.float32).view(-1, 1, 1).to(device)
            noisy_scaled = noisy * norm
            
            # 计算损失（损失函数会自己调用模型）
            loss, _, (c_A, a_A, c_B, a_B) = unsupervised_dat_loss_artifact_v2(
                model,
                noisy_scaled,
                SAMPLING_RATE,
                mask_base=params_dict['mask_base'],
                boost_scale=params_dict['boost_scale'],
                lambda_rec=params_dict['lambda_rec'],
                lambda_con=params_dict['lambda_con'],
                lambda_teacher=params_dict['lambda_teacher'],
                lambda_n2v=params_dict['lambda_n2v'],
                lambda_band=params_dict['lambda_band'],
                lambda_low=params_dict['lambda_low'],
                lambda_decor=params_dict['lambda_decor'],
                lambda_content=params_dict['lambda_content'],
                gamma_art_weight=params_dict['gamma_art_weight'],
                artifact_win_size=params_dict['artifact_win_size'],
                mask_neighborhood=MASK_NEIGHBORHOOD,
                teacher_cutoff=params_dict['teacher_cutoff'],
                lowpass_cutoff=params_dict['lowpass_cutoff'],
                teacher_threshold=params_dict['teacher_threshold']
            )
            
            val_losses.append(loss.item())
            val_outputs.append(c_B.cpu().numpy())  # 使用clean分支的输出
    
    # 计算统计指标
    val_loss_mean = np.mean(val_losses)
    val_loss_std = np.std(val_losses)  # 关键指标：损失的标准差
    
    # 计算输出的标准差
    all_outputs = np.concatenate(val_outputs, axis=0)
    output_std = np.std(all_outputs)
    
    return val_loss_mean, val_loss_std, output_std


def generate_param_grid():
    """
    生成参数搜索空间定义（连续范围）
    使用 (min, max, num_samples) 格式定义线性空间
    
    返回: 参数范围字典
    """
    param_ranges = {
        # ========== 学习率相关 ==========
        # (最小值, 最大值, 采样点数, 是否对数空间)
        'learning_rate': (1e-4, 1e-2, 50, True),  # 对数空间：1e-4 到 0.01
        'weight_decay': (1e-6, 1e-4, 30, True),   # 对数空间：1e-6 到 1e-4
        'min_lr': (1e-6, 5e-4, 30, True),         # 对数空间：1e-6 到 5e-4
        'grad_clip': (0.5, 3.0, 20, False),       # 线性空间：0.5 到 3.0
        
        # ========== 损失函数权重 ==========
        'lambda_rec': (0.5, 1.0, 20, False),       # 重建损失：0.5 到 1.0
        'lambda_con': (1.0, 2.0, 20, False),       # 一致性损失：1.0 到 2.0
        'lambda_teacher': (0.1, 0.6, 25, False),   # Teacher损失：0.1 到 0.6
        'lambda_n2v': (0.2, 0.5, 15, False),       # Noise2Void损失：0.2 到 0.5
        'lambda_band': (0.3, 0.8, 20, False),      # 频带损失：0.3 到 0.8
        'lambda_low': (0.05, 0.3, 20, False),      # 低频损失：0.05 到 0.3
        'lambda_decor': (0.2, 0.6, 20, False),     # 去相关损失：0.2 到 0.6
        'lambda_content': (0.05, 0.2, 15, False),  # 内容损失：0.05 到 0.2
        
        # ========== Artifact-aware 参数 ==========
        'mask_base': (0.05, 0.25, 25, False),           # 基础掩蔽率：0.05 到 0.25
        'boost_scale': (0.1, 0.5, 25, False),           # 伪影增强系数：0.1 到 0.5
        'gamma_art_weight': (0.5, 2.5, 30, False),      # 伪影加权因子：0.5 到 2.5
        
        # ========== 内部算法参数 ==========
        'artifact_win_size': (50, 250, 13, False),      # 伪影窗口大小：50-250，步长16左右
        'teacher_cutoff': (2.0, 12.0, 30, False),       # Teacher高通截止：2-12 Hz
        'lowpass_cutoff': (2.0, 8.0, 25, False),        # 低通截止：2-8 Hz
        'teacher_threshold': (0.5, 0.9, 25, False),     # Teacher阈值：0.5 到 0.9
        
        # ========== 其他 ==========
        'use_scheduler': ([True], 1, False, False),     # 固定值
    }
    
    return param_ranges


def sample_from_ranges(param_ranges, n_samples=50):
    """
    从参数范围中采样
    
    Args:
        param_ranges: 参数范围定义
        n_samples: 采样数量
    
    Returns:
        采样的参数列表
    """
    sampled_params = []
    
    for _ in range(n_samples):
        params = {}
        for key, range_def in param_ranges.items():
            if key == 'use_scheduler':
                # 固定值
                params[key] = True
            elif len(range_def) == 4:
                min_val, max_val, num_points, use_log = range_def
                
                if use_log:
                    # 对数空间采样
                    value = np.exp(np.random.uniform(np.log(min_val), np.log(max_val)))
                else:
                    # 线性空间采样
                    value = np.random.uniform(min_val, max_val)
                
                # 对于整数参数（通过参数名判断）
                if 'win_size' in key or 'neighborhood' in key:
                    value = int(value)
                
                params[key] = value
        
        sampled_params.append(params)
    
    return sampled_params


def print_param_ranges_info(param_ranges):
    """打印参数范围信息"""
    print('\n' + '='*80)
    print('参数搜索空间定义（连续范围）')
    print('='*80)
    
    for param_name, range_def in param_ranges.items():
        if param_name == 'use_scheduler':
            print(f'  {param_name:20s}: 固定值 True')
        else:
            min_val, max_val, num_points, use_log = range_def
            space_type = "对数空间" if use_log else "线性空间"
            print(f'  {param_name:20s}: [{min_val:.6f}, {max_val:.6f}] ({num_points}点, {space_type})')
    
    print('='*80)


def save_results(results, save_path='tuning_results.json'):
    """保存调参结果"""
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f'\n结果已保存到: {save_path}')


def main():
    """主函数"""
    print('='*80)
    print('DAT-Net 无监督训练自动调参')
    print('目标：最小化真实数据集上的标准差')
    print('='*80)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 加载数据
    train_x, val_x = load_and_split_data()
    
    train_dataset = UnsupervisedEEGDataset(train_x)
    val_dataset = UnsupervisedEEGDataset(val_x)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 生成参数范围定义
    param_ranges = generate_param_grid()
    
    # 打印参数范围信息
    print_param_ranges_info(param_ranges)
    
    # 随机采样参数（从连续范围中采样）
    n_trials = 50  # 调整这个数字来控制搜索规模
    print(f'\n将进行 {n_trials} 次随机参数搜索')
    print('每次快速训练 100 epochs 进行初步评估')
    print('从连续参数空间中随机采样，覆盖整个范围\n')
    
    sampled_params = sample_from_ranges(param_ranges, n_samples=n_trials)
    
    # 调参循环
    results = []
    best_std = float('inf')
    best_params = None
    
    for trial_idx, params in enumerate(sampled_params):
        print(f'\n{"="*80}')
        print(f'Trial {trial_idx + 1}/{n_trials}')
        print(f'{"="*80}')
        print('参数配置:')
        for key, value in params.items():
            print(f'  {key}: {value}')
        
        try:
            # 快速训练和评估
            start_time = time()
            val_loss_mean, val_loss_std, output_std = quick_train_and_evaluate(
                params, train_loader, val_loader, device, epochs=40
            )
            elapsed_time = time() - start_time
            
            # 记录结果
            result = {
                'trial': trial_idx + 1,
                'params': params,
                'val_loss_mean': float(val_loss_mean),
                'val_loss_std': float(val_loss_std),
                'output_std': float(output_std),
                'time': elapsed_time
            }
            results.append(result)
            
            # 打印结果
            print(f'\n结果:')
            print(f'  验证损失均值: {val_loss_mean:.6f}')
            print(f'  验证损失标准差: {val_loss_std:.6f}  ← 关键指标')
            print(f'  输出标准差: {output_std:.6f}')
            print(f'  训练时间: {elapsed_time:.2f}s')
            
            # 更新最佳结果
            if val_loss_std < best_std:
                best_std = val_loss_std
                best_params = params.copy()
                print(f'  ★ 新的最佳标准差！')
            
        except Exception as e:
            print(f'\n错误: {e}')
            print('跳过此参数组合...')
            continue
    
    # 输出最佳参数
    print('\n' + '='*80)
    print('调参完成！')
    print('='*80)
    
    if best_params is None:
        print('\n❌ 所有试验都失败了！')
        print('可能的原因：')
        print('  1. 数据加载问题')
        print('  2. 模型初始化问题')
        print('  3. 内存不足')
        print('  4. 参数配置错误')
        print('\n建议：')
        print('  - 检查数据路径和格式')
        print('  - 降低 BATCH_SIZE')
        print('  - 查看上面的错误信息')
        
        # 仍然保存失败的结果记录
        save_results({
            'best_params': None,
            'best_std': None,
            'all_trials': results
        })
        return
    
    print(f'\n最佳标准差: {best_std:.6f}')
    print('\n最佳参数配置:')
    for key, value in best_params.items():
        print(f'  {key}: {value}')
    
    # 保存结果
    save_results({
        'best_params': best_params,
        'best_std': float(best_std),
        'all_trials': results
    })
    
    print('\n建议：将最佳参数更新到 real_data_config.py 中')


if __name__ == '__main__':
    main()
