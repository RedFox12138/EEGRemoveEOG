"""
自动化训练脚本 - 消融实验
训练不同配置的模型变体
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import json
from time import time

# 导入配置和工具
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ablation_config import *
from model_wrapper import create_model
from loss_wrapper import unsupervised_dat_loss_ablation

# 导入原始config以获取训练参数
v2_dir = os.path.join(os.path.dirname(current_dir), 'DAT-Net-Unsupervised-v2')
sys.path.insert(0, v2_dir)
import config as base_config


class UnsupervisedEEGDataset(Dataset):
    """无监督EEG数据集"""
    def __init__(self, noisy, clean=None):
        self.noisy = noisy
        self.clean = clean
        self.has_clean = clean is not None

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # Max-Abs归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        if self.has_clean:
            clean = self.clean[idx]
            return noisy.astype('float32') / norm, clean.astype('float32'), norm
        else:
            return noisy.astype('float32') / norm, norm


def set_seed(seed):
    """设置随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_paths():
    """设置路径（使用相对路径避免中文路径问题）"""
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f'当前工作目录: {os.getcwd()}')
    
    # 使用相对路径创建checkpoints目录
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'[创建目录] {checkpoint_dir}')
    
    # 验证目录是否存在
    if not os.path.exists(checkpoint_dir):
        raise RuntimeError(f"无法创建检查点目录: {checkpoint_dir}")
    
    return checkpoint_dir


def get_checkpoint_dir():
    """获取checkpoint目录（已废弃，使用 setup_paths）"""
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir


def get_data():
    """加载训练和验证数据"""
    train_x = scipy.io.loadmat(base_config.TRAIN_CONTAMINATED_PATH)[base_config.DATA_KEY]
    val_x = scipy.io.loadmat(base_config.VAL_CONTAMINATED_PATH)[base_config.DATA_KEY]
    try:
        val_y = scipy.io.loadmat(base_config.VAL_PURE_PATH)[base_config.PURE_KEY]
    except Exception:
        val_y = None
    return train_x, val_x, val_y


def train_epoch(model, device, loader, optimizer, ablation_config):
    """训练一个epoch"""
    model.train()
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    
    for batch in loader:
        if len(batch) == 3:
            noisy, _, norm = batch
        else:
            noisy, norm = batch
        noisy = noisy.float().unsqueeze(1).to(device)
        norm = norm.float().to(device).view(-1, 1, 1)
        
        # 恢复原始幅度
        noisy_scaled = noisy * norm

        optimizer.zero_grad()
        total_loss_batch, loss_dict, _ = unsupervised_dat_loss_ablation(
            model=model,
            eeg_raw_input=noisy_scaled,
            fs=base_config.SAMPLING_RATE,
            ablation_config=ablation_config,
            mask_base=base_config.MASK_BASE,
            boost_scale=base_config.BOOST_SCALE,
            lambda_rec=base_config.LAMBDA_REC,
            lambda_con=base_config.LAMBDA_CON,
            lambda_teacher=base_config.LAMBDA_TEACHER,
            lambda_n2v=base_config.LAMBDA_N2V,
            lambda_band=base_config.LAMBDA_BAND,
            lambda_low=base_config.LAMBDA_LOW,
            lambda_decor=base_config.LAMBDA_DECOR,
            lambda_content=base_config.LAMBDA_CONTENT,
            gamma_art_weight=base_config.GAMMA_ART_WEIGHT,
            artifact_win_size=base_config.ARTIFACT_WIN_SIZE,
            mask_neighborhood=base_config.MASK_NEIGHBORHOOD,
            teacher_cutoff=base_config.TEACHER_CUTOFF,
            lowpass_cutoff=base_config.LOWPASS_CUTOFF,
            teacher_threshold=base_config.TEACHER_THRESHOLD,
        )
        total_loss_batch.backward()
        if base_config.GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), base_config.GRAD_CLIP)
        optimizer.step()

        for key in accumulated_losses.keys():
            accumulated_losses[key] += loss_dict.get(key, 0.0)
        num_batches += 1

    return {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}


def validate(model, device, loader, ablation_config, has_clean_labels=False):
    """验证模型"""
    model.eval()
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                noisy, clean, norm = batch
            else:
                noisy, norm = batch
                clean = None
            noisy = noisy.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            
            noisy_scaled = noisy * norm

            total_loss_batch, loss_dict, (c_A, a_A, c_B, a_B) = unsupervised_dat_loss_ablation(
                model=model,
                eeg_raw_input=noisy_scaled,
                fs=base_config.SAMPLING_RATE,
                ablation_config=ablation_config,
                mask_base=base_config.MASK_BASE,
                boost_scale=base_config.BOOST_SCALE,
                lambda_rec=base_config.LAMBDA_REC,
                lambda_con=base_config.LAMBDA_CON,
                lambda_teacher=base_config.LAMBDA_TEACHER,
                lambda_n2v=base_config.LAMBDA_N2V,
                lambda_band=base_config.LAMBDA_BAND,
                lambda_low=base_config.LAMBDA_LOW,
                lambda_decor=base_config.LAMBDA_DECOR,
                lambda_content=base_config.LAMBDA_CONTENT,
                gamma_art_weight=base_config.GAMMA_ART_WEIGHT,
                artifact_win_size=base_config.ARTIFACT_WIN_SIZE,
                mask_neighborhood=base_config.MASK_NEIGHBORHOOD,
                teacher_cutoff=base_config.TEACHER_CUTOFF,
                lowpass_cutoff=base_config.LOWPASS_CUTOFF,
                teacher_threshold=base_config.TEACHER_THRESHOLD,
            )

            for key in accumulated_losses.keys():
                accumulated_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1
            
            if has_clean_labels and clean is not None:
                all_preds.append(c_B.squeeze(1).cpu().numpy())
                all_targets.append(clean.numpy())
    
    # 计算metrics（如果有真实标签）
    metrics = None
    if has_clean_labels and len(all_targets) > 0:
        try:
            root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
            from metrics_utils import compute_all_metrics
            all_preds = np.concatenate(all_preds, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            metrics = compute_all_metrics(all_preds, all_targets, fs=base_config.SAMPLING_RATE)
        except Exception:
            metrics = None
    
    avg_losses = {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}
    return avg_losses, metrics


def train_single_experiment(experiment_name, ablation_config, device, train_loader, val_loader):
    """
    训练单个消融实验
    
    Args:
        experiment_name: 实验名称
        ablation_config: 消融配置字典
        device: 训练设备
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
    
    Returns:
        训练历史记录
    """
    print(f"\n{'='*80}")
    print(f"开始训练实验: {experiment_name}")
    print(f"描述: {ablation_config['description']}")
    print(f"{'='*80}")
    
    # 设置随机种子确保可重复性
    set_seed(RANDOM_SEED)
    
    # 创建模型
    model = create_model(ablation_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")
    
    # 创建优化器
    optimizer = optim.Adam(
        model.parameters(), 
        lr=base_config.LEARNING_RATE, 
        weight_decay=base_config.WEIGHT_DECAY
    )
    
    # 学习率调度器（线性衰减）
    if base_config.USE_LR_SCHEDULER:
        # 使用线性衰减策略：从 LEARNING_RATE 衰减到 MIN_LR
        def lr_lambda(epoch):
            # 线性插值：lr = LEARNING_RATE * alpha，其中 alpha 从 1.0 衰减到 MIN_LR/LEARNING_RATE
            min_factor = base_config.MIN_LR / base_config.LEARNING_RATE
            factor = 1.0 - (1.0 - min_factor) * (epoch / base_config.EPOCHS)
            return max(min_factor, factor)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None
    
    # 训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rrmse': [],
        'lr': [],
        'best_val_loss': float('inf'),
        'best_val_rrmse': float('inf'),
        'best_loss_epoch': 0,
        'best_rrmse_epoch': 0
    }
    
    # 早停计数器
    patience_counter = 0
    
    # 获取数据集名称用于保存
    dataset_name = base_config.DATASET_NAME
    
    # 模型保存路径（使用相对路径，与train_unsupervised.py一致）
    checkpoint_dir = 'checkpoints'
    
    model_save_path_rrmse = os.path.join(checkpoint_dir, f'ablation_{experiment_name}_{dataset_name}_best_rrmse.pth')
    model_save_path_loss = os.path.join(checkpoint_dir, f'ablation_{experiment_name}_{dataset_name}_best_loss.pth')
    
    print(f"RRMSE最佳模型路径: {model_save_path_rrmse}")
    print(f"Loss最佳模型路径: {model_save_path_loss}")
    
    # 训练循环
    for epoch in range(base_config.EPOCHS):
        epoch_start_time = time()
        
        # 训练
        train_losses = train_epoch(model, device, train_loader, optimizer, ablation_config)
        
        # 验证（获取RRMSE指标）
        val_losses, val_metrics = validate(model, device, val_loader, ablation_config, has_clean_labels=True)
        
        # 更新学习率
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = base_config.LEARNING_RATE
        
        # 记录历史
        history['train_loss'].append(train_losses['total'])
        history['val_loss'].append(val_losses['total'])
        history['lr'].append(current_lr)
        
        # 提取RRMSE（如果有）
        current_rrmse = val_metrics.get('RRMSE', float('inf')) if val_metrics else float('inf')
        history['val_rrmse'].append(current_rrmse)
        
        # ========== 双重保存策略：同时保存RRMSE最佳和Loss最佳模型 ==========
        # 早停判断基于Loss（消融实验的主要评估标准）
        improved_loss = False
        
        # 1. 基于Loss保存（主要指标，用于早停）
        if val_losses['total'] < history['best_val_loss']:
            history['best_val_loss'] = val_losses['total']
            history['best_loss_epoch'] = epoch + 1
            improved_loss = True
            
            checkpoint_loss = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': history['best_val_loss'],
                'best_val_rrmse': history['best_val_rrmse'],
                'val_loss': val_losses,
                'val_metrics': val_metrics if val_metrics is not None else {},
                'save_reason': f'Loss最佳: {history["best_val_loss"]:.6f}'
            }
            torch.save(checkpoint_loss, model_save_path_loss)
        
        # 2. 基于RRMSE保存（辅助指标，仅记录不影响早停）
        if val_metrics is not None and 'RRMSE' in val_metrics:
            if current_rrmse < history['best_val_rrmse']:
                history['best_val_rrmse'] = current_rrmse
                history['best_rrmse_epoch'] = epoch + 1
                
                checkpoint_rrmse = {
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': history['best_val_loss'],
                    'best_val_rrmse': history['best_val_rrmse'],
                    'val_loss': val_losses,
                    'val_metrics': val_metrics,
                    'save_reason': f'RRMSE最佳: {history["best_val_rrmse"]:.6f}'
                }
                torch.save(checkpoint_rrmse, model_save_path_rrmse)
        
        # 更新patience计数器（仅基于Loss）
        if improved_loss:
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 打印进度
        epoch_time = time() - epoch_start_time
        if (epoch + 1) % 10 == 0 or epoch == 0:
            rrmse_str = f"RRMSE: {current_rrmse:.6f}" if val_metrics else "RRMSE: N/A"
            print(f"Epoch [{epoch+1}/{base_config.EPOCHS}] "
                  f"Train Loss: {train_losses['total']:.6f} | "
                  f"Val Loss: {val_losses['total']:.6f} | "
                  f"{rrmse_str} | "
                  f"LR: {current_lr:.6f} | "
                  f"Time: {epoch_time:.2f}s | "
                  f"Best Loss: {history['best_val_loss']:.6f} (Epoch {history['best_loss_epoch']}) | "
                  f"Best RRMSE: {history['best_val_rrmse']:.6f} (Epoch {history['best_rrmse_epoch']})")
        
        # 早停（基于验证集Loss）
        if patience_counter >= base_config.PATIENCE:
            print(f"\n早停触发！验证集Loss连续{base_config.PATIENCE}个epoch未改善。")
            print(f"当前最佳验证Loss: {history['best_val_loss']:.6f} (Epoch {history['best_loss_epoch']})")
            break
    
    print(f"\n{'='*80}")
    print(f"实验 [{experiment_name}] 训练完成!")
    print(f"{'='*80}")
    print(f"最佳验证损失: {history['best_val_loss']:.6f} (Epoch {history['best_loss_epoch']}) ← 用于消融评估")
    print(f"最佳RRMSE: {history['best_val_rrmse']:.6f} (Epoch {history['best_rrmse_epoch']}) ← 仅供参考")
    print(f"\n保存的模型:")
    print(f"  ✓ Loss最佳: {model_save_path_loss}")
    print(f"  ○ RRMSE最佳: {model_save_path_rrmse}")
    print(f"{'='*80}")
    
    return history


def main(selected_experiments=None, skip_existing=False):
    """主训练流程
    
    Args:
        selected_experiments: 要执行的实验名称列表，None表示全部执行
        skip_existing: 是否跳过已有模型的实验
    """
    print("="*80)
    print("DAT-Net 消融实验 - 自动化训练")
    print("="*80)
    
    # 设置路径（使用相对路径避免中文路径问题）
    checkpoint_dir = setup_paths()
    
    # 打印配置概览
    print_ablation_summary()
    
    # 确定要执行的实验列表
    if selected_experiments is not None:
        experiments_to_run = [exp for exp in selected_experiments if exp in ABLATION_ORDER]
        print(f"\n✅ 指定执行的实验: {experiments_to_run}")
    else:
        experiments_to_run = ABLATION_ORDER
        print(f"\n执行所有实验: {len(experiments_to_run)} 个")
    
    # 如果需要跳过已存在的模型（检查best_loss模型，因为这是消融实验的主要评估模型）
    if skip_existing:
        dataset_name = base_config.DATASET_NAME
        skipped = []
        remaining = []
        for exp_name in experiments_to_run:
            model_path_loss = os.path.join('checkpoints', f'ablation_{exp_name}_{dataset_name}_best_loss.pth')
            if os.path.exists(model_path_loss):
                skipped.append(exp_name)
                print(f"  ⏭️  跳过 [{exp_name}]（Loss最佳模型已存在）")
            else:
                remaining.append(exp_name)
        experiments_to_run = remaining
        if skipped:
            print(f"\n已跳过 {len(skipped)} 个实验: {skipped}")
        if not experiments_to_run:
            print("\n✅ 所有指定的实验都已完成，无需训练！")
            return
    
    print(f"\n将要训练的实验: {experiments_to_run}")
    
    # 设备选择
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n加载数据...")
    train_x, val_x, val_y = get_data()
    print(f"训练集: {train_x.shape}")
    print(f"验证集: {val_x.shape}")
    
    # 创建数据集和数据加载器
    train_dataset = UnsupervisedEEGDataset(train_x, clean=None)
    val_dataset = UnsupervisedEEGDataset(val_x, clean=val_y)
    train_loader = DataLoader(train_dataset, batch_size=base_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=base_config.BATCH_SIZE, shuffle=False)
    
    # 训练所有实验
    all_histories = {}
    
    for exp_name in experiments_to_run:
        exp_config = get_ablation_config(exp_name)
        history = train_single_experiment(
            experiment_name=exp_name,
            ablation_config=exp_config,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader
        )
        all_histories[exp_name] = history
    
    # 保存训练历史
    history_path = os.path.join(ABLATION_ROOT, 'training_history.json')
    with open(history_path, 'w') as f:
        # 转换为可JSON序列化的格式
        serializable_histories = {}
        for name, hist in all_histories.items():
            serializable_histories[name] = {
                'train_loss': [float(x) for x in hist['train_loss']],
                'val_loss': [float(x) for x in hist['val_loss']],
                'lr': [float(x) for x in hist['lr']],
                'best_val_loss': float(hist['best_val_loss']),
                'best_epoch': int(hist['best_epoch'])
            }
        json.dump(serializable_histories, f, indent=2)
    print(f"\n训练历史已保存至: {history_path}")
    
    # 打印汇总
    print("\n" + "="*80)
    print("所有实验训练完成！汇总如下：")
    print("="*80)
    print(f"{'实验名称':<30} {'最佳验证损失':<15} {'最佳Epoch':<10}")
    print("-"*80)
    for exp_name in ABLATION_ORDER:
        hist = all_histories[exp_name]
        print(f"{exp_name:<30} {hist['best_val_loss']:<15.6f} {hist['best_epoch']:<10}")
    print("="*80)


if __name__ == '__main__':
    main()
