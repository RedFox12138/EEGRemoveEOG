"""
DAT-Net 真实数据集无监督训练脚本
使用 Artifact-aware + Dual-branch consistency (unsupervised_dat_loss_artifact_v2)
针对真实数据集（无标签）的无监督训练
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from time import time

# 添加路径以导入相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # DAT-Net-Unsupervised-v2
grandparent_dir = os.path.dirname(parent_dir)  # 我自己的方法
datnet_dir = os.path.join(grandparent_dir, 'DAT-Net')

# 添加必要的路径
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, parent_dir)  # 添加 DAT-Net-Unsupervised-v2 目录
sys.path.insert(0, current_dir)  # 添加当前目录

from model import DATNet
from unsupervised_artifact_v2 import unsupervised_dat_loss_artifact_v2

# 导入配置
from real_data_config import *


class UnsupervisedEEGDataset(Dataset):
    """
    无监督 EEG 数据集（无需标签）
    """
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
    """
    加载真实数据集并划分为训练集和验证集
    
    返回:
        train_x: 训练集数据
        val_x: 验证集数据
    """
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
        # 打印所有可用的 key
        available_keys = [k for k in data_dict.keys() if not k.startswith('__')]
        raise ValueError(f'无法找到数据！可用的 keys: {available_keys}\n请在 real_data_config.py 中修改 DATA_KEY')
    
    print(f'  数据形状: {data.shape}')
    
    # 验证数据形状
    if len(data.shape) != 2:
        raise ValueError(f'数据形状错误！期望 (n_samples, window_size)，但得到 {data.shape}')
    
    n_samples, sample_length = data.shape
    print(f'  样本数量: {n_samples}')
    print(f'  样本长度: {sample_length}')
    
    if sample_length != WINDOW_SIZE:
        print(f'  ⚠️ 警告: 样本长度 ({sample_length}) 与配置的窗口大小 ({WINDOW_SIZE}) 不匹配')
        print(f'  请在 real_data_config.py 中调整 WINDOW_SIZE')
    
    # 随机打乱并划分数据
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(n_samples)
    
    # 计算划分点
    # 比例 7:1:2 (Train:Val:Test)
    # Train: 0% -> 70%
    # Val:   70% -> 80% (70+10)
    # Test:  80% -> 100% (不在此处使用)
    
    train_end = int(n_samples * TRAIN_RATIO)  # 0.7
    val_end = int(n_samples * (TRAIN_RATIO + VAL_RATIO)) # 0.8
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    
    train_x = data[train_indices]
    val_x = data[val_indices]
    
    print(f'\n数据集划分完成 (Train:Val:Test = 7:1:2):')
    print(f'  训练集: {train_x.shape[0]} 样本')
    print(f'  验证集: {val_x.shape[0]} 样本')
    print(f'  测试集: {n_samples - val_end} 样本 (保留)')
    
    return train_x, val_x


def train_epoch(model, device, loader, optimizer):
    """训练一个 epoch"""
    model.train()
    # 累积所有损失项
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    
    for noisy, norm in loader:
        noisy = noisy.float().unsqueeze(1).to(device)
        norm = norm.float().to(device).view(-1, 1, 1)
        noisy_scaled = noisy * norm

        optimizer.zero_grad()
        total_loss_batch, loss_dict, _ = unsupervised_dat_loss_artifact_v2(
            model=model,
            eeg_raw_input=noisy_scaled,
            fs=SAMPLING_RATE,
            mask_base=MASK_BASE,
            boost_scale=BOOST_SCALE,
            lambda_rec=LAMBDA_REC,
            lambda_con=LAMBDA_CON,
            lambda_teacher=LAMBDA_TEACHER,
            lambda_n2v=LAMBDA_N2V,
            lambda_band=LAMBDA_BAND,
            lambda_low=LAMBDA_LOW,
            lambda_decor=LAMBDA_DECOR,
            lambda_content=LAMBDA_CONTENT,
            gamma_art_weight=GAMMA_ART_WEIGHT,
            artifact_win_size=ARTIFACT_WIN_SIZE,
            mask_neighborhood=MASK_NEIGHBORHOOD,
            teacher_cutoff=TEACHER_CUTOFF,
            lowpass_cutoff=LOWPASS_CUTOFF,
            teacher_threshold=TEACHER_THRESHOLD,
        )
        total_loss_batch.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # 累积所有损失项
        for key in accumulated_losses.keys():
            accumulated_losses[key] += loss_dict.get(key, 0.0)
        num_batches += 1

    # 返回平均损失
    return {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}


def validate(model, device, loader):
    """验证模型（无监督，只计算损失）"""
    model.eval()
    # 累积所有损失项
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    
    with torch.no_grad():
        for noisy, norm in loader:
            noisy = noisy.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            noisy_scaled = noisy * norm

            total_loss_batch, loss_dict, _ = unsupervised_dat_loss_artifact_v2(
                model=model,
                eeg_raw_input=noisy_scaled,
                fs=SAMPLING_RATE,
                mask_base=MASK_BASE,
                boost_scale=BOOST_SCALE,
                lambda_rec=LAMBDA_REC,
                lambda_con=LAMBDA_CON,
                lambda_teacher=LAMBDA_TEACHER,
                lambda_n2v=LAMBDA_N2V,
                lambda_band=LAMBDA_BAND,
                lambda_low=LAMBDA_LOW,
                lambda_decor=LAMBDA_DECOR,
                lambda_content=LAMBDA_CONTENT,
                gamma_art_weight=GAMMA_ART_WEIGHT,
                artifact_win_size=ARTIFACT_WIN_SIZE,
                mask_neighborhood=MASK_NEIGHBORHOOD,
                teacher_cutoff=TEACHER_CUTOFF,
                lowpass_cutoff=LOWPASS_CUTOFF,
                teacher_threshold=TEACHER_THRESHOLD,
            )
            
            # 累积所有损失项
            for key in accumulated_losses.keys():
                accumulated_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1

    # 返回平均损失
    return {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}


def main():
    print("="*80)
    print("DAT-Net 真实数据集无监督训练")
    print("Disentangling Attention Temporal-Network")
    print("无标签真实数据 - 完全无监督学习")
    print("="*80)
    
    # 打印配置
    print_config()
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'\n使用设备: {device}')
    
    # 切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f'当前工作目录: {os.getcwd()}')
    
    # 创建必要的目录
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f'检查点目录: {CHECKPOINT_DIR}')
    print(f'结果目录: {RESULTS_DIR}')
    
    # 加载并划分数据
    train_x, val_x = load_and_split_data()
    
    # 创建数据集和加载器
    print('\n创建数据加载器...')
    train_dataset = UnsupervisedEEGDataset(train_x)
    val_dataset = UnsupervisedEEGDataset(val_x)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f'  训练批次数: {len(train_loader)}')
    print(f'  验证批次数: {len(val_loader)}')
    
    # 创建模型
    print('\n创建模型...')
    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'  模型参数量: {model.count_parameters():,}')
    
    # 优化器和学习率调度器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_LR_SCHEDULER:
        def warmup_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
                # 余弦退火，但保证不低于 MIN_LR
                cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
                min_factor = MIN_LR / LEARNING_RATE
                return max(min_factor, cosine_factor)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = None
    
    # 尝试加载checkpoint以继续训练
    start_epoch = 1
    best_val_loss = float('inf')
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth')
    if os.path.isfile(checkpoint_path):
        try:
            print(f'\n找到训练checkpoint: {checkpoint_path}')
            checkpoint = torch.load(checkpoint_path, map_location=device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                # 新格式checkpoint：包含完整训练状态
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if scheduler is not None and 'scheduler_state_dict' in checkpoint:
                    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_val_loss = checkpoint['best_val_loss']
                print(f'✓ 成功加载完整checkpoint')
                print(f'  - 从epoch {start_epoch} 继续训练')
                print(f'  - 最佳验证损失: {best_val_loss:.6f}')
            else:
                # 旧格式：只有模型权重
                model.load_state_dict(checkpoint)
                print(f'✓ 加载了模型权重（旧格式）')
                print(f'  - 从epoch 1 开始，但使用已训练的模型权重')
        except Exception as e:
            print(f'⚠️ 加载checkpoint失败: {e}')
            print('  从头开始训练')
    elif os.path.isfile(MODEL_SAVE_PATH):
        try:
            print(f'\n找到最佳模型: {MODEL_SAVE_PATH}')
            best_model = torch.load(MODEL_SAVE_PATH, map_location=device)
            model.load_state_dict(best_model)
            print(f'✓ 加载了最佳模型权重')
            print(f'  - 从epoch 1 开始，但使用最佳模型权重')
        except Exception as e:
            print(f'⚠️ 加载最佳模型失败: {e}')
            print('  从头开始训练')
    else:
        print(f'\n未找到checkpoint，从头开始训练')
    # 训练循环
    print('\n' + '='*80)
    print('开始训练')
    print('='*80)
    
    patience_counter = 0
    start_time = time()
    
    for epoch in range(start_epoch, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('='*80)
        
        # 训练
        train_loss = train_epoch(model, device, train_loader, optimizer)
        
        # 打印详细的训练损失分解
        print(f'Train Loss: {train_loss["total"]:.6f}')
        print(f'  - Rec: {train_loss.get("rec", 0):.6f}  Con: {train_loss.get("con", 0):.6f}')
        print(f'  - Teacher: {train_loss.get("teacher_clean", 0):.6f}/{train_loss.get("teacher_art", 0):.6f}')
        print(f'  - N2V: {train_loss.get("n2v", 0):.6f}  Band: {train_loss.get("band", 0):.6f}')
        print(f'  - Low: {train_loss.get("low", 0):.6f}  Decor: {train_loss.get("decor", 0):.6f}  Content: {train_loss.get("content", 0):.6f}')
        
        # 验证
        val_loss = validate(model, device, val_loader)
        print(f'\nVal Loss: {val_loss["total"]:.6f}')
        print(f'  - Rec: {val_loss.get("rec", 0):.6f}  Con: {val_loss.get("con", 0):.6f}')
        print(f'  - N2V: {val_loss.get("n2v", 0):.6f}  Band: {val_loss.get("band", 0):.6f}')
        
        # 保存最佳模型（每个epoch都保存训练checkpoint）
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            print(f'\n[*] 验证损失降低: {best_val_loss:.6f}')
            print(f'保存最佳模型到: {MODEL_SAVE_PATH}')
            # 保存最佳模型（只保存模型权重，保持原格式）
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 每个epoch保存完整的训练checkpoint（用于断点续训）
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
        }
        if scheduler is not None:
            checkpoint['scheduler_state_dict'] = scheduler.state_dict()
        
        checkpoint_path = os.path.join(CHECKPOINT_DIR, 'training_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            print(f'\nLearning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # 打印时间
        elapsed = time() - start_time
        print(f'Elapsed Time: {int(elapsed//60)}min {int(elapsed%60)}s')
        
        # 早停
        if patience_counter >= PATIENCE:
            print(f'\n早停触发！{PATIENCE} 个epoch内无改善。')
            break
    
    # 保存最终模型
    print(f'\n保存最终模型到: {FINAL_MODEL_PATH}')
    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    
    print('\n' + '='*80)
    print('训练完成！')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'最佳模型: {MODEL_SAVE_PATH}')
    print(f'最终模型: {FINAL_MODEL_PATH}')
    print('='*80)


if __name__ == '__main__':
    main()
