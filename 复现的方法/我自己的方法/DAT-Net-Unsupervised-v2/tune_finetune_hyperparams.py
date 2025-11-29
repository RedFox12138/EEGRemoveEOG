"""
微调超参数调优脚本
使用 Optuna 搜索最优的微调参数配置
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna
from time import time

# 添加路径以导入相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

from model import DATNet

# 导入metrics
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}


# ========== 固定配置 ==========
BATCH_SIZE = 256
SAMPLING_RATE = 200.0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPOCHS = 1000  # 调优时减少epoch数以加快速度
PRETRAINED_PATH = 'DAT-Net-Unsupervised-v2_best.pth'


class SupervisedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / norm, dtype=torch.float32)
        clean_norm = torch.tensor(clean.astype('float32') / norm, dtype=torch.float32)
        
        return noisy_norm, clean_norm, norm


def get_data():
    """加载20%训练数据和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    full_train_x = scipy.io.loadmat(f'{data_dir}/Train_Contaminated.mat')['data']
    full_train_y = scipy.io.loadmat(f'{data_dir}/Train_Pure.mat')['data']
    
    # 取前20%数据
    num_samples = int(len(full_train_x) * 0.2)
    train_x = full_train_x[:num_samples]
    train_y = full_train_y[:num_samples]
    
    # 验证集
    val_x = scipy.io.loadmat(f'{data_dir}/Val_Contaminated.mat')['data']
    val_y = scipy.io.loadmat(f'{data_dir}/Val_Pure.mat')['data']
    
    return train_x, train_y, val_x, val_y


def get_layerwise_params(model, lr_encoder, lr_bottleneck, lr_decoder, lr_output):
    """获取分层学习率参数组"""
    encoder_params = []
    bottleneck_params = []
    decoder_params = []
    output_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'bottleneck' in name:
            bottleneck_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            output_params.append(param)
    
    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': lr_encoder, 'name': 'encoder'})
    if bottleneck_params:
        param_groups.append({'params': bottleneck_params, 'lr': lr_bottleneck, 'name': 'bottleneck'})
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': lr_decoder, 'name': 'decoder'})
    if output_params:
        param_groups.append({'params': output_params, 'lr': lr_output, 'name': 'output'})
    
    return param_groups


def train_and_validate(model, train_loader, val_loader, optimizer, schedulers, 
                       grad_clip, epochs):
    """训练和验证"""
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        train_losses = []
        
        for noisy, clean, norm in train_loader:
            noisy = noisy.float().unsqueeze(1).to(DEVICE)
            clean = clean.float().unsqueeze(1).to(DEVICE)
            norm = norm.float().to(DEVICE).view(-1, 1, 1)
            
            noisy_scaled = noisy * norm
            clean_scaled = clean * norm
            
            optimizer.zero_grad()
            
            eeg_clean, _ = model(noisy_scaled)
            
            # MSE损失（与finetune_adaptive.py保持一致）
            loss = F.mse_loss(eeg_clean, clean_scaled)
            
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            
            train_losses.append(loss.item())
        
        # 学习率衰减
        if schedulers:
            for sch in schedulers:
                sch.step()
        
        # 验证
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            for noisy, clean, norm in val_loader:
                noisy = noisy.float().unsqueeze(1).to(DEVICE)
                clean = clean.float().unsqueeze(1).to(DEVICE)
                norm = norm.float().to(DEVICE).view(-1, 1, 1)
                
                noisy_scaled = noisy * norm
                clean_scaled = clean * norm
                
                eeg_clean, _ = model(noisy_scaled)
                loss = F.mse_loss(eeg_clean, clean_scaled)
                val_losses.append(loss.item())
        
        val_loss = np.mean(val_losses)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss


def objective(trial):
    """Optuna目标函数"""
    print(f"\n{'='*70}")
    print(f"Trial {trial.number}")
    print('='*70)
    
    # ========== 超参数搜索空间 ==========
    # 分层学习率
    lr_encoder = trial.suggest_float('lr_encoder', 1e-5, 1e-3, log=True)
    lr_bottleneck = trial.suggest_float('lr_bottleneck', 1e-4, 5e-3, log=True)
    lr_decoder = trial.suggest_float('lr_decoder', 1e-4, 1e-2, log=True)
    lr_output = trial.suggest_float('lr_output', 1e-4, 5e-3, log=True)
    
    # 优化器参数
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-4, log=True)
    grad_clip = trial.suggest_float('grad_clip', 0.5, 2.0)
    
    # 学习率衰减策略
    use_lr_decay = trial.suggest_categorical('use_lr_decay', [True, False])
    lr_decay_factor = trial.suggest_float('lr_decay_factor', 0.05, 0.15) if use_lr_decay else 0.1
    
    print(f"\n参数配置:")
    print(f"  LR - Encoder: {lr_encoder:.2e}, Bottleneck: {lr_bottleneck:.2e}")
    print(f"  LR - Decoder: {lr_decoder:.2e}, Output: {lr_output:.2e}")
    print(f"  Weight Decay: {weight_decay:.2e}, Grad Clip: {grad_clip:.2f}")
    print(f"  LR Decay: {use_lr_decay}, Factor: {lr_decay_factor:.3f}\n")
    
    # ========== 加载数据 ==========
    train_x, train_y, val_x, val_y = get_data()
    train_dataset = SupervisedDataset(train_x, train_y)
    val_dataset = SupervisedDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # ========== 创建模型 ==========
    model = DATNet(in_channels=1, base_channels=32).to(DEVICE)
    
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=DEVICE))
    else:
        print(f'⚠️ 未找到预训练模型: {PRETRAINED_PATH}')
        return float('inf')
    
    # ========== 设置优化器 ==========
    param_groups = get_layerwise_params(model, lr_encoder, lr_bottleneck, lr_decoder, lr_output)
    optimizer = optim.Adam(param_groups, weight_decay=weight_decay)
    
    # ========== 设置学习率调度器 ==========
    schedulers = []
    if use_lr_decay:
        for i, group in enumerate(optimizer.param_groups):
            init_lr = group['lr']
            min_lr = init_lr * lr_decay_factor
            
            def make_lr_lambda(init_lr, min_lr):
                def lr_lambda(epoch):
                    progress = epoch / EPOCHS
                    cos_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    lr = min_lr + (init_lr - min_lr) * cos_factor
                    return lr / init_lr
                return lr_lambda
            
            schedulers.append(optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=make_lr_lambda(init_lr, min_lr), last_epoch=-1
            ))
    else:
        schedulers = None
    
    # ========== 训练 ==========
    best_val_loss = train_and_validate(
        model, train_loader, val_loader, optimizer, schedulers,
        grad_clip, EPOCHS
    )
    
    print(f"\nTrial {trial.number} 完成 - Best Val Loss: {best_val_loss:.6f}")
    
    return best_val_loss


def main():
    print('='*70)
    print('DAT-Net-Unsupervised-v2 微调超参数调优')
    print('='*70)
    print(f'使用设备: {DEVICE}')
    print(f'预训练模型: {PRETRAINED_PATH}')
    print(f'调优轮数: {EPOCHS} epochs')
    print('='*70)
    
    # 创建Optuna study
    study = optuna.create_study(
        direction='minimize',
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=20)
    )
    
    # 开始优化
    n_trials = 50
    print(f'\n开始超参数搜索 (共 {n_trials} trials)...\n')
    
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    # ========== 输出结果 ==========
    print('\n' + '='*70)
    print('超参数调优完成!')
    print('='*70)
    
    print('\n最佳参数:')
    best_params = study.best_params
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f'  {key}: {value:.6f}')
        else:
            print(f'  {key}: {value}')
    
    print(f'\n最佳验证损失: {study.best_value:.6f}')
    
    # 保存结果
    results_file = 'finetune_best_hyperparams.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write('='*70 + '\n')
        f.write('DAT-Net-Unsupervised-v2 微调最佳超参数\n')
        f.write('='*70 + '\n\n')
        
        f.write('最佳参数:\n')
        for key, value in best_params.items():
            if isinstance(value, float):
                f.write(f'  {key}: {value:.6f}\n')
            else:
                f.write(f'  {key}: {value}\n')
        
        f.write(f'\n最佳验证损失: {study.best_value:.6f}\n')
        
        f.write('\n\n建议配置 (复制到 finetune_adaptive.py):\n')
        f.write('='*70 + '\n')
        f.write(f"STAGE2_LR_ENCODER = {best_params['lr_encoder']:.2e}\n")
        f.write(f"STAGE2_LR_BOTTLENECK = {best_params['lr_bottleneck']:.2e}\n")
        f.write(f"STAGE2_LR_DECODER = {best_params['lr_decoder']:.2e}\n")
        f.write(f"STAGE2_LR_OUTPUT = {best_params['lr_output']:.2e}\n")
        f.write(f"WEIGHT_DECAY = {best_params['weight_decay']:.2e}\n")
        f.write(f"GRAD_CLIP = {best_params['grad_clip']:.2f}\n")
        f.write(f"USE_LR_DECAY = {best_params['use_lr_decay']}\n")
        if best_params['use_lr_decay']:
            f.write(f"LR_DECAY_FACTOR = {best_params['lr_decay_factor']:.3f}  # 最小LR = 初始LR * factor\n")
    
    print(f'\n最佳参数已保存到: {results_file}')
    print('='*70)


if __name__ == '__main__':
    main()
