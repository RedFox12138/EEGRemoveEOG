"""
DAT-Net-Unsupervised-v2 微调脚本（分层学习率）
使用20%训练数据进行有监督微调

采用分层学习率策略:

- Encoder: 1e-4 (慢速微调，保留预训练特征)
- Bottleneck: 3e-4 (中速调整)
- Decoder: 5e-4 (较快适应)
- 输出头: 1e-3 (快速优化)
- 轮数: 10000 epochs
- 目的: 全模型精细调整，达到最佳性能
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

# 导入配置
from config import *

# 导入metrics
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass


# ========== 微调超参数 ==========
# BATCH_SIZE, SAMPLING_RATE 等基本配置从 config.py 导入

# ========== 微调阶段：分层学习率 ==========
STAGE2_EPOCHS = 1000
# STAGE2_LR_ENCODER = 1e-4     # Encoder慢速微调
# STAGE2_LR_BOTTLENECK = 3e-3  # Bottleneck中速
# STAGE2_LR_DECODER = 5e-3     # Decoder较快
# STAGE2_LR_OUTPUT = 1e-3      # 输出头最快
# USE_WARMUP = False           # 不使用warmup

WARMUP_EPOCHS = 10           # warmup轮数
PATIENCE = 1000      # 早停耐心值

STAGE2_LR_ENCODER = 0.0005590547219468103
STAGE2_LR_BOTTLENECK =  5.054618790576543e-04
STAGE2_LR_DECODER = 0.0001178199097716924
STAGE2_LR_OUTPUT =0.00010610643578857668
WEIGHT_DECAY = 1.7038831253031843e-04
GRAD_CLIP = 1.933899743786938
USE_LR_DECAY = True



class SupervisedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # Max-Abs归一化（与原始版本一致）
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = (noisy / norm).astype('float32')
        clean_norm = (clean / norm).astype('float32')
        
        return torch.tensor(noisy_norm), torch.tensor(clean_norm), norm

    def __getitem___old(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化（注意：clean也要用noisy的norm来归一化，保持一致）
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / norm, dtype=torch.float32)
        clean_norm = torch.tensor(clean.astype('float32') / norm, dtype=torch.float32)  # clean也要归一化
        
        return noisy_norm, clean_norm, norm


def get_data():
    """加载微调数据和验证数据"""
    print(f"\n加载微调数据集（{int(FINETUNE_RATIO*100)}%训练数据）...")
    
    # 检查是否有预生成的微调数据集（半模拟数据集）
    if os.path.exists(FINETUNE_CONTAMINATED_PATH) and os.path.exists(FINETUNE_PURE_PATH):
        # 使用预先生成的微调数据集（均匀采样自5种SNR）
        print(f"✓ 检测到预生成的微调数据集")
        train_x = scipy.io.loadmat(FINETUNE_CONTAMINATED_PATH)[DATA_KEY]
        train_y = scipy.io.loadmat(FINETUNE_PURE_PATH)[PURE_KEY]
        print(f"✓ 加载微调数据: {train_x.shape}")
        print(f"  来源: 从7种SNR中均匀采样{int(FINETUNE_RATIO*100)}%训练数据")
    else:
        # 回退到旧逻辑：从完整训练集前面取比例数据（全模拟数据集）
        print(f"✓ 未找到预生成的微调数据集，使用传统方式（从完整训练集前面取数据）")
        full_train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
        full_train_y = scipy.io.loadmat(TRAIN_PURE_PATH)[PURE_KEY]
        
        # 取前N%数据
        num_samples = int(len(full_train_x) * FINETUNE_RATIO)
        train_x = full_train_x[:num_samples]
        train_y = full_train_y[:num_samples]
        print(f"✓ 加载微调数据: {train_x.shape}")
        print(f"  来源: 完整训练集的前{int(FINETUNE_RATIO*100)}%")
    
    # 验证集
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    val_y = scipy.io.loadmat(VAL_PURE_PATH)[PURE_KEY]  # 修复：使用PURE_KEY而不是DATA_KEY
    print(f"✓ 加载验证数据: {val_x.shape}")
    
    return train_x, train_y, val_x, val_y


def get_test_data():
    """加载测试数据（仅在单一测试集模式下使用，多SNR模式下跳过）"""
    if TEST_CONTAMINATED_PATH is not None:
        test_x = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]
        test_y = scipy.io.loadmat(TEST_PURE_PATH)[PURE_KEY]  # 修复：使用PURE_KEY
        return test_x, test_y
    else:
        # 多SNR模式，微调过程中不使用测试集
        return None, None


def freeze_backbone(model):
    """
    冻结Backbone（Encoder全部 + Bottleneck）
    只让Decoder和输出头可训练
    
    Returns:
        trainable_params: 可训练参数数量
        frozen_params: 冻结参数数量
    """
    frozen_params = 0
    
    # 冻结所有Encoder层
    for name, param in model.named_parameters():
        if 'encoder' in name or 'bottleneck' in name:
            param.requires_grad = False
            frozen_params += param.numel()
        else:
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return trainable_params, frozen_params


def unfreeze_all(model):
    """解冻所有层"""
    for param in model.parameters():
        param.requires_grad = True


def _strip_module_prefix(state_dict):
    """如果 state_dict 的 key 带有 `module.` 前缀，去掉它（DataParallel 情况）。"""
    new_state = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state[k[len('module.'):]] = v
        else:
            new_state[k] = v
    return new_state


def load_checkpoint_to_model(model, path, device, strict=False):
    """安全地将 checkpoint 加载到 model 中。

    支持 checkpoint 为 dict（包含 'model_state_dict' 或 'state_dict'）或直接为 state_dict。
    自动去掉 DataParallel 的 'module.' 前缀，并以非严格模式加载以避免键名不匹配导致崩溃，
    同时打印缺失/多余键的信息供调试。
    """
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # 可能是直接保存的 state_dict
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # 如果 key 带 module. 前缀则去掉
    if isinstance(state_dict, dict) and any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = _strip_module_prefix(state_dict)

    # 以非严格模式加载并打印不匹配信息
    load_res = model.load_state_dict(state_dict, strict=False)
    missing = getattr(load_res, 'missing_keys', None)
    unexpected = getattr(load_res, 'unexpected_keys', None)
    if missing:
        print(f'⚠️  加载 checkpoint 时缺失的参数 ({len(missing)}): {missing[:5]}{"..." if len(missing)>5 else ""}')
    if unexpected:
        print(f'⚠️  加载 checkpoint 时多余的参数 ({len(unexpected)}): {unexpected[:5]}{"..." if len(unexpected)>5 else ""}')
    print(f'✓ 从 {path} 加载权重（非严格模式）')


def get_layerwise_params_stage2(model):
    """
    获取第二阶段的分层学习率参数组
    
    分层策略:
    - Encoder: 1e-4 (慢速微调，保留预训练特征)
    - Bottleneck: 3e-4 (中速调整)
    - Decoder: 5e-4 (较快适应)
    - 输出头: 1e-3 (快速优化)
    
    Returns:
        param_groups: 参数组列表
    """
    encoder_params = []
    bottleneck_params = []
    decoder_params = []
    output_params = []
    
    # 分类参数
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'encoder' in name:
            encoder_params.append(param)
        elif 'bottleneck' in name:
            bottleneck_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:  # 输出头等
            output_params.append(param)
    
    # 构建参数组
    param_groups = []
    if encoder_params:
        param_groups.append({'params': encoder_params, 'lr': STAGE2_LR_ENCODER, 'name': 'encoder'})
    if bottleneck_params:
        param_groups.append({'params': bottleneck_params, 'lr': STAGE2_LR_BOTTLENECK, 'name': 'bottleneck'})
    if decoder_params:
        param_groups.append({'params': decoder_params, 'lr': STAGE2_LR_DECODER, 'name': 'decoder'})
    if output_params:
        param_groups.append({'params': output_params, 'lr': STAGE2_LR_OUTPUT, 'name': 'output'})
    
    return param_groups


def train_epoch(model, device, loader, optimizer, epoch=1):
    """有监督训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, (noisy, clean, norm) in enumerate(loader):
        noisy = noisy.float().unsqueeze(1).to(device)
        clean = clean.float().unsqueeze(1).to(device)
        norm = norm.float().to(device).view(-1, 1, 1)
        
        # 恢复原始幅度: max-abs反归一化
        noisy_scaled = noisy * norm
        clean_scaled = clean * norm
        
        # 调试：打印第一个batch的数据范围
        if epoch == 1 and batch_idx == 0:
            print(f"\n[调试信息]")
            print(f"  noisy (归一化): min={noisy.min():.4f}, max={noisy.max():.4f}")
            print(f"  noisy_scaled (原始): min={noisy_scaled.min():.4f}, max={noisy_scaled.max():.4f}")
            print(f"  clean (归一化): min={clean.min():.4f}, max={clean.max():.4f}")
            print(f"  clean_scaled (原始): min={clean_scaled.min():.4f}, max={clean_scaled.max():.4f}\n")
        
        optimizer.zero_grad()
        
        # 前向传播（模型返回 eeg_clean 和 eog_artifact）
        eeg_clean, _ = model(noisy_scaled)
        
        # 调试：打印第一个batch的输出范围
        if epoch == 1 and batch_idx == 0:
            print(f"  eeg_clean (输出): min={eeg_clean.min():.4f}, max={eeg_clean.max():.4f}\n")
        
        # MSE损失（只使用clean分支）
        loss = F.mse_loss(eeg_clean, clean_scaled)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)  # 使用调优后的梯度裁剪
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / max(1, num_batches)


def validate(model, device, loader):
    """在验证集上评估"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for noisy, clean, norm in loader:
            noisy = noisy.float().unsqueeze(1).to(device)
            clean = clean.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            
            # 恢复原始幅度: max-abs反归一化
            noisy_scaled = noisy * norm
            clean_scaled = clean * norm
            
            # 前向传播（模型返回 eeg_clean 和 eog_artifact）
            eeg_clean, _ = model(noisy_scaled)
            
            # 计算验证损失（在原始尺度）
            loss = F.mse_loss(eeg_clean, clean_scaled)
            total_loss += loss.item()
            num_batches += 1
            
            # 用于计算指标
            output_scaled = eeg_clean.squeeze(1).cpu().numpy()
            clean_scaled_np = clean_scaled.squeeze(1).cpu().numpy()
            
            all_preds.append(output_scaled)
            all_targets.append(clean_scaled_np)
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评价指标
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    
    avg_loss = total_loss / max(1, num_batches)
    return avg_loss, metrics


def test_on_testset(model, device, model_suffix):
    """在测试集上评估并保存结果"""
    print('\n' + '='*70)
    print('在测试集上评估微调后的模型')
    print('='*70)
    
    # 加载测试数据
    test_x, test_y = get_test_data()
    
    # 如果是多SNR模式，跳过测试集评估
    if test_x is None:
        print('⚠️  多SNR测试集配置，请使用 test_finetuned.py 进行完整测试')
        return
    
    print(f'测试集样本数: {len(test_x)}')
    
    test_dataset = SupervisedDataset(test_x, test_y)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)
    
    # 评估
    model.eval()
    all_preds = []
    all_eog_preds = []
    all_targets = []
    sample_count = 0
    start = time()
    
    with torch.no_grad():
        for noisy, clean, norm in test_loader:
            sample_count += noisy.shape[0]
            
            # ✅ 与训练时保持一致：输入归一化数据，在循环中乘回 norm，然后传入模型
            noisy_t = noisy.float().unsqueeze(1).to(device)  # (B, 1, L) - 归一化的
            norm_t = norm.float().view(-1, 1, 1).to(device)  # (B, 1, 1)
            noisy_scaled = noisy_t * norm_t  # 恢复原始尺度（与训练时一致）
            
            # 前向传播 - 使用原始尺度的数据（与训练时一致）
            eeg_clean, eog_artifact = model(noisy_scaled)
            
            # 模型输出已经是原始尺度，targets也要恢复到原始尺度
            all_preds.append(eeg_clean.squeeze(1).cpu().numpy())
            all_eog_preds.append(eog_artifact.squeeze(1).cpu().numpy())
            # clean是归一化的，需要乘回norm恢复原始尺度
            all_targets.append(clean.cpu().numpy() * norm.cpu().numpy().reshape(-1, 1))
    
    total_time = time() - start
    time_per_sample = total_time / max(1, sample_count)
    
    # 合并结果
    all_preds = np.concatenate(all_preds, axis=0)
    all_eog_preds = np.concatenate(all_eog_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    print(f'推理完成! 单样本时间: {time_per_sample*1000:.3f} ms')
    
    # 计算评价指标
    print('\n计算评价指标...')
    metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    print_metrics(metrics, prefix='测试集')
    
    # 验证解耦一致性
    print('\n验证解耦一致性...')
    reconstructed = all_preds + all_eog_preds
    original = test_x
    consistency_error = np.mean((reconstructed - original) ** 2)
    print(f'重建一致性MSE: {consistency_error:.6f}')
    
    # 保存结果
    out_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, f'DAT-Net-Unsupervised-v2_finetuned_{model_suffix}_predictions.mat')
    scipy.io.savemat(save_path, {
        'predictions': all_preds,
        'eog_artifacts': all_eog_preds,
        'time_per_sample': time_per_sample,
    })
    print(f'\n预测结果已保存: {save_path}')
    print('='*70)


def main():
    print('='*70)
    print('DAT-Net-Unsupervised-v2 微调（分层学习率）')
    print('='*70)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)

    # 加载数据
    train_x, train_y, val_x, val_y = get_data()
    print('\n微调数据集:', train_x.shape)
    print('验证集:', val_x.shape)

    train_dataset = SupervisedDataset(train_x, train_y)
    val_dataset = SupervisedDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型
    model = DATNet(in_channels=1, base_channels=32).to(device)

    # 优先加载上次的最佳微调模型，否则加载预训练模型
    best_finetune_path = 'DAT-Net-Unsupervised-v2_finetuned_best_30%鏁版嵁.pth'
    pretrained_path = 'checkpoints/datnet_unsupervised_v2_semi_simulated_best_rrmse_1.pth'

    if os.path.exists(best_finetune_path):
        load_checkpoint_to_model(model, best_finetune_path, device, strict=False)
        print(f'✓ 从上次最佳微调模型继续: {best_finetune_path}')
    elif os.path.exists(pretrained_path):
        load_checkpoint_to_model(model, pretrained_path, device, strict=False)
        print(f'✓ 加载预训练模型: {pretrained_path}')
    else:
        print(f'⚠️  未找到最佳微调模型和预训练模型')
        print('从头开始微调...')

    total_params = model.count_parameters()
    print(f'\n模型总参数量: {total_params:,}')

    # 分层学习率训练
    print('\n' + '='*70)
    print('【分层学习率训练】')
    print('='*70)

    param_groups = get_layerwise_params_stage2(model)
    optimizer = optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)

    # 根据调优结果决定是否使用学习率衰减
    schedulers = []
    if USE_LR_DECAY:
        # 为每个参数组设置独立的学习率调度器（余弦退火，最低为初始值的十分之一）
        for i, group in enumerate(optimizer.param_groups):
            init_lr = group['lr']
            min_lr = init_lr / 10
            def make_lr_lambda(init_lr, min_lr):
                def lr_lambda(epoch):
                    # 余弦退火到 min_lr
                    progress = epoch / STAGE2_EPOCHS
                    cos_factor = 0.5 * (1 + np.cos(np.pi * progress))
                    lr = min_lr + (init_lr - min_lr) * cos_factor
                    return lr / init_lr
                return lr_lambda
            schedulers.append(optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=make_lr_lambda(init_lr, min_lr), last_epoch=-1))
    else:
        schedulers = None

    best_val_loss = float('inf')
    best_val_rrmse = float('inf')  # RRMSE越小越好
    no_improve_count = 0
    start_time = time()

    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer, epoch)
        val_loss, val_metrics = validate(model, device, val_loader)
        val_rrmse = val_metrics.get('RRMSE', float('inf'))  # 获取RRMSE指标

        # 如果使用学习率衰减，则更新学习率
        if schedulers is not None:
            for sch in schedulers:
                sch.step()

        # 打印所有参数组的学习率
        lr_str = ' | '.join([f"{group['name']}: {group['lr']:.2e}" for group in optimizer.param_groups])
        print(f'Epoch {epoch:3d}/{STAGE2_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | RRMSE: {val_rrmse:.4f} | LR: {lr_str}')

        # 保存最佳模型（基于RRMSE）
        if val_rrmse < best_val_rrmse:
            best_val_rrmse = val_rrmse
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'DAT-Net-Unsupervised-v2_finetuned_best_30%数据.pth')
            print(f'  ✅ 保存最佳模型 (RRMSE: {val_rrmse:.4f}, Val Loss: {val_loss:.6f})')
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 早停
        if no_improve_count >= PATIENCE:
            print(f'\nRRMSE连续{no_improve_count}轮无改善，提前停止训练')
            break

    total_elapsed = time() - start_time

    # 保存最终模型
    torch.save(model.state_dict(), 'DAT-Net-Unsupervised-v2_finetuned_final.pth')

    print('\n' + '='*70)
    print('微调完成!')
    print('='*70)
    print(f'总用时: {total_elapsed/60:.2f}分钟')
    print(f'最佳验证RRMSE: {best_val_rrmse:.4f}')
    print(f'对应验证损失: {best_val_loss:.6f}')

    print(f'\n保存的模型文件:')
    print(f'  - DAT-Net-Unsupervised-v2_finetuned_best.pth (最佳模型)')
    print(f'  - DAT-Net-Unsupervised-v2_finetuned_final.pth (最终模型)')

    # 加载最佳模型并在测试集上评估
    best_model_path = 'DAT-Net-Unsupervised-v2_finetuned_best_30%数据.pth'
    if os.path.exists(best_model_path):
        print(f'\n加载最佳模型进行测试集评估: {best_model_path}')
        load_checkpoint_to_model(model, best_model_path, device, strict=False)
        test_on_testset(model, device, 'best')
    else:
        print('\n⚠️  未找到最佳模型，跳过测试集评估')


if __name__ == '__main__':
    main()
