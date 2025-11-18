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

# 导入metrics
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass


# ========== 超参数配置 ==========
BATCH_SIZE = 256
SAMPLING_RATE = 200.0
WEIGHT_DECAY = 1e-5

# ========== 微调阶段：分层学习率 ==========
STAGE2_EPOCHS = 10000
STAGE2_LR_ENCODER = 1e-4     # Encoder慢速微调
STAGE2_LR_BOTTLENECK = 3e-4  # Bottleneck中速
STAGE2_LR_DECODER = 5e-4     # Decoder较快
STAGE2_LR_OUTPUT = 1e-3      # 输出头最快



class SupervisedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # 归一化
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_norm = torch.tensor(noisy.astype('float32') / norm, dtype=torch.float32)
        clean_norm = torch.tensor(clean.astype('float32') / norm, dtype=torch.float32)
        
        return noisy_norm, clean_norm, norm


def get_data():
    """加载20%训练数据和验证数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    
    # 加载完整训练集
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


def get_test_data():
    """加载测试数据"""
    data_dir = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
    test_x = scipy.io.loadmat(f'{data_dir}/Test_Contaminated.mat')['data']
    test_y = scipy.io.loadmat(f'{data_dir}/Test_Pure.mat')['data']
    return test_x, test_y


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


def train_epoch(model, device, loader, optimizer):
    """有监督训练一个epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for noisy, clean, _ in loader:
        noisy = noisy.float().unsqueeze(1).to(device)  # (B, 1, L)
        clean = clean.float().unsqueeze(1).to(device)  # (B, 1, L)
        
        optimizer.zero_grad()
        
        # 前向传播（模型返回 eeg_clean 和 eog_artifact）
        eeg_clean, _ = model(noisy)
        
        # MSE损失（只使用clean分支）
        loss = F.mse_loss(eeg_clean, clean)
        
        loss.backward()
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
            clean_norm = clean.float().unsqueeze(1).to(device)
            
            # 前向传播（模型返回 eeg_clean 和 eog_artifact）
            eeg_clean, _ = model(noisy)
            
            # 计算验证损失（在归一化空间）
            loss = F.mse_loss(eeg_clean, clean_norm)
            total_loss += loss.item()
            num_batches += 1
            
            # 恢复到原始尺度用于计算指标
            output_scaled = eeg_clean.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
            clean_scaled = clean.cpu().numpy() * norm.numpy().reshape(-1, 1)
            
            all_preds.append(output_scaled)
            all_targets.append(clean_scaled)
    
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
            noisy = noisy.float().unsqueeze(1).to(device)
            
            # 前向传播
            eeg_clean, eog_artifact = model(noisy)
            
            # 恢复到原始尺度
            output_scaled = eeg_clean.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
            eog_scaled = eog_artifact.squeeze(1).cpu().numpy() * norm.numpy().reshape(-1, 1)
            clean_scaled = clean.cpu().numpy() * norm.numpy().reshape(-1, 1)
            
            all_preds.append(output_scaled)
            all_eog_preds.append(eog_scaled)
            all_targets.append(clean_scaled)
    
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
    print('\n微调数据集（30%训练集）:', train_x.shape)
    print('验证集:', val_x.shape)

    train_dataset = SupervisedDataset(train_x, train_y)
    val_dataset = SupervisedDataset(val_x, val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型并加载预训练权重
    model = DATNet(in_channels=1, base_channels=40).to(device)

    pretrained_path = 'DAT-Net-Unsupervised-v2_best.pth'
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path, map_location=device))
        print(f'✓ 加载预训练模型: {pretrained_path}')
    else:
        print(f'⚠️  未找到预训练模型: {pretrained_path}')
        print('从头开始微调...')

    total_params = model.count_parameters()
    print(f'\n模型总参数量: {total_params:,}')

    # 分层学习率训练
    print('\n' + '='*70)
    print('【分层学习率训练】')
    print('='*70)

    param_groups = get_layerwise_params_stage2(model)
    optimizer = optim.Adam(param_groups, weight_decay=WEIGHT_DECAY)

    best_val_loss = float('inf')
    best_val_cc = -1.0
    no_improve_count = 0

    for epoch in range(1, STAGE2_EPOCHS + 1):
        train_loss = train_epoch(model, device, train_loader, optimizer)
        val_loss, val_metrics = validate(model, device, val_loader)
        val_cc = val_metrics.get('CC', 0)

        # 每轮都打印（需要密切监控）
        print(f'Epoch {epoch:3d}/{STAGE2_EPOCHS} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | CC: {val_cc:.4f}')

        # 保存最佳模型（基于验证损失）
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_cc = val_cc
            torch.save(model.state_dict(), 'DAT-Net-Unsupervised-v2_finetuned_best.pth')
            print(f'  ✅ 保存最佳模型 (Val Loss: {val_loss:.6f}, CC: {val_cc:.4f})')
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 早停（连续30轮无改善）
        if no_improve_count >= 30:
            print(f'\n验证损失连续{no_improve_count}轮无改善，提前停止训练')
            break

    total_elapsed = time() - time()

    # 保存最终模型
    torch.save(model.state_dict(), 'DAT-Net-Unsupervised-v2_finetuned_final.pth')

    print('\n' + '='*70)
    print('微调完成!')
    print('='*70)
    print(f'总用时: {total_elapsed/60:.2f}分钟')
    print(f'最佳验证损失: {best_val_loss:.6f}')
    print(f'最佳验证CC: {best_val_cc:.4f}')

    print(f'\n保存的模型文件:')
    print(f'  - DAT-Net-Unsupervised-v2_finetuned_best.pth (最佳模型)')
    print(f'  - DAT-Net-Unsupervised-v2_finetuned_final.pth (最终模型)')

    # 加载最佳模型并在测试集上评估
    best_model_path = 'DAT-Net-Unsupervised-v2_finetuned_best.pth'
    if os.path.exists(best_model_path):
        print(f'\n加载最佳模型进行测试集评估: {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        test_on_testset(model, device, 'best')
    else:
        print('\n⚠️  未找到最佳模型，跳过测试集评估')


if __name__ == '__main__':
    main()
