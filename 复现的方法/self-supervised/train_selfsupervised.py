"""
Self-Supervised EEG Denoising 训练脚本
基于 Self-Supervised-EEG-Denoising-main 的模型架构
使用半模拟数据集进行训练
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

# 添加路径以导入metrics
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # 复现的方法
sys.path.insert(0, parent_dir)
sys.path.insert(0, current_dir)

# 导入模型和工具
from model_selfsupervised import DenoiseEEG
from utils_selfsupervised import get_pearson_correlation, get_snr, trrmse_metric, frrmse_metric

# 导入metrics
from metrics_utils import compute_all_metrics, print_metrics

# 导入数据配置
from data_config import *


# ========== 超参数配置（与原始Self-Supervised一致）==========
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-4  # 修正：源码默认为 1e-3，此前为 1e-4 可能导致收敛慢
MASK_RATIO = 0.4  # 增大掩码比例，强迫模型更多地进行预测而非复制

# 我们的数据尺寸（从 data_config 中获取，适配不同数据集）
INPUT_CHANNELS = 1  # 单通道EEG
SEQ_LEN = WINDOW_SIZE  # 序列长度（从配置文件获取，自动适配）
HIDDEN_DIM = 128    # 隐藏层维度


# ==========================================
# [Helper Functions] DAT-Net Artifact Estimation
# ==========================================
def _moving_average(x, w):
    pad = w // 2
    x_padded = F.pad(x, (pad, pad), mode='reflect')
    # 截断以匹配原始长度
    if x_padded.shape[-1] > x.shape[-1]:
       x_padded = x_padded[..., :x.shape[-1]] 
    
    # 简单的卷积实现移动平均
    kernel = torch.ones(1, 1, w, device=x.device, dtype=x.dtype) / w
    # Reshape x 用于 conv1d: (B*C, 1, L)
    B, C, L = x.shape
    x_reshaped = x.reshape(-1, 1, L)
    
    out = F.conv1d(x_padded.reshape(-1, 1, x_padded.shape[-1]), kernel, padding=0)
    
    # 确保输出尺寸一致
    if out.shape[-1] != L:
        out = F.interpolate(out, size=L, mode='linear', align_corners=False)
        
    return out.view(B, C, L)

def _fft_lowpass(x, fs, cutoff=4.0):
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    mask = (freqs <= cutoff).float()
    
    xf = torch.fft.rfft(x, dim=-1)
    xf_filtered = xf * mask
    x_low = torch.fft.irfft(xf_filtered, n=n, dim=-1)
    return x_low

def _fft_highpass(x, fs, cutoff=8.0):
    n = x.shape[-1]
    freqs = torch.fft.rfftfreq(n, d=1/fs).to(x.device)
    mask = (freqs >= cutoff).float()
    
    xf = torch.fft.rfft(x, dim=-1)
    xf_filtered = xf * mask
    x_high = torch.fft.irfft(xf_filtered, n=n, dim=-1)
    return x_high

def compute_artifact_prob_simple(x, fs=SAMPLING_RATE):
    """简化的 artifacts 概率估计 (基于幅度、变化率、低频能量)"""
    B, C, L = x.shape
    eps = 1e-8
    
    # 1. 局部幅度（EOG通常幅度大）
    amp = _moving_average(torch.abs(x), 64)
    
    # 2. 低频能量占比（EOG集中在 <4Hz）
    x_low = _fft_lowpass(x, fs, cutoff=4.0)
    p_low = _moving_average(x_low**2, 64)
    p_total = _moving_average(x**2, 64) + eps
    ratio_low = p_low / p_total
    
    # 3. 归一化并组合
    def normalize(t):
        return (t - t.min()) / (t.max() - t.min() + eps)
    
    score = normalize(amp) + normalize(ratio_low)
    
    # Sigmoid 映射到概率 [0, 1]
    # 经验阈值：score > mean 可能是伪影
    prob = torch.sigmoid(5 * (score - score.mean()))
    return prob


class SelfSupervisedDataset(Dataset):
    """
    自监督训练数据集
    只需要受污染的EEG信号，不需要干净标签
    """
    def __init__(self, noisy_data):
        self.noisy = noisy_data

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        
        # 归一化（与DAT-Net-v2一致）
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        
        noisy_normalized = noisy.astype('float32') / norm
        
        return torch.tensor(noisy_normalized, dtype=torch.float32), norm


class SupervisedDataset(Dataset):
    """
    监督验证数据集
    用于评估模型去伪影效果
    """
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


def mask_input(data, mask_ratio=0.5, block_size=20):
    """
    块掩码 (Block Masking) 策略
    不只是随机遮挡单个点，而是遮挡连续的时间片段。
    这能防止模型简单地通过相邻点插值来重构低频伪影(EOG)。
    
    Args:
        data: (B, C, L)
        mask_ratio: 总体掩码比例
        block_size: 每个块的长度 (点数)
    """
    batch_size, num_channels, seq_len = data.shape
    
    # 创建一个全0的mask
    mask = torch.zeros(batch_size, num_channels, seq_len, device=data.device, dtype=torch.bool)
    
    # 计算需要掩码的总点数
    num_mask_points = int(seq_len * mask_ratio)
    # 计算需要多少个块
    num_blocks = num_mask_points // block_size
    
    for b in range(batch_size):
        for c in range(num_channels):
            # 随机选择块的起始位置
            start_indices = torch.randint(0, seq_len - block_size, (num_blocks,))
            for start_idx in start_indices:
                mask[b, c, start_idx : start_idx + block_size] = True
                
    masked_data = data.clone()
    
    # 使用 0 填充 (对于块掩码，0填充通常比噪声填充更难，迫使模型生成结构)
    # 或者使用均值填充
    masked_data[mask] = 0
    
    return masked_data, mask


def mae_loss(reconstructed, original, mask):
    """
    掩码自编码器损失（修正版）
    只计算被掩码位置的损失，强迫模型利用上下文信息进行预测
    """
    reconstructed = reconstructed.to(mask.device)
    original = original.to(mask.device)
    
    # 只计算 mask 为 True (被遮挡) 的部分的 loss
    # mask 是 bool 类型，可以直接作为索引
    loss = F.mse_loss(reconstructed[mask], original[mask])
    return loss


def train_epoch(model, device, loader, optimizer, mask_ratio):
    """
    一致性学习训练（Consistency Learning）
    基于论文源代码的逻辑：
    1. 随机掩码输入（设置随机点为0）
    2. 两次前向传播：raw_out = model(raw_input), masked_out = model(masked_input)
    3. 损失 = MSE(raw_out, masked_out) + MSE(raw_out, raw_input)
    """
    model.train()
    total_loss = 0

    for data, norm in loader:
        data = data.to(device).float()
        if len(data.shape) == 2:
            data = data.unsqueeze(1)

        optimizer.zero_grad()

        # [改进1] 数据增强：随机垂直翻转
        # EEG 信号关于0点对称，翻转不改变信号性质，但能增加数据多样性
        if torch.rand(1).item() < 0.5:
            data = -data

        # [改进Retained] Block Masking (比例提升至 0.30)
        # 进一步提升掩码强度，防止模型学习恒等映射 (Identity Mapping)。
        batch_size, channels, seq_len = data.shape
        mask = torch.zeros(batch_size, channels, seq_len, device=device)
        
        target_ratio = 0.30
        # 增加Block尺寸以覆盖完整的EOG伪影 (约0.2s - 0.6s)
        current_block_size = torch.randint(60, 160, (1,)).item()
        
        num_blocks = int((seq_len * target_ratio) / current_block_size)
        if num_blocks < 1: num_blocks = 1
            
        for b in range(batch_size):
            for c in range(channels):
                # 随机生成块的起始位置
                starts = torch.randint(0, seq_len - current_block_size, (num_blocks,), device=device)
                for start in starts:
                    mask[b, c, start : start + current_block_size] = 1.0
        
        mask = mask > 0.5
        masked_data = data.clone()
        masked_data[mask] = 0

        # 两次前向传播
        raw_out = model(data)           
        masked_out = model(masked_data) 

        # 恢复原始尺度
        norm_scale = norm.to(device).view(-1, 1, 1)
        raw_out_denorm = raw_out * norm_scale
        masked_out_denorm = masked_out * norm_scale
        data_denorm = data * norm_scale
        
        # [Revised Strategy v3 - Aggressive Denoising] 
        # 用户需求：效果再强一些。
        # 策略：进一步降低对伪影区域输入的信任，大幅提升对一致性的依赖。
        
        # 1. 计算伪影概率用于加权
        p_art = compute_artifact_prob_simple(data_denorm, fs=SAMPLING_RATE).detach()
        
        # 2. 计算 SNR (用于调节全局权重)
        n = seq_len
        fft_vals = torch.fft.rfft(data_denorm, dim=-1)
        psd = torch.abs(fft_vals) ** 2
        freqs = torch.fft.rfftfreq(n, d=1/SAMPLING_RATE).to(device)
        
        mask_sig = (freqs >= 8) & (freqs <= 30)
        mask_noi = ((freqs >= 1) & (freqs <= 4)) | (freqs > 30)
        
        p_sig = torch.sum(psd[..., mask_sig], dim=-1)
        p_noi = torch.sum(psd[..., mask_noi], dim=-1) + 1e-6
        
        batch_snr = 10 * torch.log10(p_sig / p_noi + 1e-6).mean().item()
        
        # Map SNR to Lambda
        adaptive_lambda = -0.09 * batch_snr + 1.1
        adaptive_lambda = max(0.2, min(2.5, adaptive_lambda))

        # 3. 加权 Loss 计算
        
        # A. Reconstruction Loss (Aggressive Weighting)
        # 在检测到伪影的区域，权重由 1.0 降至 0.1 (系数 0.9)。
        # 这意味着模型在眨眼处几乎不需要去拟合输入信号，避免了 "学会眨眼"。
        rec_weight = 1.0 - 0.9 * p_art
        loss_rec = (rec_weight * (raw_out_denorm - data_denorm) ** 2).mean()
        
        # B. Consistency Loss (Super Boosted)
        # 在伪影区域，一致性权重增加 2.0 倍 (即总权重 3.0)。
        # 逻辑：既然输入不可信（有伪影），那就完全依赖 "Masked Branch" 的预测结果。
        # 而 Masked Branch 的输入遮挡了伪影，所以它预测的是单纯的 EEG 上下文。
        con_weight = 1.0 + 2.0 * p_art
        loss_con = (con_weight * (raw_out_denorm - masked_out_denorm) ** 2).mean()
        
        # 4. 辅助 Loss
        # 频域 & 差分
        fft_raw = torch.fft.rfft(raw_out_denorm, dim=-1, norm='ortho')
        fft_masked = torch.fft.rfft(masked_out_denorm, dim=-1, norm='ortho')
        loss_freq = F.mse_loss(torch.abs(fft_raw), torch.abs(fft_masked))
        
        diff_raw = raw_out_denorm[..., 1:] - raw_out_denorm[..., :-1]
        diff_masked = masked_out_denorm[..., 1:] - masked_out_denorm[..., :-1]
        loss_diff = F.mse_loss(diff_raw, diff_masked)

        # 总损失
        # 提升 diff loss 权重 (0.5 -> 1.0)，进一步抑制高频残留与毛刺
        loss = loss_rec + (adaptive_lambda * loss_con) + (0.2 * loss_freq) + (1.0 * loss_diff)

        total_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    return total_loss / len(loader)


def validate(model, device, loader, compute_full_metrics=True):
    """
    验证模型性能
    
    Args:
        compute_full_metrics: 是否计算所有指标（包括耗时的PSD等）。
                              设为False时只计算 Loss, RRMSE, CC。
    """
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad(): 
        for data, clean, norm in loader:
            # 添加通道维度
            if len(data.shape) == 2:
                data = data.unsqueeze(1)
            if len(clean.shape) == 2:
                clean = clean.unsqueeze(1)
            
            x, y = data.to(device), clean.to(device)
            norm_scale = norm.float().to(device).view(-1, 1, 1)

            # 前向传播 (输入归一化数据)
            reconstructed = model(x)
            
            # 反归一化到原始尺度 (Raw Scale)
            reconstructed_denorm = reconstructed * norm_scale
            y_denorm = y * norm_scale

            # 计算验证 Loss (在原始尺度计算)
            # 1. 去噪 Loss (与 Clean 信号对比)
            loss_denoise = F.mse_loss(reconstructed_denorm, y_denorm)
            
            total_loss += loss_denoise.item()

            # 收集预测和目标（用于计算完整指标）
            all_preds.append(reconstructed_denorm.squeeze(1).cpu().numpy())
            all_targets.append(y_denorm.squeeze(1).cpu().numpy())
    
    # 合并所有批次
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 计算评估指标
    if compute_full_metrics:
        # 计算所有指标（可能耗时）
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)
    else:
        # 只计算快速指标 (RRMSE, CC) 以节省时间
        # 手动快速计算，跳过频谱分析
        from metrics_utils import compute_rrmse, compute_cc
        rrmse = compute_rrmse(all_targets, all_preds)
        cc = compute_cc(all_targets, all_preds)
        metrics = {'RRMSE': rrmse, 'CC': cc}
    
    num_batches = len(loader)
    avg_loss = total_loss / num_batches
    
    return avg_loss, metrics


def get_data():
    """
    加载数据（与DAT-Net-v2一致）

    """
    # 训练集只需要污染数据（自监督）
    train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    
    # 验证集需要干净标签（用于评估）
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    val_y = scipy.io.loadmat(VAL_PURE_PATH)[PURE_KEY]
    
    return train_x, val_x, val_y


def main():
    print('='*70)
    print('Self-Supervised EEG Denoising 训练')
    print('='*70)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('使用设备:', device)
    
    # 加载数据
    print('\n加载数据...')
    train_x, val_x, val_y = get_data()
    print(f'训练集样本数: {len(train_x)}')
    print(f'验证集样本数: {len(val_x)}')
    print(f'数据维度: {train_x.shape}')
    
    # 创建数据集和加载器
    train_dataset = SelfSupervisedDataset(train_x)
    val_dataset = SupervisedDataset(val_x, val_y)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 创建模型
    print('\n创建模型...')
    model = DenoiseEEG(
        in_channels=INPUT_CHANNELS,
        length=SEQ_LEN,
        n_feat=HIDDEN_DIM
    ).to(device)
    
    print(f'模型参数量: {model.count_parameters():,}')
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # 学习率调度器：余弦退火
    # T_max=EPOCHS: 周期长度
    # eta_min=1e-6: 最小学习率
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    
    # 训练循环
    best_val_loss = float('inf')
    best_rrmse = float('inf')
    best_metrics = {}
    patience = 100
    patience_counter = 0
    
    # 自动加载已有的best模型继续训练
    best_model_path = 'Self-Supervised_best_loss.pth'
    if os.path.exists(best_model_path):
        print(f"\n发现已有模型: {best_model_path}")
        try:
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            print(f"✓ 成功加载模型，将从已有最佳模型继续训练")
        except Exception as e:
            print(f"⚠ 加载模型失败: {e}")
            print("将从头开始训练")
    else:
        print(f"\n未找到已有模型: {best_model_path}")
        print("将从头开始训练")
    
    start_training_time = time()
    
    print('\n开始训练...')
    print(f'训练轮数: {EPOCHS}')
    print(f'批次大小: {BATCH_SIZE}')
    print(f'学习率: {LEARNING_RATE}')
    print(f'掩码比例: {MASK_RATIO}')
    print('='*70)
    
    for epoch in range(1, EPOCHS + 1):
        epoch_start = time()
        
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('-'*70)
        
        # 训练
        avg_loss = train_epoch(model, device, train_loader, optimizer, MASK_RATIO)
        
        # 更新学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        
        print(f'Train Loss: {avg_loss:.6f} | LR: {current_lr:.2e}')
        
        # 验证
        # 优化：每个epoch只进行轻量级验证，避免耗时的PSD计算
        val_loss, val_metrics = validate(model, device, val_loader, compute_full_metrics=(epoch % 5 == 0 or epoch == EPOCHS))
        
        print(f'Val Loss:   {val_loss:.6f}')
        
        # 打印详细的验证指标
        if val_metrics:
            rrmse = val_metrics.get('RRMSE', 0.0)
            cc = val_metrics.get('CC', 0.0)
            
            # 只有在计算了完整指标时才打印完整信息
            if epoch % 5 == 0 or epoch == EPOCHS:
                snr = val_metrics.get('SNR', 0.0)
                prd = val_metrics.get('PRD', 0.0)
                print(f'  RRMSE: {rrmse:.6f}  CC: {cc:.6f}')
                print(f'  SNR:   {snr:.4f} dB  PRD: {prd:.6f}')
            else:
                print(f'  RRMSE: {rrmse:.6f}  CC: {cc:.6f} (Lite Check)')
            
            # 更新最佳模型（基于 Validation Loss）
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), 'Self-Supervised_best_loss.pth')
                print(f'  ✓ 保存最佳 Loss 模型! (Val Loss: {val_loss:.6f})')
            else:
                patience_counter += 1

            # 更新最佳模型（基于 RRMSE）
            if rrmse < best_rrmse:
                best_rrmse = rrmse
                best_metrics = val_metrics.copy()
                torch.save(model.state_dict(), 'Self-Supervised_best_rrmse.pth')
                print(f'  ✓ 保存最佳 RRMSE 模型! (RRMSE: {rrmse:.6f})')
        
        # 每轮保存当前模型 (覆盖旧文件)
        torch.save(model.state_dict(), 'Self-Supervised_current.pth')

        epoch_time = time() - epoch_start
        elapsed = time() - start_training_time
        print(f'Epoch Time: {epoch_time:.1f}s  |  Elapsed: {int(elapsed//60)}min {int(elapsed%60)}s')
        
        # Early stopping
        if patience_counter >= patience:
            print(f'\n早停触发！{patience} 个epoch内无改善。')
            break
    
    # 保存最终模型
    torch.save(model.state_dict(), 'Self-Supervised_final.pth')
    
    print('\n' + '='*70)
    print('训练完成!')
    if best_metrics:
        print('\n最佳验证指标:')
        print_metrics(best_metrics, prefix='  ')
    print('='*70)


if __name__ == '__main__':
    main()