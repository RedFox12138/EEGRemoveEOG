"""
损失函数包装器，支持消融实验中的条件禁用损失项
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# 导入原始的 unsupervised_artifact_v2 模块
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
v2_dir = os.path.join(parent_dir, 'DAT-Net-Unsupervised-v2')
sys.path.insert(0, v2_dir)

from unsupervised_artifact_v2 import (
    compute_artifact_prob_v2,
    generate_masked_input_artifact_aware,
    _fft_highpass,
    _fft_lowpass
)


def generate_random_block_mask(eeg_input: torch.Tensor, mask_ratio: float = 0.5, block_size: int = 20):
    """
    常规随机块掩蔽策略（严格参考self-supervised）
    优化版本：使用向量化操作替代三重循环，速度提升2-5倍，计算结果完全一致
    
    Args:
        eeg_input: (B, 1, L) 输入EEG信号
        mask_ratio: 掩蔽比例（默认0.5，与self-supervised一致）
        block_size: 每个块的长度（默认20）
    
    Returns:
        masked_input: (B, 1, L) 掩蔽后的输入
        mask: (B, 1, L) 掩蔽位置（True表示被掩蔽）
    """
    batch_size, num_channels, seq_len = eeg_input.shape
    device = eeg_input.device
    
    # 创建全False的mask（与self-supervised一致）
    mask = torch.zeros(batch_size, num_channels, seq_len, device=device, dtype=torch.bool)
    
    # 计算需要掩蔽的总点数和块数
    num_mask_points = int(seq_len * mask_ratio)
    num_blocks = num_mask_points // block_size
    
    if seq_len > block_size and num_blocks > 0:
        # 向量化优化：一次性为所有 batch 和 channel 生成随机起始位置
        # Shape: (batch_size, num_channels, num_blocks)
        start_indices = torch.randint(
            0, seq_len - block_size, 
            (batch_size, num_channels, num_blocks), 
            device=device
        )
        
        # 生成块内偏移量 [0, 1, ..., block_size-1]，shape: (block_size,)
        offsets = torch.arange(block_size, device=device)
        
        # 广播计算所有掩蔽位置的索引
        # start_indices: (B, C, num_blocks, 1)
        # offsets: (block_size,)
        # 结果: (B, C, num_blocks, block_size)
        mask_indices = start_indices.unsqueeze(-1) + offsets
        
        # 使用高级索引设置掩蔽位置
        # 为每个 (b, c) 对展开索引
        batch_idx = torch.arange(batch_size, device=device).view(-1, 1, 1, 1)
        channel_idx = torch.arange(num_channels, device=device).view(1, -1, 1, 1)
        
        # 广播并展平索引
        batch_idx = batch_idx.expand(batch_size, num_channels, num_blocks, block_size).reshape(-1)
        channel_idx = channel_idx.expand(batch_size, num_channels, num_blocks, block_size).reshape(-1)
        mask_indices_flat = mask_indices.reshape(-1)
        
        # 一次性设置所有掩蔽位置
        mask[batch_idx, channel_idx, mask_indices_flat] = True
    
    # 复制输入并掩蔽（使用0填充，与self-supervised一致）
    masked_input = eeg_input.clone()
    masked_input[mask] = 0.0
    
    return masked_input, mask


def unsupervised_dat_loss_ablation(
    model: nn.Module,
    eeg_raw_input: torch.Tensor,
    fs: float,
    ablation_config: dict,
    mask_base: float = 0.1857,
    boost_scale: float = 0.2341,
    lambda_rec: float = 0.8121,
    lambda_con: float = 1.2613,
    lambda_n2v: float = 0.2440,
    lambda_teacher: float = 0.4036,
    lambda_band: float = 0.1,
    lambda_low: float = 0.05,
    lambda_decor: float = 0.1,
    lambda_content: float = 0.05,
    gamma_art_weight: float = 1.0,
    artifact_win_size: int = 64,
    mask_neighborhood: int = 5,
    teacher_cutoff: float = 8.0,
    lowpass_cutoff: float = 4.0,
    teacher_threshold: float = 0.7,
    random_mask_ratio: float = 0.5,
    random_block_size: int = 20,
):
    """
    支持消融实验的无监督损失函数
    根据 ablation_config 选择性地启用或禁用各个损失项
    
    损失分类:
    - 重建类损失: loss_rec, loss_n2v
    - 频域先验类损失: loss_teacher, loss_band
    - 正则化类损失: loss_low, loss_decor, loss_content
    
    Args:
        model: 神经网络模型
        eeg_raw_input: 原始EEG输入 (B, 1, L)
        fs: 采样率
        ablation_config: 消融配置字典
        其他参数与原始函数一致
    
    Returns:
        total_loss: 总损失
        loss_dict: 各项损失的字典
        outputs: (c_A, a_A, c_B, a_B) 模型输出
    """
    device = eeg_raw_input.device
    
    # 从消融配置中读取启用状态
    use_n2v = ablation_config.get('use_n2v', True)
    use_teacher = ablation_config.get('use_teacher', True)
    use_band = ablation_config.get('use_band', True)
    use_regularization = ablation_config.get('use_regularization', True)
    use_random_masking = ablation_config.get('use_random_masking', False)
    
    # ---------- 构造分支输入 ----------
    if use_random_masking:
        # 使用常规随机块掩蔽
        x_A, mask_A = generate_random_block_mask(
            eeg_raw_input, 
            mask_ratio=random_mask_ratio, 
            block_size=random_block_size
        )
    else:
        # 使用artifact-aware掩蔽
        x_A, mask_A = generate_masked_input_artifact_aware(
            eeg_raw_input, fs, 
            mask_base=mask_base, 
            boost_scale=boost_scale, 
            neighborhood=mask_neighborhood
        )
    
    # 分支 B 直接用原始输入
    x_B = eeg_raw_input
    mask_B = torch.zeros_like(mask_A, dtype=mask_A.dtype, device=device)

    # 前向传播
    c_A, a_A = model(x_A)
    c_B, a_B = model(x_B)

    y_A = c_A + a_A
    y_B = c_B + a_B

    # ---------- 损失项 1: 分支 B 的重建损失（简单MSE） ----------
    loss_rec = F.mse_loss(y_B, eeg_raw_input)

    # ---------- 损失项 2: N2V 掩蔽重建（重建类） ----------
    if use_n2v and lambda_n2v > 0.0:
        mask_bool = mask_A.bool()
        if mask_bool.sum() > 0:
            sq_mask = (y_A - eeg_raw_input)[mask_bool] ** 2
            loss_n2v = sq_mask.mean()
        else:
            loss_n2v = torch.tensor(0.0, device=device)
    else:
        loss_n2v = torch.tensor(0.0, device=device)

    # ---------- 损失项 3: Teacher损失（频域先验类） ----------
    if use_teacher and lambda_teacher > 0.0:
        p_art = compute_artifact_prob_v2(eeg_raw_input, fs, win_size=artifact_win_size, lowpass_cutoff=lowpass_cutoff)
        x_hp = _fft_highpass(eeg_raw_input, fs, cutoff=teacher_cutoff)
        x_art = eeg_raw_input - x_hp
        
        mask_teacher = (p_art > teacher_threshold)
        if mask_teacher.sum() > 0:
            loss_teacher_clean = F.mse_loss(c_B[mask_teacher], x_hp[mask_teacher])
            loss_teacher_art = F.mse_loss(a_B[mask_teacher], x_art[mask_teacher])
        else:
            loss_teacher_clean = torch.tensor(0.0, device=device)
            loss_teacher_art = torch.tensor(0.0, device=device)
        loss_teacher = loss_teacher_clean + loss_teacher_art
    else:
        loss_teacher_clean = torch.tensor(0.0, device=device)
        loss_teacher_art = torch.tensor(0.0, device=device)
        loss_teacher = torch.tensor(0.0, device=device)

    # ---------- 损失项 4: 频带先验（频域先验类） ----------
    if use_band and lambda_band > 0.0:
        c_hp = _fft_highpass(c_B, fs, cutoff=teacher_cutoff)
        x_hp_ref = _fft_highpass(eeg_raw_input, fs, cutoff=teacher_cutoff)
        loss_band_clean = F.mse_loss(c_hp, x_hp_ref)
        
        a_hp = _fft_highpass(a_B, fs, cutoff=teacher_cutoff)
        loss_band_art = (a_hp ** 2).mean()
        
        loss_band = loss_band_clean + loss_band_art
    else:
        loss_band = torch.tensor(0.0, device=device)

    # ---------- 损失项 5-7: 正则化类损失 ----------
    if use_regularization:
        # 5. 低频平滑先验
        if lambda_low > 0.0:
            c_diff1 = c_B[:, :, 1:] - c_B[:, :, :-1]
            c_diff2 = c_diff1[:, :, 1:] - c_diff1[:, :, :-1]
            loss_low = (c_diff2 ** 2).mean()
        else:
            loss_low = torch.tensor(0.0, device=device)
        
        # 6. 解耦损失
        if lambda_decor > 0.0:
            reconstruction_error = ((c_B + a_B - eeg_raw_input) ** 2).mean()
            
            c_flat = c_B.view(c_B.size(0), -1)
            a_flat = a_B.view(a_B.size(0), -1)
            c_norm = F.normalize(c_flat, p=2, dim=1)
            a_norm = F.normalize(a_flat, p=2, dim=1)
            similarity = (c_norm * a_norm).sum(dim=1)
            similarity_penalty = F.relu(similarity).mean()
            
            loss_decor = reconstruction_error + 0.1 * similarity_penalty
        else:
            loss_decor = torch.tensor(0.0, device=device)
        
        # 7. 内容保持损失
        if lambda_content > 0.0:
            c_flat = c_B.view(c_B.size(0), -1)
            x_flat = eeg_raw_input.view(eeg_raw_input.size(0), -1)
            c_norm = F.normalize(c_flat, p=2, dim=1)
            x_norm = F.normalize(x_flat, p=2, dim=1)
            cosine_sim = (c_norm * x_norm).sum(dim=1).mean()
            loss_content = 1.0 - cosine_sim
        else:
            loss_content = torch.tensor(0.0, device=device)
    else:
        loss_low = torch.tensor(0.0, device=device)
        loss_decor = torch.tensor(0.0, device=device)
        loss_content = torch.tensor(0.0, device=device)
    # ---------- 总损失（按损失分类累加） ----------
    # 重建类损失（始终保留loss_rec）
    total_loss = lambda_rec * loss_rec
    if use_n2v:
        total_loss += lambda_n2v * loss_n2v
    
    # 频域先验类损失
    if use_teacher:
        total_loss += lambda_teacher * loss_teacher
    if use_band:
        total_loss += lambda_band * loss_band
    
    # 正则化类损失
    if use_regularization:
        total_loss += lambda_low * loss_low
        total_loss += lambda_decor * loss_decor
        total_loss += lambda_content * loss_content

    # ---------- 返回损失字典 ----------
    loss_dict = {
        'total': float(total_loss.detach().cpu().item()),
        'rec': float(loss_rec.detach().cpu().item()),
        'n2v': float(loss_n2v.detach().cpu().item()),
        'teacher_clean': float(loss_teacher_clean.detach().cpu().item()),
        'teacher_art': float(loss_teacher_art.detach().cpu().item()),
        'band': float(loss_band.detach().cpu().item()),
        'low': float(loss_low.detach().cpu().item()),
        'decor': float(loss_decor.detach().cpu().item()),
        'content': float(loss_content.detach().cpu().item()),
    }

    return total_loss, loss_dict, (c_A, a_A, c_B, a_B)


if __name__ == '__main__':
    print('测试消融损失函数...')
    
    # 创建测试数据
    B, L = 2, 512
    fs = 200.0
    x = torch.randn(B, 1, L)
    
    # 创建简单模型
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
        def forward(self, x):
            return x * 0.5, x * 0.5
    
    model = SimpleModel()
    
    # 测试完整配置
    print("\n1. 测试完整配置（所有组件启用）:")
    config_full = {
        'use_n2v': True,
        'use_teacher': True,
        'use_band': True,
        'use_regularization': True,
    }
    loss, loss_dict, _ = unsupervised_dat_loss_ablation(
        model, x, fs, config_full,
        lambda_n2v=0.1, lambda_teacher=0.1, lambda_band=0.1,
        lambda_low=0.05, lambda_decor=0.1, lambda_content=0.05
    )
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  重建类: rec={loss_dict['rec']:.4f}, n2v={loss_dict['n2v']:.4f}")
    print(f"  频域先验类: teacher_clean={loss_dict['teacher_clean']:.4f}, teacher_art={loss_dict['teacher_art']:.4f}, band={loss_dict['band']:.4f}")
    print(f"  正则化类: low={loss_dict['low']:.4f}, decor={loss_dict['decor']:.4f}, content={loss_dict['content']:.4f}")
    
    # 测试仅重建类损失
    print("\n2. 测试仅重建类损失:")
    config_minimal = {
        'use_n2v': True,
        'use_teacher': False,
        'use_band': False,
        'use_regularization': False,
    }
    loss, loss_dict, _ = unsupervised_dat_loss_ablation(
        model, x, fs, config_minimal,
        lambda_n2v=0.1
    )
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  各项损失: {loss_dict}")
    
    # 测试移除频域先验类
    print("\n3. 测试移除频域先验类损失:")
    config_no_freq = {
        'use_n2v': True,
        'use_teacher': False,
        'use_band': False,
        'use_regularization': True,
    }
    loss, loss_dict, _ = unsupervised_dat_loss_ablation(
        model, x, fs, config_no_freq,
        lambda_n2v=0.1, lambda_low=0.05, lambda_decor=0.1, lambda_content=0.05
    )
    print(f"  Total Loss: {loss_dict['total']:.4f}")
    print(f"  各项损失: {loss_dict}")
    
    print("\n测试完成！")
