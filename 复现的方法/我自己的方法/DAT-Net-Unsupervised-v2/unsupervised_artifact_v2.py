"""Artifact-aware 无监督 Version 2 (dual-branch)

实现说明见项目里提出的设计：
- dual-branch 自监督：分支 A 使用 artifact-aware 掩蔽视角，分支 B 使用原始输入
- 分支 B 负责重建原始信号（带伪影加权）
- 对 clean/art 通道在 A/B 之间施加一致性约束
- 支持可选 N2V 重建损失（在掩蔽位置）和伪老师 (high-pass/low-pass)

该文件复用 `unsupervised_artifact_v1.py` 中的工具函数。
"""
from typing import Tuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 确保可以导入已有的工具函数（来自 DAT-Net-Unsupervised 目录）
here = os.path.dirname(__file__)
parent = os.path.normpath(os.path.join(here, '..'))
# 确保能导入同级的 DAT-Net-Unsupervised 中的工具函数
unsup_v1_dir = os.path.normpath(os.path.join(parent, 'DAT-Net-Unsupervised'))
if os.path.isdir(unsup_v1_dir):
    sys.path.insert(0, unsup_v1_dir)
# 仍然保留 parent 以便需要时导入其他模块
sys.path.insert(0, parent)

try:
    from unsupervised_artifact_v1 import (
        compute_artifact_prob,
        generate_masked_input_artifact_aware,
        _fft_highpass,
    )
except Exception as e:
    raise ImportError(f"无法导入 unsupervised_artifact_v1 中的工具函数: {e}")


def unsupervised_dat_loss_artifact_v2(
    model: nn.Module,
    eeg_raw_input: torch.Tensor,
    fs: float,
    mask_base: float = 0.1857,
    boost_scale: float = 0.2341,
    lambda_rec: float = 0.8121,
    lambda_con: float = 1.2613,
    lambda_teacher: float = 0.4036,
    lambda_n2v: float = 0.2440,
    lambda_band: float = 0.0553,
    lambda_low: float = 0.0577,
    lambda_decor: float = 0.3153,
    lambda_content: float = 0.6763,
    gamma_art_weight: float = 1.0,
):
    """
    Artifact-aware 无监督 Version 2（dual-branch）
    - 分支 A/B 使用不同的掩蔽视角（这里 A 掩蔽，B 使用原始输入）
    - 分支 B 负责重建原始 EEG（带伪影加权）
    - A/B 在 clean 和 artifact 通道上做一致性约束
    - 伪影区域有更高的损失权重，并在高置信伪影区域使用 high-pass/low-pass teacher

    返回: total_loss, loss_dict, (c_A, a_A, c_B, a_B)
    """
    device = eeg_raw_input.device

    # ---------- 构造分支输入 ----------
    x_A, mask_A = generate_masked_input_artifact_aware(
        eeg_raw_input, fs, mask_base=mask_base, boost_scale=boost_scale
    )
    # 方案1: 分支 B 直接用原始输入
    x_B = eeg_raw_input
    mask_B = torch.zeros_like(mask_A, dtype=mask_A.dtype, device=device)

    # 前向传播
    c_A, a_A = model(x_A)
    c_B, a_B = model(x_B)

    y_A = c_A + a_A
    y_B = c_B + a_B

    # ---------- 伪影概率与加权 ----------
    p_art = compute_artifact_prob(eeg_raw_input, fs)  # (B,1,L)
    w = 1.0 + gamma_art_weight * p_art

    # ---------- 损失项 1: 分支 B 的重建损失（加权 MSE） ----------
    loss_rec = (w * (y_B - eeg_raw_input) ** 2).sum() / (w.sum() + 1e-8)

    # ---------- 损失项 2: A/B 一致性损失（clean & artifact 通道） ----------
    diff_c = c_A - c_B
    diff_a = a_A - a_B

    # 对一致性损失也可以按伪影概率加权
    weight_con = (1.0 + gamma_art_weight * p_art)
    loss_con_clean = (weight_con * (diff_c ** 2)).mean()
    loss_con_art = (weight_con * (diff_a ** 2)).mean()
    loss_con = loss_con_clean + loss_con_art

    # ---------- 损失项 3: 可选的 N2V 掩蔽重建（在分支 A 的 mask 上） ----------
    if lambda_n2v > 0.0:
        mask_bool = mask_A.bool()
        if mask_bool.sum() > 0:
            w_mask = w[mask_bool]
            sq_mask = (y_A - eeg_raw_input)[mask_bool] ** 2
            loss_n2v = (w_mask * sq_mask).sum() / (w_mask.sum() + 1e-8)
        else:
            loss_n2v = torch.tensor(0.0, device=device)
    else:
        loss_n2v = torch.tensor(0.0, device=device)

    # ---------- 损失项 4: Artifact-aware teacher（high-pass/low-pass） ----------
    x_hp = _fft_highpass(eeg_raw_input, fs, cutoff=8.0)
    x_art = eeg_raw_input - x_hp

    mask_teacher = (p_art > 0.7)
    if mask_teacher.sum() > 0:
        loss_teacher_clean = F.mse_loss(c_B[mask_teacher], x_hp[mask_teacher])
        loss_teacher_art = F.mse_loss(a_B[mask_teacher], x_art[mask_teacher])
    else:
        loss_teacher_clean = torch.tensor(0.0, device=device)
        loss_teacher_art = torch.tensor(0.0, device=device)
    loss_teacher = loss_teacher_clean + loss_teacher_art

    # ---------- 损失项 5: 频带先验与解耦 ----------
    # 5.1 频带先验：clean应该保留高频EEG特征，artifact应该保留低频伪影特征
    if lambda_band > 0.0:
        # Clean通道应该与高通滤波后的信号相似
        c_hp = _fft_highpass(c_B, fs, cutoff=8.0)
        x_hp_ref = _fft_highpass(eeg_raw_input, fs, cutoff=8.0)
        loss_band_clean = F.mse_loss(c_hp, x_hp_ref)
        
        # Artifact通道应该是低频为主
        a_hp = _fft_highpass(a_B, fs, cutoff=8.0)
        loss_band_art = (a_hp ** 2).mean()  # 惩罚artifact的高频成分
        
        loss_band = loss_band_clean + loss_band_art
    else:
        loss_band = torch.tensor(0.0, device=device)
    
    # 5.2 低频平滑先验：clean通道应该相对平滑（温和约束）
    if lambda_low > 0.0:
        # 计算二阶差分（更温和），只惩罚过度的高频噪声
        # 二阶差分能更好地保留有用的高频EEG成分
        c_diff1 = c_B[:, :, 1:] - c_B[:, :, :-1]
        c_diff2 = c_diff1[:, :, 1:] - c_diff1[:, :, :-1]
        loss_low = (c_diff2 ** 2).mean()
    else:
        loss_low = torch.tensor(0.0, device=device)
    
    # 5.3 解耦损失：确保分解的合理性
    if lambda_decor > 0.0:
        # 方法1：确保clean + artifact = input的约束（主要约束）
        reconstruction_error = ((c_B + a_B - eeg_raw_input) ** 2).mean()
        
        # 方法2：温和的正交约束（只在相似度很高时惩罚）
        c_flat = c_B.view(c_B.size(0), -1)
        a_flat = a_B.view(a_B.size(0), -1)
        
        # 归一化
        c_norm = F.normalize(c_flat, p=2, dim=1)
        a_norm = F.normalize(a_flat, p=2, dim=1)
        
        # 计算相似度，使用ReLU只惩罚正相关
        similarity = (c_norm * a_norm).sum(dim=1)
        similarity_penalty = F.relu(similarity).mean()  # 只惩罚正相关
        
        # 重建一致性是主要约束，正交性是辅助约束
        loss_decor = reconstruction_error + 0.1 * similarity_penalty
    else:
        loss_decor = torch.tensor(0.0, device=device)
    
    # 5.4 内容保持损失：防止clean通道过度改变输入信号
    if lambda_content > 0.0:
        # clean通道应该与输入信号保持一定相似度（防止过度去噪）
        # 使用余弦相似度鼓励方向一致
        c_flat = c_B.view(c_B.size(0), -1)
        x_flat = eeg_raw_input.view(eeg_raw_input.size(0), -1)
        
        c_norm = F.normalize(c_flat, p=2, dim=1)
        x_norm = F.normalize(x_flat, p=2, dim=1)
        
        # 余弦相似度，越大越好
        cosine_sim = (c_norm * x_norm).sum(dim=1).mean()
        loss_content = 1.0 - cosine_sim  # 转换为损失（越小越好）
    else:
        loss_content = torch.tensor(0.0, device=device)

    # ---------- 总损失 ----------
    total_loss = (
        lambda_rec * loss_rec
        + lambda_con * loss_con
        + lambda_n2v * loss_n2v
        + lambda_teacher * loss_teacher
        + lambda_band * loss_band
        + lambda_low * loss_low
        + lambda_decor * loss_decor
        + lambda_content * loss_content
    )

    loss_dict = {
        'total': float(total_loss.detach().cpu().item()),
        'rec': float(loss_rec.detach().cpu().item()),
        'con': float(loss_con.detach().cpu().item()),
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
    # 简单自检
    print('运行 unsupervised_artifact_v2 的本地自检...')
    B = 2
    L = 512
    fs = 200.0
    x = torch.randn(B, 1, L)

    # 尝试导入 project 中的 DATNet
    try:
        # 寻找上上级目录的 DAT-Net
        repo_root = os.path.normpath(os.path.join(here, '..'))
        sys.path.insert(0, os.path.normpath(os.path.join(repo_root, '..', 'DAT-Net')))
        from model import DATNet
        model = DATNet(in_channels=1, base_channels=32)
    except Exception:
        # 回退到简单模型
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x * 0.5, x * 0.5
        model = SimpleModel()

    model = model
    total_loss, loss_dict, preds = unsupervised_dat_loss_artifact_v2(
        model, x, fs, mask_base=0.1, boost_scale=0.3, lambda_n2v=0.1
    )

    print('total_loss:', float(total_loss.detach().cpu().item()))
    print('loss_dict:', loss_dict)
