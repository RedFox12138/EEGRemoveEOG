"""Artifact-aware 无监督 Version 1 实现

包含：
- compute_artifact_prob
- generate_masked_input_artifact_aware
- unsupervised_dat_loss_artifact_v1

该文件只使用 PyTorch，无额外依赖，包含一个简单的本地测试。
"""
from typing import Tuple
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F


def _moving_average(x: torch.Tensor, win: int) -> torch.Tensor:
    """用一维卷积实现滑动平均（same padding）。
    x: (B, 1, L)
    """
    B, C, L = x.shape
    device = x.device
    kernel = torch.ones(1, 1, win, device=device) / float(win)
    pad = win // 2
    x_padded = F.pad(x, (pad, win - pad - 1)) if win % 2 == 0 else F.pad(x, (pad, pad))
    out = F.conv1d(x_padded, kernel)
    return out


def _fft_lowpass(x: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    """基于 FFT 的简单低通滤波（保持原始长度）。
    x: (B, 1, L)
    返回与 x 相同形状的实值张量
    """
    B, C, L = x.shape
    # rfft 频率轴
    Xf = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n=L, d=1.0 / fs, device=x.device)
    mask = (freqs <= cutoff).to(x.dtype)
    Xf = Xf * mask.view(1, 1, -1)
    x_low = torch.fft.irfft(Xf, n=L, dim=-1)
    return x_low


def compute_artifact_prob(x: torch.Tensor, fs: float, win_size: int = 64) -> torch.Tensor:
    """
    输入:
        x: (B, 1, L) 原始单通道 EEG（含伪影）
        fs: 采样率
    输出:
        p_art: (B, 1, L)，每个时间点是伪影的概率，范围 [0, 1]
    """
    eps = 1e-8
    B, C, L = x.shape
    device = x.device

    # 1) 局部幅度: 局部平均绝对值
    amp = _moving_average(torch.abs(x), win_size)

    # 2) 局部变化速度: 平滑后的 |x(t+1)-x(t)| 的窗口平均
    diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
    diff = F.pad(diff, (0, 1))  # 恢复长度
    diff = _moving_average(diff, win_size)

    # 3) 低频能量占比 r(t)
    x_low = _fft_lowpass(x, fs, cutoff=4.0)
    power_low = _moving_average(x_low ** 2, win_size)
    power_total = _moving_average(x ** 2, win_size)
    r = power_low / (power_total + eps)

    # 4) 归一化（用 MAD）
    def mad_normalize(a: torch.Tensor):
        # a: (B,1,L)
        med = a.median(dim=-1, keepdim=True).values
        mad = (a - med).abs().median(dim=-1, keepdim=True).values
        mad = mad.clamp(min=eps)
        return (a - med) / mad

    amp_n = mad_normalize(amp)
    diff_n = mad_normalize(diff)
    r_n = mad_normalize(r)

    # 5) 线性加权得到分数 s(t)
    s = amp_n + diff_n + r_n

    # 6) 非线性映射：减去 70% 分位数阈值，再 sigmoid
    try:
        tau = torch.quantile(s, 0.7, dim=-1, keepdim=True)
    except Exception:
        # 兼容旧版本 torch
        tau_vals = []
        s_np = s.detach().cpu().numpy()
        import numpy as _np

        for i in range(s_np.shape[0]):
            tau_vals.append(_np.quantile(s_np[i, 0, :], 0.7))
        tau = torch.tensor(_np.array(tau_vals), device=device, dtype=s.dtype).view(B, 1, 1)

    alpha = 10.0
    p = torch.sigmoid(alpha * (s - tau))
    p = p.clamp(0.0, 1.0)
    return p


def generate_masked_input_artifact_aware(
    x: torch.Tensor,
    fs: float,
    mask_base: float = 0.1,
    boost_scale: float = 0.3,
    neighborhood: int = 5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    输入:
        x: (B, 1, L) 原始 EEG
        fs: 采样率
        mask_base: 基础掩蔽比例，比如 0.1
        boost_scale: 伪影区域额外增加的掩蔽强度
        neighborhood: 替换值的邻域半径
    输出:
        x_in: (B, 1, L) 掩蔽后的输入
        mask: (B, 1, L) 0/1 掩码，1 表示该点被“挖洞”
    """
    B, C, L = x.shape
    device = x.device

    p_art = compute_artifact_prob(x, fs, win_size=min(64, max(3, L // 8)))
    p_mask = (mask_base + boost_scale * p_art).clamp(0.0, 1.0)

    # Bernoulli 采样
    mask = torch.bernoulli(p_mask).to(device)

    # 如果 neighborhood == 0, 直接置为 0（避免选择自身）
    if neighborhood <= 0:
        x_in = x.clone()
        x_in[mask.bool()] = 0.0
        return x_in, mask

    # 随机偏移，范围 [-neighborhood, neighborhood], 排除 0
    offsets = torch.randint(-neighborhood, neighborhood + 1, (B, L), device=device)
    zeros_idx = (offsets == 0)
    # 将 0 值替换为 +1（或者 -1），保证不选择自身
    offsets[zeros_idx] = 1

    # 构造索引并收集值
    base_idx = torch.arange(L, device=device).view(1, L).expand(B, L)
    gather_idx = (base_idx + offsets).clamp(0, L - 1)  # (B, L)

    x_s = x.view(B, L)
    gathered = torch.gather(x_s, 1, gather_idx)  # (B, L)

    x_in = x_s.clone()
    mask_bool = mask.view(B, L).bool()
    x_in[mask_bool] = gathered[mask_bool]
    x_in = x_in.view(B, 1, L)

    return x_in, mask


def _fft_highpass(x: torch.Tensor, fs: float, cutoff: float) -> torch.Tensor:
    B, C, L = x.shape
    Xf = torch.fft.rfft(x, dim=-1)
    freqs = torch.fft.rfftfreq(n=L, d=1.0 / fs, device=x.device)
    mask = (freqs >= cutoff).to(x.dtype)
    Xf = Xf * mask.view(1, 1, -1)
    x_hp = torch.fft.irfft(Xf, n=L, dim=-1)
    return x_hp


def unsupervised_dat_loss_artifact_v1(
    model: nn.Module,
    eeg_raw_input: torch.Tensor,
    fs: float,
    mask_base: float = 0.1,
    boost_scale: float = 0.3,
    lambda_n2v: float = 1.0,
    lambda_cons: float = 1.0,
    lambda_teacher: float = 1.0,
    lambda_band: float = 0.0,
    lambda_low: float = 0.0,
    lambda_decor: float = 0.0,
    gamma_art_weight: float = 1.0,
):
    """
    Artifact-aware 无监督 Version 1
    返回: total_loss, loss_dict, eeg_clean_pred, eog_artifact_pred
    """
    device = eeg_raw_input.device

    x_in, mask = generate_masked_input_artifact_aware(
        eeg_raw_input, fs, mask_base=mask_base, boost_scale=boost_scale
    )

    eeg_clean_pred, eog_artifact_pred = model(x_in)
    y_pred = eeg_clean_pred + eog_artifact_pred

    p_art = compute_artifact_prob(eeg_raw_input, fs)
    w = 1.0 + gamma_art_weight * p_art

    # N2V 风格重建损失: 仅在 mask==1 上，加权 MSE
    sq = (y_pred - eeg_raw_input) ** 2
    mask_bool = mask.bool()
    if mask_bool.sum() > 0:
        w_mask = w[mask_bool]
        sq_mask = sq[mask_bool]
        loss_n2v = (w_mask * sq_mask).sum() / (w_mask.sum() + 1e-8)
    else:
        loss_n2v = torch.tensor(0.0, device=device)

    # 全局一致性损失
    loss_cons = F.mse_loss(y_pred, eeg_raw_input)

    # 伪老师: high-pass (>=8Hz) 作为近似干净 EEG
    x_hp = _fft_highpass(eeg_raw_input, fs, cutoff=8.0)
    x_art = eeg_raw_input - x_hp

    mask_teacher = (p_art > 0.7)
    if mask_teacher.sum() > 0:
        loss_teacher_clean = F.mse_loss(
            eeg_clean_pred[mask_teacher], x_hp[mask_teacher]
        )
        loss_teacher_art = F.mse_loss(
            eog_artifact_pred[mask_teacher], x_art[mask_teacher]
        )
    else:
        loss_teacher_clean = torch.tensor(0.0, device=device)
        loss_teacher_art = torch.tensor(0.0, device=device)

    # 可选项默认 0
    loss_band = torch.tensor(0.0, device=device)
    loss_low = torch.tensor(0.0, device=device)
    loss_decor = torch.tensor(0.0, device=device)

    total_loss = (
        lambda_n2v * loss_n2v
        + lambda_cons * loss_cons
        + lambda_teacher * (loss_teacher_clean + loss_teacher_art)
        + lambda_band * loss_band
        + lambda_low * loss_low
        + lambda_decor * loss_decor
    )

    loss_dict = {
        "total": float(total_loss.detach().cpu().item()),
        "n2v": float(loss_n2v.detach().cpu().item()),
        "cons": float(loss_cons.detach().cpu().item()),
        "teacher_clean": float(loss_teacher_clean.detach().cpu().item()),
        "teacher_art": float(loss_teacher_art.detach().cpu().item()),
        "band": float(loss_band.detach().cpu().item()),
        "low": float(loss_low.detach().cpu().item()),
        "decor": float(loss_decor.detach().cpu().item()),
    }

    return total_loss, loss_dict, eeg_clean_pred, eog_artifact_pred


if __name__ == "__main__":
    # 简单本地测试，加载 DATNet（在兄弟目录 DAT-Net 中）
    here = os.path.dirname(__file__)
    datnet_dir = os.path.normpath(os.path.join(here, "..", "DAT-Net"))
    sys.path.insert(0, datnet_dir)
    try:
        from model import DATNet
    except Exception:
        # 如果无法导入，请使用一个简单的替代模型进行测试
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x * 0.5, x * 0.5

        DATNet = SimpleModel

    device = torch.device("cpu")
    model = DATNet() if isinstance(DATNet, type) else DATNet
    model = model.to(device)

    B = 2
    L = 512
    fs = 256.0
    x = torch.randn(B, 1, L, device=device)

    total_loss, loss_dict, eeg_clean_pred, eog_artifact_pred = unsupervised_dat_loss_artifact_v1(
        model, x, fs
    )

    print("total_loss:", total_loss.item())
    print("loss_dict:", loss_dict)
