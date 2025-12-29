
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import numpy as np
import json
from datetime import datetime

# --- 路径设置 ---
# 添加必要的路径以导入相关模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, current_dir)
sys.path.append(os.path.dirname(os.path.dirname(current_dir)))

# --- 从原始训练脚本导入组件 ---
from model import DATNet
from unsupervised_artifact_v2 import unsupervised_dat_loss_artifact_v2
from train_unsupervised import UnsupervisedEEGDataset, get_data, print_metrics
# 导入评价指标计算函数
sys.path.append(os.path.join(parent_dir, '..'))
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass

# --- 全局常量配置 ---
BATCH_SIZE = 256
EPOCHS = 100  # 在调优时减少epoch数量以加快速度
SAMPLING_RATE = 200.0
GRAD_CLIP = 1.0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 全局变量：记录当前最佳参数
BEST_PARAMS_LOG_FILE = 'best_params_live.json'
global_best_rrmse = float('inf')

def save_best_params(trial_number, rrmse, loss, params):
    """
    实时保存当前最佳参数到日志文件
    """
    best_params = {
        'trial_number': trial_number,
        'best_rrmse': float(rrmse),
        'corresponding_loss': float(loss),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'parameters': {k: float(v) for k, v in params.items()}
    }
    
    with open(BEST_PARAMS_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎯 发现更佳参数！已更新日志: {BEST_PARAMS_LOG_FILE}")
    print(f"   Trial #{trial_number}: RRMSE={rrmse:.6f}, Loss={loss:.6f}")
# 方法1: 限制PyTorch最大显存使用（推荐）
# torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 限制使用60%的GPU显存（约6GB/10GB）

def objective(trial):
    """
    Optuna 的目标函数，用于评估一组超参数。
    """
    global global_best_rrmse
    
    print(f"\n--- 开始 Trial {trial.number} ---")
    
    # --- 1. 定义超参数搜索空间 ---
    # 学习率和优化器参数
    lr = trial.suggest_float('learning_rate', 1e-4, 3e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    
    # 损失权重 - 扩大搜索范围
    lambda_rec = trial.suggest_float('lambda_rec', 0.3, 6.0)
    lambda_con = trial.suggest_float('lambda_con', 0.3, 4.0)
    lambda_teacher = trial.suggest_float('lambda_teacher', 0.05, 1.5)
    lambda_n2v = trial.suggest_float('lambda_n2v', 0.005, 0.8)
    lambda_band = trial.suggest_float('lambda_band', 0.005, 0.8)
    lambda_low = trial.suggest_float('lambda_low', 0.005, 0.3)
    lambda_decor = trial.suggest_float('lambda_decor', 0.05, 1.2)
    lambda_content = trial.suggest_float('lambda_content', 0.05, 1.5)

    # Artifact-aware 掩蔽参数 - 扩大搜索范围
    mask_base = trial.suggest_float('mask_base', 0.03, 0.35)
    boost_scale = trial.suggest_float('boost_scale', 0.05, 0.7)
    gamma_art_weight = trial.suggest_float('gamma_art_weight', 0.3, 3.0)
    
    # compute_artifact_prob 内部参数
    # win_size: 用于计算伪影概率的滑动窗口大小
    artifact_win_size = trial.suggest_int('artifact_win_size', 40, 300, step=16)
    
    # generate_masked_input_artifact_aware 内部参数
    # neighborhood: N2V风格掩蔽时的邻域半径
    mask_neighborhood = trial.suggest_int('mask_neighborhood', 2, 12)
    
    # _fft_highpass/lowpass cutoff 参数 - 扩大搜索范围
    # teacher_cutoff: teacher信号分离的高通滤波截止频率
    teacher_cutoff = trial.suggest_float('teacher_cutoff', 3.0, 15.0)
    
    # lowpass_cutoff: compute_artifact_prob中低频能量计算的截止频率
    lowpass_cutoff = trial.suggest_float('lowpass_cutoff', 1.5, 8.0)
    
    # teacher阈值: 决定哪些区域应用teacher损失 - 扩大搜索范围
    teacher_threshold = trial.suggest_float('teacher_threshold', 0.4, 0.95)

    # --- 2. 设置模型和数据 ---
    train_x, val_x, val_y = get_data()
    train_dataset = UnsupervisedEEGDataset(train_x)
    val_dataset = UnsupervisedEEGDataset(val_x, clean=val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DATNet(in_channels=1, base_channels=40).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 3. 训练和验证循环 ---
    best_val_loss = float('inf')
    best_val_rrmse = float('inf')  # 跟踪最佳RRMSE

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in train_loader:
            noisy, norm = batch
            noisy = noisy.float().unsqueeze(1).to(DEVICE)
            norm = norm.float().to(DEVICE).view(-1, 1, 1)
            noisy_scaled = noisy * norm

            optimizer.zero_grad()
            total_loss_batch, _, _ = unsupervised_dat_loss_artifact_v2(
                model=model, eeg_raw_input=noisy_scaled, fs=SAMPLING_RATE,
                mask_base=mask_base, boost_scale=boost_scale,
                lambda_rec=lambda_rec, lambda_con=lambda_con, lambda_teacher=lambda_teacher,
                lambda_n2v=lambda_n2v, lambda_band=lambda_band, lambda_low=lambda_low,
                lambda_decor=lambda_decor, lambda_content=lambda_content,
                gamma_art_weight=gamma_art_weight,
                artifact_win_size=artifact_win_size,
                mask_neighborhood=mask_neighborhood,
                teacher_cutoff=teacher_cutoff,
                lowpass_cutoff=lowpass_cutoff,
                teacher_threshold=teacher_threshold
            )
            total_loss_batch.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # --- 验证：同时计算loss和RRMSE ---
        model.eval()
        val_losses = []
        all_preds = []
        all_clean = []
        
        with torch.no_grad():
            for batch in val_loader:
                noisy, clean, norm = batch if len(batch) == 3 else (batch[0], None, batch[1])
                noisy = noisy.float().unsqueeze(1).to(DEVICE)
                norm = norm.float().to(DEVICE).view(-1, 1, 1)
                noisy_scaled = noisy * norm
                
                # 计算loss
                val_loss_batch, _, outputs = unsupervised_dat_loss_artifact_v2(
                    model=model, eeg_raw_input=noisy_scaled, fs=SAMPLING_RATE,
                    mask_base=mask_base, boost_scale=boost_scale,
                    lambda_rec=lambda_rec, lambda_con=lambda_con, lambda_teacher=lambda_teacher,
                    lambda_n2v=lambda_n2v, lambda_band=lambda_band, lambda_low=lambda_low,
                    lambda_decor=lambda_decor, lambda_content=lambda_content,
                    gamma_art_weight=gamma_art_weight,
                    artifact_win_size=artifact_win_size,
                    mask_neighborhood=mask_neighborhood,
                    teacher_cutoff=teacher_cutoff,
                    lowpass_cutoff=lowpass_cutoff,
                    teacher_threshold=teacher_threshold
                )
                val_losses.append(val_loss_batch.item())
                
                # 收集预测和真实标签以计算RRMSE
                if clean is not None:
                    c_B = outputs[0]  # 模型输出的清洁信号
                    all_preds.append(c_B.squeeze(1).cpu().numpy())
                    all_clean.append(clean.numpy())
        
        current_val_loss = np.mean(val_losses)
        
        # 计算RRMSE（如果有真实标签）
        current_rrmse = float('inf')
        if len(all_clean) > 0:
            all_preds_np = np.concatenate(all_preds, axis=0)
            all_clean_np = np.concatenate(all_clean, axis=0)
            metrics = compute_all_metrics(all_preds_np, all_clean_np, fs=SAMPLING_RATE)
            current_rrmse = metrics.get('RRMSE', float('inf'))
        
        # 更新最佳RRMSE（主要目标）
        if current_rrmse < best_val_rrmse:
            best_val_rrmse = current_rrmse
        
        # 更新最佳loss（次要监控）
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

        # Optuna剪枝：基于RRMSE进行剪枝
        # 注意：只有在epoch > 30后才开始剪枝，避免过早剪枝
        if epoch > 30 and current_rrmse != float('inf'):
            trial.report(current_rrmse, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}. RRMSE={current_rrmse:.6f}")
                raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} 完成 ---")
    print(f"  - 最佳RRMSE: {best_val_rrmse:.6f}")
    print(f"  - 最佳Loss: {best_val_loss:.6f}")
    
    # 检查是否是全局最佳RRMSE，如果是则保存参数
    if best_val_rrmse < global_best_rrmse:
        global_best_rrmse = best_val_rrmse
        # 保存当前最佳参数到实时日志
        save_best_params(
            trial_number=trial.number,
            rrmse=best_val_rrmse,
            loss=best_val_loss,
            params=trial.params
        )
    
    return best_val_rrmse  # 返回RRMSE作为优化目标


def main():
    print('='*70)
    print('DAT-Net 无监督训练超参数调优 (Optuna)')
    print('优化目标: RRMSE (去噪性能) - 主要目标')
    print('次要监控: Loss (训练稳定性)')
    print(f'使用设备: {DEVICE}')
    print('='*70)

    # 创建一个研究(study)并开始优化
    # 优化目标：最小化RRMSE（去噪效果最好）
    # 使用更宽松的剪枝策略，避免过早终止有潜力的trial
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=15,    # 前15个trial不剪枝，用于建立基准
        n_warmup_steps=40,      # 每个trial前40个epoch不剪枝
        interval_steps=5        # 每5个epoch检查一次是否需要剪枝
    )
    
    study = optuna.create_study(
        direction='minimize',
        study_name='dat-net-unsupervised-v2-tuning-rrmse',
        storage='sqlite:///dat-net-tuning_12_26.db',
        load_if_exists=True,
        pruner=pruner
    )

    # n_trials: 总共要运行的试验次数
    study.optimize(objective, n_trials=100)

    print("\n调优完成!")
    print("最佳 Trial:")
    trial = study.best_trial

    print(f"  > Value (最小验证损失): {trial.value:.6f}")
    print("  > Params: ")
    for key, value in trial.params.items():
        print(f"    - {key}: {value:.4f}")

    # 可视化
    try:
        # 尝试生成并保存可视化图表
        fig = optuna.visualization.plot_optimization_history(study)
        fig.write_image("optuna_history.png")

        fig = optuna.visualization.plot_param_importances(study)
        fig.write_image("optuna_importances.png")
        
        print("\n已生成'optuna_history.png'和'optuna_importances.png'图表。")
        print("需要安装plotly和kaleido: pip install plotly kaleido")
    except Exception as e:
        print(f"\n无法生成可视化图表: {e}")
        print("请确保已安装 'plotly' 和 'kaleido' (pip install plotly kaleido)。")


if __name__ == '__main__':
    main()
