
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import optuna
import numpy as np

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

# --- 全局常量配置 ---
BATCH_SIZE = 256
EPOCHS = 100  # 在调优时减少epoch数量以加快速度
SAMPLING_RATE = 200.0
GRAD_CLIP = 1.0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def objective(trial):
    """
    Optuna 的目标函数，用于评估一组超参数。
    """
    print(f"\n--- 开始 Trial {trial.number} ---")
    
    # --- 1. 定义超参数搜索空间 ---
    # 学习率
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    # 损失权重
    lambda_rec = trial.suggest_float('lambda_rec', 0.5, 5.0)
    lambda_con = trial.suggest_float('lambda_con', 0.5, 3.0)
    lambda_teacher = trial.suggest_float('lambda_teacher', 0.1, 1.0)
    lambda_n2v = trial.suggest_float('lambda_n2v', 0.01, 0.5)
    lambda_band = trial.suggest_float('lambda_band', 0.01, 0.5)
    lambda_low = trial.suggest_float('lambda_low', 0.01, 0.2)
    lambda_decor = trial.suggest_float('lambda_decor', 0.1, 0.8)
    lambda_content = trial.suggest_float('lambda_content', 0.1, 1.0)

    # Artifact-aware 掩蔽参数
    mask_base = trial.suggest_float('mask_base', 0.05, 0.25)
    boost_scale = trial.suggest_float('boost_scale', 0.1, 0.5)

    # --- 2. 设置模型和数据 ---
    train_x, val_x, val_y = get_data()
    train_dataset = UnsupervisedEEGDataset(train_x)
    val_dataset = UnsupervisedEEGDataset(val_x, clean=val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DATNet(in_channels=1, base_channels=32).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # --- 3. 训练和验证循环 ---
    best_val_loss = float('inf')

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
                lambda_decor=lambda_decor, lambda_content=lambda_content
            )
            total_loss_batch.backward()
            if GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # --- 验证 ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                noisy, _, norm = batch if len(batch) == 3 else (batch[0], None, batch[1])
                noisy = noisy.float().unsqueeze(1).to(DEVICE)
                norm = norm.float().to(DEVICE).view(-1, 1, 1)
                noisy_scaled = noisy * norm
                
                val_loss_batch, _, _ = unsupervised_dat_loss_artifact_v2(
                    model=model, eeg_raw_input=noisy_scaled, fs=SAMPLING_RATE,
                    mask_base=mask_base, boost_scale=boost_scale,
                    lambda_rec=lambda_rec, lambda_con=lambda_con, lambda_teacher=lambda_teacher,
                    lambda_n2v=lambda_n2v, lambda_band=lambda_band, lambda_low=lambda_low,
                    lambda_decor=lambda_decor, lambda_content=lambda_content
                )
                val_losses.append(val_loss_batch.item())
        
        current_val_loss = np.mean(val_losses)
        
        # 更新最佳损失
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss

        # Optuna剪枝：如果当前trial表现不佳，提前终止
        trial.report(current_val_loss, epoch)
        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch}.")
            raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} 完成 ---")
    print(f"  - 最佳验证损失: {best_val_loss:.6f}")
    
    return best_val_loss


def main():
    print('='*70)
    print('DAT-Net 无监督训练超参数调优 (Optuna)')
    print(f'使用设备: {DEVICE}')
    print('='*70)

    # 创建一个研究(study)并开始优化
    # storage: 指定数据库URL，用于保存和恢复研究状态
    # study_name: 研究的名称，同一名称的研究会从上次中断的地方继续
    study = optuna.create_study(
        direction='minimize',
        study_name='dat-net-unsupervised-v2-tuning',
        storage='sqlite:///dat-net-tuning.db',  # 使用SQLite数据库保存结果
        load_if_exists=True  # 如果数据库中已存在同名研究，则加载它
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
