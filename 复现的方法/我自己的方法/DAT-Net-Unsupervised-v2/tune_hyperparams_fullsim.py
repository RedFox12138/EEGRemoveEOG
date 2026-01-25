
import os
import sys
# 在导入任何本地配置前，设置环境变量以选择数据集
os.environ['DATNET_DATASET_NAME'] = 'fully_simulated'

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
from train_unsupervised import UnsupervisedEEGDataset, print_metrics

# 导入配置（包含数据集相关配置）
from config import SAMPLING_RATE, DATASET_NAME, WINDOW_SIZE, DATA_KEY, PURE_KEY, TEST_SNR_LEVELS, TEST_SNR_PATHS

# 导入评价指标计算函数
sys.path.append(os.path.join(parent_dir, '..'))
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass

# ========== 数据集选择 ==========
# 数据集通过顶部的环境变量 os.environ['DATNET_DATASET_NAME'] 自动控制
# 当前模式: fully_simulated (全模拟数据集)
# ================================

# --- 全局常量配置 ---
BATCH_SIZE = 128
EPOCHS = 30# 在调优时减少epoch数量以加快速度
# SAMPLING_RATE 已从 config.py 导入，会根据数据集自动适配
GRAD_CLIP = 1.0
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 预训练模型路径配置
# 根据数据集类型自动选择对应的最佳RRMSE模型
if DATASET_NAME == 'fully_simulated':
    PRETRAINED_MODEL_PATH = 'checkpoints/datnet_unsupervised_v2_fully_simulated_best_rrmse.pth'
else:
    PRETRAINED_MODEL_PATH = 'checkpoints/datnet_unsupervised_v2_fully_simulated_best_rrmse.pth'

print(f"将基于预训练模型进行参数调优: {PRETRAINED_MODEL_PATH}")

# 全局变量：记录当前最佳参数
BEST_PARAMS_LOG_FILE = 'best_params_live_fully_simulated.json'
BEST_MODEL_CHECKPOINT = 'checkpoints/tune_best_model_fully_simulated.pth'  # 保存调优过程中的最佳模型
# 存储最佳指标: {'avg_rrmse': float, 'per_snr': {snr: rrmse}}
global_best_metrics = {'avg_rrmse': float('inf'), 'per_snr': {}}


def load_previous_best_params():
    """
    加载上一次的最佳参数和模型checkpoint作为调优基准
    """
    if not os.path.exists(BEST_PARAMS_LOG_FILE):
        print(f"⚠️  未找到上一次的最佳参数文件: {BEST_PARAMS_LOG_FILE}")
        print("将从预训练模型开始搜索...")
        return None
    
    try:
        with open(BEST_PARAMS_LOG_FILE, 'r', encoding='utf-8') as f:
            best_params_data = json.load(f)
        
        previous_params = best_params_data.get('parameters', {})
        # 兼容旧格式（只有best_rrmse）和新格式（包含per_snr_rrmse）
        rrmse = best_params_data.get('best_rrmse', float('inf'))
        per_snr = best_params_data.get('per_snr_rrmse', {})
        
        # 如果没有per_snr数据，将在后续通过checkpoint重新评估获取
        metrics = {'avg_rrmse': rrmse, 'per_snr': per_snr}
        
        previous_checkpoint = best_params_data.get('checkpoint_path', None)
        
        print(f"✅ 成功加载上一次的最佳参数:")
        print(f"   - Avg RRMSE: {rrmse:.6f}")
        print(f"   - Trial: {best_params_data.get('trial_number', 'N/A')}")
        print(f"   - 时间: {best_params_data.get('timestamp', 'N/A')}")
        if per_snr:
            print(f"   - Per SNR: {per_snr}")
            
        if previous_checkpoint and os.path.exists(previous_checkpoint):
            print(f"   - Checkpoint: {previous_checkpoint}")
            print("将从上一次最佳模型继续调优...\n")
        else:
            print("   - Checkpoint: 不存在，将从预训练模型开始\n")
            previous_checkpoint = None
        
        return previous_params, metrics, previous_checkpoint
    except Exception as e:
        print(f"⚠️  加载最佳参数失败: {e}")
        print("将从预训练模型开始搜索...")
        return None


def save_best_params(trial_number, metrics, loss, params, model=None):
    """
    实时保存当前最佳参数和模型到日志文件
    metrics: {'avg_rrmse': float, 'per_snr': {snr: rrmse}}
    """
    # 保存模型checkpoint
    checkpoint_path = None
    if model is not None:
        os.makedirs(os.path.dirname(BEST_MODEL_CHECKPOINT), exist_ok=True)
        torch.save({
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'loss': loss,
            'trial_number': trial_number,
            'parameters': params
        }, BEST_MODEL_CHECKPOINT)
        checkpoint_path = BEST_MODEL_CHECKPOINT
    
    best_params = {
        'trial_number': trial_number,
        'best_rrmse': float(metrics['avg_rrmse']),
        'per_snr_rrmse': {str(k): float(v) for k, v in metrics['per_snr'].items()},
        'corresponding_loss': float(loss),
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'checkpoint_path': checkpoint_path,
        'parameters': {k: float(v) for k, v in params.items()}
    }
    
    with open(BEST_PARAMS_LOG_FILE, 'w', encoding='utf-8') as f:
        json.dump(best_params, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎯 发现更佳参数！已更新日志: {BEST_PARAMS_LOG_FILE}")
    print(f"   Trial #{trial_number}: Avg RRMSE={metrics['avg_rrmse']:.6f}, Loss={loss:.6f}")
    print(f"   Per SNR: {metrics.get('per_snr', {})}")
    if checkpoint_path:
        print(f"   模型已保存: {checkpoint_path}")

def load_pretrained_model(model, pretrained_path):
    """
    加载预训练模型权重
    """
    if not os.path.exists(pretrained_path):
        print(f"⚠️  警告: 预训练模型文件不存在: {pretrained_path}")
        print("将从头开始训练...")
        return model
    
    try:
        checkpoint = torch.load(pretrained_path, map_location=DEVICE)
        # 兼容不同的保存格式
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"✅ 成功加载预训练模型 (带checkpoint): {pretrained_path}")
            if 'rrmse' in checkpoint:
                print(f"   预训练模型RRMSE: {checkpoint['rrmse']:.6f}")
        else:
            model.load_state_dict(checkpoint)
            print(f"✅ 成功加载预训练模型: {pretrained_path}")
    except Exception as e:
        print(f"⚠️  加载预训练模型失败: {e}")
        print("将从头开始训练...")
    
    return model

# 方法1: 限制PyTorch最大显存使用（推荐）
# torch.cuda.set_per_process_memory_fraction(1.0, 0)  # 限制使用60%的GPU显存（约6GB/10GB）

def load_test_data_by_snr(snr_db):
    """
    根据SNR加载测试数据
    """
    import scipy.io
    contaminated_path = TEST_SNR_PATHS[snr_db]['contaminated']
    pure_path = TEST_SNR_PATHS[snr_db]['pure']
    
    test_input = scipy.io.loadmat(contaminated_path)[DATA_KEY]
    test_output = scipy.io.loadmat(pure_path)[PURE_KEY]
    return test_input, test_output


def get_data():
    """
    加载训练和所有SNR的测试数据
    """
    import scipy.io
    from config import TRAIN_CONTAMINATED_PATH
    
    # 训练数据
    train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    
    # 🔧 修改：加载所有定义的SNR测试集
    test_data_dict = {}
    
    # 过滤掉无法加载的SNR
    print(f"\n📊 正在加载测试集 (Target SNRs: {TEST_SNR_LEVELS})...")
    
    for snr in TEST_SNR_LEVELS:
        try:
            # 确保 key 存在于 paths 中
            if snr not in TEST_SNR_PATHS:
                continue
                
            tx, ty = load_test_data_by_snr(snr)
            test_data_dict[snr] = (tx, ty)
            print(f"   ✅ Loaded SNR={snr}dB")
        except Exception as e:
            print(f"   ⚠️  Failed to load SNR={snr}dB: {e}")
    
    return train_x, test_data_dict

def evaluate_model_metrics(model, test_loaders_dict):
    """
    在所有测试集上评估模型，返回 Avg RRMSE 和 Per-SNR RRMSE
    """
    model.eval()
    results = {}
    total_rrmse = 0
    count = 0
    
    with torch.no_grad():
        for snr, loader in test_loaders_dict.items():
            all_preds = []
            all_clean = []
            
            for batch in loader:
                # 兼容不同的batch结构
                noisy, clean, norm = batch if len(batch) == 3 else (batch[0], None, batch[1])
                
                noisy = noisy.float().unsqueeze(1).to(DEVICE)
                norm = norm.float().to(DEVICE).view(-1, 1, 1)
                
                # 归一化输入
                noisy_scaled = noisy * norm
                
                # 推理
                c_out, _ = model(noisy_scaled)
                
                if clean is not None:
                    # c_out 是原始幅度
                    all_preds.append(c_out.squeeze(1).cpu().numpy())
                    all_clean.append(clean.numpy())
            
            # 计算该SNR下的RRMSE
            if len(all_clean) > 0:
                p = np.concatenate(all_preds, axis=0)
                t = np.concatenate(all_clean, axis=0)
                m = compute_all_metrics(p, t, SAMPLING_RATE)
                rrmse = m.get('RRMSE', float('inf'))
                
                # 记录结果 (key转为string以方便json序列化)
                results[str(snr)] = rrmse
                total_rrmse += rrmse
                count += 1
                
    avg_rrmse = total_rrmse / count if count > 0 else float('inf')
    return {'avg_rrmse': avg_rrmse, 'per_snr': results}


def objective(trial, previous_best_params=None, previous_checkpoint=None):
    """
    Optuna 的目标函数，用于评估一组超参数。
    如果提供了previous_best_params，将在其周围进行搜索。
    如果提供了previous_checkpoint，将从该checkpoint继续训练。
    """
    global global_best_metrics
    
    print(f"\n--- 开始 Trial {trial.number} ---")
    
    # --- 1. 定义超参数搜索空间 ---
    # 如果有上一次的最佳参数，在其周围搜索；否则使用默认范围
    
    def suggest_param_near(name, default_low, default_high, previous_value=None, 
                          variation=0.3, log=False, param_type='float', step=None):
        """
        在上一次最佳值附近搜索，如果没有上一次值则使用默认范围
        variation: 搜索范围为上一次值的 ±variation * 值
        """
        if previous_value is not None:
            # 在上一次最佳值周围搜索
            if log:
                # 对数空间
                low = max(default_low, previous_value * (1 - variation))
                high = min(default_high, previous_value * (1 + variation))
            else:
                # 线性空间
                value_range = previous_value * variation
                low = max(default_low, previous_value - value_range)
                high = min(default_high, previous_value + value_range)
            
            if param_type == 'int':
                if step:
                    return trial.suggest_int(name, int(low), int(high), step=step)
                else:
                    return trial.suggest_int(name, int(low), int(high))
            else:
                return trial.suggest_float(name, low, high, log=log)
        else:
            # 使用默认范围
            if param_type == 'int':
                if step:
                    return trial.suggest_int(name, default_low, default_high, step=step)
                else:
                    return trial.suggest_int(name, default_low, default_high)
            else:
                return trial.suggest_float(name, default_low, default_high, log=log)
    
    # 学习率和优化器参数
    lr = suggest_param_near('learning_rate', 1e-6, 1e-3,
                           previous_best_params.get('learning_rate') if previous_best_params else None,
                           variation=0.5, log=True)
    weight_decay = suggest_param_near('weight_decay', 1e-6, 1e-2,
                                     previous_best_params.get('weight_decay') if previous_best_params else None,
                                     variation=0.5, log=True)
    
    # 损失权重 - 在上一次最佳值周围搜索
    lambda_rec = suggest_param_near('lambda_rec', 0, 100,
                                   previous_best_params.get('lambda_rec') if previous_best_params else None,
                                   variation=0.3)
    lambda_con = suggest_param_near('lambda_con', 0, 0,
                                   previous_best_params.get('lambda_con') if previous_best_params else None,
                                   variation=0.3)
    lambda_teacher = suggest_param_near('lambda_teacher', 0.05, 100,
                                       previous_best_params.get('lambda_teacher') if previous_best_params else None,
                                       variation=0.3)
    lambda_n2v = suggest_param_near('lambda_n2v', 0, 100,
                                   previous_best_params.get('lambda_n2v') if previous_best_params else None,
                                   variation=0.3)
    lambda_band = suggest_param_near('lambda_band', 0, 100,
                                    previous_best_params.get('lambda_band') if previous_best_params else None,
                                    variation=0.3)
    lambda_low = suggest_param_near('lambda_low', 0, 0,
                                   previous_best_params.get('lambda_low') if previous_best_params else None,
                                   variation=0.3)
    lambda_decor = suggest_param_near('lambda_decor', 0, 0,
                                     previous_best_params.get('lambda_decor') if previous_best_params else None,
                                     variation=0.3)
    lambda_content = suggest_param_near('lambda_content', 0, 0,
                                       previous_best_params.get('lambda_content') if previous_best_params else None,
                                       variation=0.3)

    # Artifact-aware 掩蔽参数
    mask_base = suggest_param_near('mask_base', 0.03, 10,
                                  previous_best_params.get('mask_base') if previous_best_params else None,
                                  variation=0.3)
    boost_scale = suggest_param_near('boost_scale', 0.05, 10,
                                    previous_best_params.get('boost_scale') if previous_best_params else None,
                                    variation=0.3)
    gamma_art_weight = suggest_param_near('gamma_art_weight', 0.3, 10,
                                         previous_best_params.get('gamma_art_weight') if previous_best_params else None,
                                         variation=0.3)
    
    # compute_artifact_prob 内部参数
    artifact_win_size = suggest_param_near('artifact_win_size', 40, 300,
                                          previous_best_params.get('artifact_win_size') if previous_best_params else None,
                                          variation=0.3, param_type='int', step=16)

    # generate_masked_input_artifact_aware 内部参数
    mask_neighborhood = suggest_param_near('mask_neighborhood', 1, 100,
                                          previous_best_params.get('mask_neighborhood') if previous_best_params else None,
                                          variation=0.3, param_type='int')
    
    # _fft_highpass/lowpass cutoff 参数
    teacher_cutoff = suggest_param_near('teacher_cutoff', 3.0, 15.0,
                                       previous_best_params.get('teacher_cutoff') if previous_best_params else None,
                                       variation=0.3)
    
    lowpass_cutoff = suggest_param_near('lowpass_cutoff', 1.5, 8.0,
                                       previous_best_params.get('lowpass_cutoff') if previous_best_params else None,
                                       variation=0.3)
    
    # teacher阈值
    teacher_threshold = suggest_param_near('teacher_threshold', 0.2, 0.95,
                                          previous_best_params.get('teacher_threshold') if previous_best_params else None,
                                          variation=0.2)

    # --- 2. 设置模型和数据 ---
    # 🔧 修改：支持多SNR测试
    train_x, test_data_dict = get_data()
    train_dataset = UnsupervisedEEGDataset(train_x)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 构建所有SNR的测试Loader
    test_loaders = {}
    for snr, (tx, ty) in test_data_dict.items():
        ds = UnsupervisedEEGDataset(tx, clean=ty)
        test_loaders[snr] = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    # 创建模型并加载权重
    model = DATNet(in_channels=1, base_channels=32).to(DEVICE)
    
    # 优先使用上一次调优的最佳模型，否则使用预训练模型
    if previous_checkpoint and os.path.exists(previous_checkpoint):
        print(f"  📦 从上一次最佳checkpoint加载: {os.path.basename(previous_checkpoint)}")
        model = load_pretrained_model(model, previous_checkpoint)
    else:
        print(f"  📦 从预训练模型加载: {os.path.basename(PRETRAINED_MODEL_PATH)}")
        model = load_pretrained_model(model, PRETRAINED_MODEL_PATH)
    
    # ⭐ 关键：加载后立即验证模型初始性能
    print(f"  🔍 验证加载的模型在测试集上的初始性能 (所有SNR)...")
    model.eval()
    initial_metrics = evaluate_model_metrics(model, test_loaders)
    print(f"  ✅ 初始 Avg RRMSE: {initial_metrics['avg_rrmse']:.6f}")
    
    # 如果没有全局基准，使用当前初始模型作为基准
    if global_best_metrics['avg_rrmse'] == float('inf'):
         print("  ⚠️  未找到全局基准，将当前初始模型设为基准")
         global_best_metrics = initial_metrics
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # --- 3. 训练和验证循环 ---
    best_val_loss = float('inf')
    best_trial_avg_rrmse = float('inf')

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

        # --- 在测试集上评估 ---
        model.eval()
        
        # 1. 计算Loss (仅用第一个测试集做参考)
        val_losses = []
        if test_loaders:
            first_loader = next(iter(test_loaders.values()))
            with torch.no_grad():
                for batch in first_loader:
                    noisy, clean, norm = batch if len(batch) == 3 else (batch[0], None, batch[1])
                    noisy = noisy.float().unsqueeze(1).to(DEVICE)
                    norm = norm.float().to(DEVICE).view(-1, 1, 1)
                    noisy_scaled = noisy * norm
                    
                    val_loss_batch, _, _ = unsupervised_dat_loss_artifact_v2(
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
        
        current_val_loss = np.mean(val_losses) if val_losses else float('inf')
        
        # 2. 计算多SNR指标
        current_metrics = evaluate_model_metrics(model, test_loaders)
        current_avg_rrmse = current_metrics['avg_rrmse']
        current_per_snr = current_metrics['per_snr']
        
        # 记录本Trial最佳 (用于Optuna返回)
        if current_avg_rrmse < best_trial_avg_rrmse:
            best_trial_avg_rrmse = current_avg_rrmse
            
        # 记录最佳Loss
        if current_val_loss < best_val_loss:
            best_val_loss = current_val_loss
        
        # --- 择优保存逻辑 (Average Improvement) ---
        # 只要平均RRMSE优于(小于)历史最佳时，就保存
        baseline_avg_rrmse = global_best_metrics.get('avg_rrmse', float('inf'))
        is_better = current_avg_rrmse < baseline_avg_rrmse
        
        if is_better:
            print(f"  ✅ Epoch {epoch}: 发现全局更优模型 (Avg RRMSE: {baseline_avg_rrmse:.6f} -> {current_avg_rrmse:.6f})")
            
            global_best_metrics = current_metrics
            save_best_params(
                trial_number=trial.number,
                metrics=current_metrics,
                loss=current_val_loss,
                params=trial.params,
                model=model
            )
            print(f"  💾 已立即保存到磁盘: {BEST_MODEL_CHECKPOINT}")

        # Optuna剪枝：基于平均RRMSE
        if epoch > 30:
            trial.report(current_avg_rrmse, epoch)
            if trial.should_prune():
                print(f"Trial {trial.number} pruned at epoch {epoch}. Avg RRMSE={current_avg_rrmse:.6f}")
                raise optuna.exceptions.TrialPruned()

    print(f"--- Trial {trial.number} 完成 ---")
    print(f"  - Trial最佳Avg RRMSE: {best_trial_avg_rrmse:.6f}")
    
    return best_trial_avg_rrmse


def main():
    global global_best_metrics
    
    print('='*70)
    print('DAT-Net 无监督训练超参数调优 (Optuna) - 基于预训练模型')
    print('⚠️  注意：使用测试集进行调优（而非验证集）')
    print('优化目标: RRMSE (去噪性能) - 主要目标')
    print('次要监控: Loss (训练稳定性)')
    print(f'使用设备: {DEVICE}')
    print(f'数据集: {DATASET_NAME}')
    print(f'预训练模型: {PRETRAINED_MODEL_PATH}')
    print('='*70)
    
    # 加载上一次的最佳参数和模型
    previous_result = load_previous_best_params()
    previous_best_params = None
    previous_checkpoint = None
    if previous_result is not None:
        previous_best_params, previous_metrics, previous_checkpoint = previous_result
        global_best_metrics = previous_metrics  # 设置全局最佳metrics基准
        print(f"将从 Avg RRMSE={previous_metrics['avg_rrmse']:.6f} 的基础上继续优化\n")

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
        study_name='dat-net-unsupervised-v2-tuning-rrmse-full',
        storage='sqlite:///dat-net-tuning_fully_simulated.db',
        load_if_exists=True,
        pruner=pruner
    )

    # n_trials: 总共要运行的试验次数
    # 使用lambda函数传递previous_best_params和previous_checkpoint
    study.optimize(lambda trial: objective(trial, previous_best_params, previous_checkpoint), n_trials=200)

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
