"""
综合评估与微调脚本 (Comprehensive Evaluation & Finetuning)

功能：
1. 读取全模拟与半模拟的最佳Checkpoint
2. 执行交叉评估 (Full on Full, Semi on Semi, Full on Semi, Semi on Full)
3. 执行微调 (Finetune 20% Full -> Test, Finetune 20% Semi -> Test)

使用方法：修改下方的“执行开关”变量来控制要运行的任务
"""
import os
import sys
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
from datetime import datetime

# --- 路径设置 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
grandparent_dir = os.path.dirname(parent_dir)

# 1. 导入 dataset_config (在 复现的方法 目录下)
sys.path.insert(0, grandparent_dir)
from dataset_config import get_dataset_config

# 2. 导入 DATNet (在 DAT-Net 目录下)
datnet_dir = os.path.join(parent_dir, 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, current_dir)
from model import DATNet

# 3. 导入 Metrics (在 复现的方法 目录下)
try:
    from metrics_utils import compute_all_metrics
except ImportError:
    def compute_all_metrics(pred, target, fs): return {'RRMSE': float('inf')}

# ========== 执行开关 (SWITCHES) ==========
DO_TEST_FULL_ON_FULL = False      # 1. 用全模拟模型测试全模拟数据
DO_TEST_SEMI_ON_SEMI = False      # 2. 用半模拟模型测试半模拟数据
DO_TEST_FULL_ON_SEMI = True      # 3. 用全模拟模型测试半模拟数据
DO_TEST_SEMI_ON_FULL = False      # 4. 用半模拟模型测试全模拟数据

DO_FINETUNE_FULL = False          # 5. 微调全模拟模型 (20%全模拟数据) -> 测试全模拟数据
DO_FINETUNE_SEMI = False          # 6. 微调半模拟模型 (20%半模拟数据) -> 测试半模拟数据
# ========================================

# ========== 配置常量 ==========
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
BATCH_SIZE = 64 # 测试时Batch Size可以大一点
SAMPLING_RATE = 250.0  # 假设为250，会尝试从config读取

# 微调参数 (来自 finetune_adaptive.py)
FINETUNE_RATIO = 0.2
FINETUNE_EPOCHS = 1000
FINETUNE_BATCH_SIZE = 64
LR_ENCODER = 0.0005590547219468103
LR_BOTTLENECK = 5.054618790576543e-04
LR_DECODER = 0.0001178199097716924
LR_OUTPUT = 0.00010610643578857668
FINETUNE_WEIGHT_DECAY = 1.7038831253031843e-04
FINETUNE_GRAD_CLIP = 1.933899743786938

# 结果保存目录
RESULTS_DIR = 'comprehensive_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

class SupervisedDataset(Dataset):
    def __init__(self, noisy, clean):
        self.noisy = noisy
        self.clean = clean

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        clean = self.clean[idx]
        
        # Max-Abs归一化
        norm = np.max(np.abs(noisy))
        if norm == 0: norm = 1.0
        
        noisy_norm = (noisy / norm).astype('float32')
        clean_norm = (clean / norm).astype('float32')
        
        return torch.tensor(noisy_norm), torch.tensor(clean_norm), norm

def find_best_checkpoint(mode):
    """
    查找最新的最佳模型 checkpoint
    mode: 'fully_simulated' 或 'semi_simulated'
    优先级: tune_best_model_*.pth > datnet_unsupervised_v2_*_best_rrmse.pth
    """
    ckpt_dir = os.path.join(current_dir, 'checkpoints')
    
    # 1. 尝试 Optuna 调优后的最佳模型
    tuned_ckpt = os.path.join(ckpt_dir, f'tune_best_model_{mode}.pth')
    if os.path.exists(tuned_ckpt):
        print(f"[{mode}] Found Tuned Checkpoint: {os.path.basename(tuned_ckpt)}")
        return tuned_ckpt
        
    # 2. 尝试默认最佳模型
    default_ckpt = os.path.join(ckpt_dir, f'datnet_unsupervised_v2_{mode}_best_rrmse_1.pth')
    if os.path.exists(default_ckpt):
        print(f"[{mode}] Found Default Checkpoint: {os.path.basename(default_ckpt)}")
        return default_ckpt
        
    print(f"[{mode}] ⚠️ No checkpoint found!")
    return None

def load_data_for_mode(mode):
    """根据模式('fully_simulated'/'semi_simulated')加载数据配置"""
    config = get_dataset_config(mode)
    return config

def load_test_data(config):
    """加载测试数据 (多SNR支持)"""
    test_data = {}
    if 'test_snr_levels' in config and 'test_snr_paths' in config:
        snrs = config['test_snr_levels']
        paths = config['test_snr_paths']
        for snr in snrs:
            if snr in paths:
                try:
                    p_cont = paths[snr]['contaminated']
                    p_pure = paths[snr]['pure']
                    d_key = config['data_key']
                    p_key = config.get('pure_key', d_key)
                    
                    x = scipy.io.loadmat(p_cont)[d_key]
                    y = scipy.io.loadmat(p_pure)[p_key]
                    test_data[snr] = (x, y)
                except Exception as e:
                    print(f"  Error loading SNR {snr}: {e}")
    else:
        # 单一测试集 fallback (如果有的话)
        pass 
    return test_data

def evaluate_model(model, test_data_dict, task_name):
    """评估模型并返回结果"""
    print(f"\n--- Evaluating: {task_name} ---")
    model.eval()
    results = {}
    
    avg_rrmse = 0
    count = 0
    
    with torch.no_grad():
        for snr, (test_x, test_y) in test_data_dict.items():
            ds = SupervisedDataset(test_x, test_y)
            dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)
            
            preds = []
            targets = []
            
            for noisy_t, clean_t, norm in dl:
                noisy_t = noisy_t.float().unsqueeze(1).to(DEVICE)
                norm = norm.float().to(DEVICE).view(-1, 1, 1)
                
                # 按照 test_unsupervised.py 的逻辑，即使Dataset做了归一化
                # 模型输入也应该是恢复到原始尺度的信号
                noisy_restored = noisy_t * norm
                
                out_tuple = model(noisy_restored)
                # DATNet returns (eeg_clean, eog_artifact)
                if isinstance(out_tuple, tuple):
                    out = out_tuple[0]
                else:
                    out = out_tuple
                
                # 模型输出即为原始尺度，无需再乘 norm
                # 这里的 out 已经是去噪后的原始信号
                pred_numpy = out.squeeze(1).detach().cpu().numpy()
                
                # clean_t 是 (B, L) 归一化的，需要恢复为原始Target
                norm_np = norm.squeeze().cpu().numpy()
                target_restored = clean_t.numpy() * norm_np[:, None]
                
                preds.append(pred_numpy)
                targets.append(target_restored)
                
            preds = np.concatenate(preds, axis=0)
            targets = np.concatenate(targets, axis=0)
            
            metrics = compute_all_metrics(preds, targets, SAMPLING_RATE)
            rrmse = metrics.get('RRMSE', float('inf'))
            results[str(snr)] = rrmse
            avg_rrmse += rrmse
            count += 1
            print(f"  SNR {snr}dB: RRMSE = {rrmse:.4f}")

    final_avg = avg_rrmse / count if count > 0 else float('inf')
    print(f"  > Average RRMSE: {final_avg:.4f}")
    
    # 保存结果
    save_path = os.path.join(RESULTS_DIR, f'{task_name}.json')
    with open(save_path, 'w') as f:
        json.dump({'avg_rrmse': final_avg, 'per_snr': results}, f, indent=2)
    print(f"  Saved results to {save_path}")
    return final_avg

def finetune_model(model, train_x, train_y, val_data_dict, task_name):
    """微调模型"""
    print(f"\n--- Finetuning: {task_name} ---")
    
    # 构建微调数据集
    train_ds = SupervisedDataset(train_x, train_y)
    train_loader = DataLoader(train_ds, batch_size=FINETUNE_BATCH_SIZE, shuffle=True)
    
    # 分层学习率
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
        {'params': model.bottleneck.parameters(), 'lr': LR_BOTTLENECK},
        {'params': model.decoder.parameters(), 'lr': LR_DECODER},
        {'params': model.output_conv.parameters(), 'lr': LR_OUTPUT}
    ]
    optimizer = optim.AdamW(param_groups, weight_decay=FINETUNE_WEIGHT_DECAY)
    
    model.train()
    best_loss = float('inf')
    
    for epoch in range(1, FINETUNE_EPOCHS + 1):
        total_loss = 0
        batch_count = 0
        
        for noisy, clean, norm in train_loader:
            noisy = noisy.float().unsqueeze(1).to(DEVICE)
            clean = clean.float().unsqueeze(1).to(DEVICE)
            norm = norm.float().to(DEVICE).view(-1, 1, 1)
            
            # 恢复原始尺度
            noisy_restored = noisy * norm
            clean_restored = clean * norm
            
            optimizer.zero_grad()
            output_tuple = model(noisy_restored)
            
            if isinstance(output_tuple, tuple):
                output = output_tuple[0]  # 只使用 EEG clean 输出进行监督微调
            else:
                output = output_tuple
            
            # 使用原始尺度的 Target 计算 Loss
            loss = nn.MSELoss()(output, clean_restored)
            loss.backward()
            
            if FINETUNE_GRAD_CLIP > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), FINETUNE_GRAD_CLIP)
                
            optimizer.step()
            total_loss += loss.item()
            batch_count += 1
            
        avg_loss = total_loss / batch_count if batch_count > 0 else 0
        if epoch % 100 == 0:
            print(f"  Epoch {epoch}/{FINETUNE_EPOCHS}: Loss = {avg_loss:.6f}")
            
    # 保存微调后的模型
    save_path = os.path.join(os.path.join(current_dir, 'checkpoints'), f'{task_name}_model.pth')
    torch.save(model.state_dict(), save_path)
    print(f"  Finetuned model saved to {save_path}")
    return model

def main():
    print("========================================")
    print("   DAT-Net Comprehensive Evaluation   ")
    print("========================================")
    
    FULL = 'fully_simulated'
    SEMI = 'semi_simulated'
    
    # 1. 加载 Config
    config_full = load_data_for_mode(FULL)
    config_semi = load_data_for_mode(SEMI)
    
    # 2. 加载数据
    print("Loading datasets...")
    test_data_full = load_test_data(config_full)
    test_data_semi = load_test_data(config_semi)
    
    # 加载训练数据用于微调 (取前20%)
    def load_train_subset(config):
        path_cont = config['train_contaminated_path']
        path_pure = config['train_pure_path']
        d_key = config['data_key']
        p_key = config.get('pure_key', d_key)
        
        tx = scipy.io.loadmat(path_cont)[d_key]
        ty = scipy.io.loadmat(path_pure)[p_key]
        
        n_samples = int(len(tx) * FINETUNE_RATIO)
        return tx[:n_samples], ty[:n_samples]
    
    if DO_FINETUNE_FULL:
        train_full_x, train_full_y = load_train_subset(config_full)
    if DO_FINETUNE_SEMI:
        train_semi_x, train_semi_y = load_train_subset(config_semi)
        
    print("Datasets loaded.")
    
    # 3. 加载模型 Checkpoints
    ckpt_full = find_best_checkpoint(FULL)
    ckpt_semi = find_best_checkpoint(SEMI)
    
    # --- 任务执行 ---
    
    # 模型加载 Helper
    def get_model(ckpt_path):
        if not ckpt_path: return None
        m = DATNet().to(DEVICE)
        state = torch.load(ckpt_path, map_location=DEVICE)
        if 'model_state_dict' in state: state = state['model_state_dict']
        m.load_state_dict(state)
        return m

    # Task 1: Full on Full
    if DO_TEST_FULL_ON_FULL and ckpt_full:
        model = get_model(ckpt_full)
        evaluate_model(model, test_data_full, '1_Full_Model_on_Full_Data')

    # Task 2: Semi on Semi
    if DO_TEST_SEMI_ON_SEMI and ckpt_semi:
        model = get_model(ckpt_semi)
        evaluate_model(model, test_data_semi, '2_Semi_Model_on_Semi_Data')

    # Task 3: Full on Semi
    if DO_TEST_FULL_ON_SEMI and ckpt_full:
        model = get_model(ckpt_full)
        evaluate_model(model, test_data_semi, '3_Full_Model_on_Semi_Data')

    # Task 4: Semi on Full
    if DO_TEST_SEMI_ON_FULL and ckpt_semi:
        model = get_model(ckpt_semi)
        evaluate_model(model, test_data_full, '4_Semi_Model_on_Full_Data')

    # Task 5: Finetune Full -> Test Full
    if DO_FINETUNE_FULL and ckpt_full:
        model = get_model(ckpt_full)
        model = finetune_model(model, train_full_x, train_full_y, test_data_full, 'finetuned_full')
        evaluate_model(model, test_data_full, '5_Finetuned_Full_Model_on_Full_Data')

    # Task 6: Finetune Semi -> Test Semi
    if DO_FINETUNE_SEMI and ckpt_semi:
        model = get_model(ckpt_semi)
        model = finetune_model(model, train_semi_x, train_semi_y, test_data_semi, 'finetuned_semi')
        evaluate_model(model, test_data_semi, '6_Finetuned_Semi_Model_on_Semi_Data')

    print("\nAll tasks completed.")

if __name__ == '__main__':
    main()
