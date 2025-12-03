"""
DAT-Net 无监督训练脚本（Version 2）
使用 Artifact-aware + Dual-branch consistency (unsupervised_dat_loss_artifact_v2)
"""
import os
import sys
import scipy.io
import numpy as np
import torch
import torch.optim as optim
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
from unsupervised_artifact_v2 import unsupervised_dat_loss_artifact_v2

# 导入配置
from config import *

# 可选metrics导入
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics, print_metrics
except Exception:
    def compute_all_metrics(pred, target, fs): return {}
    def print_metrics(m, prefix=""): pass


class UnsupervisedEEGDataset(Dataset):
    def __init__(self, noisy, clean=None):
        self.noisy = noisy
        self.clean = clean
        self.has_clean = clean is not None

    def __len__(self):
        return len(self.noisy)

    def __getitem__(self, idx):
        noisy = self.noisy[idx]
        norm = np.max(np.abs(noisy))
        if norm == 0:
            norm = 1.0
        if self.has_clean:
            clean = self.clean[idx]
            return noisy.astype('float32') / norm, clean.astype('float32'), norm
        else:
            return noisy.astype('float32') / norm, norm


def get_data():
    train_x = scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]
    val_x = scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]
    try:
        val_y = scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]
    except Exception:
        val_y = None
    return train_x, val_x, val_y


def train_epoch(model, device, loader, optimizer):
    model.train()
    # 累积所有损失项
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    
    for batch in loader:
        if len(batch) == 3:
            noisy, _, norm = batch
        else:
            noisy, norm = batch
        noisy = noisy.float().unsqueeze(1).to(device)
        norm = norm.float().to(device).view(-1, 1, 1)
        noisy_scaled = noisy * norm

        optimizer.zero_grad()
        total_loss_batch, loss_dict, _ = unsupervised_dat_loss_artifact_v2(
            model=model,
            eeg_raw_input=noisy_scaled,
            fs=SAMPLING_RATE,
            mask_base=MASK_BASE,
            boost_scale=BOOST_SCALE,
            lambda_rec=LAMBDA_REC,
            lambda_con=LAMBDA_CON,
            lambda_teacher=LAMBDA_TEACHER,
            lambda_n2v=LAMBDA_N2V,
            lambda_band=LAMBDA_BAND,
            lambda_low=LAMBDA_LOW,
            lambda_decor=LAMBDA_DECOR,
            lambda_content=LAMBDA_CONTENT,
            gamma_art_weight=GAMMA_ART_WEIGHT,
            artifact_win_size=ARTIFACT_WIN_SIZE,
            mask_neighborhood=MASK_NEIGHBORHOOD,
            teacher_cutoff=TEACHER_CUTOFF,
            lowpass_cutoff=LOWPASS_CUTOFF,
            teacher_threshold=TEACHER_THRESHOLD,
        )
        total_loss_batch.backward()
        if GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        # 累积所有损失项
        for key in accumulated_losses.keys():
            accumulated_losses[key] += loss_dict.get(key, 0.0)
        num_batches += 1

    # 返回平均损失
    return {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}


def validate(model, device, loader, has_clean_labels=False):
    model.eval()
    # 累积所有损失项
    accumulated_losses = {
        'total': 0.0, 'rec': 0.0, 'con': 0.0, 'n2v': 0.0,
        'teacher_clean': 0.0, 'teacher_art': 0.0,
        'band': 0.0, 'low': 0.0, 'decor': 0.0, 'content': 0.0
    }
    num_batches = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                noisy, clean, norm = batch
            else:
                noisy, norm = batch
                clean = None
            noisy = noisy.float().unsqueeze(1).to(device)
            norm = norm.float().to(device).view(-1, 1, 1)
            noisy_scaled = noisy * norm

            total_loss_batch, loss_dict, (c_A, a_A, c_B, a_B) = unsupervised_dat_loss_artifact_v2(
                model=model,
                eeg_raw_input=noisy_scaled,
                fs=SAMPLING_RATE,
                mask_base=MASK_BASE,
                boost_scale=BOOST_SCALE,
                lambda_rec=LAMBDA_REC,
                lambda_con=LAMBDA_CON,
                lambda_teacher=LAMBDA_TEACHER,
                lambda_n2v=LAMBDA_N2V,
                lambda_band=LAMBDA_BAND,
                lambda_low=LAMBDA_LOW,
                lambda_decor=LAMBDA_DECOR,
                lambda_content=LAMBDA_CONTENT,
                gamma_art_weight=GAMMA_ART_WEIGHT,
                artifact_win_size=ARTIFACT_WIN_SIZE,
                mask_neighborhood=MASK_NEIGHBORHOOD,
                teacher_cutoff=TEACHER_CUTOFF,
                lowpass_cutoff=LOWPASS_CUTOFF,
                teacher_threshold=TEACHER_THRESHOLD,
            )

            # 累积所有损失项
            for key in accumulated_losses.keys():
                accumulated_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1

            if has_clean_labels and clean is not None:
                all_preds.append(c_B.squeeze(1).cpu().numpy())
                all_targets.append(clean.numpy())

    metrics = None
    if has_clean_labels and len(all_targets) > 0:
        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        metrics = compute_all_metrics(all_preds, all_targets, fs=SAMPLING_RATE)

    # 返回平均损失
    avg_losses = {key: val / max(1, num_batches) for key, val in accumulated_losses.items()}
    return avg_losses, metrics


def main():
    print('='*70)
    print('DAT-Net 无监督训练 (Version 2)')
    print('='*70)
    print_config()  # 打印配置信息
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('\n使用设备:', device)

    train_x, val_x, val_y = get_data()
    print('训练集:', train_x.shape)
    print('验证集:', val_x.shape)

    train_dataset = UnsupervisedEEGDataset(train_x, clean=None)
    val_dataset = UnsupervisedEEGDataset(val_x, clean=val_y)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = DATNet(in_channels=1, base_channels=32).to(device)
    print(f'模型参数量: {model.count_parameters():,}')

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    if USE_LR_SCHEDULER:
        def warmup_lambda(epoch):
            if epoch < WARMUP_EPOCHS:
                return (epoch + 1) / WARMUP_EPOCHS
            else:
                progress = (epoch - WARMUP_EPOCHS) / max(1, (EPOCHS - WARMUP_EPOCHS))
                # 余弦退火，但保证不低于 MIN_LR
                cosine_factor = 0.5 * (1.0 + np.cos(np.pi * progress))
                min_factor = MIN_LR / LEARNING_RATE
                return max(min_factor, cosine_factor)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup_lambda)
    else:
        scheduler = None

    best_val_loss = float('inf')
    patience_counter = 0
    start_time = time()

    for epoch in range(1, EPOCHS + 1):
        print(f'\nEpoch [{epoch}/{EPOCHS}]')
        print('='*70)
        train_loss = train_epoch(model, device, train_loader, optimizer)
        
        # 打印详细的训练损失分解
        print(f'Train Loss: {train_loss["total"]:.6f}')
        print(f'  - Rec: {train_loss.get("rec", 0):.6f}  Con: {train_loss.get("con", 0):.6f}')
        print(f'  - Teacher: {train_loss.get("teacher_clean", 0):.6f}/{train_loss.get("teacher_art", 0):.6f}')
        print(f'  - N2V: {train_loss.get("n2v", 0):.6f}  Band: {train_loss.get("band", 0):.6f}')
        print(f'  - Low: {train_loss.get("low", 0):.6f}  Decor: {train_loss.get("decor", 0):.6f}  Content: {train_loss.get("content", 0):.6f}')

        val_loss, val_metrics = validate(model, device, val_loader, has_clean_labels=(val_y is not None))
        print(f'\nVal Loss: {val_loss["total"]:.6f}')
        print(f'  - Rec: {val_loss.get("rec", 0):.6f}  Con: {val_loss.get("con", 0):.6f}')
        
        if val_metrics is not None:
            print_metrics(val_metrics, prefix='验证集')

        # 使用验证损失作为保存标准（无监督训练的正确做法）
        improved = False
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            improved = True
            print(f'\n✓ 验证损失降低: {best_val_loss:.6f}')
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        if scheduler is not None:
            scheduler.step()
            print(f'\nLearning Rate: {optimizer.param_groups[0]["lr"]:.6f}')

        elapsed = time() - start_time
        print(f'Elapsed Time: {int(elapsed//60)}min {int(elapsed%60)}s')

        if patience_counter >= PATIENCE:
            print(f'\n早停触发！{PATIENCE} 个epoch内无改善。')
            break

    torch.save(model.state_dict(), FINAL_MODEL_PATH)
    print(f'训练完成，模型已保存到: {MODEL_SAVE_PATH}')
    print(f'最终模型已保存到: {FINAL_MODEL_PATH}')


if __name__ == '__main__':
    main()
