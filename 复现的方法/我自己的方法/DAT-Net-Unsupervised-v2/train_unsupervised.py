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
        
        # Max-Abs归一化（与原始版本一致）
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
        val_y = scipy.io.loadmat(VAL_PURE_PATH)[PURE_KEY]
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
        
        # 恢复原始幅度: max-abs反归一化
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
            
            # 恢复原始幅度: max-abs反归一化
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

    # 使用相对路径避免中文路径问题
    # 先切换到脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    print(f'当前工作目录: {os.getcwd()}')
    
    # 使用相对路径创建checkpoints目录
    checkpoint_dir = 'checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f'[创建目录] {checkpoint_dir}')
    
    # 定义模型保存路径（使用相对路径）
    model_save_path_rrmse = os.path.join(checkpoint_dir, f'datnet_unsupervised_v2_{DATASET_NAME}_best_rrmse.pth')
    model_save_path_loss = os.path.join(checkpoint_dir, f'datnet_unsupervised_v2_{DATASET_NAME}_best_loss.pth')
    model_save_path_current = os.path.join(checkpoint_dir, f'datnet_unsupervised_v2_{DATASET_NAME}_current.pth')
    final_model_path = os.path.join(checkpoint_dir, f'datnet_unsupervised_v2_{DATASET_NAME}_final.pth')
    
    print(f'检查点目录: {checkpoint_dir}')
    print(f'检查点目录存在: {os.path.exists(checkpoint_dir)}')
    print(f'RRMSE最佳模型: {model_save_path_rrmse}')
    print(f'Loss最佳模型: {model_save_path_loss}')
    print(f'最终模型路径: {final_model_path}')

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
        # 使用线性衰减策略：从 LEARNING_RATE 衰减到 MIN_LR
        def lr_lambda(epoch):
            # 线性插值：lr = LEARNING_RATE * alpha，其中 alpha 从 1.0 衰减到 MIN_LR/LEARNING_RATE
            min_factor = MIN_LR / LEARNING_RATE
            factor = 1.0 - (1.0 - min_factor) * (epoch / EPOCHS)
            return max(min_factor, factor)
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    else:
        scheduler = None

    # ========== 尝试加载上次的最佳模型（断点续训） ==========
    best_val_loss = float('inf')
    best_val_rrmse = float('inf')
    start_epoch = 1  # 始终从epoch 1开始
    patience_counter = 0
    rrmse_no_improve_counter = 0  # RRMSE连续不降的计数器
    
    # 根据配置选择要加载的模型
    resume_path = None
    if RESUME_FROM == 'rrmse' and os.path.exists(model_save_path_rrmse):
        resume_path = model_save_path_rrmse
        resume_type = 'RRMSE最佳模型'
    elif RESUME_FROM == 'loss' and os.path.exists(model_save_path_loss):
        resume_path = model_save_path_loss
        resume_type = 'Loss最佳模型'
    elif RESUME_FROM == 'auto':
        # 自动选择：优先RRMSE，其次Loss
        if os.path.exists(model_save_path_rrmse):
            resume_path = model_save_path_rrmse
            resume_type = 'RRMSE最佳模型（自动选择）'
        elif os.path.exists(model_save_path_loss):
            resume_path = model_save_path_loss
            resume_type = 'Loss最佳模型（自动选择）'
    
    if resume_path is not None:
        print(f'\n[检测到已有模型] {resume_type}')
        print(f'路径: {resume_path}')
        try:
            checkpoint = torch.load(resume_path, map_location=device)
            
            # 兼容旧的保存格式（只有state_dict）和新格式（包含训练状态）
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                # 恢复最佳指标，防止新训练覆盖之前的更好结果
                best_val_loss = checkpoint.get('best_val_loss', float('inf'))
                best_val_rrmse = checkpoint.get('best_val_rrmse', float('inf'))
                previous_epoch = checkpoint.get('epoch', 0)
                
                # 恢复优化器状态（如果存在）
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    print(f'[加载模型、优化器和最佳指标]')
                else:
                    print(f'[加载模型权重和最佳指标]')
                
                print(f'  - 上次训练到: Epoch {previous_epoch}')
                print(f'  - 继承 Best Loss: {best_val_loss:.6f}')
                print(f'  - 继承 Best RRMSE: {best_val_rrmse:.6f}')
                if 'save_reason' in checkpoint:
                    print(f'  - 保存原因: {checkpoint["save_reason"]}')
            else:
                # 旧格式，只加载模型权重
                model.load_state_dict(checkpoint)
                print('[加载模型权重] 旧格式checkpoint，最佳指标从头开始')
            
            print(f'✓ 成功加载模型，将从Epoch 1开始继续训练\n')
        except Exception as e:
            print(f'✗ 加载模型失败: {e}')
            print('将从头开始训练\n')
    else:
        print(f'\n[未找到已有模型] 将从头开始训练')
        print(f'RESUME_FROM设置: {RESUME_FROM}\n')
    
    start_time = time()

    for epoch in range(start_epoch, EPOCHS + 1):
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

        # ========== 双重保存策略：同时保存RRMSE最佳和Loss最佳模型 ==========
        improved = False
        
        # 1. 基于RRMSE保存（如果有真实标签）
        if val_metrics is not None and 'RRMSE' in val_metrics:
            current_rrmse = val_metrics['RRMSE']
            if current_rrmse < best_val_rrmse:
                best_val_rrmse = current_rrmse
                improved = True
                print(f'\n[*] RRMSE降低至 {best_val_rrmse:.6f}')
                
                # 保存RRMSE最佳模型
                save_dir = os.path.dirname(model_save_path_rrmse) if os.path.dirname(model_save_path_rrmse) else '.'
                if save_dir != '.' and not os.path.exists(save_dir):
                    os.makedirs(save_dir, exist_ok=True)
                
                checkpoint_rrmse = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_val_rrmse': best_val_rrmse,
                    'val_loss': val_loss,
                    'val_metrics': val_metrics,
                    'save_reason': f'RRMSE最佳: {best_val_rrmse:.6f}'
                }
                torch.save(checkpoint_rrmse, model_save_path_rrmse)
                print(f'[保存RRMSE最佳模型] {model_save_path_rrmse}')
                rrmse_no_improve_counter = 0  # 重置计数器
        
        # 2. 基于Loss保存（始终跟踪）
        if val_loss['total'] < best_val_loss:
            best_val_loss = val_loss['total']
            improved = True
            print(f'\n[*] 验证损失降低至 {best_val_loss:.6f}')
            
            # 保存Loss最佳模型
            save_dir = os.path.dirname(model_save_path_loss) if os.path.dirname(model_save_path_loss) else '.'
            if save_dir != '.' and not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            
            checkpoint_loss = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_rrmse': best_val_rrmse,
                'val_loss': val_loss,
                'val_metrics': val_metrics if val_metrics is not None else {},
                'save_reason': f'Loss最佳: {best_val_loss:.6f}'
            }
            torch.save(checkpoint_loss, model_save_path_loss)
            print(f'[保存Loss最佳模型] {model_save_path_loss}')
        
        # 更新patience计数器
        if improved:
            patience_counter = 0
        else:
            patience_counter += 1
        
        # ========== RRMSE回退策略 ==========
        # 如果有真实标签，检查RRMSE是否连续不降
        if val_metrics is not None and 'RRMSE' in val_metrics:
            current_rrmse = val_metrics['RRMSE']
            if current_rrmse >= best_val_rrmse:
                rrmse_no_improve_counter += 1
                print(f'[RRMSE未改善] 连续 {rrmse_no_improve_counter}/{RRMSE_ROLLBACK_PATIENCE} 个epoch')
                
                # 达到回退阈值，重新加载最佳RRMSE模型
                if rrmse_no_improve_counter >= RRMSE_ROLLBACK_PATIENCE and os.path.exists(model_save_path_rrmse):
                    print(f'\n[触发回退] RRMSE连续{RRMSE_ROLLBACK_PATIENCE}个epoch未改善，回退到最佳模型')
                    try:
                        checkpoint = torch.load(model_save_path_rrmse, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        if 'optimizer_state_dict' in checkpoint:
                            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                            print('  ✓ 已恢复模型和优化器状态')
                        else:
                            print('  ✓ 已恢复模型权重（优化器状态未保存）')
                        print(f'  - 回退至Epoch {checkpoint.get("epoch", "未知")}的最佳模型')
                        print(f'  - Best RRMSE: {checkpoint.get("best_val_rrmse", "未知"):.6f}')
                        rrmse_no_improve_counter = 0  # 重置计数器
                    except Exception as e:
                        print(f'  ✗ 回退失败: {e}')

        # 每轮保存当前模型 (覆盖旧文件)
        checkpoint_current = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'best_val_rrmse': best_val_rrmse,
            'val_loss': val_loss,
            'val_metrics': val_metrics if val_metrics is not None else {},
            'save_reason': f'Current Epoch {epoch}'
        }
        torch.save(checkpoint_current, model_save_path_current)

        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(f'\nLearning Rate: {current_lr:.8f} (初始:{LEARNING_RATE:.6f} → 最小:{MIN_LR:.6f})')

        elapsed = time() - start_time
        print(f'Elapsed Time: {int(elapsed//60)}min {int(elapsed%60)}s')

        if patience_counter >= PATIENCE:
            print(f'\n早停触发！{PATIENCE} 个epoch内无改善。')
            break

    # 保存最终模型（使用相对路径）
    final_save_dir = os.path.dirname(final_model_path) if os.path.dirname(final_model_path) else '.'
    if final_save_dir != '.' and not os.path.exists(final_save_dir):
        os.makedirs(final_save_dir, exist_ok=True)
        print(f'[创建目录] {final_save_dir}')
    print(f'\n保存最终模型到: {final_model_path}')
    torch.save(model.state_dict(), final_model_path)
    print(f'训练完成！')
    print(f'RRMSE最佳模型: {model_save_path_rrmse}')
    print(f'Loss最佳模型: {model_save_path_loss}')
    print(f'最终模型: {final_model_path}')


if __name__ == '__main__':
    main()
