"""
EEGIFNet测试脚本 - 使用ASNet数据集格式
保持原有网络结构,使用标准化和反标准化逻辑
"""
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from EEGIFNet_1200 import MA_INet, MA_MNet
from config import cal_ACC_tensor, cal_RRMSE_tensor, cal_SNR
from train_new import EEGDataset, get_data
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from time import time

# 导入数据集配置
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_config import *


def test_model(I_model, M_model, device, test_loader):
    """
    测试模型并计算各项指标
    返回每个样本的ACC, RRMSE, SNR列表以及所有预测结果
    """
    I_model.eval()
    M_model.eval()

    acc_list = []
    rrmse_list = []
    snr_list = []
    
    # 用于可视化的指标
    acc_e_list = []
    acc_n_list = []
    
    # 收集所有预测结果
    all_predictions = []

    test_step_num = 0
    sum_acc = 0
    sum_rrmse = 0
    sum_snr = 0

    with torch.no_grad():
        for batch_idx, (x, y, norm_factors) in enumerate(test_loader):
            test_step_num += 1

            x = x.float().to(device)
            y = y.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1)  # ✅ (batch, 1) 与ASNet一致

            # ⚠️ EEGIFNet需要通道维度
            x_with_channel = x.unsqueeze(1)  # (batch, time) -> (batch, 1, time)

            # 计算噪声目标
            z = x - y

            # 模型预测 (输出: batch, time)
            e_outputs, n_outputs = I_model(x_with_channel)
            outputs = M_model(x_with_channel, e_outputs, n_outputs)

            # ⚠️ 反归一化到原始尺度（与ASNet一致）
            outputs_denorm = outputs * norm_factors
            e_outputs_denorm = e_outputs * norm_factors
            n_outputs_denorm = n_outputs * norm_factors
            
            # y和z已经是原始尺度
            y_denorm = y
            z_denorm = z
            
            # 收集预测结果
            all_predictions.append(outputs_denorm.cpu().numpy())

            # 计算反归一化后的指标 (对每个样本)
            for i in range(outputs_denorm.shape[0]):
                acc = cal_ACC_tensor(
                    outputs_denorm[i:i+1].detach(), 
                    y_denorm[i:i+1].detach()
                ).item()
                
                rrmse = cal_RRMSE_tensor(
                    outputs_denorm[i:i+1].detach(), 
                    y_denorm[i:i+1].detach()
                ).item()
                
                snr = cal_SNR(
                    outputs_denorm[i:i+1], 
                    y_denorm[i:i+1]
                )
                
                acc_e = cal_ACC_tensor(
                    e_outputs_denorm[i:i+1].detach(),
                    y_denorm[i:i+1].detach()
                ).item()
                
                acc_n = cal_ACC_tensor(
                    n_outputs_denorm[i:i+1].detach(),
                    z_denorm[i:i+1].detach()
                ).item()

                acc_list.append(acc)
                rrmse_list.append(rrmse)
                snr_list.append(snr)
                acc_e_list.append(acc_e)
                acc_n_list.append(acc_n)

                sum_acc += acc
                sum_rrmse += rrmse
                sum_snr += snr

    # 合并所有预测结果
    all_predictions = np.concatenate(all_predictions, axis=0)
    
    # 计算平均值
    num_samples = len(acc_list)
    avg_acc = sum_acc / num_samples
    avg_rrmse = sum_rrmse / num_samples
    avg_snr = sum_snr / num_samples

    print("\n" + "="*80)
    print("测试结果:")
    print("="*80)
    print(f"样本数量: {num_samples}")
    print(f"平均ACC: {avg_acc:.6f}")
    print(f"平均RRMSE: {avg_rrmse:.6f}")
    print(f"平均SNR: {avg_snr:.2f} dB")
    print("="*80)

    return all_predictions, acc_list, rrmse_list, snr_list, acc_e_list, acc_n_list


def visualize_results(I_model, M_model, device, test_loader, save_dir, num_samples=10):
    """
    可视化部分测试结果
    """
    I_model.eval()
    M_model.eval()

    os.makedirs(save_dir, exist_ok=True)
    
    sample_count = 0

    with torch.no_grad():
        for batch_idx, (x, y, norm_factors) in enumerate(test_loader):
            if sample_count >= num_samples:
                break

            x = x.float().to(device)
            y = y.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1, 1)

            # 模型预测
            e_outputs, n_outputs = I_model(x)
            outputs = M_model(x, e_outputs, n_outputs)

            # 反归一化
            # norm_factors: (batch, 1, 1), outputs: (batch, time)
            norm_factors_2d = norm_factors.squeeze(-1)  # (batch, 1)
            outputs_denorm = outputs * norm_factors_2d
            y_denorm = y.squeeze() * norm_factors_2d
            x_denorm = x.squeeze() * norm_factors_2d
            e_outputs_denorm = e_outputs * norm_factors_2d
            n_outputs_denorm = n_outputs * norm_factors_2d

            # 转换为numpy
            x_np = x_denorm.cpu().numpy()
            y_np = y_denorm.cpu().numpy()
            e_out_np = e_outputs_denorm.cpu().numpy()
            out_np = outputs_denorm.cpu().numpy()
            n_out_np = n_outputs_denorm.cpu().numpy()

            # 计算 x - n_outputs 的去噪结果
            denoised_by_subtraction = x_np - n_out_np

            # 对每个样本可视化
            for i in range(x_np.shape[0]):
                if sample_count >= num_samples:
                    break

                plt.figure(figsize=(15, 10))

                # 子图1: INet的EEG分支输出
                plt.subplot(3, 1, 1)
                plt.plot(x_np[i], label='Contaminated EEG', alpha=0.7)
                plt.plot(e_out_np[i], label='INet EEG Output', alpha=0.8)
                plt.plot(y_np[i], label='Clean EEG (Ground Truth)', alpha=0.8)
                plt.legend(loc='upper right')
                plt.title('(a) INet EEG Branch Output')
                plt.grid(True, alpha=0.3)
                plt.xticks([])

                # 子图2: 通过减去noise分支得到的去噪结果
                plt.subplot(3, 1, 2)
                plt.plot(x_np[i], label='Contaminated EEG', alpha=0.7)
                plt.plot(denoised_by_subtraction[i], label='Denoised (X - Noise)', alpha=0.8)
                plt.plot(y_np[i], label='Clean EEG (Ground Truth)', alpha=0.8)
                plt.legend(loc='upper right')
                plt.title('(b) Denoised by Subtracting Noise Branch')
                plt.grid(True, alpha=0.3)
                plt.xticks([])

                # 子图3: MNet融合输出
                plt.subplot(3, 1, 3)
                plt.plot(x_np[i], label='Contaminated EEG', alpha=0.7)
                plt.plot(out_np[i], label='MNet Fusion Output', alpha=0.8)
                plt.plot(y_np[i], label='Clean EEG (Ground Truth)', alpha=0.8)
                plt.legend(loc='upper right')
                plt.title('(c) MNet Fusion Output')
                plt.grid(True, alpha=0.3)
                plt.xlabel('Time Points')

                plt.tight_layout()
                save_path = os.path.join(save_dir, f'sample_{sample_count:03d}.png')
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()

                print(f"保存可视化结果: {save_path}")
                sample_count += 1

    print(f"共保存了 {sample_count} 个可视化结果")


def main():
    parser = argparse.ArgumentParser(description='EEGIFNet Testing with ASNet Dataset')
    parser.add_argument('--data_path', type=str, 
                        default=DATA_DIR,  # 从data_config导入
                        help='数据集路径')
    parser.add_argument('--batch_size', type=int, default=256, help='批大小')
    parser.add_argument('--device', type=str, default='cuda:0', help='使用的设备')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoint', help='模型保存目录')
    parser.add_argument('--inet_model', type=str, default='EEGIFNet_INet_best.pkl', 
                        help='INet模型文件名')
    parser.add_argument('--mnet_model', type=str, default='EEGIFNet_MNet_best.pkl', 
                        help='MNet模型文件名')
    parser.add_argument('--result_dir', type=str, default='./result', help='结果保存目录')
    parser.add_argument('--visualize', action='store_true', help='是否生成可视化结果')
    parser.add_argument('--num_vis', type=int, default=10, help='可视化样本数量')
    args = parser.parse_args()

    # 创建结果目录
    os.makedirs(args.result_dir, exist_ok=True)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据
    print("加载数据集...")
    _, _, test_loader, input_length = get_data(args.data_path, args.batch_size)
    print(f"数据时间点: {input_length}")

    # 初始化模型 (适配input_length)
    print("初始化模型...")
    I_model = MA_INet(input_length=input_length).to(device)
    M_model = MA_MNet().to(device)

    # 加载模型权重
    inet_path = os.path.join(args.checkpoint_dir, args.inet_model)
    mnet_path = os.path.join(args.checkpoint_dir, args.mnet_model)

    if not os.path.exists(inet_path) or not os.path.exists(mnet_path):
        print(f"错误: 找不到模型文件!")
        print(f"  INet: {inet_path}")
        print(f"  MNet: {mnet_path}")
        return

    print(f"加载模型权重...")
    print(f"  INet: {inet_path}")
    print(f"  MNet: {mnet_path}")
    
    I_model.load_state_dict(torch.load(inet_path, map_location=device))
    M_model.load_state_dict(torch.load(mnet_path, map_location=device))

    # 测试模型
    print("\n开始测试...")
    start_time = time()
    predictions, acc_list, rrmse_list, snr_list, acc_e_list, acc_n_list = test_model(
        I_model, M_model, device, test_loader
    )
    test_time = time() - start_time
    time_per_sample = test_time / len(predictions)
    
    # 保存预测结果为.mat格式
    import scipy.io
    pred_output_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(pred_output_dir, exist_ok=True)
    
    pred_save_path = os.path.join(pred_output_dir, 'EEGIFNet_predictions.mat')
    scipy.io.savemat(pred_save_path, {
        'predictions': predictions,
        'time_per_sample': time_per_sample
    })
    
    print(f"\n预测结果已保存为.mat格式: {pred_save_path}")
    print(f"预测结果形状: {predictions.shape}")
    print(f"单样本推理时间: {time_per_sample*1000:.3f}ms")


    # 保存结果到CSV
    print(f"\n保存结果到 {args.result_dir}")
    
    results_df = pd.DataFrame({
        'ACC': acc_list,
        'RRMSE': rrmse_list,
        'SNR_dB': snr_list,
        'ACC_e': acc_e_list,
        'ACC_n': acc_n_list
    })
    
    results_df.to_csv(os.path.join(args.result_dir, 'EEGIFNet_test_results.csv'), index=False)
    
    # 保存统计信息
    stats_df = pd.DataFrame({
        'Metric': ['ACC', 'RRMSE', 'SNR_dB'],
        'Mean': [np.mean(acc_list), np.mean(rrmse_list), np.mean(snr_list)],
        'Std': [np.std(acc_list), np.std(rrmse_list), np.std(snr_list)],
        'Min': [np.min(acc_list), np.min(rrmse_list), np.min(snr_list)],
        'Max': [np.max(acc_list), np.max(rrmse_list), np.max(snr_list)]
    })
    
    stats_df.to_csv(os.path.join(args.result_dir, 'EEGIFNet_test_statistics.csv'), index=False)
    print(f"  详细结果: EEGIFNet_test_results.csv")
    print(f"  统计信息: EEGIFNet_test_statistics.csv")

    # 打印统计信息
    print("\n统计信息:")
    print(stats_df.to_string(index=False))

    # 可视化
    if args.visualize:
        print(f"\n生成可视化结果 (前{args.num_vis}个样本)...")
        vis_dir = os.path.join(args.result_dir, 'visualizations')
        visualize_results(I_model, M_model, device, test_loader, vis_dir, args.num_vis)

    print("\n✓ 完成！请运行统一指标计算脚本来评估所有方法。")


if __name__ == '__main__':
    main()
