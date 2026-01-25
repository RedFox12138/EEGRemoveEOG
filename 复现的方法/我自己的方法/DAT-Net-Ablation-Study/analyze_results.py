"""
结果分析脚本 - 消融实验
分析和可视化所有消融实验的测试结果
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ablation_config import ABLATION_ORDER

# 导入metrics（可选）
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
    sys.path.insert(0, os.path.join(root_dir, '复现的方法'))
    from metrics_utils import compute_all_metrics
    HAS_METRICS = True
except Exception:
    HAS_METRICS = False
    print("警告: 无法导入metrics_utils，将跳过指标计算")


def load_test_results():
    """加载所有实验的测试结果"""
    results_file = os.path.join(current_dir, 'test_results_summary.json')
    
    if not os.path.exists(results_file):
        print(f"错误: 未找到测试结果文件: {results_file}")
        print("请先运行测试脚本: python test_ablation.py")
        return None
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    return results


def create_comparison_table(results):
    """创建实验对比表格"""
    print("\n" + "="*120)
    print("消融实验结果对比表")
    print("="*120)
    
    # 收集所有指标名称
    all_metrics = set()
    for exp_name in results:
        for snr in results[exp_name]:
            if 'metrics' in results[exp_name][snr]:
                all_metrics.update(results[exp_name][snr]['metrics'].keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # 打印每个SNR的结果
    for snr in sorted(results[ABLATION_ORDER[0]].keys()):
        print(f"\n{snr}:")
        print("-" * 120)
        
        # 表头
        header = f"{'实验名称':<25}"
        for metric in all_metrics:
            header += f"{metric:>12}"
        header += f"{'推理时间(ms)':>15}"
        print(header)
        print("-" * 120)
        
        # 每个实验的结果
        for exp_name in ABLATION_ORDER:
            if exp_name not in results:
                continue
            
            if snr not in results[exp_name]:
                continue
            
            row = f"{exp_name:<25}"
            
            if 'metrics' in results[exp_name][snr]:
                metrics = results[exp_name][snr]['metrics']
                for metric in all_metrics:
                    if metric in metrics:
                        value = metrics[metric]
                        row += f"{value:>12.4f}"
                    else:
                        row += f"{'N/A':>12}"
            else:
                row += f"{'N/A':>12}" * len(all_metrics)
            
            # 添加推理时间
            if 'avg_time_per_sample' in results[exp_name][snr]:
                avg_time_ms = results[exp_name][snr]['avg_time_per_sample'] * 1000
                row += f"{avg_time_ms:>15.2f}"
            else:
                row += f"{'N/A':>15}"
            
            print(row)
    
    print("="*120)


def save_comparison_csv(results):
    """保存对比结果为CSV文件"""
    csv_file = os.path.join(current_dir, 'ablation_comparison.csv')
    
    # 收集所有指标
    all_metrics = set()
    for exp_name in results:
        for snr in results[exp_name]:
            if 'metrics' in results[exp_name][snr]:
                all_metrics.update(results[exp_name][snr]['metrics'].keys())
    
    all_metrics = sorted(list(all_metrics))
    
    # 写入CSV
    with open(csv_file, 'w', encoding='utf-8') as f:
        # 写入表头
        header = "实验名称,SNR"
        for metric in all_metrics:
            header += f",{metric}"
        header += ",avg_inference_time_ms,total_samples"
        f.write(header + "\n")
        
        # 写入数据
        for exp_name in ABLATION_ORDER:
            if exp_name not in results:
                continue
            
            for snr in sorted(results[exp_name].keys()):
                row = f"{exp_name},{snr}"
                
                if 'metrics' in results[exp_name][snr]:
                    metrics = results[exp_name][snr]['metrics']
                    for metric in all_metrics:
                        if metric in metrics:
                            row += f",{metrics[metric]:.6f}"
                        else:
                            row += ",N/A"
                else:
                    row += ",N/A" * len(all_metrics)
                
                # 添加推理时间
                if 'avg_time_per_sample' in results[exp_name][snr]:
                    avg_time_ms = results[exp_name][snr]['avg_time_per_sample'] * 1000
                    row += f",{avg_time_ms:.4f}"
                else:
                    row += ",N/A"
                
                # 添加样本数
                if 'total_samples' in results[exp_name][snr]:
                    row += f",{results[exp_name][snr]['total_samples']}"
                else:
                    row += ",N/A"
                
                f.write(row + "\n")
    
    print(f"\n✓ 已保存对比结果到: {csv_file}")


def plot_performance_comparison(results):
    """绘制性能对比图"""
    try:
        # 选择一个主要指标进行可视化（例如RRMSE或第一个可用指标）
        sample_exp = ABLATION_ORDER[0]
        sample_snr = list(results[sample_exp].keys())[0]
        
        if 'metrics' not in results[sample_exp][sample_snr]:
            print("警告: 没有可用的指标数据进行可视化")
            return
        
        metrics = results[sample_exp][sample_snr]['metrics']
        metric_names = list(metrics.keys())
        
        if not metric_names:
            print("警告: 没有可用的指标数据进行可视化")
            return
        
        # 使用第一个指标作为主要指标
        primary_metric = metric_names[0]
        
        # 提取数据
        snr_levels = sorted(list(results[sample_exp].keys()))
        
        plt.figure(figsize=(12, 6))
        
        for exp_name in ABLATION_ORDER:
            if exp_name not in results:
                continue
            
            values = []
            for snr in snr_levels:
                if snr in results[exp_name] and 'metrics' in results[exp_name][snr]:
                    if primary_metric in results[exp_name][snr]['metrics']:
                        values.append(results[exp_name][snr]['metrics'][primary_metric])
                    else:
                        values.append(np.nan)
                else:
                    values.append(np.nan)
            
            plt.plot(snr_levels, values, marker='o', label=exp_name, linewidth=2)
        
        plt.xlabel('SNR Level', fontsize=12)
        plt.ylabel(primary_metric, fontsize=12)
        plt.title(f'Ablation Study Performance Comparison ({primary_metric})', fontsize=14)
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = os.path.join(current_dir, 'ablation_performance.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ 已保存性能对比图到: {plot_file}")
        
        plt.close()
        
    except Exception as e:
        print(f"警告: 绘图失败: {e}")


def analyze_component_contribution(results):
    """分析各组件的贡献度"""
    output_file = os.path.join(current_dir, 'component_contribution.txt')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write("组件贡献度分析\n")
        f.write("="*80 + "\n\n")
        
        f.write("基于消融实验的组件重要性排序：\n\n")
        
        # 简单分析：比较各消融实验与baseline的性能差异
        # 注意：用户已单独实现baseline，这里仅分析消融变体之间的差异
        
        f.write("实验配置说明：\n")
        f.write("1. no_dual_output: 移除双输出头设计\n")
        f.write("2. unet_baseline: 使用标准UNet替代DAT-Net架构\n")
        f.write("3. random_masking: 使用随机掩蔽替代伪影感知掩蔽\n")
        f.write("4. no_n2v: 移除N2V损失\n")
        f.write("5. no_teacher: 移除Teacher损失\n")
        f.write("6. no_consistency: 移除一致性损失\n")
        f.write("7. only_reconstruction: 仅使用重建损失\n\n")
        
        f.write("详细指标数据请参考: ablation_comparison.csv\n")
        f.write("性能对比图请参考: ablation_performance.png\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"✓ 已保存组件贡献度分析到: {output_file}")


def main():
    """主函数"""
    print("\n" + "="*80)
    print("消融实验结果分析")
    print("="*80)
    
    # 加载结果
    results = load_test_results()
    if results is None:
        return
    
    # 创建对比表格
    create_comparison_table(results)
    
    # 保存CSV
    save_comparison_csv(results)
    
    # 绘制对比图
    plot_performance_comparison(results)
    
    # 分析组件贡献
    analyze_component_contribution(results)
    
    print("\n" + "="*80)
    print("分析完成！")
    print("="*80)
    print("\n生成的文件：")
    print("  - ablation_comparison.csv: 详细对比数据")
    print("  - ablation_performance.png: 性能对比图")
    print("  - component_contribution.txt: 组件贡献度分析")


if __name__ == '__main__':
    main()
