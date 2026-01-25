"""
消融实验主控制脚本
一键运行完整的消融实验流程：训练所有模型变体 + 测试并生成.mat文件

使用方法:
    python run_ablation_study.py --train --test
    python run_ablation_study.py --train  # 仅训练
    python run_ablation_study.py --test   # 仅测试（需要已有训练好的模型）
"""
import os
import sys
import argparse
from time import time
from datetime import datetime

# 添加当前目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from ablation_config import print_ablation_summary, ABLATION_ORDER


def run_training(selected_experiments=None, skip_existing=False):
    """运行训练流程
    
    Args:
        selected_experiments: 要执行的实验名称列表，None表示全部执行
        skip_existing: 是否跳过已有模型的实验
    """
    print("\n" + "="*80)
    print("第一阶段: 自动化训练所有消融实验变体")
    print("="*80)
    
    # 导入并运行训练脚本
    from train_ablation import main as train_main
    
    train_start = time()
    train_main(selected_experiments=selected_experiments, skip_existing=skip_existing)
    train_time = time() - train_start
    
    print(f"\n训练阶段完成！总耗时: {train_time/60:.2f} 分钟 ({train_time/3600:.2f} 小时)")
    return train_time


def run_testing(selected_experiments=None):
    """运行测试流程
    
    Args:
        selected_experiments: 要执行的实验名称列表，None表示全部执行
    """
    print("\n" + "="*80)
    print("第二阶段: 自动化测试所有消融实验变体并生成.mat文件")
    print("="*80)
    
    # 导入并运行测试脚本
    from test_ablation import main as test_main
    
    test_start = time()
    test_main(selected_experiments=selected_experiments)
    test_time = time() - test_start
    
    print(f"\n测试阶段完成！总耗时: {test_time/60:.2f} 分钟")
    return test_time


def main():
    """主控制函数"""
    parser = argparse.ArgumentParser(
        description='DAT-Net 消融实验主控制脚本',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_ablation_study.py --train --test                    # 完整流程（训练+测试）
  python run_ablation_study.py --train                           # 仅训练
  python run_ablation_study.py --test                            # 仅测试
  python run_ablation_study.py --summary                         # 显示实验配置概览
  python run_ablation_study.py --train --experiments unet_baseline no_n2v  # 仅训练指定实验
  python run_ablation_study.py --train --skip-existing           # 跳过已训练的实验
        """
    )
    
    parser.add_argument('--train', action='store_true', 
                       help='执行训练阶段')
    parser.add_argument('--test', action='store_true', 
                       help='执行测试阶段')
    parser.add_argument('--summary', action='store_true', 
                       help='显示消融实验配置概览')
    parser.add_argument('--experiments', nargs='+', 
                       help='指定要执行的实验名称（多个实验用空格分隔）')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                       help='自动跳过已有Loss最佳模型的实验（默认启用，使用--no-skip-existing禁用）')
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false',
                       help='强制重新训练所有实验（即使模型已存在）')
    
    args = parser.parse_args()
    
    # 如果没有指定任何参数，默认运行完整流程
    if not (args.train or args.test or args.summary):
        args.train = True
        args.test = True
    
    # 打印欢迎信息
    print("="*80)
    print("DAT-Net 消融实验自动化系统")
    print("="*80)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 打印数据集配置
    print("\n" + "-"*80)
    print("数据集配置信息:")
    print("-"*80)
    
    # 导入配置
    v2_dir = os.path.join(os.path.dirname(current_dir), 'DAT-Net-Unsupervised-v2')
    sys.path.insert(0, v2_dir)
    import config as base_config
    
    print(f"数据集类型: {base_config.DATASET_NAME}")
    print(f"采样率: {base_config.SAMPLING_RATE} Hz")
    print(f"窗口大小: {base_config.WINDOW_SIZE} 样本点")
    print(f"\n训练数据:")
    print(f"  污染数据: {base_config.TRAIN_CONTAMINATED_PATH}")
    print(f"  纯净数据: {base_config.TRAIN_PURE_PATH}")
    print(f"\n验证数据:")
    print(f"  污染数据: {base_config.VAL_CONTAMINATED_PATH}")
    print(f"  纯净数据: {base_config.VAL_PURE_PATH}")
    
    if base_config.TEST_SNR_LEVELS:
        print(f"\n测试集SNR级别: {base_config.TEST_SNR_LEVELS}")
    elif base_config.TEST_CONTAMINATED_PATH:
        print(f"\n测试数据:")
        print(f"  污染数据: {base_config.TEST_CONTAMINATED_PATH}")
        print(f"  纯净数据: {base_config.TEST_PURE_PATH}")
    
    print("-"*80)
    
    # 打印训练配置
    print("\n" + "-"*80)
    print("训练配置参数:")
    print("-"*80)
    print(f"训练轮数: {base_config.EPOCHS}")
    print(f"批次大小: {base_config.BATCH_SIZE}")
    print(f"学习率: {base_config.LEARNING_RATE}")
    print(f"最小学习率: {base_config.MIN_LR}")
    print(f"权重衰减: {base_config.WEIGHT_DECAY}")
    print(f"使用学习率调度: {base_config.USE_LR_SCHEDULER}")
    print(f"梯度裁剪: {base_config.GRAD_CLIP}")
    print(f"早停耐心值: {base_config.PATIENCE}")
    print(f"RRMSE回退耐心值: {base_config.RRMSE_ROLLBACK_PATIENCE}")
    
    print(f"\n损失函数权重:")
    print(f"  LAMBDA_REC (重建): {base_config.LAMBDA_REC}")
    print(f"  LAMBDA_CON (一致性): {base_config.LAMBDA_CON}")
    print(f"  LAMBDA_TEACHER (教师): {base_config.LAMBDA_TEACHER}")
    print(f"  LAMBDA_N2V (N2V): {base_config.LAMBDA_N2V}")
    print(f"  LAMBDA_BAND (频带): {base_config.LAMBDA_BAND}")
    print(f"  LAMBDA_LOW (低频平滑): {base_config.LAMBDA_LOW}")
    print(f"  LAMBDA_DECOR (解耦): {base_config.LAMBDA_DECOR}")
    print(f"  LAMBDA_CONTENT (内容保持): {base_config.LAMBDA_CONTENT}")
    
    print(f"\n其他超参数:")
    print(f"  MASK_BASE: {base_config.MASK_BASE}")
    print(f"  BOOST_SCALE: {base_config.BOOST_SCALE}")
    print(f"  GAMMA_ART_WEIGHT: {base_config.GAMMA_ART_WEIGHT}")
    print(f"  TEACHER_CUTOFF: {base_config.TEACHER_CUTOFF} Hz")
    print(f"  LOWPASS_CUTOFF: {base_config.LOWPASS_CUTOFF} Hz")
    print(f"  TEACHER_THRESHOLD: {base_config.TEACHER_THRESHOLD}")
    
    print("-"*80)
    
    # 显示实验概览
    if args.summary or args.train or args.test:
        print_ablation_summary()
    
    if args.summary:
        return
    
    # 验证并处理实验选择
    selected_experiments = None
    if args.experiments:
        # 验证实验名称
        invalid_exps = [exp for exp in args.experiments if exp not in ABLATION_ORDER]
        if invalid_exps:
            print(f"\n❌ 错误: 无效的实验名称: {invalid_exps}")
            print(f"可用的实验名称: {ABLATION_ORDER}")
            return
        selected_experiments = args.experiments
        print(f"\n✅ 将执行以下实验: {selected_experiments}")
    
    if args.skip_existing:
        print(f"\n✅ 将自动跳过已有模型的实验")
    
    # 记录总耗时
    total_start = time()
    train_time = 0
    test_time = 0
    
    # 执行训练
    if args.train:
        train_time = run_training(
            selected_experiments=selected_experiments,
            skip_existing=args.skip_existing
        )
    
    # 执行测试
    if args.test:
        test_time = run_testing(selected_experiments=selected_experiments)
    
    # 完成汇总
    total_time = time() - total_start
    
    print("\n" + "="*80)
    print("消融实验全部完成！")
    print("="*80)
    
    if args.train:
        print(f"训练耗时: {train_time/60:.2f} 分钟 ({train_time/3600:.2f} 小时)")
    if args.test:
        print(f"测试耗时: {test_time/60:.2f} 分钟")
    
    print(f"总耗时: {total_time/60:.2f} 分钟 ({total_time/3600:.2f} 小时)")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n实验结果保存位置:")
    print(f"  - 模型检查点: {os.path.join(current_dir, 'checkpoints')}")
    print(f"  - 测试结果(.mat文件): {os.path.join(current_dir, 'results')}")
    print(f"  - 训练历史: {os.path.join(current_dir, 'training_history.json')}")
    print(f"  - 测试汇总: {os.path.join(current_dir, 'test_results_summary.json')}")
    
    print("\n下一步:")
    print("  1. 查看 training_history.json 了解训练过程")
    print("  2. 查看 test_results_summary.json 了解测试性能")
    print("  3. 使用现有的评估工具分析 results/ 目录中的.mat文件")
    print("  4. 可视化不同消融配置的性能对比")
    
    print("="*80)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n实验被用户中断！")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
