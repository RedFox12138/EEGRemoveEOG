"""
DAT-Net 无监督学习可解释性可视化 - 主入口脚本

用于毕业设计：展示模型、无监督过程每一步的可解释性
通过绘图方式展示各个模块的直观作用

使用方法:
    python main.py --task all                    # 运行所有可视化任务
    python main.py --task artifact_probability   # 运行单个任务
    python main.py --sample_idx 5                # 指定样本索引
    python main.py --report                      # 生成HTML报告
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from utils import initialize, load_model, load_test_data
from config import VIS_CONFIG, DEBUG_CONFIG


# 任务映射表 - 仅保留核心可视化任务
TASK_MAPPING = {
    'artifact_probability': {
        'id': 1,
        'name': '伪影概率计算可视化',
        'module': 'visualizations.vis_artifact_probability',
        'function': 'visualize_artifact_probability',
        'description': '展示如何计算每个时间点的伪影概率',
    },
    'masking_strategy': {
        'id': 2,
        'name': '掩蔽策略可视化',
        'module': 'visualizations.vis_masking_strategy',
        'function': 'visualize_masking_strategy',
        'description': '对比随机掩蔽vs伪影感知掩蔽',
    },
    'denoising_results': {
        'id': 3,
        'name': '去噪效果可视化',
        'module': 'visualizations.vis_denoising_results',
        'function': 'visualize_denoising_results',
        'description': '展示典型样本的处理结果',
    },
}


def print_banner():
    """打印欢迎横幅"""
    banner = """
    ╔═══════════════════════════════════════════════════════════════╗
    ║                                                               ║
    ║        DAT-Net 无监督学习可解释性可视化系统                    ║
    ║        Interpretability Visualization System                  ║
    ║                                                               ║
    ║        用于毕业设计：展示模型每一步的可解释性                  ║
    ║                                                               ║
    ╚═══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def list_tasks():
    """列出所有可用任务"""
    print("\n可用的可视化任务：\n")
    print(f"{'ID':<4} {'任务名称':<30} {'描述'}")
    print("=" * 80)
    
    for task_key, task_info in sorted(TASK_MAPPING.items(), key=lambda x: x[1]['id']):
        print(f"{task_info['id']:<4} {task_info['name']:<30} {task_info['description']}")
    
    print("\n" + "=" * 80)
    print("\n使用示例:")
    print("  python main.py --task artifact_probability --sample_idx 0")
    print("  python main.py --task all")
    print("  python main.py --list")
    print()


def run_task(task_key, sample_idx=None, model=None, data_loader=None, device=None, **kwargs):
    """
    运行单个可视化任务
    
    Args:
        task_key: 任务键名
        sample_idx: 样本索引
        model: 模型（如果为None则加载）
        data_loader: 数据加载器（如果为None则加载）
        device: 设备（如果为None则自动检测）
        **kwargs: 其他参数
    """
    if task_key not in TASK_MAPPING:
        print(f"❌ 错误: 未知任务 '{task_key}'")
        print("使用 --list 查看所有可用任务")
        return False
    
    task_info = TASK_MAPPING[task_key]
    
    print(f"\n{'='*70}")
    print(f"📊 任务 {task_info['id']}: {task_info['name']}")
    print(f"{'='*70}")
    print(f"描述: {task_info['description']}\n")
    
    try:
        # 动态导入模块
        module_name = task_info['module']
        function_name = task_info['function']
        
        # 尝试导入
        try:
            module = __import__(module_name, fromlist=[function_name])
            vis_function = getattr(module, function_name)
        except (ImportError, AttributeError) as e:
            print(f"⚠️  该任务尚未实现")
            print(f"   模块: {module_name}")
            print(f"   函数: {function_name}")
            if DEBUG_CONFIG['verbose']:
                print(f"   错误: {e}")
            return False
        
        # 加载模型和数据（如果未提供）
        if model is None or data_loader is None or device is None:
            print("加载模型和数据...")
            if model is None:
                model, device = load_model(verbose=False)
            if data_loader is None:
                data_loader = load_test_data(verbose=False)
        
        # 运行可视化函数
        if sample_idx is not None:
            vis_function(model=model, data_loader=data_loader, device=device,
                        sample_idx=sample_idx, **kwargs)
        else:
            vis_function(model=model, data_loader=data_loader, device=device, **kwargs)
        
        print(f"\n✓ 任务完成: {task_info['name']}\n")
        return True
        
    except Exception as e:
        print(f"\n❌ 任务执行失败: {e}")
        if DEBUG_CONFIG['verbose']:
            import traceback
            traceback.print_exc()
        return False


def run_all_tasks(sample_idx=None):
    """
    运行所有可视化任务
    
    Args:
        sample_idx: 样本索引
    """
    print("\n" + "="*70)
    print("🚀 开始运行所有可视化任务")
    print("="*70 + "\n")
    
    # 预加载模型和数据（避免重复加载）
    print("加载模型和数据...")
    model, device = load_model(verbose=True)
    data_loader = load_test_data(verbose=True)
    print()
    
    success_count = 0
    failed_count = 0
    skipped_count = 0
    
    for task_key in sorted(TASK_MAPPING.keys(), key=lambda x: TASK_MAPPING[x]['id']):
        result = run_task(task_key, sample_idx=sample_idx, 
                         model=model, data_loader=data_loader, device=device)
        if result is True:
            success_count += 1
        elif result is False:
            skipped_count += 1
        else:
            failed_count += 1
    
    # 总结
    print("\n" + "="*70)
    print("📈 执行总结")
    print("="*70)
    print(f"✓ 成功: {success_count}")
    print(f"⚠  跳过: {skipped_count}")
    print(f"❌ 失败: {failed_count}")
    print(f"总计: {success_count + failed_count + skipped_count}")
    print("="*70 + "\n")


def generate_report():
    """生成HTML可视化报告"""
    print("\n" + "="*70)
    print("📝 生成交互式可视化报告")
    print("="*70 + "\n")
    
    try:
        from visualizations.vis_report import generate_html_report
        report_path = generate_html_report()
        print(f"\n✓ 报告已生成: {report_path}\n")
    except ImportError:
        print("⚠️  报告生成功能尚未实现")
        print("   模块: visualizations.vis_report")
    except Exception as e:
        print(f"❌ 报告生成失败: {e}")
        if DEBUG_CONFIG['verbose']:
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='DAT-Net 无监督学习可解释性可视化系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python main.py --list                              # 列出所有任务
  python main.py --task artifact_probability         # 运行单个任务
  python main.py --task all                          # 运行所有任务
  python main.py --task artifact_probability -s 5    # 指定样本索引
  python main.py --report                            # 生成HTML报告
        """
    )
    
    parser.add_argument(
        '--task', '-t',
        type=str,
        default=None,
        help='要运行的任务名称 (使用 "all" 运行所有任务)'
    )
    
    parser.add_argument(
        '--sample_idx', '-s',
        type=int,
        default=None,
        help=f'样本索引 (默认: {VIS_CONFIG["default_sample_idx"]})'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='列出所有可用任务'
    )
    
    parser.add_argument(
        '--report', '-r',
        action='store_true',
        help='生成HTML可视化报告'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='显示详细输出'
    )
    
    args = parser.parse_args()
    
    # 更新调试配置
    if args.verbose:
        DEBUG_CONFIG['verbose'] = True
    
    # 打印横幅
    print_banner()
    
    # 初始化环境
    initialize()
    
    # 处理命令
    if args.list:
        list_tasks()
        return
    
    if args.report:
        generate_report()
        return
    
    if args.task:
        # 确定样本索引
        sample_idx = args.sample_idx if args.sample_idx is not None else VIS_CONFIG['default_sample_idx']
        
        if args.task.lower() == 'all':
            run_all_tasks(sample_idx=sample_idx)
        else:
            run_task(args.task, sample_idx=sample_idx)
    else:
        # 没有指定任务，显示帮助
        parser.print_help()
        print()
        list_tasks()


if __name__ == '__main__':
    main()
