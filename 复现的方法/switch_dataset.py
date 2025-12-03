"""
数据集快速切换工具
用于在不同数据集之间快速切换配置
"""
import os
import re
import sys

# 配置文件路径
GLOBAL_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_config.py')
DATNET_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   '我自己的方法', 'DAT-Net-Unsupervised-v2', 'config.py')

AVAILABLE_DATASETS = ['semi_simulated', 'fully_simulated']


def get_current_dataset(config_path):
    """获取配置文件中当前设置的数据集"""
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 查找DEFAULT_DATASET或DATASET_NAME
    match = re.search(r"(DEFAULT_DATASET|DATASET_NAME)\s*=\s*['\"](\w+)['\"]", content)
    if match:
        return match.group(2)
    return None


def set_dataset(config_path, dataset_name, var_name='DEFAULT_DATASET'):
    """设置配置文件中的数据集"""
    if dataset_name not in AVAILABLE_DATASETS:
        print(f"错误: 未知数据集 '{dataset_name}'")
        print(f"可用数据集: {', '.join(AVAILABLE_DATASETS)}")
        return False
    
    with open(config_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 替换数据集设置
    pattern = rf"({var_name}\s*=\s*['\"])(\w+)(['\"])"
    new_content = re.sub(pattern, rf"\1{dataset_name}\3", content)
    
    if new_content == content:
        print(f"警告: 未找到{var_name}配置项")
        return False
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True


def show_status():
    """显示当前所有配置的状态"""
    print("=" * 80)
    print("数据集配置状态")
    print("=" * 80)
    
    # 全局配置
    if os.path.exists(GLOBAL_CONFIG_PATH):
        current = get_current_dataset(GLOBAL_CONFIG_PATH)
        print(f"\n全局配置 (dataset_config.py):")
        print(f"  当前数据集: {current}")
    else:
        print(f"\n⚠️  找不到全局配置文件: {GLOBAL_CONFIG_PATH}")
    
    # DAT-Net配置
    if os.path.exists(DATNET_CONFIG_PATH):
        current = get_current_dataset(DATNET_CONFIG_PATH)
        print(f"\nDAT-Net-Unsupervised-v2 配置:")
        print(f"  当前数据集: {current}")
    else:
        print(f"\n⚠️  找不到DAT-Net配置文件: {DATNET_CONFIG_PATH}")
    
    print("\n可用数据集:")
    for ds in AVAILABLE_DATASETS:
        print(f"  - {ds}")
    print("=" * 80)


def switch_global(dataset_name):
    """切换全局数据集配置"""
    print(f"\n切换全局数据集到: {dataset_name}")
    if set_dataset(GLOBAL_CONFIG_PATH, dataset_name, 'DEFAULT_DATASET'):
        print("✓ 全局配置已更新")
        return True
    else:
        print("✗ 全局配置更新失败")
        return False


def switch_datnet(dataset_name):
    """切换DAT-Net数据集配置"""
    print(f"\n切换DAT-Net数据集到: {dataset_name}")
    if set_dataset(DATNET_CONFIG_PATH, dataset_name, 'DATASET_NAME'):
        print("✓ DAT-Net配置已更新")
        return True
    else:
        print("✗ DAT-Net配置更新失败")
        return False


def switch_all(dataset_name):
    """同时切换所有配置"""
    print(f"\n切换所有配置到数据集: {dataset_name}")
    print("=" * 80)
    
    success1 = switch_global(dataset_name)
    success2 = switch_datnet(dataset_name)
    
    print("\n" + "=" * 80)
    if success1 and success2:
        print("✓ 所有配置已成功切换!")
        print(f"\n现在所有模块将使用: {dataset_name}")
        
        # 显示数据集信息
        try:
            sys.path.insert(0, os.path.dirname(GLOBAL_CONFIG_PATH))
            from dataset_config import get_dataset_config
            config = get_dataset_config(dataset_name)
            print(f"\n数据集信息:")
            print(f"  名称: {config['name']}")
            print(f"  采样率: {config['sampling_rate']} Hz")
            print(f"  窗口大小: {config['window_size']} 样本")
            print(f"  数据目录: {config['data_dir']}")
        except:
            pass
    else:
        print("⚠️  部分配置更新失败")
    print("=" * 80)


def interactive_mode():
    """交互式模式"""
    while True:
        print("\n" + "=" * 80)
        print("数据集配置管理工具 - 交互模式")
        print("=" * 80)
        print("\n请选择操作:")
        print("  1. 查看当前状态")
        print("  2. 切换到半模拟数据集 (semi_simulated, 200Hz)")
        print("  3. 切换到全模拟数据集 (fully_simulated, 250Hz)")
        print("  4. 仅切换全局配置")
        print("  5. 仅切换DAT-Net配置")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-5): ").strip()
        
        if choice == '0':
            print("\n退出程序。")
            break
        elif choice == '1':
            show_status()
        elif choice == '2':
            switch_all('semi_simulated')
        elif choice == '3':
            switch_all('fully_simulated')
        elif choice == '4':
            ds = input("输入数据集名称: ").strip()
            switch_global(ds)
        elif choice == '5':
            ds = input("输入数据集名称: ").strip()
            switch_datnet(ds)
        else:
            print("⚠️  无效选项,请重试")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据集配置管理工具')
    parser.add_argument('--status', action='store_true', help='显示当前配置状态')
    parser.add_argument('--switch', type=str, choices=AVAILABLE_DATASETS, 
                       help='切换到指定数据集')
    parser.add_argument('--global-only', action='store_true', 
                       help='仅切换全局配置')
    parser.add_argument('--datnet-only', action='store_true', 
                       help='仅切换DAT-Net配置')
    parser.add_argument('--interactive', '-i', action='store_true', 
                       help='进入交互模式')
    
    args = parser.parse_args()
    
    # 如果没有参数,显示状态
    if len(sys.argv) == 1:
        show_status()
        print("\n提示: 使用 --help 查看所有选项,或使用 -i 进入交互模式")
        return
    
    if args.interactive:
        interactive_mode()
    elif args.status:
        show_status()
    elif args.switch:
        if args.global_only:
            switch_global(args.switch)
        elif args.datnet_only:
            switch_datnet(args.switch)
        else:
            switch_all(args.switch)


if __name__ == '__main__':
    main()
