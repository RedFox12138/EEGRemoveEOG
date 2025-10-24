"""
快速运行示例脚本
用于快速测试EEGIFNet的新训练和测试流程
"""
import os
import subprocess
import sys

def print_section(title):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def run_command(cmd, description):
    print(f">>> {description}")
    print(f">>> 命令: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0

def main():
    print_section("EEGIFNet 快速运行示例")
    
    # 检查数据路径
    data_path = r"D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据"
    contaminated_file = os.path.join(data_path, "Contaminated.npy")
    pure_file = os.path.join(data_path, "Pure_Data.npy")
    
    print("检查数据文件...")
    if os.path.exists(contaminated_file) and os.path.exists(pure_file):
        print(f"✓ 找到数据文件:")
        print(f"  - {contaminated_file}")
        print(f"  - {pure_file}")
    else:
        print(f"✗ 未找到数据文件！请检查路径:")
        print(f"  - {data_path}")
        return
    
    # 选择运行模式
    print_section("选择运行模式")
    print("1. 训练模型 (快速测试: 5个epoch)")
    print("2. 训练模型 (完整训练: 80个epoch)")
    print("3. 测试模型 (需要已有训练好的模型)")
    print("4. 测试模型 (带可视化)")
    print("5. 查看帮助")
    
    choice = input("\n请选择 (1-5): ").strip()
    
    if choice == "1":
        # 快速训练测试
        print_section("快速训练测试 (5个epoch)")
        cmd = (
            f'python train_new.py '
            f'--data_path "{data_path}" '
            f'--epochs 5 '
            f'--batch_size 256 '
            f'--lr 5e-5 '
            f'--device cuda:0'
        )
        success = run_command(cmd, "开始快速训练")
        if success:
            print("\n✓ 快速训练完成！模型保存在 ./checkpoint/")
        
    elif choice == "2":
        # 完整训练
        print_section("完整训练 (80个epoch)")
        confirm = input("这将需要较长时间，确认继续? (y/n): ").strip().lower()
        if confirm == 'y':
            cmd = (
                f'python train_new.py '
                f'--data_path "{data_path}" '
                f'--epochs 80 '
                f'--batch_size 256 '
                f'--lr 5e-5 '
                f'--device cuda:0'
            )
            success = run_command(cmd, "开始完整训练")
            if success:
                print("\n✓ 训练完成！模型保存在 ./checkpoint/")
        else:
            print("已取消")
    
    elif choice == "3":
        # 测试模型
        print_section("测试模型")
        
        # 检查模型文件
        inet_path = "./checkpoint/EEGIFNet_INet_best.pkl"
        mnet_path = "./checkpoint/EEGIFNet_MNet_best.pkl"
        
        if not os.path.exists(inet_path) or not os.path.exists(mnet_path):
            print(f"✗ 未找到模型文件！")
            print(f"  需要: {inet_path}")
            print(f"       {mnet_path}")
            print(f"\n请先运行训练 (选项1或2)")
            return
        
        print(f"✓ 找到模型文件")
        cmd = (
            f'python test_new.py '
            f'--data_path "{data_path}" '
            f'--batch_size 256 '
            f'--device cuda:0'
        )
        success = run_command(cmd, "开始测试")
        if success:
            print("\n✓ 测试完成！结果保存在 ./result/")
            print("  - EEGIFNet_test_results.csv (详细结果)")
            print("  - EEGIFNet_test_statistics.csv (统计信息)")
    
    elif choice == "4":
        # 测试模型 + 可视化
        print_section("测试模型 (带可视化)")
        
        # 检查模型文件
        inet_path = "./checkpoint/EEGIFNet_INet_best.pkl"
        mnet_path = "./checkpoint/EEGIFNet_MNet_best.pkl"
        
        if not os.path.exists(inet_path) or not os.path.exists(mnet_path):
            print(f"✗ 未找到模型文件！请先训练模型")
            return
        
        num_vis = input("可视化多少个样本? (默认10): ").strip()
        if not num_vis:
            num_vis = "10"
        
        print(f"✓ 将可视化 {num_vis} 个样本")
        cmd = (
            f'python test_new.py '
            f'--data_path "{data_path}" '
            f'--batch_size 256 '
            f'--device cuda:0 '
            f'--visualize '
            f'--num_vis {num_vis}'
        )
        success = run_command(cmd, "开始测试和可视化")
        if success:
            print("\n✓ 测试完成！")
            print("  - 结果: ./result/")
            print("  - 可视化: ./result/visualizations/")
    
    elif choice == "5":
        # 显示帮助
        print_section("帮助信息")
        print("训练参数:")
        print("  --data_path: 数据路径")
        print("  --batch_size: 批大小 (默认256)")
        print("  --lr: 学习率 (默认5e-5)")
        print("  --epochs: 训练轮数 (默认80)")
        print("  --device: 设备 (默认cuda:0)")
        print("  --save_dir: 模型保存目录 (默认./checkpoint)")
        print("\n测试参数:")
        print("  --data_path: 数据路径")
        print("  --batch_size: 批大小")
        print("  --device: 设备")
        print("  --checkpoint_dir: 模型目录 (默认./checkpoint)")
        print("  --visualize: 生成可视化")
        print("  --num_vis: 可视化数量")
        print("\n查看详细说明: README_NEW.md")
    
    else:
        print("无效的选择")
    
    print_section("完成")

if __name__ == "__main__":
    main()
