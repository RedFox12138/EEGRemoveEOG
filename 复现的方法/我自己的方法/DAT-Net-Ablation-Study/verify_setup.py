"""
系统验证脚本
用于验证消融实验环境是否配置正确
"""
import os
import sys

def check_dependencies():
    """检查依赖是否已安装"""
    print("检查Python依赖...")
    required_packages = ['torch', 'numpy', 'scipy']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (缺失)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n缺失的包: {', '.join(missing_packages)}")
        print("请运行: pip install " + " ".join(missing_packages))
        return False
    return True


def check_directories():
    """检查必要的目录结构"""
    print("\n检查目录结构...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    required_dirs = {
        'DAT-Net': os.path.join(parent_dir, 'DAT-Net'),
        'DAT-Net-Unsupervised-v2': os.path.join(parent_dir, 'DAT-Net-Unsupervised-v2'),
    }
    
    all_exist = True
    for name, path in required_dirs.items():
        if os.path.isdir(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (不存在: {path})")
            all_exist = False
    
    return all_exist


def check_files():
    """检查必要的文件"""
    print("\n检查关键文件...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    
    required_files = {
        'model.py': os.path.join(parent_dir, 'DAT-Net', 'model.py'),
        'unsupervised_artifact_v2.py': os.path.join(parent_dir, 'DAT-Net-Unsupervised-v2', 'unsupervised_artifact_v2.py'),
        'config.py': os.path.join(parent_dir, 'DAT-Net-Unsupervised-v2', 'config.py'),
    }
    
    all_exist = True
    for name, path in required_files.items():
        if os.path.isfile(path):
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} (不存在: {path})")
            all_exist = False
    
    return all_exist


def test_imports():
    """测试关键模块导入"""
    print("\n测试模块导入...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, current_dir)
    
    tests = [
        ('ablation_config', 'from ablation_config import ABLATION_CONFIGS'),
        ('model_wrapper', 'from model_wrapper import create_model'),
        ('loss_wrapper', 'from loss_wrapper import unsupervised_dat_loss_ablation'),
    ]
    
    all_passed = True
    for name, import_stmt in tests:
        try:
            exec(import_stmt)
            print(f"  ✓ {name}")
        except Exception as e:
            print(f"  ✗ {name} (错误: {e})")
            all_passed = False
    
    return all_passed


def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    try:
        import torch
        from model_wrapper import create_model
        
        # 测试完整配置
        config_full = {'use_tcn': True}
        model = create_model(config_full)
        print(f"  ✓ 创建完整模型（使用TCN）")
        
        # 测试消融配置
        config_no_tcn = {'use_tcn': False}
        model = create_model(config_no_tcn)
        print(f"  ✓ 创建消融模型（不使用TCN）")
        
        # 测试前向传播
        x = torch.randn(2, 1, 512)
        c, a = model(x)
        if c.shape == (2, 1, 512) and a.shape == (2, 1, 512):
            print(f"  ✓ 模型前向传播正常")
            return True
        else:
            print(f"  ✗ 输出形状不正确: clean={c.shape}, artifact={a.shape}")
            return False
            
    except Exception as e:
        print(f"  ✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_function():
    """测试损失函数"""
    print("\n测试损失函数...")
    try:
        import torch
        from loss_wrapper import unsupervised_dat_loss_ablation
        
        # 创建简单模型用于测试
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
            def forward(self, x):
                return x * 0.5, x * 0.5
        
        model = SimpleModel()
        x = torch.randn(2, 1, 512)
        
        # 测试完整配置
        config_full = {
            'use_consistency': True,
            'use_artifact_weight': True,
            'use_n2v': True,
            'use_teacher': True,
            'use_band': True,
            'use_low': True,
            'use_decor': True,
            'use_content': True,
        }
        
        loss, loss_dict, _ = unsupervised_dat_loss_ablation(
            model, x, 200.0, config_full, lambda_n2v=0.1
        )
        print(f"  ✓ 完整配置损失计算: {loss.item():.4f}")
        
        # 测试最小配置
        config_minimal = {k: False for k in config_full.keys()}
        loss, loss_dict, _ = unsupervised_dat_loss_ablation(
            model, x, 200.0, config_minimal
        )
        print(f"  ✓ 最小配置损失计算: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ 损失函数测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主验证流程"""
    print("="*70)
    print("DAT-Net 消融实验系统验证")
    print("="*70)
    
    results = {
        '依赖检查': check_dependencies(),
        '目录结构': check_directories(),
        '关键文件': check_files(),
        '模块导入': test_imports(),
        '模型创建': test_model_creation(),
        '损失函数': test_loss_function(),
    }
    
    print("\n" + "="*70)
    print("验证结果汇总")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{test_name:<15} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ 所有检查通过！系统已就绪。")
        print("\n下一步：")
        print("  1. 查看配置: python ablation_config.py")
        print("  2. 运行实验: python run_ablation_study.py")
    else:
        print("✗ 部分检查失败，请修复上述问题后重试。")
        print("\n建议：")
        print("  1. 检查是否安装了所有依赖")
        print("  2. 确认目录结构正确")
        print("  3. 查看详细错误信息")
    print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
