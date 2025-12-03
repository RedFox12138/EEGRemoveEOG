"""
批量更新脚本 - 将所有方法的数据路径替换为配置导入
增强版：更智能的替换和验证
"""
import os
import re
import glob
import shutil
from datetime import datetime

# 定义替换规则（按优先级排序）
REPLACEMENTS = [
    # 1. 数据目录路径
    (
        r"data_dir\s*=\s*r['\"]D:\\\\Pycharm_Projects\\\\EOG Remove\\\\生成半模拟数据\\\\已经生成好的数据['\"]",
        "# 数据目录已迁移到 data_config.py\n    # data_dir = DATA_DIR  # 现在从配置文件导入",
        "替换数据目录定义"
    ),
    
    # 2. 文件路径 - 使用f-string的情况
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Train_Contaminated\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(TRAIN_CONTAMINATED_PATH)[DATA_KEY]",
        "替换训练集污染数据加载(f-string)"
    ),
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Train_Pure\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(TRAIN_PURE_PATH)[DATA_KEY]",
        "替换训练集纯净数据加载(f-string)"
    ),
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Val_Contaminated\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(VAL_CONTAMINATED_PATH)[DATA_KEY]",
        "替换验证集污染数据加载(f-string)"
    ),
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Val_Pure\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(VAL_PURE_PATH)[DATA_KEY]",
        "替换验证集纯净数据加载(f-string)"
    ),
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Test_Contaminated\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]",
        "替换测试集污染数据加载(f-string)"
    ),
    (
        r"scipy\.io\.loadmat\(f['\"]{{data_dir}}/Test_Pure\.mat['\"]\)\[['\"]data['\"]\]",
        "scipy.io.loadmat(TEST_PURE_PATH)[DATA_KEY]",
        "替换测试集纯净数据加载(f-string)"
    ),
    
    # 3. 采样率
    (
        r"compute_all_metrics\(([^,]+),\s*([^,]+),\s*fs\s*=\s*200\s*\)",
        r"compute_all_metrics(\1, \2, fs=SAMPLING_RATE)",
        "替换metrics计算中的采样率"
    ),
]

def add_import_if_needed(content, file_path):
    """在文件中添加配置导入(如果还没有)"""
    if 'from data_config import' in content or 'import data_config' in content:
        return content  # 已经有导入了
    
    # 查找合适的位置插入导入
    lines = content.split('\n')
    insert_pos = 0
    
    # 找到最后一个import语句的位置
    for i, line in enumerate(lines):
        if line.strip().startswith('import ') or line.strip().startswith('from '):
            insert_pos = i + 1
    
    # 插入导入语句
    import_statement = "from data_config import *  # 数据集配置"
    lines.insert(insert_pos, import_statement)
    
    return '\n'.join(lines)

def update_file(file_path):
    """更新单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 应用所有替换规则
        for pattern, replacement in REPLACEMENTS:
            if callable(replacement):
                content = re.sub(pattern, replacement, content)
            else:
                content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化,添加导入
        if content != original_content:
            content = add_import_if_needed(content, file_path)
            
            # 保存文件
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return True, "已更新"
        else:
            return False, "无需更新"
    
    except Exception as e:
        return False, f"错误: {str(e)}"

def main():
    """主函数"""
    base_path = r'D:\Pycharm_Projects\EOG Remove\复现的方法'
    
    # 要更新的方法目录
    methods = [
        'ASNet-main',
        'EEGIFNet-master', 
        'MicroWaveNet',
        'noise2void',
        'self-supervised',
        'Self2Self',
    ]
    
    print("=" * 80)
    print("批量更新数据路径配置")
    print("=" * 80)
    
    total_files = 0
    updated_files = 0
    
    for method in methods:
        method_path = os.path.join(base_path, method)
        if not os.path.exists(method_path):
            print(f"\n⚠️  目录不存在: {method}")
            continue
        
        print(f"\n处理: {method}")
        print("-" * 80)
        
        # 查找所有.py文件
        py_files = glob.glob(os.path.join(method_path, '*.py'))
        
        for py_file in py_files:
            filename = os.path.basename(py_file)
            
            # 跳过配置文件本身和一些特殊文件
            if filename in ['data_config.py', 'config.py', '__init__.py', 'generate_configs.py']:
                continue
            
            total_files += 1
            updated, message = update_file(py_file)
            
            if updated:
                updated_files += 1
                print(f"  ✓ {filename}: {message}")
            else:
                print(f"  - {filename}: {message}")
    
    print("\n" + "=" * 80)
    print(f"完成! 共处理 {total_files} 个文件, 更新了 {updated_files} 个文件")
    print("=" * 80)
    print("\n⚠️  注意: 请手动检查更新后的文件,确保逻辑正确!")
    print("建议使用 git diff 查看具体更改")

if __name__ == '__main__':
    main()
