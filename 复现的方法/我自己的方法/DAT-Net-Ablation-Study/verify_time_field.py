"""
验证测试结果MAT文件中是否包含正确的时间字段
"""
import os
import scipy.io
import glob

def verify_time_fields():
    """检查results目录中的MAT文件是否有time_per_sample字段"""
    # 项目根目录的results文件夹
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results_dir = os.path.join(project_root, 'results')
    
    if not os.path.exists(results_dir):
        print(f"❌ results目录不存在: {results_dir}")
        return
    
    print(f"扫描目录: {results_dir}")
    print("="*80)
    
    # 查找消融实验的MAT文件
    pattern = os.path.join(results_dir, '*_SNR*dB.mat')
    mat_files = glob.glob(pattern)
    
    if not mat_files:
        print("⚠️  未找到任何测试结果文件")
        print(f"   搜索模式: {pattern}")
        return
    
    print(f"找到 {len(mat_files)} 个测试结果文件\n")
    
    has_time_field_count = 0
    missing_time_field = []
    
    for mat_file in sorted(mat_files)[:5]:  # 只检查前5个文件
        filename = os.path.basename(mat_file)
        print(f"检查文件: {filename}")
        
        try:
            data = scipy.io.loadmat(mat_file)
            
            # 检查关键字段
            has_time_per_sample = 'time_per_sample' in data
            has_avg_time_per_sample = 'avg_time_per_sample' in data
            has_denoised = 'denoised_data' in data
            
            if has_time_per_sample:
                time_value = data['time_per_sample'][0, 0]
                print(f"  ✓ time_per_sample: {time_value:.6f}s ({time_value*1000:.2f}ms)")
                has_time_field_count += 1
            else:
                print(f"  ✗ 缺少 time_per_sample")
                missing_time_field.append(filename)
            
            if has_avg_time_per_sample:
                avg_time_value = data['avg_time_per_sample'][0, 0]
                print(f"  ✓ avg_time_per_sample: {avg_time_value:.6f}s")
            else:
                print(f"  ✗ 缺少 avg_time_per_sample")
            
            if has_denoised:
                print(f"  ✓ denoised_data: {data['denoised_data'].shape}")
            else:
                print(f"  ⚠️  缺少 denoised_data")
            
            print()
            
        except Exception as e:
            print(f"  ❌ 读取失败: {e}\n")
    
    print("="*80)
    print("汇总:")
    print(f"  检查文件数: {min(len(mat_files), 5)}")
    print(f"  包含 time_per_sample: {has_time_field_count}")
    
    if missing_time_field:
        print(f"\n⚠️  以下文件缺少 time_per_sample 字段:")
        for f in missing_time_field:
            print(f"    - {f}")
        print("\n建议: 重新运行测试以生成包含时间字段的结果文件")
    else:
        print("\n✅ 所有检查的文件都包含正确的时间字段！")
        print("   可以正常使用 compute_all_metrics.m 计算指标")

if __name__ == '__main__':
    verify_time_fields()
