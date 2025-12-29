"""
生成多种SNR级别的半模拟数据集

该脚本基于纯净EEG、垂直眼电(VEOG)和水平眼电(HEOG)生成不同SNR级别的数据集。
SNR公式：SNR = 10 × log(RMS(x) / RMS(λ·n))
其中：
- x: 纯净EEG信号
- n: 伪影（眼电，VEOG和HEOG等权重混合）
- λ: 调节伪影强度的系数
- y: 污染的EEG信号（y = x + λ·n）

生成SNR: -6, -4, -2, 0, 2 dB
训练集、验证集、测试集分开保存，每个SNR的测试集单独保存

关键修正：
1. 确保sim1对应veog_1和heog_1（同一个体）
2. 数据格式为(N, 1200)，每个通道单独一行
3. VEOG和HEOG等权重混合：(veog + heog) / 2
4. 对所有通道使用统一的λ值
"""

import scipy.io
import numpy as np
import re
import os


def calculate_lambda_for_snr(pure_signal, artifact_signal, target_snr_db):
    """
    计算达到目标SNR所需的λ系数

    修正依据：
    基于标准SNR定义: SNR_dB = 10 * log10(Power_signal / Power_noise)
                           = 20 * log10(RMS_signal / RMS_noise)
    由此推导: λ = RMS_signal / (RMS_artifact * 10^(SNR_dB / 20))

    参数:
        pure_signal: 纯净EEG信号 [4, window_size]
        artifact_signal: 伪影信号（眼电） [window_size]
        target_snr_db: 目标SNR (dB)

    返回:
        lambda_value: 伪影强度系数
    """
    # 计算纯净信号的整体RMS（所有通道）
    rms_pure = np.sqrt(np.mean(pure_signal ** 2))
    rms_artifact = np.sqrt(np.mean(artifact_signal ** 2))

    # 【关键修正】这里的分母指数除数由 10 改为 20
    # 这样计算出的 dB 值才是符合物理学定义的标准分贝
    lambda_value = rms_pure / (rms_artifact * (10 ** (target_snr_db / 20.0)))

    return lambda_value

def process_subject_data(pure_eeg, veog_signal, heog_signal, window_size, step):
    """
    处理单个个体的数据，确保EEG和EOG分段数量一致
    
    参数:
        pure_eeg: 纯净EEG数据 [4, time]
        veog_signal: VEOG信号 [time]
        heog_signal: HEOG信号 [time]
        window_size: 窗口大小
        step: 步长
    
    返回:
        pure_segments: 纯净EEG分段列表（每个通道单独一行）
        eog_segments: 混合EOG分段列表
    """
    # 确保所有信号长度一致
    min_len = min(pure_eeg.shape[1], len(veog_signal), len(heog_signal))
    pure_eeg = pure_eeg[:, :min_len]
    veog_signal = veog_signal[:min_len]
    heog_signal = heog_signal[:min_len]
    
    # VEOG和HEOG等权重混合
    mixed_eog = (veog_signal + heog_signal) / 2.0
    
    # 计算分段数量
    num_windows = (min_len - window_size) // step + 1
    
    pure_segments = []
    eog_segments = []
    
    for j in range(num_windows):
        start = j * step
        end = start + window_size
        if end <= min_len:
            # 提取EOG分段
            eog_seg = mixed_eog[start:end]
            eog_segments.append(eog_seg)
            
            # 提取每个通道的EEG分段
            for ch in range(4):
                pure_seg = pure_eeg[ch, start:end]
                pure_segments.append(pure_seg)
    
    return pure_segments, eog_segments

def load_all_subject_data(pure_data, veog_data, heog_data, window_size, step):
    """
    加载所有个体的数据，确保EEG和EOG配对且分段数量一致
    
    参数:
        pure_data: 纯净EEG数据字典
        veog_data: VEOG数据字典
        heog_data: HEOG数据字典
        window_size: 窗口大小
        step: 步长
    
    返回:
        all_pure_segments: 所有纯净EEG分段列表 [num_segments, window_size]
        all_eog_segments: 所有EOG分段列表 [num_samples, window_size]（每4个pure对应1个eog）
    """
    # 提取并排序纯净EEG的key
    pure_keys = [k for k in pure_data.keys() if k.startswith('sim') and not k.startswith('__')]
    pure_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))
    
    all_pure_segments = []
    all_eog_segments = []
    subjects_processed = 0
    
    for pure_key in pure_keys:
        # 从pure_key提取编号（例如从'sim1_resampled'提取'1'）
        subject_num = re.search(r'\d+', pure_key).group()
        
        # 构造对应的VEOG和HEOG的key
        veog_key = f'veog_{subject_num}'
        heog_key = f'heog_{subject_num}'
        
        if veog_key not in veog_data or heog_key not in heog_data:
            print(f"警告: 找不到 {veog_key} 或 {heog_key}，跳过个体 {subject_num}")
            continue
        
        # 提取该个体的数据
        pure_eeg = pure_data[pure_key][:4, :]  # 取前4个通道
        veog_signal = veog_data[veog_key][0, :]
        heog_signal = heog_data[heog_key][0, :]
        
        # 处理该个体的数据
        pure_segs, eog_segs = process_subject_data(
            pure_eeg, veog_signal, heog_signal, window_size, step
        )
        
        all_pure_segments.extend(pure_segs)
        all_eog_segments.extend(eog_segs)
        subjects_processed += 1
    
    print(f"成功处理 {subjects_processed} 个个体")
    return all_pure_segments, all_eog_segments

def generate_contaminated_data_with_snr(pure_segments, eog_segments, target_snr_db):
    """
    根据目标SNR生成污染数据
    
    参数:
        pure_segments: 纯净EEG分段列表 [num_segments, window_size]
        eog_segments: EOG分段列表 [num_segments, window_size]
        target_snr_db: 目标SNR (dB)
    
    返回:
        contaminated_segments: 污染的EEG分段列表 [num_segments, window_size]
        lambda_values: 每个样本组使用的λ值列表
    """
    # 每4个连续的pure_segments对应同一个样本的4个通道
    num_samples = len(pure_segments) // 4
    num_eog_segments = len(eog_segments)
    
    if num_samples != num_eog_segments:
        print(f"警告: 样本数不匹配，pure样本数={num_samples}, eog样本数={num_eog_segments}")
        num_samples = min(num_samples, num_eog_segments)
    
    contaminated_segments = []
    lambda_values = []
    
    for i in range(num_samples):
        # 取出同一个样本的4个通道
        start_idx = i * 4
        end_idx = start_idx + 4
        pure_sample_4ch = np.array(pure_segments[start_idx:end_idx])  # [4, window_size]
        eog_seg = eog_segments[i]  # [window_size]
        
        # 对整个样本（所有通道）计算统一的λ值
        lambda_val = calculate_lambda_for_snr(pure_sample_4ch, eog_seg, target_snr_db)
        lambda_values.append(lambda_val)
        
        # 对每个通道添加相同比例的伪影
        for ch in range(4):
            pure_channel = pure_sample_4ch[ch]
            contaminated_channel = pure_channel + lambda_val * eog_seg
            contaminated_segments.append(contaminated_channel)
    
    return contaminated_segments, lambda_values

def generate_finetune_datasets(train_pure_array, train_eog, snr_levels, 
                               finetune_ratios, output_dir):
    """
    生成无监督微调数据集（从5种SNR中均匀采样）
    
    参数:
        train_pure_array: 训练集纯净数据 (N, 1200) - 单个SNR的
        train_eog: 训练集EOG数据列表 (每个样本)
        snr_levels: SNR级别列表 (dB)
        finetune_ratios: 微调数据集比例列表 (例如 [0.1, 0.2, 0.3])
        output_dir: 输出目录
    """
    num_snr_levels = len(snr_levels)
    samples_per_snr = len(train_eog)  # 每个SNR的样本数
    channels_per_sample = 4
    
    print(f"\n微调数据集配置:")
    print(f"  每个SNR的样本数: {samples_per_snr}")
    print(f"  SNR级别: {snr_levels}")
    print(f"  微调比例: {finetune_ratios}")
    
    for ratio in finetune_ratios:
        print(f"\n--- 生成 {int(ratio*100)}% 微调数据集 ---")
        
        # 计算每个SNR需要采样的样本数（均匀分布）
        samples_per_snr_finetune = int(samples_per_snr * ratio)
        total_samples = samples_per_snr_finetune * num_snr_levels
        
        print(f"  每个SNR采样: {samples_per_snr_finetune} 个样本")
        print(f"  总样本数: {total_samples} 个样本")
        
        # 为每个SNR级别生成对应的污染数据
        finetune_contaminated_list = []
        finetune_pure_list = []
        
        # 使用固定种子以确保可重复性
        np.random.seed(42 + int(ratio * 100))
        
        for snr_idx, snr_db in enumerate(snr_levels):
            # 随机采样该SNR的样本索引
            sample_indices = np.random.choice(samples_per_snr, 
                                            samples_per_snr_finetune, 
                                            replace=False)
            sample_indices = sorted(sample_indices)  # 排序以保持一致性
            
            # 提取对应的纯净数据和EOG数据
            pure_samples = []
            eog_samples = []
            for idx in sample_indices:
                # 每个样本对应4个通道
                start = idx * channels_per_sample
                end = start + channels_per_sample
                pure_samples.extend(train_pure_array[start:end])
                eog_samples.append(train_eog[idx])
            
            # 生成该SNR的污染数据
            contaminated, lambdas = generate_contaminated_data_with_snr(
                pure_samples, eog_samples, snr_db
            )
            
            finetune_contaminated_list.append(np.array(contaminated))
            finetune_pure_list.append(np.array(pure_samples))
            
            print(f"    SNR={snr_db:+3d}dB: {len(contaminated)} 个通道, 平均λ={np.mean(lambdas):.6f}")
        
        # 合并所有SNR的数据
        finetune_contaminated = np.concatenate(finetune_contaminated_list, axis=0)
        finetune_pure = np.concatenate(finetune_pure_list, axis=0)
        
        # 保存微调数据集
        finetune_cont_path = f'{output_dir}/Finetune_{int(ratio*100)}percent_Contaminated.mat'
        finetune_pure_path = f'{output_dir}/Finetune_{int(ratio*100)}percent_Pure.mat'
        
        scipy.io.savemat(finetune_cont_path, {'data': finetune_contaminated})
        scipy.io.savemat(finetune_pure_path, {'data': finetune_pure})
        
        print(f"  ✓ 保存污染数据: {finetune_contaminated.shape} -> {finetune_cont_path}")
        print(f"  ✓ 保存纯净数据: {finetune_pure.shape} -> {finetune_pure_path}")
        
        # 验证维度
        assert finetune_contaminated.shape == finetune_pure.shape, \
            f"维度不匹配: {finetune_contaminated.shape} != {finetune_pure.shape}"
        assert finetune_contaminated.shape[0] == total_samples * channels_per_sample, \
            f"样本数不匹配: {finetune_contaminated.shape[0]} != {total_samples * channels_per_sample}"
    
    print(f"\n✓ 所有微调数据集生成完成")

def process_multi_snr_dataset(pure_path, veog_path, heog_path, output_dir, 
                              snr_levels=[-8,-6, -4, -2, 0, 2,4],
                              train_ratio=0.8, val_ratio=0.1):
    """
    生成多种SNR级别的数据集
    
    参数:
        pure_path: 纯净EEG数据路径
        veog_path: VEOG数据路径
        heog_path: HEOG数据路径
        output_dir: 输出目录
        snr_levels: SNR级别列表 (dB)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    # 数据参数
    sample_rate = 200  # Hz
    window_duration = 6  # seconds
    window_size = sample_rate * window_duration
    step = window_size  # 无重叠
    
    # 加载数据
    print("加载数据...")
    pure_data = scipy.io.loadmat(pure_path)
    veog_data = scipy.io.loadmat(veog_path)
    heog_data = scipy.io.loadmat(heog_path)
    
    # 加载所有个体的数据
    print("\n处理所有个体的数据...")
    pure_segments, eog_segments = load_all_subject_data(
        pure_data, veog_data, heog_data, window_size, step
    )
    
    num_samples = len(eog_segments)
    num_pure_segments = len(pure_segments)
    
    print(f"纯净EEG总分段数: {num_pure_segments}")
    print(f"EOG总样本数: {num_samples}")
    print(f"验证: {num_pure_segments} = {num_samples} × 4? {num_pure_segments == num_samples * 4}")
    
    if num_pure_segments != num_samples * 4:
        raise ValueError("数据不一致：纯净EEG分段数应该是EOG样本数的4倍")
    
    # 随机打乱数据（使用固定的随机种子以保证可重复性）
    print("\n打乱数据...")
    np.random.seed(42)
    sample_indices = np.random.permutation(num_samples)
    
    # 根据样本索引重排数据
    pure_segments_shuffled = []
    eog_segments_shuffled = []
    for idx in sample_indices:
        # 每个样本的4个通道
        start = idx * 4
        end = start + 4
        pure_segments_shuffled.extend(pure_segments[start:end])
        eog_segments_shuffled.append(eog_segments[idx])
    
    pure_segments = pure_segments_shuffled
    eog_segments = eog_segments_shuffled
    
    # 计算划分点（基于样本数，不是分段数）
    train_end_sample = int(train_ratio * num_samples)
    val_end_sample = int((train_ratio + val_ratio) * num_samples)
    
    # 划分数据集
    train_pure = pure_segments[:train_end_sample * 4]
    val_pure = pure_segments[train_end_sample * 4:val_end_sample * 4]
    test_pure = pure_segments[val_end_sample * 4:]
    
    train_eog = eog_segments[:train_end_sample]
    val_eog = eog_segments[train_end_sample:val_end_sample]
    test_eog = eog_segments[val_end_sample:]
    
    print(f"\n数据集划分:")
    print(f"训练集: {len(train_pure)} 个分段 ({len(train_eog)} 个样本)")
    print(f"验证集: {len(val_pure)} 个分段 ({len(val_eog)} 个样本)")
    print(f"测试集: {len(test_pure)} 个分段 ({len(test_eog)} 个样本)")
    print(f"比例验证: {len(train_eog)/num_samples:.1%} / {len(val_eog)/num_samples:.1%} / {len(test_eog)/num_samples:.1%}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 为每个SNR级别生成数据
    all_train_contaminated = []
    all_val_contaminated = []
    
    for snr_db in snr_levels:
        print(f"\n{'='*60}")
        print(f"生成 SNR = {snr_db} dB 的数据集...")
        print(f"{'='*60}")
        
        # 生成训练集
        print("生成训练集污染数据...")
        train_contaminated, train_lambdas = generate_contaminated_data_with_snr(
            train_pure, train_eog, snr_db
        )
        train_contaminated_array = np.array(train_contaminated)
        print(f"训练集: {train_contaminated_array.shape}, 平均λ: {np.mean(train_lambdas):.6f}")
        all_train_contaminated.append(train_contaminated_array)
        
        # 生成验证集
        print("生成验证集污染数据...")
        val_contaminated, val_lambdas = generate_contaminated_data_with_snr(
            val_pure, val_eog, snr_db
        )
        val_contaminated_array = np.array(val_contaminated)
        print(f"验证集: {val_contaminated_array.shape}, 平均λ: {np.mean(val_lambdas):.6f}")
        all_val_contaminated.append(val_contaminated_array)
        
        # 生成测试集
        print("生成测试集污染数据...")
        test_contaminated, test_lambdas = generate_contaminated_data_with_snr(
            test_pure, test_eog, snr_db
        )
        test_contaminated_array = np.array(test_contaminated)
        print(f"测试集: {test_contaminated_array.shape}, 平均λ: {np.mean(test_lambdas):.6f}")
        
        # 保存该SNR级别的测试集
        test_cont_filename = f'{output_dir}/Test_Contaminated_SNR{snr_db}dB.mat'
        scipy.io.savemat(test_cont_filename, {'data': test_contaminated_array})
        print(f"保存测试集污染数据: {test_cont_filename}")
        
        test_pure_filename = f'{output_dir}/Test_Pure_SNR{snr_db}dB.mat'
        test_pure_array = np.array(test_pure)
        scipy.io.savemat(test_pure_filename, {'data': test_pure_array})
        print(f"保存测试集纯净数据: {test_pure_filename}")
    
    # 合并所有SNR的训练集和验证集
    print(f"\n{'='*60}")
    print("合并所有SNR的训练集和验证集...")
    print(f"{'='*60}")
    
    all_train_contaminated = np.concatenate(all_train_contaminated, axis=0)
    all_val_contaminated = np.concatenate(all_val_contaminated, axis=0)
    
    # 纯净数据也要重复对应的次数（与SNR级别数量相同）
    num_snr_levels = len(snr_levels)
    train_pure_array = np.array(train_pure)
    val_pure_array = np.array(val_pure)
    
    # 重复纯净数据以匹配污染数据的维度
    all_train_pure = np.tile(train_pure_array, (num_snr_levels, 1))
    all_val_pure = np.tile(val_pure_array, (num_snr_levels, 1))
    
    print(f"\n纯净数据重复 {num_snr_levels} 次以匹配污染数据维度")
    print(f"训练集纯净: {train_pure_array.shape} -> {all_train_pure.shape}")
    print(f"验证集纯净: {val_pure_array.shape} -> {all_val_pure.shape}")
    
    # 保存训练集数据
    scipy.io.savemat(f'{output_dir}/Train_Pure.mat', {'data': all_train_pure})
    print(f"\n保存训练集纯净数据: {all_train_pure.shape}")
    
    scipy.io.savemat(f'{output_dir}/Train_Contaminated.mat', {'data': all_train_contaminated})
    print(f"保存训练集污染数据（所有SNR）: {all_train_contaminated.shape}")
    
    # 保存验证集数据
    scipy.io.savemat(f'{output_dir}/Val_Pure.mat', {'data': all_val_pure})
    print(f"\n保存验证集纯净数据: {all_val_pure.shape}")
    
    scipy.io.savemat(f'{output_dir}/Val_Contaminated.mat', {'data': all_val_contaminated})
    print(f"保存验证集污染数据（所有SNR）: {all_val_contaminated.shape}")
    
    # 生成微调数据集（10%, 20%, 30%的训练集，从5种SNR中均匀采样）
    print(f"\n{'='*60}")
    print("生成无监督微调数据集...")
    print(f"{'='*60}")
    
    finetune_ratios = [0.1, 0.2, 0.3]
    generate_finetune_datasets(train_pure_array, train_eog, snr_levels, 
                              finetune_ratios, output_dir)
    
    print(f"\n{'='*60}")
    print("所有数据生成完成！")
    print(f"{'='*60}")
    print(f"\n文件列表:")
    print(f"  - Train_Pure.mat: 训练集纯净数据（重复{num_snr_levels}次） {all_train_pure.shape}")
    print(f"  - Train_Contaminated.mat: 训练集污染数据（所有SNR混合） {all_train_contaminated.shape}")
    print(f"  - Val_Pure.mat: 验证集纯净数据（重复{num_snr_levels}次） {all_val_pure.shape}")
    print(f"  - Val_Contaminated.mat: 验证集污染数据（所有SNR混合） {all_val_contaminated.shape}")
    for ratio in finetune_ratios:
        print(f"  - Finetune_{int(ratio*100)}percent_Contaminated.mat: {int(ratio*100)}%微调数据集（污染）")
        print(f"  - Finetune_{int(ratio*100)}percent_Pure.mat: {int(ratio*100)}%微调数据集（纯净）")
    for snr_db in snr_levels:
        print(f"  - Test_Contaminated_SNR{snr_db}dB.mat: SNR={snr_db}dB的测试集污染数据")
        print(f"  - Test_Pure_SNR{snr_db}dB.mat: SNR={snr_db}dB的测试集纯净数据")
    
    print(f"\n数据维度验证:")
    print(f"  训练集: {all_train_contaminated.shape} == {all_train_pure.shape}? {all_train_contaminated.shape == all_train_pure.shape}")
    print(f"  验证集: {all_val_contaminated.shape} == {all_val_pure.shape}? {all_val_contaminated.shape == all_val_pure.shape}")

if __name__ == '__main__':
    # 生成多SNR数据集
    process_multi_snr_dataset(
        pure_path='Pure_Data.mat',
        veog_path='VEOG.mat',
        heog_path='HEOG.mat',
        output_dir='已经生成好的数据/multi_snr',
        snr_levels=[-8,-6, -4, -2, 0, 2,4],
        train_ratio=0.8,
        val_ratio=0.1
    )
