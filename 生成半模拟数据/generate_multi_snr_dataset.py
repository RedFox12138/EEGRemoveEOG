"""
生成多种SNR级别的半模拟数据集 (带详细统计日志版)

该脚本基于纯净EEG、垂直眼电(VEOG)和水平眼电(HEOG)生成不同SNR级别的数据集。
不改变原始处理逻辑，仅增加了详细的数据量统计输出。
"""

import scipy.io
import numpy as np
import re
import os

def calculate_lambda_for_snr(pure_signal, artifact_signal, target_snr_db):
    """计算达到目标SNR所需的λ系数 (逻辑未变)"""
    rms_pure = np.sqrt(np.mean(pure_signal ** 2))
    rms_artifact = np.sqrt(np.mean(artifact_signal ** 2))
    # 避免除以零
    if rms_artifact == 0:
        return 0
    lambda_value = rms_pure / (rms_artifact * (10 ** (target_snr_db / 20.0)))
    return lambda_value

def process_subject_data(pure_eeg, veog_signal, heog_signal, window_size, step):
    """处理单个个体的数据 (逻辑未变)"""
    min_len = min(pure_eeg.shape[1], len(veog_signal), len(heog_signal))
    pure_eeg = pure_eeg[:, :min_len]
    veog_signal = veog_signal[:min_len]
    heog_signal = heog_signal[:min_len]

    mixed_eog = (veog_signal + heog_signal) / 2.0
    num_windows = (min_len - window_size) // step + 1

    pure_segments = []
    eog_segments = []

    for j in range(num_windows):
        start = j * step
        end = start + window_size
        if end <= min_len:
            eog_seg = mixed_eog[start:end]
            eog_segments.append(eog_seg)
            for ch in range(4):
                pure_seg = pure_eeg[ch, start:end]
                pure_segments.append(pure_seg)

    return pure_segments, eog_segments

def load_all_subject_data(pure_data, veog_data, heog_data, window_size, step):
    """加载所有个体的数据 (增加统计日志)"""
    pure_keys = [k for k in pure_data.keys() if k.startswith('sim') and not k.startswith('__')]
    pure_keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    all_pure_segments = []
    all_eog_segments = []

    # [新增统计变量]
    subjects_processed = 0
    subjects_skipped = 0
    total_raw_seconds = 0
    fs = 200 # 假设采样率，仅用于统计时长显示

    print(f"\n{'='*20} 原始数据扫描 {'='*20}")
    print(f"检测到潜在个体数量: {len(pure_keys)}")

    for pure_key in pure_keys:
        subject_num = re.search(r'\d+', pure_key).group()
        veog_key = f'veog_{subject_num}'
        heog_key = f'heog_{subject_num}'

        if veog_key not in veog_data or heog_key not in heog_data:
            print(f"  [跳过] 个体 {subject_num}: 缺少 VEOG 或 HEOG 数据")
            subjects_skipped += 1
            continue

        pure_eeg = pure_data[pure_key][:4, :]
        veog_signal = veog_data[veog_key][0, :]
        heog_signal = heog_data[heog_key][0, :]

        # 统计该个体的原始时长
        current_len = min(pure_eeg.shape[1], len(veog_signal), len(heog_signal))
        total_raw_seconds += (current_len / fs)

        pure_segs, eog_segs = process_subject_data(
            pure_eeg, veog_signal, heog_signal, window_size, step
        )

        all_pure_segments.extend(pure_segs)
        all_eog_segments.extend(eog_segs)
        subjects_processed += 1

    print(f"{'-'*60}")
    print(f"扫描结果:")
    print(f"  - 成功匹配个体数: {subjects_processed}")
    print(f"  - 跳过/缺失个体数: {subjects_skipped}")
    print(f"  - 原始有效信号总时长: 约 {total_raw_seconds/3600:.2f} 小时 ({total_raw_seconds:.1f} 秒)")
    print(f"  - 切分生成基础样本数 (Base Samples): {len(all_eog_segments)}")
    print(f"  - 切分生成基础通道数 (Base Channels): {len(all_pure_segments)} (应该是样本数的4倍)")
    print(f"{'='*60}\n")

    return all_pure_segments, all_eog_segments

def generate_contaminated_data_with_snr(pure_segments, eog_segments, target_snr_db):
    """根据目标SNR生成污染数据 (逻辑未变)"""
    num_samples = len(pure_segments) // 4
    num_eog_segments = len(eog_segments)

    if num_samples != num_eog_segments:
        num_samples = min(num_samples, num_eog_segments)

    contaminated_segments = []
    lambda_values = []

    for i in range(num_samples):
        start_idx = i * 4
        end_idx = start_idx + 4
        pure_sample_4ch = np.array(pure_segments[start_idx:end_idx])
        eog_seg = eog_segments[i]

        lambda_val = calculate_lambda_for_snr(pure_sample_4ch, eog_seg, target_snr_db)
        lambda_values.append(lambda_val)

        for ch in range(4):
            pure_channel = pure_sample_4ch[ch]
            contaminated_channel = pure_channel + lambda_val * eog_seg
            contaminated_segments.append(contaminated_channel)

    return contaminated_segments, lambda_values

def generate_finetune_datasets(train_pure_array, train_eog, snr_levels,
                               finetune_ratios, output_dir):
    """生成无监督微调数据集 (增加详细统计)"""
    samples_per_snr = len(train_eog)

    print(f"\n{'='*20} 微调数据集生成统计 {'='*20}")
    print(f"基础池 (Training Set Base): {samples_per_snr} 样本")

    for ratio in finetune_ratios:
        samples_per_snr_finetune = int(samples_per_snr * ratio)
        total_samples = samples_per_snr_finetune * len(snr_levels)

        print(f"\n>>> 生成比例 {int(ratio*100)}% :")
        print(f"  - 采样策略: 从每种SNR等级中随机抽取 {samples_per_snr_finetune} 个样本")
        print(f"  - 总计: {len(snr_levels)} SNR等级 × {samples_per_snr_finetune} 样本 = {total_samples} 个总样本")
        print(f"  - 总通道数: {total_samples * 4}")

        finetune_contaminated_list = []
        finetune_pure_list = []
        np.random.seed(42 + int(ratio * 100))

        for snr_db in snr_levels:
            # (这里省略中间的生成逻辑，与原代码一致，只为了不重复占用篇幅)
            # ...逻辑保持不变...
            sample_indices = np.random.choice(samples_per_snr, samples_per_snr_finetune, replace=False)
            sample_indices = sorted(sample_indices)

            pure_samples = []
            eog_samples = []
            for idx in sample_indices:
                start = idx * 4
                end = start + 4
                pure_samples.extend(train_pure_array[start:end])
                eog_samples.append(train_eog[idx])

            contaminated, _ = generate_contaminated_data_with_snr(pure_samples, eog_samples, snr_db)
            finetune_contaminated_list.append(np.array(contaminated))
            finetune_pure_list.append(np.array(pure_samples))

        finetune_contaminated = np.concatenate(finetune_contaminated_list, axis=0)
        finetune_pure = np.concatenate(finetune_pure_list, axis=0)

        # 保存
        finetune_cont_path = f'{output_dir}/Finetune_{int(ratio*100)}percent_Contaminated.mat'
        finetune_pure_path = f'{output_dir}/Finetune_{int(ratio*100)}percent_Pure.mat'
        scipy.io.savemat(finetune_cont_path, {'data': finetune_contaminated})
        scipy.io.savemat(finetune_pure_path, {'data': finetune_pure})

def process_multi_snr_dataset(pure_path, veog_path, heog_path, output_dir,
                              snr_levels=[-8,-6, -4, -2, 0, 2,4],
                              train_ratio=0.8, val_ratio=0.1):
    # 数据参数
    sample_rate = 200
    window_duration = 6
    window_size = sample_rate * window_duration
    step = window_size

    print("加载数据...")
    pure_data = scipy.io.loadmat(pure_path)
    veog_data = scipy.io.loadmat(veog_path)
    heog_data = scipy.io.loadmat(heog_path)

    # 1. 加载和切分
    pure_segments, eog_segments = load_all_subject_data(
        pure_data, veog_data, heog_data, window_size, step
    )

    num_samples = len(eog_segments)
    num_pure_segments = len(pure_segments)

    # 2. 打乱
    np.random.seed(42)
    sample_indices = np.random.permutation(num_samples)

    pure_segments_shuffled = []
    eog_segments_shuffled = []
    for idx in sample_indices:
        start = idx * 4
        end = start + 4
        pure_segments_shuffled.extend(pure_segments[start:end])
        eog_segments_shuffled.append(eog_segments[idx])

    pure_segments = pure_segments_shuffled
    eog_segments = eog_segments_shuffled

    # 3. 划分
    train_end_sample = int(train_ratio * num_samples)
    val_end_sample = int((train_ratio + val_ratio) * num_samples)

    train_pure = pure_segments[:train_end_sample * 4]
    val_pure = pure_segments[train_end_sample * 4:val_end_sample * 4]
    test_pure = pure_segments[val_end_sample * 4:]

    train_eog = eog_segments[:train_end_sample]
    val_eog = eog_segments[train_end_sample:val_end_sample]
    test_eog = eog_segments[val_end_sample:]

    # [新增统计日志] 基础数据集划分情况
    print(f"\n{'='*20} 基础样本划分 (无SNR扩充前) {'='*20}")
    print(f"总基础样本数 (Base Samples): {num_samples}")
    print(f"  - 训练集 (Training): {len(train_eog)} 样本 ({(len(train_eog)/num_samples):.1%})")
    print(f"  - 验证集 (Validation): {len(val_eog)} 样本 ({(len(val_eog)/num_samples):.1%})")
    print(f"  - 测试集 (Test):       {len(test_eog)} 样本 ({(len(test_eog)/num_samples):.1%})")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    all_train_contaminated = []
    all_val_contaminated = []

    # [新增统计日志] 收集每层的数量用于最后汇总
    snr_stats = []

    for snr_db in snr_levels:
        # 生成训练集
        train_contaminated, _ = generate_contaminated_data_with_snr(train_pure, train_eog, snr_db)
        train_contaminated_array = np.array(train_contaminated)
        all_train_contaminated.append(train_contaminated_array)

        # 生成验证集
        val_contaminated, _ = generate_contaminated_data_with_snr(val_pure, val_eog, snr_db)
        val_contaminated_array = np.array(val_contaminated)
        all_val_contaminated.append(val_contaminated_array)

        # 生成测试集 (并单独保存)
        test_contaminated, _ = generate_contaminated_data_with_snr(test_pure, test_eog, snr_db)
        test_contaminated_array = np.array(test_contaminated)

        test_cont_filename = f'{output_dir}/Test_Contaminated_SNR{snr_db}dB.mat'
        scipy.io.savemat(test_cont_filename, {'data': test_contaminated_array})

        test_pure_filename = f'{output_dir}/Test_Pure_SNR{snr_db}dB.mat'
        test_pure_array = np.array(test_pure)
        scipy.io.savemat(test_pure_filename, {'data': test_pure_array})

        # 记录该SNR的统计信息
        snr_stats.append({
            'snr': snr_db,
            'train_count': train_contaminated_array.shape[0] // 4,
            'val_count': val_contaminated_array.shape[0] // 4,
            'test_count': test_contaminated_array.shape[0] // 4
        })

    # 合并
    all_train_contaminated = np.concatenate(all_train_contaminated, axis=0)
    all_val_contaminated = np.concatenate(all_val_contaminated, axis=0)

    # 纯净数据扩充
    num_snr_levels = len(snr_levels)
    train_pure_array = np.array(train_pure)
    val_pure_array = np.array(val_pure)
    all_train_pure = np.tile(train_pure_array, (num_snr_levels, 1))
    all_val_pure = np.tile(val_pure_array, (num_snr_levels, 1))

    # 保存
    scipy.io.savemat(f'{output_dir}/Train_Pure.mat', {'data': all_train_pure})
    scipy.io.savemat(f'{output_dir}/Train_Contaminated.mat', {'data': all_train_contaminated})
    scipy.io.savemat(f'{output_dir}/Val_Pure.mat', {'data': all_val_pure})
    scipy.io.savemat(f'{output_dir}/Val_Contaminated.mat', {'data': all_val_contaminated})

    # 微调数据集
    generate_finetune_datasets(train_pure_array, train_eog, snr_levels,
                              [0.1, 0.2, 0.3], output_dir)

    # [新增统计日志] 最终全量统计表
    print(f"\n{'#'*30} 最终数据集统计报告 {'#'*30}")
    print(f"SNR 等级列表 ({len(snr_levels)}级): {snr_levels} dB")
    print(f"{'SNR(dB)':<10} | {'Train(样本)':<15} | {'Val(样本)':<15} | {'Test(样本)':<15}")
    print("-" * 65)

    total_train = 0
    total_val = 0
    total_test = 0

    for stat in snr_stats:
        print(f"{stat['snr']:<10} | {stat['train_count']:<15} | {stat['val_count']:<15} | {stat['test_count']:<15}")
        total_train += stat['train_count']
        total_val += stat['val_count']
        total_test += stat['test_count']

    print("-" * 65)
    print(f"{'Total':<10} | {total_train:<15} | {total_val:<15} | {total_test:<15}")
    print(f"{'Samples':<10} | {'(混合所有SNR)':<15} | {'(混合所有SNR)':<15} | {'(分文件保存)':<15}")

    print(f"\n{'='*20} 文件规格汇总 {'='*20}")
    print(f"1. Train_Contaminated.mat: {all_train_contaminated.shape} (通道数 x 时间点)")
    print(f"   => 包含 {all_train_contaminated.shape[0]//4} 个多通道样本")
    print(f"2. Val_Contaminated.mat:   {all_val_contaminated.shape}")
    print(f"   => 包含 {all_val_contaminated.shape[0]//4} 个多通道样本")
    print(f"3. Test Files (每种SNR):")
    print(f"   => 每个文件包含 {snr_stats[0]['test_count']} 个样本 ({snr_stats[0]['test_count']*4} 通道)")
    print(f"{'#'*80}")

if __name__ == '__main__':
    process_multi_snr_dataset(
        pure_path='Pure_Data.mat',
        veog_path='VEOG.mat',
        heog_path='HEOG.mat',
        output_dir='已经生成好的数据/multi_snr',
        snr_levels=[-8,-6, -4, -2, 0, 2,4],
        train_ratio=0.8,
        val_ratio=0.1
    )