import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.signal import firwin, lfilter, filtfilt


def find_intervals(binary_array):
    """查找二值数组中的连续True区间"""
    diff = np.diff(np.concatenate(([False], binary_array, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]
    return starts, ends


def eog_removal_corrected(eeg, fs=250, visualize=False):
    """
    单通道脑电信号眼电去除函数 - 严格按照文献实现
    
    参考文献: 《单通道脑电信号中眼电干扰的自动分离方法》
    
    参数：
        eeg : 输入脑电信号（1D数组）
        fs : 采样率（默认250Hz）
        visualize : 是否显示处理过程（默认False）
    返回：
        clean_eeg : 去除眼电后的脑电信号
    """
    # ===== 第1步：长时差分计算 =====
    # 文献: k = 0.2*fs (约200ms窗口)
    k = int(0.2 * fs)
    diff_eeg = np.zeros_like(eeg)
    diff_eeg[k:] = eeg[k:] - eeg[:-k]

    # ===== 第2步：振幅包络提取 =====
    # 平方和降采样
    M = 8  # 文献中的降采样因子
    squared = diff_eeg ** 2
    downsampled = squared[::M]

    # 低通滤波（FIR滤波器，零相位）
    # 文献: 21阶FIR滤波器, 4Hz截止频率
    N = 21
    cutoff = 4 / (fs / M / 2)  # 归一化截止频率
    h = firwin(N, cutoff, window='hamming')
    
    # 使用零相位滤波
    filtered = filtfilt(h, [1.0], downsampled)

    # 上采样恢复原始长度并开方
    upsampled = np.repeat(np.maximum(filtered, 0), M)[:len(eeg)]
    envelope = np.sqrt(upsampled)

    # ===== 第3步：双门限眼电检测 =====
    # 文献: 使用中位数绝对偏差(MAD)计算阈值
    env_med = np.median(envelope)
    env_mad = np.median(np.abs(envelope - env_med))
    
    # 标准化MAD
    sigma = env_mad / 0.6745 if env_mad > 0 else np.std(envelope)
    
    # 文献中的阈值设置
    Th = env_med + 3.0 * sigma  # 高阈值
    Tl = env_med + 0.5 * sigma  # 低阈值
    
    print(f"[阈值检测] 中位数:{env_med:.4f}, σ:{sigma:.4f}, Th:{Th:.4f}, Tl:{Tl:.4f}")

    # ⚠️ 修正1: 使用高阈值Th进行初检,而不是固定值1
    thresholded = envelope > Th  # ✅ 修正
    starts, ends = find_intervals(thresholded)

    # 扩展检测区间（使用低阈值Tl）
    valid_intervals = []
    for s, e in zip(starts, ends):
        # 向前扩展
        new_s = s
        while new_s > 0 and envelope[new_s] > Tl:
            new_s -= 1
        
        # 向后扩展
        new_e = e
        while new_e < len(envelope) - 1 and envelope[new_e] > Tl:
            new_e += 1
        
        # 文献: 去除持续时间短于50ms的区间
        if (new_e - new_s) > int(0.05 * fs):
            valid_intervals.append((new_s, new_e))

    # 合并相邻区间（间隔<200ms）
    def merge_intervals(intervals, max_gap):
        if not intervals:
            return []
        intervals = sorted(intervals, key=lambda x: x[0])
        merged = []
        cur_s, cur_e = intervals[0]
        for ns, ne in intervals[1:]:
            if ns - cur_e <= max_gap:
                cur_e = max(cur_e, ne)
            else:
                merged.append((cur_s, cur_e))
                cur_s, cur_e = ns, ne
        merged.append((cur_s, cur_e))
        return merged

    final_intervals = merge_intervals(valid_intervals, max_gap=int(0.2 * fs))
    print(f"[区间检测] 检测到 {len(final_intervals)} 个眼电伪迹区间")

    # ===== 第4步：小波变换眼电分离 =====
    wavelet = 'sym5'  # 文献使用sym5小波
    desired_level = 8  # 文献: 8层分解
    clean_eeg = eeg.copy()

    w = pywt.Wavelet(wavelet)

    for idx, (s, e) in enumerate(final_intervals):
        win_len = e - s
        # 添加重叠以避免边界效应
        overlap = min(int(0.25 * fs), win_len // 2)
        s_ext = max(0, s - overlap)
        e_ext = min(len(eeg), e + overlap)

        segment = eeg[s_ext:e_ext]
        N_segment = len(segment)  # ⚠️ 修正2: 保存原始长度用于n计算

        # 计算最大分解层数
        max_level = pywt.dwt_max_level(N_segment, w.dec_len)
        if max_level < 1:
            print(f"[警告] 区间{idx+1}太短,跳过处理")
            continue
        
        level = min(desired_level, max_level)

        # 小波分解
        coeffs = pywt.wavedec(segment, w, level=level)

        # ⚠️ 文献核心算法:
        # 1. 保留近似系数cA (coeffs[0])
        # 2. 对每层细节系数cDj应用硬阈值,去除背景脑电成分
        # 3. 重构后得到眼电伪迹估计
        new_coeffs = [coeffs[0]]  # 保留近似系数
        
        for j in range(1, len(coeffs)):
            # 文献公式(4): α = 2.5 (j<3), α = 2.0 (j≥3)
            alpha = 2.5 if j < 3 else 2.0
            
            # ⚠️ 修正3: 文献公式(3) - n应该用原始信号长度N,不是当前层长度
            n = int(N_segment / (level + 2 - j) ** alpha)
            n = max(0, min(n, len(coeffs[j]) - 1))

            # 计算通用阈值 (文献公式5)
            coeff_abs = np.abs(coeffs[j])
            sigma_j = np.median(coeff_abs) / 0.6745 if np.any(coeff_abs != 0) else 0.0
            universal_thr = sigma_j * np.sqrt(2 * np.log(max(len(coeff_abs), 2)))

            # 计算BM阈值 (Birgé-Massart阈值,文献公式3)
            sorted_coeff = np.sort(coeff_abs)[::-1]
            bm_thr = sorted_coeff[n] if n < len(sorted_coeff) else 0.0
            
            # 取两者最小值
            thr = min(bm_thr, universal_thr)

            # 硬阈值处理
            new_c = pywt.threshold(coeffs[j], value=thr, mode='hard') if sigma_j > 0 else coeffs[j]
            new_coeffs.append(new_c)
            
            print(f"  区间{idx+1} 层{j}: n={n}, σ={sigma_j:.4f}, BM_thr={bm_thr:.4f}, U_thr={universal_thr:.4f}, 采用={thr:.4f}")

        # 重构得到眼电伪迹估计
        # 文献逻辑: 硬阈值后保留的是眼电成分(低频),去除的是脑电高频成分
        eog_estimate = pywt.waverec(new_coeffs, w)
        
        # 长度对齐
        if len(eog_estimate) > len(segment):
            eog_estimate = eog_estimate[:len(segment)]
        elif len(eog_estimate) < len(segment):
            eog_estimate = np.pad(eog_estimate, (0, len(segment) - len(eog_estimate)), mode='edge')

        # 文献公式(1): 净化信号 = 原信号 - 眼电估计
        clean_segment = segment - eog_estimate

        # 应用过渡窗以平滑边界
        transition = min(100, overlap)
        if transition > 0:
            window = np.ones(len(clean_segment))
            window[:transition] = np.cos(np.linspace(np.pi / 2, 0, transition)) ** 2
            window[-transition:] = np.cos(np.linspace(0, np.pi / 2, transition)) ** 2
            clean_segment = clean_segment * window + segment * (1 - window)

        # 边界融合
        if transition > 0:
            start = s_ext + transition // 2
            end = e_ext - transition // 2
            if start < end and start < len(clean_eeg) and end <= len(clean_eeg):
                seg_start = transition // 2
                seg_end = len(clean_segment) - transition // 2
                if seg_start < seg_end:
                    clean_eeg[start:end] = clean_segment[seg_start:seg_end]
        else:
            clean_eeg[s_ext:e_ext] = clean_segment

    # ===== 可视化 =====
    if visualize:
        plt.figure(figsize=(20, 16))

        # 1. 原始信号
        plt.subplot(5, 1, 1)
        plt.plot(eeg, label='Original EEG', linewidth=1)
        plt.title('Step 1: Original EEG Signal', fontsize=14, pad=15)
        plt.ylabel('Amplitude (μV)')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 2. 长时差分
        plt.subplot(5, 1, 2)
        plt.plot(diff_eeg, color='orange', label=f'Long-term Differential (k={k})', linewidth=1)
        plt.title('Step 2: Long-term Differential Signal', fontsize=14, pad=15)
        plt.ylabel('Amplitude')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 3. 振幅包络
        plt.subplot(5, 1, 3)
        plt.plot(envelope, label='Amplitude Envelope (4Hz LPF)', linewidth=1.5, color='purple')
        plt.title('Step 3: Amplitude Envelope', fontsize=14, pad=15)
        plt.ylabel('Envelope')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 4. 双门限检测
        plt.subplot(5, 1, 4)
        plt.plot(envelope, label='Envelope', linewidth=1, color='purple')
        plt.axhline(Th, color='r', linestyle='--', linewidth=2, label=f'High Threshold (Th={Th:.2f})')
        plt.axhline(Tl, color='g', linestyle='--', linewidth=2, label=f'Low Threshold (Tl={Tl:.2f})')
        
        # 标注检测区间
        for idx, (s, e) in enumerate(final_intervals):
            plt.axvspan(s, e, alpha=0.3, color='red', label='EOG Artifact' if idx == 0 else '')
        
        plt.title('Step 4: Dual-Threshold EOG Detection', fontsize=14, pad=15)
        plt.ylabel('Envelope')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        # 5. 最终结果对比
        plt.subplot(5, 1, 5)
        plt.plot(eeg, label='Original EEG', alpha=0.6, linewidth=1)
        plt.plot(clean_eeg, label='Cleaned EEG', linewidth=1.5, color='green')
        
        # 标注处理区间
        for s, e in final_intervals:
            plt.axvspan(s, e, alpha=0.2, color='yellow')
        
        plt.title('Step 5: Final Result (Original vs Cleaned)', fontsize=14, pad=15)
        plt.xlabel('Samples')
        plt.ylabel('Amplitude (μV)')
        plt.legend(loc='upper right')
        plt.grid(alpha=0.3)

        plt.tight_layout()
        plt.show()

    return clean_eeg


# 辅助函数
def compare_implementations(eeg, fs=250):
    """对比原实现和修正后的实现"""
    from SingleDenoise import eog_removal
    
    print("="*60)
    print("对比原实现 vs 修正实现")
    print("="*60)
    
    print("\n[原实现]")
    result_old = eog_removal(eeg, fs, visualize=False)
    
    print("\n[修正实现]")
    result_new = eog_removal_corrected(eeg, fs, visualize=False)
    
    # 计算差异
    diff = np.abs(result_old - result_new)
    print(f"\n[差异统计]")
    print(f"  最大差异: {np.max(diff):.4f}")
    print(f"  平均差异: {np.mean(diff):.4f}")
    print(f"  差异标准差: {np.std(diff):.4f}")
    
    # 可视化对比
    plt.figure(figsize=(15, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(eeg, label='Original', alpha=0.7)
    plt.plot(result_old, label='Old Implementation', alpha=0.7)
    plt.title('Original Implementation')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 2)
    plt.plot(eeg, label='Original', alpha=0.7)
    plt.plot(result_new, label='Corrected Implementation', alpha=0.7)
    plt.title('Corrected Implementation')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.subplot(3, 1, 3)
    plt.plot(diff, label='Absolute Difference', color='red')
    plt.title('Difference between Two Implementations')
    plt.xlabel('Samples')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    return result_old, result_new


if __name__ == '__main__':
    # 测试代码
    print("单通道眼电去除算法 - 文献修正版")
    print("加载测试数据...")
    
    # 你可以用自己的数据替换
    # eeg = np.load('your_eeg_data.npy')
    
    # 或者生成模拟数据
    fs = 250
    t = np.arange(0, 10, 1/fs)
    eeg = np.sin(2*np.pi*10*t) + 0.5*np.random.randn(len(t))
    
    # 添加模拟眼电伪迹
    eeg[500:700] += 5 * np.sin(2*np.pi*2*t[500:700])
    
    print(f"信号长度: {len(eeg)} 采样点 ({len(eeg)/fs:.1f}秒)")
    print(f"采样率: {fs} Hz")
    
    # 运行修正算法
    clean_eeg = eog_removal_corrected(eeg, fs, visualize=True)
    
    print("\n处理完成!")
