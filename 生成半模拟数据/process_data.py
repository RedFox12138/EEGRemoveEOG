### 该脚本用于处理EEG数据集中的.mat文件，提取指定通道的数据，并将其保存为.npy文件。

import scipy.io
import numpy as np
import re

def process_and_save_data(mat_path, output_path, prefix, key_suffix):
    try:
        mat_data = scipy.io.loadmat(mat_path)
    except FileNotFoundError:
        print(f"Error: {mat_path} not found.")
        return

    # 数据参数
    sample_rate = 200  # Hz
    window_duration = 6  # seconds
    window_size = sample_rate * window_duration
    # No overlap as per new requirement, so step is window_size
    step = window_size

    # 提取并排序相关的key
    keys = [k for k in mat_data.keys() if k.startswith(prefix)]
    keys.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    all_segments = []

    for key in keys:
        # 每个数组的尺寸是19*n
        sample_data = mat_data[key]
        # 取前4个通道
        sample_data_4ch = sample_data[:4, :]

        num_channels, signal_len = sample_data_4ch.shape
        for i in range(num_channels):
            signal = sample_data_4ch[i, :]

            num_windows = (signal_len - window_size) // step + 1
            for j in range(num_windows):
                start = j * step
                end = start + window_size
                if end <= signal_len:
                    segment = signal[start:end]
                    all_segments.append(segment)

    np.save(output_path, np.array(all_segments))
    print(f"Processed data saved to {output_path}")
    print(f"Shape of saved data: {np.array(all_segments).shape}")

if __name__ == '__main__':
    # 处理受污染的数据
    process_and_save_data(
        'A semi-simulated EEGEOG dataset for the comparison of EOG artifact rejection techniques/Contaminated_Data.mat',
        'A semi-simulated EEGEOG dataset for the comparison of EOG artifact rejection techniques/Contaminated.npy',
        'sim',
        '_con'
    )

    # 处理纯净的数据
    process_and_save_data(
        'A semi-simulated EEGEOG dataset for the comparison of EOG artifact rejection techniques/Pure_Data.mat',
        'A semi-simulated EEGEOG dataset for the comparison of EOG artifact rejection techniques/Pure_Data.npy',
        'sim',
        '_pure'
    )
