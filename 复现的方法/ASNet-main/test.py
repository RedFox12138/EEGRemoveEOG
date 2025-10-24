import scipy
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from ASNet import ASNet

BATCH_SIZE = 50

class EEGDataset(Dataset):
    def __init__(self, noisy_signals, clean_signals, is_train=False):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        # 归一化 noisy 信号
        # 找到绝对值的最大值作为归一化因子
        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0 # 避免除以零

        noisy_normalized = noisy / norm_factor

        # 不需要在这里添加通道维度，ASNet的forward会自动处理
        # noisy_normalized = noisy_normalized[np.newaxis, :]
        # clean = clean[np.newaxis, :]

        return noisy_normalized, clean, norm_factor

def get_data():
    # 加载已经分割好的数据
    # raw_eeg_segments = np.load('../Contaminated.npy', allow_pickle=True)
    # clean_eeg_segments = np.load('../Pure_Data.npy', allow_pickle=True)
    raw_eeg_segments = scipy.io.loadmat('D:/Pycharm_Projects/EOG Remove/生成全模拟数据/已经生成好的数据/Contaminated.mat')['contaminatedEEG']
    clean_eeg_segments = scipy.io.loadmat('D:/Pycharm_Projects/EOG Remove/生成全模拟数据/已经生成好的数据/Pure_Data.mat')['pureEEG']

    # 数据集拆分 (例如, 80% 训练, 10% 验证, 10% 测试)
    num_samples = len(raw_eeg_segments)
    train_end = int(num_samples * 0.8)
    verify_end = int(num_samples * 0.9)

    train_input = raw_eeg_segments[:train_end]
    verify_input = raw_eeg_segments[train_end:verify_end]
    test_input = raw_eeg_segments[verify_end:]

    train_output = clean_eeg_segments[:train_end]
    verify_output = clean_eeg_segments[train_end:verify_end]
    test_output = clean_eeg_segments[verify_end:]

    train_dataset = EEGDataset(train_input, train_output, is_train=True)
    verify_dataset = EEGDataset(verify_input, verify_output, is_train=False)
    test_dataset = EEGDataset(test_input, test_output, is_train=False)

    train_loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    verify_loader = Data.DataLoader(
        dataset=verify_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = Data.DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )
    return train_loader, verify_loader, test_loader




def test(model, device, test_loader, num, num_x, input_z, output_z, pre_z):
    model.eval()
    step_num=0
    with torch.no_grad():
        for batch_idx, (test_input, test_output, norm_factors) in enumerate(test_loader):
            test_input=test_input.float().to(device)
            test_output=test_output.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1) # 调整形状以进行广播 [batch, 1]
            
            output = model(test_input)
            
            # 反向恢复幅度
            output_restored = output * norm_factors
            
            output_restored = output_restored.detach().cpu()
            test_output = test_output.detach().cpu()
            test_input = test_input.detach().cpu()
            
            # 数据已经是2D的，不需要squeeze
            test_input_squeezed = test_input
            test_output_squeezed = test_output
            output_squeezed = output_restored
            
            batch_size_actual = test_input.size(0)
            start_idx = step_num * BATCH_SIZE
            end_idx = start_idx + batch_size_actual
            
            input_z[start_idx:end_idx] = test_input_squeezed
            output_z[start_idx:end_idx] = test_output_squeezed
            pre_z[start_idx:end_idx] = output_squeezed
            step_num += 1


train_loader, verify_loader, test_loader = get_data()
model = ASNet()
model_name = 'ASNet'

print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

begin_time = time()
if os.path.exists(model_name + '.pkl'):
    print('load model')
    model.load_state_dict(torch.load(model_name + '.pkl'))
else:
    print('Warning: Model file not found. Using untrained model.')

# 计算测试集的实际大小
num_test_samples = len(test_loader.dataset)
print(f'Number of test samples: {num_test_samples}')

# 获取序列长度 (从第一个样本获取)
first_sample = test_loader.dataset[0][0]  # 获取第一个输入样本
sequence_length = first_sample.shape[0] if first_sample.ndim == 1 else first_sample.shape[1]  # 获取序列长度
print(f'Sequence length: {sequence_length}')

test_input_z = torch.zeros(num_test_samples, sequence_length)
test_output_z = torch.zeros(num_test_samples, sequence_length)
pre_z = torch.zeros(num_test_samples, sequence_length)

test(model, device, test_loader, num_test_samples, num_test_samples//10, test_input_z, test_output_z, pre_z)

# 绘制结果
i = 100 if num_test_samples > 100 else 0
x = np.linspace(0, 2, sequence_length)
l0, = plt.plot(x, test_input_z[i])
l1, = plt.plot(x, test_output_z[i])
l2, = plt.plot(x, pre_z[i])
plt.legend([l0, l1, l2], ['Contaminated EEG', 'Pure EEG', 'Corrected EEG'], loc='upper right')
plt.xlabel('Time (s)')  # 设置x轴标签
plt.ylabel('Amplitude(mV)')  # 设置y轴标签
plt.show()


