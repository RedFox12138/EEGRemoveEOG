import scipy
import torch
import torch.optim as optim
import torch.utils.data as Data
import torch.nn as nn
import os
import numpy as np
from time import time
from torch.utils.data import Dataset

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


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    step_num=0
    loss_epoh=0
    for batch_idx, (train_input, train_output, norm_factors) in enumerate(train_loader):
        step_num +=1
        train_input=train_input.float().to(device)
        train_output=train_output.float().to(device)
        norm_factors = norm_factors.float().to(device).view(-1, 1) # 调整形状以进行广播 [batch, 1]

        optimizer.zero_grad()
        output = model(train_input)

        # 反向恢复幅度
        output_restored = output * norm_factors

        loss = loss_f(output_restored, train_output)
        loss_epoh+=loss.item()
        loss.backward()
        optimizer.step()
         
    print(loss_epoh/step_num)
    return loss_epoh/step_num
            
def verify(model, device, verify_loader, optimizer, epoch):
    model.eval()
    step_num=0
    loss_epoh=0
    with torch.no_grad():
        for batch_idx, (verify_input, verify_output, norm_factors) in enumerate(verify_loader):
            step_num +=1
            verify_input=verify_input.float().to(device)
            verify_output=verify_output.float().to(device)
            norm_factors = norm_factors.float().to(device).view(-1, 1) # 调整形状以进行广播 [batch, 1]

            output = model(verify_input)

            # 反向恢复幅度
            output_restored = output * norm_factors

            loss = loss_f(output_restored, verify_output)
            loss_epoh+=loss.item()

    print(loss_epoh/step_num)
    return loss_epoh/step_num


train_loader,verify_loader,test_loader = get_data()
from ASNet import ASNet
model = ASNet()
model_name = 'ASNet'
learning_rate = 5e-4
loss_f = nn.MSELoss(reduction='mean')
print("torch.cuda.is_available() = ", torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
begin_time = time()

for epoch in range(150):
    train(model, device, train_loader, optimizer, epoch)
    verify(model, device, verify_loader, optimizer, epoch)
    print('save model')
    torch.save(model.state_dict(),  model_name + '.pkl')
    training_time = time() - begin_time
    minute = int(training_time // 60)
    second = int(training_time % 60)
    print(f'{minute}:{second}')
    print('epoch')
    print(epoch)
    print('finish')
