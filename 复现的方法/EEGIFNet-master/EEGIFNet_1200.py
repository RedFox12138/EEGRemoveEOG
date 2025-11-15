"""
EEGIFNet网络结构 - 适配1200时间点
基于原始EEGIFNet，调整全连接层以适配ASNet数据集(1200时间点)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init(m):
    if isinstance(m, nn.Conv1d):
        torch.nn.init.kaiming_uniform_(m.weight, mode='fan_in')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class Conv_Block(nn.Module):
    def __init__(self, channel=64, kernel=9, stride=1, padding=4):
        super(Conv_Block, self).__init__()
        self.lay = nn.Sequential(
            nn.Conv1d(channel, channel // 2, kernel, stride, padding, bias=False),
            nn.BatchNorm1d(channel // 2),
            nn.Dropout(0.1),
            nn.Sigmoid(),
        )
        self.lay2 = nn.Sequential(
            nn.Conv1d(channel, channel, kernel, stride, padding),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(channel, channel, kernel, stride, padding),
            nn.BatchNorm1d(channel),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(channel, channel // 2, kernel, stride, padding),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.lay2(x)


class Interaction_Block(nn.Module):
    def __init__(self, channel=64, outchannel=8):
        super(Interaction_Block, self).__init__()
        self.Conv_n2s = Conv_Block(channel*2)
        self.Conv_s2n = Conv_Block(channel*2)
        self.lay_s = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=outchannel, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        self.lay_n = nn.Sequential(
            nn.Conv1d(in_channels=channel, out_channels=outchannel, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(outchannel),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
    
    def forward(self, F_RA_s, F_RA_n):
        F_cat = torch.cat((F_RA_n, F_RA_s), dim=1)

        Mask_n = self.Conv_n2s(F_cat)
        Mask_s = self.Conv_s2n(F_cat)

        H_n2s = F_RA_n * Mask_n
        H_s2n = F_RA_s * Mask_s

        H_n2s = self.lay_s(H_n2s)
        H_s2n = self.lay_n(H_s2n)
        F_RA_S = torch.cat((F_RA_s, H_n2s), dim=1)
        F_RA_N = torch.cat((F_RA_n, H_s2n), dim=1)

        return F_RA_S, F_RA_N


class Exchange(nn.Module):
    def __init__(self, K=5):
        super(Exchange, self).__init__()
        self.K = K
    
    def forward(self, e, n, bn_e, bn_n, bn_threshold=0.01):
        bn1, bn2 = bn_e.weight.abs(), bn_n.weight.abs()
        _, idx1 = bn1.topk(self.K, largest=False, sorted=False)
        _, idx2 = bn2.topk(self.K, largest=False, sorted=False)
        
        x1 = e.clone()
        x2 = n.clone()
        x1[:, idx1, :] = n[:, idx1, :]
        x2[:, idx2, :] = e[:, idx2, :]
        
        return x1, x2


class MA_INet(nn.Module):
    """
    INet - 适配1200时间点
    
    架构:
    - 输入: (B, 1, 1200)
    - Conv1 (stride=2): (B, 32, 600)
    - Conv2 (stride=2): (B, 32, 300)
    - Conv3 (stride=2): (B, 32, 150)
    - GRU: (B, 150, 64)
    - Flatten: (B, 9600)
    - FC: (B, 1200)
    """
    def __init__(self, input_length=1200):
        super(MA_INet, self).__init__()
        
        self.input_length = input_length

        # EEG分支
        self.c1_e = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm1_e = nn.BatchNorm1d(num_features=32)
        self.relu1_e = nn.ReLU()
        self.drop1_e = nn.Dropout(p=0.1)
        self.drop = nn.Dropout(p=0.1)

        # Noise分支
        self.c1_n = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm1_n = nn.BatchNorm1d(num_features=32)
        self.relu1_n = nn.ReLU()
        self.drop1_n = nn.Dropout(p=0.1)
        
        self.ex1 = Exchange()
        self.ex2 = Exchange()
        self.ex3 = Exchange()
        self.ex1 = Exchange()
        self.ex2 = Exchange()
        self.ex3 = Exchange()
        self.i1 = Interaction_Block(32)

        # 第二层 (源码MA_INet不使用Interaction Block，输入通道数保持32)
        self.c2_e = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm2_e = nn.BatchNorm1d(num_features=32)
        self.relu2_e = nn.ReLU()

        self.c2_n = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm2_n = nn.BatchNorm1d(num_features=32)
        self.relu2_n = nn.ReLU()

        self.i2 = Interaction_Block(32)

        # 第三层
        self.c3_e = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm3_e = nn.BatchNorm1d(num_features=32)
        self.relu3_e = nn.ReLU()

        self.c3_n = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=2, padding=4)
        self.batnorm3_n = nn.BatchNorm1d(num_features=32)
        self.relu3_n = nn.ReLU()

        self.i3 = Interaction_Block(32)

        # GRU层
        self.rnn_e = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.rnn_n = nn.GRU(input_size=32, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        self.i4 = Interaction_Block(64)

        self.i4 = Interaction_Block(64)

        # 计算经过3次stride=2卷积后的长度
        # 1200 -> 600 -> 300 -> 150
        conv_output_length = input_length // 8  # 150
        rnn_output_dim = 64  # bidirectional GRU: 32*2=64
        fc_input_dim = conv_output_length * rnn_output_dim  # 150 * 64 = 9600

        # 全连接层 - 适配1200时间点
        self.f1_e = nn.Linear(fc_input_dim, input_length)
        self.f1_n = nn.Linear(fc_input_dim, input_length)

        self.fc1 = nn.Linear(input_length * 2, input_length)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=0.1)

        self.fc2 = nn.Linear(input_length, input_length)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.1)

        self.fc3 = nn.Linear(input_length, input_length)
        self.fc4 = nn.Linear(input_length, input_length)
        self.fc5 = nn.Linear(input_length, input_length)

    def forward(self, x):
        # 第一层卷积 + BN + 激活
        e = self.c1_e(x)
        e = self.batnorm1_e(e)
        n = self.c1_n(x)
        n = self.batnorm1_n(n)

        e = self.relu1_e(e)
        e = self.drop1_e(e)
        n = self.relu1_n(n)
        n = self.drop1_n(n)

        # ⚠️ 源码MA_INet注释掉了Interaction Block（与main.py一致）
        #e, n = self.ex1(e, n, self.batnorm1_e, self.batnorm1_n)
        #e, n = self.i1(e, n)

        # 第二层
        e = self.c2_e(e)
        e = self.batnorm2_e(e)
        n = self.c2_n(n)
        n = self.batnorm2_n(n)

        e = self.relu2_e(e)
        e = self.drop1_e(e)
        n = self.relu2_n(n)
        n = self.drop1_n(n)

        # ⚠️ 源码MA_INet注释掉了Interaction Block
        #e, n = self.ex2(e, n, self.batnorm2_e, self.batnorm2_n)
        #e, n = self.i2(e, n)

        # 第三层
        e = self.c3_e(e)
        e = self.batnorm3_e(e)
        n = self.c3_n(n)
        n = self.batnorm3_n(n)
        
        e = self.relu3_e(e)
        e = self.drop1_e(e)
        n = self.relu3_n(n)
        n = self.drop1_n(n)

        # ⚠️ 源码MA_INet注释掉了Interaction Block
        #e, n = self.ex3(e, n, self.batnorm3_e, self.batnorm3_n)
        #e, n = self.i3(e, n)

        # GRU层
        e, _ = self.rnn_e(e.permute(0, 2, 1))
        n, _ = self.rnn_n(n.permute(0, 2, 1))

        # ⚠️ 源码MA_INet注释掉了i4
        #e, n = self.i4(e.permute(0, 2, 1), n.permute(0, 2, 1))

        # Flatten
        e = e.reshape(e.size(0), -1)
        n = n.reshape(n.size(0), -1)

        e = self.drop(e)
        n = self.drop(n)

        # 全连接层（⚠️ 源码MA_INet只处理e_out，n_out的中间层被注释掉）
        e_out = self.f1_e(e)
        n_out = self.f1_n(n)

        e_out = self.relu1(e_out)
        #n_out = self.relu1(n_out)  # 源码注释掉
        e_out = self.drop(e_out)
        #n_out = self.drop(n_out)    # 源码注释掉
        
        e_out = self.fc2(e_out)
        #n_out = self.fc3(n_out)     # 源码注释掉
        e_out = self.relu1(e_out)
        #n_out = self.relu1(n_out)   # 源码注释掉
        e_out = self.drop(e_out)
        #n_out = self.drop(n_out)    # 源码注释掉
        
        e_out = self.fc4(e_out)
        #n_out = self.fc5(n_out)     # 源码注释掉

        return e_out, n_out


class MA_MNet(nn.Module):
    """
    MNet - 融合网络，适配1200时间点
    
    输入:
    - x: 原始污染信号 (B, 1, 1200)
    - x1: INet预测的clean EEG (B, 1200)
    - x2: INet预测的noise (B, 1200)
    
    输出:
    - 融合后的去噪信号 (B, 1200)
    """
    def __init__(self):
        super(MA_MNet, self).__init__()
        self.lay1 = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=9, stride=1, padding=4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(in_channels=32, out_channels=1, kernel_size=9, stride=1, padding=4),
            nn.Sigmoid(),
        )

    def forward(self, x, x1, x2):
        # x1: INet的EEG输出 (B, 1200)
        # x2: INet的Noise输出 (B, 1200)
        # x: 原始输入 (B, 1, 1200)
        
        x1 = x1.unsqueeze(1)  # (B, 1, 1200)
        x2 = x - x2.unsqueeze(1)  # x - noise = clean estimate
        
        # 拼接三个信号
        mask = torch.cat((x, x1, x2), dim=1)  # (B, 3, 1200)
        mask = self.lay1(mask)  # (B, 1, 1200)
        
        # 加权融合
        out = x1 * mask + x2 * (1 - mask)
        
        return out.squeeze()


if __name__ == '__main__':
    # 测试网络
    print("测试EEGIFNet (1200时间点版本)")
    
    # 创建测试数据
    batch_size = 4
    time_points = 1200
    x = torch.randn(batch_size, 1, time_points)
    
    # 测试INet
    print("\n测试INet...")
    inet = MA_INet(input_length=time_points)
    e_out, n_out = inet(x)
    print(f"输入形状: {x.shape}")
    print(f"INet EEG输出: {e_out.shape}")
    print(f"INet Noise输出: {n_out.shape}")
    
    # 测试MNet
    print("\n测试MNet...")
    mnet = MA_MNet()
    final_out = mnet(x, e_out, n_out)
    print(f"MNet最终输出: {final_out.shape}")
    
    # 统计参数量
    inet_params = sum(p.numel() for p in inet.parameters())
    mnet_params = sum(p.numel() for p in mnet.parameters())
    total_params = inet_params + mnet_params
    
    print(f"\n参数统计:")
    print(f"INet参数量: {inet_params:,}")
    print(f"MNet参数量: {mnet_params:,}")
    print(f"总参数量: {total_params:,}")
    
    print("\n✓ 网络测试通过!")
