"""
1D Noise2Void Data Wrapper for EEG Signals
改编自原始2D N2V的数据包装器，用于处理1D时序信号的盲点掩蔽
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class N2V_Dataset1D(Dataset):
    """
    1D Noise2Void Dataset for EEG Denoising
    
    核心思想：
    - 从输入信号中随机选择部分时间点（盲点）
    - 用这些盲点周围的值来预测盲点本身的值
    - 训练时输入是掩蔽后的信号，目标是原始信号
    
    Parameters:
    -----------
    data : np.ndarray
        输入数据，shape: (n_samples, time_steps)
    perc_pix : float
        要掩蔽的时间点百分比 (default: 1.5%)
    neighborhood_radius : int
        邻域半径，用于从邻近值中替换盲点 (default: 5)
    patch_size : int
        训练时使用的patch大小，None表示使用完整序列
    manipulator : str
        盲点值替换策略: 'uniform', 'median', 'neighbor'
    """
    def __init__(self, data, perc_pix=1.5, neighborhood_radius=5, 
                 patch_size=None, manipulator='uniform', clean_data=None):
        super(N2V_Dataset1D, self).__init__()
        
        self.data = data
        self.clean_data = clean_data  # 可选的干净数据用于验证
        self.has_clean = clean_data is not None
        self.n_samples = len(data)
        self.time_steps = data.shape[1]
        self.perc_pix = perc_pix
        self.neighborhood_radius = neighborhood_radius
        self.patch_size = patch_size if patch_size else self.time_steps
        self.manipulator = manipulator
        
        # 计算每个patch要掩蔽的点数
        self.num_blind_spots = max(1, int(self.patch_size * perc_pix / 100.0))
        
        # 用于分层采样的box大小
        self.box_size = int(np.round(np.sqrt(100.0 / perc_pix)))
        
        print(f'N2V Dataset初始化:')
        print(f'  - 样本数: {self.n_samples}')
        print(f'  - 时间步长: {self.time_steps}')
        print(f'  - Patch大小: {self.patch_size}')
        print(f'  - 每个patch盲点数: {self.num_blind_spots}')
        print(f'  - 盲点百分比: {perc_pix}%')
        print(f'  - 替换策略: {manipulator}')
    
    def __len__(self):
        return self.n_samples
    
    def get_stratified_coords(self):
        """
        分层采样：将序列分成多个box，在每个box中随机选择一个点
        这样可以确保盲点在整个序列中均匀分布
        """
        box_count = int(np.ceil(self.patch_size / self.box_size))
        coords = []
        
        for i in range(box_count):
            # 在当前box内随机选择一个位置
            coord = i * self.box_size + np.random.randint(0, self.box_size)
            if coord < self.patch_size:
                coords.append(coord)
            if len(coords) >= self.num_blind_spots:
                break
        
        return np.array(coords)
    
    def get_replacement_value(self, signal, coord):
        """
        根据指定策略获取用于替换盲点的值
        
        Parameters:
        -----------
        signal : np.ndarray
            输入信号 (time_steps,)
        coord : int
            盲点位置
        """
        if self.manipulator == 'uniform':
            # 从邻域随机选择一个非中心点的值
            left = max(0, coord - self.neighborhood_radius)
            right = min(len(signal), coord + self.neighborhood_radius + 1)
            
            # 创建邻域索引，排除中心点
            neighbors = []
            for i in range(left, right):
                if i != coord:
                    neighbors.append(i)
            
            if len(neighbors) > 0:
                idx = np.random.choice(neighbors)
                return signal[idx]
            else:
                return signal[coord]  # fallback
        
        elif self.manipulator == 'median':
            # 使用邻域的中位数
            left = max(0, coord - self.neighborhood_radius)
            right = min(len(signal), coord + self.neighborhood_radius + 1)
            
            neighbors = []
            for i in range(left, right):
                if i != coord:
                    neighbors.append(signal[i])
            
            if len(neighbors) > 0:
                return np.median(neighbors)
            else:
                return signal[coord]
        
        elif self.manipulator == 'neighbor':
            # 从正态分布采样选择一个邻居
            offset = int(np.clip(np.round(np.random.normal(0, 4)), 
                                -self.neighborhood_radius, 
                                self.neighborhood_radius))
            if offset == 0:
                offset = 1 if np.random.rand() > 0.5 else -1
            
            new_coord = coord + offset
            new_coord = np.clip(new_coord, 0, len(signal) - 1)
            return signal[new_coord]
        
        else:
            raise ValueError(f"Unknown manipulator: {self.manipulator}")
    
    def __getitem__(self, idx):
        # 获取原始信号
        signal = self.data[idx].copy()
        
        # 归一化
        norm = np.max(np.abs(signal))
        if norm == 0:
            norm = 1.0
        signal_normalized = signal / norm
        
        # 创建输入和目标
        x = signal_normalized.copy()
        y = signal_normalized.copy()  # 目标是原始信号
        
        # 掩蔽矩阵：标记哪些位置被掩蔽了
        mask = np.zeros(len(signal_normalized), dtype=np.float32)
        
        # 获取盲点位置（分层采样）
        blind_spot_coords = self.get_stratified_coords()
        
        # 对每个盲点进行处理
        for coord in blind_spot_coords:
            # 获取替换值
            replacement = self.get_replacement_value(signal_normalized, coord)
            
            # 在输入中替换盲点值
            x[coord] = replacement
            
            # 标记这个位置被掩蔽了
            mask[coord] = 1.0
        
        # 转换为tensor并添加通道维度
        x = torch.from_numpy(x).float().unsqueeze(0)  # (1, time_steps)
        y = torch.from_numpy(y).float().unsqueeze(0)  # (1, time_steps)
        mask = torch.from_numpy(mask).float().unsqueeze(0)  # (1, time_steps)
        
        if self.has_clean:
            clean = self.clean_data[idx]
            clean = torch.from_numpy(clean).float()
            return x, y, mask, clean, norm
        else:
            return x, y, mask, norm


class N2V_ValidationDataset1D(Dataset):
    """
    验证集数据：预先计算所有掩蔽，加速验证过程
    """
    def __init__(self, data, perc_pix=1.5, neighborhood_radius=5,
                 manipulator='uniform', clean_data=None):
        super(N2V_ValidationDataset1D, self).__init__()
        
        self.data = data
        self.clean_data = clean_data
        self.has_clean = clean_data is not None
        
        # 预先生成所有掩蔽版本
        temp_dataset = N2V_Dataset1D(
            data, perc_pix, neighborhood_radius, 
            patch_size=data.shape[1], manipulator=manipulator
        )
        
        print('预计算验证集掩蔽...')
        self.masked_inputs = []
        self.targets = []
        self.masks = []
        self.norms = []
        
        for i in range(len(temp_dataset)):
            if clean_data is not None:
                x, y, mask, clean, norm = temp_dataset[i]
            else:
                x, y, mask, norm = temp_dataset[i]
            
            self.masked_inputs.append(x)
            self.targets.append(y)
            self.masks.append(mask)
            self.norms.append(norm)
        
        print(f'验证集预计算完成: {len(self.masked_inputs)} 样本')
    
    def __len__(self):
        return len(self.masked_inputs)
    
    def __getitem__(self, idx):
        if self.has_clean:
            clean = self.clean_data[idx]
            clean = torch.from_numpy(clean).float()
            return (self.masked_inputs[idx], self.targets[idx], 
                   self.masks[idx], clean, self.norms[idx])
        else:
            return (self.masked_inputs[idx], self.targets[idx], 
                   self.masks[idx], self.norms[idx])


def n2v_loss(pred, target, mask):
    """
    Noise2Void损失函数
    
    只在盲点位置计算损失
    
    Parameters:
    -----------
    pred : torch.Tensor
        预测输出 (batch, channels, time_steps)
    target : torch.Tensor
        目标值 (batch, channels, time_steps)
    mask : torch.Tensor
        掩蔽矩阵 (batch, channels, time_steps)
    """
    # 只在掩蔽位置计算损失
    diff = (pred - target) ** 2
    masked_loss = diff * mask
    
    # 计算平均损失（只对掩蔽位置）
    loss = masked_loss.sum() / (mask.sum() + 1e-8)
    
    return loss


if __name__ == '__main__':
    # 测试数据包装器
    print('测试N2V_Dataset1D...')
    
    # 创建模拟数据
    data = np.random.randn(100, 1200).astype(np.float32)
    
    dataset = N2V_Dataset1D(data, perc_pix=1.5, neighborhood_radius=5,
                           manipulator='uniform')
    
    print(f'\nDataset大小: {len(dataset)}')
    
    # 测试获取样本
    x, y, mask, norm = dataset[0]
    print(f'输入形状: {x.shape}')
    print(f'目标形状: {y.shape}')
    print(f'掩蔽形状: {mask.shape}')
    print(f'掩蔽点数: {mask.sum().item()}')
    print(f'归一化系数: {norm}')
    
    # 测试损失函数
    pred = torch.randn_like(y)
    loss = n2v_loss(pred, y, mask)
    print(f'\n测试损失: {loss.item():.6f}')
    
    print('\n✓ 数据包装器测试通过')
