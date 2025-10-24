# EEGIFNet 新版训练和测试脚本

## 概述

本目录包含了重构后的EEGIFNet训练和测试脚本（`train_new.py` 和 `test_new.py`），使用与ASNet-main项目一致的数据集格式和标准化逻辑，同时保持EEGIFNet原有的网络结构和训练参数。

## 主要改进

### 1. 数据加载方式
- ✅ 使用与ASNet相同的`.npy`格式数据集
- ✅ 数据划分: 80% 训练集, 10% 验证集, 10% 测试集
- ✅ 自动处理数据路径

### 2. 标准化逻辑
- ✅ **归一化**: 使用每个样本的绝对值最大值进行归一化
- ✅ **反归一化**: 在计算最终指标时，使用原始的归一化因子恢复信号幅度
- ✅ 训练和损失计算在归一化空间进行
- ✅ 评估指标（ACC, RRMSE, SNR）在反归一化空间计算

### 3. 保持原有特性
- ✅ EEGIFNet的双网络结构 (INet + MNet)
- ✅ 原始的训练参数 (batch_size=256, lr=5e-5, epochs=80)
- ✅ RMSprop优化器 (alpha=0.9)
- ✅ 三重损失函数 (loss_e + loss_n + loss_all)
- ✅ 原始的评估指标计算方式

## 文件说明

```
EEGIFNet-master/
├── train_new.py          # 新的训练脚本
├── test_new.py           # 新的测试脚本
├── EEGIFNet.py          # 网络结构定义 (不变)
├── config.py            # 配置和工具函数 (不变)
├── basenetwork.py       # 基础网络模块 (不变)
├── checkpoint/          # 模型权重保存目录
└── result/              # 测试结果保存目录
```

## 使用方法

### 训练模型

#### 基本用法
```bash
python train_new.py
```

#### 自定义参数
```bash
python train_new.py \
    --data_path "D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据" \
    --batch_size 256 \
    --lr 5e-5 \
    --epochs 80 \
    --device cuda:0 \
    --save_dir ./checkpoint
```

#### 参数说明
- `--data_path`: 数据集路径（包含Contaminated.npy和Pure_Data.npy）
- `--batch_size`: 批大小，默认256
- `--lr`: 学习率，默认5e-5
- `--epochs`: 训练轮数，默认80
- `--device`: 设备，默认cuda:0
- `--save_dir`: 模型保存目录，默认./checkpoint

#### 训练输出
训练过程会显示:
- 每个epoch的训练损失 (loss_e, loss_n, loss_all)
- 验证集上的各项指标
  - **归一化空间**: ACC, RRMSE (用于loss计算)
  - **反归一化空间**: ACC, RRMSE, SNR (真实性能)
- 自动保存最佳模型和定期checkpoint

### 测试模型

#### 基本用法
```bash
python test_new.py
```

#### 带可视化
```bash
python test_new.py --visualize --num_vis 20
```

#### 自定义参数
```bash
python test_new.py \
    --data_path "D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据" \
    --batch_size 256 \
    --device cuda:0 \
    --checkpoint_dir ./checkpoint \
    --inet_model EEGIFNet_INet_best.pkl \
    --mnet_model EEGIFNet_MNet_best.pkl \
    --result_dir ./result \
    --visualize \
    --num_vis 10
```

#### 参数说明
- `--data_path`: 数据集路径
- `--batch_size`: 批大小
- `--device`: 设备
- `--checkpoint_dir`: 模型权重目录
- `--inet_model`: INet模型文件名
- `--mnet_model`: MNet模型文件名
- `--result_dir`: 结果保存目录
- `--visualize`: 是否生成可视化结果
- `--num_vis`: 可视化样本数量

#### 测试输出
测试脚本会生成:
1. **EEGIFNet_test_results.csv**: 每个样本的详细结果
   - ACC, RRMSE, SNR_dB
   - ACC_e (INet EEG分支)
   - ACC_n (INet Noise分支)

2. **EEGIFNet_test_statistics.csv**: 统计信息
   - 均值、标准差、最小值、最大值

3. **visualizations/** (如果使用--visualize): 可视化图像
   - 每个样本生成一张包含三个子图的PNG图像
   - (a) INet EEG分支输出
   - (b) 通过减去noise分支得到的去噪结果
   - (c) MNet融合输出

## 数据集要求

数据集目录应包含以下文件:
```
已经生成好的数据/
├── Contaminated.npy    # 污染的EEG信号
└── Pure_Data.npy       # 纯净的EEG信号
```

数据格式:
- Shape: `(num_samples, time_points)`
- 每个样本应为一维时间序列
- ASNet使用的是1200个时间点 (6秒 @ 200Hz)

## 网络结构

### INet (MA_INet)
- **输入**: 污染的EEG信号 (归一化后)
- **输出**: 
  - `e_outputs`: 预测的纯净EEG信号
  - `n_outputs`: 预测的噪声信号
- **结构**: 
  - 双分支CNN-GRU结构
  - EEG分支和Noise分支并行处理
  - 3层卷积 + GRU + 全连接层

### MNet (MA_MNet)
- **输入**: 原始污染信号, INet的两个输出
- **输出**: 融合后的去噪EEG信号
- **作用**: 融合INet的预测，提升去噪性能

## 训练策略

### 损失函数
- `loss_e`: INet EEG分支与真实clean EEG的MSE
- `loss_n`: INet Noise分支与真实noise的MSE
- `loss_all`: MNet输出与真实clean EEG的MSE
- **总损失**: `loss = loss_e + loss_n + loss_all`

### 优化器
- RMSprop (lr=5e-5, alpha=0.9)
- 分别优化INet和MNet

### 评估指标
1. **ACC (Correlation Coefficient)**: 相关系数，越接近1越好
2. **RRMSE (Relative Root Mean Square Error)**: 相对均方根误差，越小越好
3. **SNR (Signal-to-Noise Ratio)**: 信噪比，越大越好

## 与原始EEGIFNet的对比

| 方面 | 原始EEGIFNet | 新版本 |
|------|-------------|--------|
| 数据格式 | 自定义路径的.npy | 与ASNet一致的.npy |
| 数据划分 | 自定义比例 | 80-10-10 |
| 标准化 | 无明确标准化 | 样本级最大值归一化 |
| 反标准化 | 未处理 | ✅ 评估时反归一化 |
| 网络结构 | MA_INet + MA_MNet | ✅ 完全相同 |
| 训练参数 | batch=256, lr=5e-5 | ✅ 完全相同 |
| 损失函数 | loss_e+loss_n+loss_all | ✅ 完全相同 |
| 评估指标 | ACC, RRMSE | ✅ ACC, RRMSE + SNR |
| 可视化 | 简单绘图 | ✅ 增强的三子图对比 |

## 典型训练流程

1. **准备数据**
   ```bash
   # 确保数据在正确路径
   ls "D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据"
   # 应该看到: Contaminated.npy, Pure_Data.npy
   ```

2. **开始训练**
   ```bash
   python train_new.py --epochs 80
   ```

3. **监控训练**
   - 观察训练和验证损失
   - 关注验证集的ACC和RRMSE (反归一化)
   - 最佳模型会自动保存到checkpoint/

4. **测试模型**
   ```bash
   python test_new.py --visualize --num_vis 20
   ```

5. **查看结果**
   - 检查result/EEGIFNet_test_statistics.csv
   - 查看result/visualizations/中的图像

## 常见问题

### Q1: 训练时显存不足
**A**: 减小batch_size
```bash
python train_new.py --batch_size 128
```

### Q2: 想使用CPU训练
**A**: 设置device为cpu
```bash
python train_new.py --device cpu
```

### Q3: 数据路径不对
**A**: 使用绝对路径指定数据位置
```bash
python train_new.py --data_path "你的数据路径"
```

### Q4: 加载模型失败
**A**: 检查checkpoint目录和文件名
```bash
python test_new.py \
    --checkpoint_dir ./checkpoint \
    --inet_model EEGIFNet_INet_best.pkl \
    --mnet_model EEGIFNet_MNet_best.pkl
```

## 性能优化建议

1. **学习率调整**: 如果loss不降，可以尝试更小的学习率
   ```bash
   python train_new.py --lr 1e-5
   ```

2. **更多训练轮数**: 如果验证指标还在提升
   ```bash
   python train_new.py --epochs 100
   ```

3. **数据增强**: 在EEGDataset类中可以添加数据增强

## 致谢

- 原始EEGIFNet论文和代码
- ASNet数据集和标准化方法

## 作者

重构和适配: 脑机接口博士生

## 许可

与原项目保持一致
