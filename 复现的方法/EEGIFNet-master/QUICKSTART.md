# EEGIFNet 新版本快速开始指南

## 🎯 核心改进

本版本将EEGIFNet适配为使用ASNet相同的数据集和标准化方法，同时完全保留原始网络结构和训练参数。

### ✅ 与ASNet一致的部分
- 数据格式：`.npy`文件 (Contaminated.npy + Pure_Data.npy)
- 数据划分：80% 训练 / 10% 验证 / 10% 测试
- 标准化：样本级最大值归一化
- 反标准化：评估时恢复原始幅度

### ✅ 保持EEGIFNet原有的部分
- 网络结构：MA_INet + MA_MNet
- 训练参数：batch_size=256, lr=5e-5, epochs=80
- 优化器：RMSprop (alpha=0.9)
- 损失函数：loss_e + loss_n + loss_all

## 🚀 快速开始

### 方法1：使用交互式脚本（推荐）
```bash
python run_example.py
```
然后根据菜单选择操作即可。

### 方法2：直接运行命令

#### 训练模型
```bash
# 快速测试（5个epoch）
python train_new.py --epochs 5

# 完整训练（80个epoch）
python train_new.py --epochs 80
```

#### 测试模型
```bash
# 基础测试
python test_new.py

# 带可视化（推荐）
python test_new.py --visualize --num_vis 20
```

## 📊 输出结果

### 训练阶段
- **控制台输出**：每个epoch的loss和指标
- **模型保存**：
  - `checkpoint/EEGIFNet_INet_best.pkl` - INet最佳权重
  - `checkpoint/EEGIFNet_MNet_best.pkl` - MNet最佳权重
  - `checkpoint/EEGIFNet_*Net_epoch*.pkl` - 定期checkpoint

### 测试阶段
- **CSV文件**：
  - `result/EEGIFNet_test_results.csv` - 每个样本的详细结果
  - `result/EEGIFNet_test_statistics.csv` - 统计摘要
  
- **可视化图像** (如果使用--visualize)：
  - `result/visualizations/sample_*.png` - 对比图
  - 每张图包含三个子图：
    - (a) INet EEG分支输出
    - (b) X - Noise分支的去噪结果
    - (c) MNet融合输出

## 📈 评估指标

所有指标在**反归一化后的信号**上计算（真实性能）：

- **ACC** (Correlation Coefficient): 相关系数，越接近1越好
- **RRMSE** (Relative RMSE): 相对均方根误差，越小越好
- **SNR** (Signal-to-Noise Ratio): 信噪比(dB)，越大越好

## ⚙️ 常用参数调整

### 如果显存不足
```bash
python train_new.py --batch_size 128
```

### 使用不同GPU
```bash
python train_new.py --device cuda:1
```

### 使用CPU
```bash
python train_new.py --device cpu
```

### 自定义数据路径
```bash
python train_new.py --data_path "你的数据路径"
python test_new.py --data_path "你的数据路径"
```

## 🔍 文件说明

| 文件 | 说明 |
|------|------|
| `train_new.py` | 新的训练脚本 |
| `test_new.py` | 新的测试脚本 |
| `run_example.py` | 交互式运行脚本 |
| `README_NEW.md` | 详细文档 |
| `EEGIFNet.py` | 网络定义（原始） |
| `config.py` | 工具函数（原始） |
| `basenetwork.py` | 基础模块（原始） |

## 💡 典型工作流程

```bash
# 1. 快速测试训练（5个epoch）
python train_new.py --epochs 5

# 2. 如果没问题，完整训练
python train_new.py --epochs 80

# 3. 测试并生成可视化
python test_new.py --visualize --num_vis 20

# 4. 查看结果
# - 打开 result/EEGIFNet_test_statistics.csv 查看统计
# - 查看 result/visualizations/ 中的图像
```

## 📝 数据要求

确保数据目录包含：
```
你的数据路径/
├── Contaminated.npy  # Shape: (N, T) - 污染信号
└── Pure_Data.npy     # Shape: (N, T) - 纯净信号
```

其中：
- N: 样本数量
- T: 时间点数量（如1200点 = 6秒@200Hz）

## 🆘 问题排查

**Q: 找不到数据文件**
```bash
# 检查数据路径
python -c "import os; print(os.path.exists('你的数据路径/Contaminated.npy'))"
```

**Q: 加载模型失败**
```bash
# 检查模型文件
ls checkpoint/EEGIFNet_*Net_best.pkl
```

**Q: CUDA out of memory**
```bash
# 减小批大小
python train_new.py --batch_size 64
```

## 📚 更多信息

详细文档请查看 `README_NEW.md`

---

**版本**: 1.0  
**适配**: EEGIFNet + ASNet数据集  
**作者**: 脑机接口博士生
