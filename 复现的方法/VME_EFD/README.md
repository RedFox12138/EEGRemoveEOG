# VME-EFD 眼动伪迹去除算法实现

本目录包含基于论文的 VME-EFD（Variational Mode Extraction + Empirical Fourier Decomposition）算法的 MATLAB 实现。

## 文件说明

- **vme_efd_denoise.m**: 主算法函数，实现 VME + EFD 的完整流程
- **test_vme_efd_sim10.m**: 可执行测试脚本，使用与 ACMD 一致的 sim10 数据
- **quick_test_vme_efd.m**: 快速验证脚本，使用合成信号测试基本功能

## 快速开始

### 快速验证（推荐首次运行）

```matlab
cd('d:/Pycharm_Projects/EOG Remove/复现的方法/VME_EFD')
quick_test_vme_efd
```

这将使用合成信号快速验证算法是否正常工作，无需准备数据。

## 依赖

- 需要 `VME_GMETV/VME/vme.m`（已在仓库中）
- 无需额外工具箱（内置了 bandpower、findpeaks、skewness 等的简化实现）

## 使用方法

### 方式 A：从工作区变量运行

1. 在 MATLAB 工作区准备数据：
   ```matlab
   % sim10_con: 污染的 EEG 信号（列向量或 channels×samples 矩阵）
   % sim10_resampled: 干净参考信号（可选，用于评价）
   ```

2. 运行测试脚本：
   ```matlab
   cd('d:/Pycharm_Projects/EOG Remove/复现的方法/VME_EFD')
   test_vme_efd_sim10
   ```

### 方式 B：从 .mat 文件加载

1. 编辑 `test_vme_efd_sim10.m` 顶部配置：
   ```matlab
   useWorkspaceVars = false;
   contFile = 'path/to/your/contaminated.mat';  % 包含 sim10_con
   cleanFile = 'path/to/your/clean.mat';        % 包含 sim10_resampled（可选）
   ```

2. 运行脚本

### 直接调用函数

```matlab
% 添加路径
addpath('d:/Pycharm_Projects/EOG Remove/复现的方法/VME_EFD');
addpath('d:/Pycharm_Projects/EOG Remove/复现的方法/VME_GMETV/VME');

% 准备参数
params = struct();
params.alpha  = 3500;      % VME 紧致系数
params.omega0 = 2.8;       % VME 初始中心频率 (Hz)
params.K = 6;              % EFD 分解层数
params.nArtifacts = 2;     % 要剔除的伪迹层数
params.lfCut = 5;          % 低频上限 (Hz)
params.verbose = true;

% 去除伪迹
[y_denoised, info] = vme_efd_denoise(contaminated_signal, sampling_rate, params);
```

## 参数说明

### VME 参数
- `alpha`: 紧致系数（默认 3500）
  - 较大值：更紧致的模态，适合窄带伪迹
  - 可设为数组进行网格搜索：`[2000, 3500, 5000]`
  
- `omega0`: 初始中心频率 (Hz，默认 2.8)
  - EOG 通常在 0.5-4 Hz，建议 2-3 Hz
  - 可设为数组进行网格搜索：`[2.0, 2.5, 2.8, 3.0]`

- `tau`: 对偶上升步长（默认 0，噪声较大时使用）
- `tol`: 收敛阈值（默认 1e-6）

### EFD 参数
- `K`: 分解层数（默认 6）
  - 论文中使用 6 层（efd1-efd6）
  
- `nArtifacts`: 要剔除的伪迹层数（默认 2）
  - 论文中通常剔除 efd3 和 efd6
  
- `lfCut`: 低频上限 (Hz，默认 5)
  - 只在频谱质心 ≤ lfCut 的层中选择伪迹
  - EOG 主要在低频，建议 3-7 Hz

## 算法流程

1. **VME 提取 EOG 段** (`xeog`)
   - 可选网格搜索优化 alpha 和 omega0
   - 目标：低频集中且与残差相关性小

2. **计算残差** `y1 = y - xeog`

3. **EFD 分解 xeog**
   - 基于幅度谱峰值确定频段边界
   - 生成 K 个零相位带通分量

4. **计算每层指标**
   - 能量、偏度、频谱质心

5. **选择伪迹层**
   - 在低频层（质心 ≤ lfCut）中
   - 选取能量最高的前 nArtifacts 个

6. **重构** `y_denoised = y1 + sum(保留的层)`

## 输出说明

### 函数返回值
- `y_denoised`: 去伪迹后的 EEG
- `info`: 结构体，包含：
  - `xeog`: VME 估计的 EOG 段
  - `y1`: 残差 (y - xeog)
  - `efd`: {Kx1} cell 数组，每层分量
  - `bands`: [Kx2] 每层频段边界 (Hz)
  - `energy`: [Kx1] 每层能量
  - `skew`: [Kx1] 每层偏度
  - `centroid`: [Kx1] 每层频谱质心 (Hz)
  - `artifactIdx`: 被剔除的层索引
  - `alpha`, `omega0`: 最终使用的 VME 参数

### 测试脚本输出
- **控制台**：
  - 数据维度诊断
  - 选择的 VME 参数
  - 每层 EFD 的频段、能量、偏度、质心
  - 评价指标（若有参考）：CC、RRMSE、SNR 提升

- **图形**：
  - Clean EEG（若有）
  - 残差 y1
  - 污染 EEG
  - 估计的 EOG 段
  - 去伪迹结果

## 故障排除

### 错误：Size 输入必须为整数（VME zeros 错误）
- **原因**：VME 要求信号长度必须为偶数（用于镜像扩展）
- **解决**：代码已自动处理
  - 奇数长度信号会自动截断最后一个样本
  - 输出时会补齐到原始长度
  - 控制台会显示截断提示

### 错误：数组大小不兼容
- **原因**：VME 调用失败或返回空结果
- **解决**：
  1. 检查信号长度（至少 100 个采样点）
  2. 检查采样率设置是否正确
  3. 尝试调整 alpha 和 omega0
  4. 查看控制台的 VME 警告信息

### 网格搜索全部失败
- 自动回退到第一组参数
- 建议手动指定单组参数进行测试

### 去噪效果不佳
- **过剔除**（EEG 失真）：
  - 减少 `nArtifacts`（如改为 1）
  - 降低 `lfCut`（如改为 3）
  
- **欠剔除**（伪迹残留）：
  - 增加 `nArtifacts`（如改为 3）
  - 提高 `lfCut`（如改为 7）
  - 调整 VME 参数（增大 alpha 或调整 omega0）

## 与论文对应关系

- **Figure 1**: 完整流程在 `vme_efd_denoise.m` 中实现
- **Section 2.2 (VME)**: 调用 `vme.m`，支持参数优化
- **Section 2.3 (EFD)**: `efd_decompose` 子函数
  - 频谱分段（基于峰值和极小值）
  - 零相位滤波器组
  - 6 层分解（efd1-efd6）
- **Figure 3**: 各层分量在 `info.efd` 中
- **Table 1**: 能量和偏度在 `info.energy` 和 `info.skew` 中
- **公式 (1)**: 重构逻辑（剔除 efd3/efd6，保留其他层）

## 性能建议

- 首次使用：单组参数（alpha=3500, omega0=2.8）
- 需要优化：小网格搜索（3×3 或 3×4 组合）
- 避免过大网格（会显著增加计算时间）

## 引用

如果使用本实现，请引用原论文中的 VME 和 EFD 相关文献。
