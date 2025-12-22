% ========================================
% MATLAB 可视化执行指南
% ========================================
%
% 本文件说明如何在MATLAB中执行可视化
%
% 重要: MATLAB只负责绘图，不负责模型预测！
% 模型预测由Python完成

%% ========================================
%% 步骤0: 用Python生成预测结果 (必需!)
%% ========================================
% 
% 在运行MATLAB之前，必须先用Python生成预测结果：
%
% 1. 打开命令行/终端
% 2. cd 到项目目录
% 3. 运行Python预测脚本:
%    python 真实数据集测试/test_real_dataset.py
%
% 这会生成文件: results/DAT-Net-Unsupervised-v2_real_data_predictions.mat
%

%% ========================================
%% 步骤1: 在MATLAB中运行测试
%% ========================================

% 打开MATLAB，切换到本目录，然后运行：
test_visualization

% 如果显示 "✓ 所有测试通过!"，说明配置正确

%% ========================================
%% 步骤2: 运行可视化
%% ========================================

%% 方法1: 运行所有可视化任务
% 可视化样本1，通道1的所有任务
main_visualization('all', 1, 1);

% 可视化样本5，通道10的所有任务
% main_visualization('all', 5, 10);

%% 方法2: 运行单个任务
% 伪影概率计算可视化 - 样本1，通道1
% vis_artifact_probability(1, 1);

% 掩蔽策略可视化 - 样本3，通道8
% vis_masking_strategy(3, 8);

% 去噪效果可视化 - 样本5，通道10
% vis_denoising_results(5, 10);

%% 方法3: 使用便捷函数
% 快速可视化样本3的所有任务（使用默认通道1）
% quick_vis('all', 3);

% 可视化样本5通道10的所有任务
% visualize_sample(5, 10);

%% ========================================
%% 步骤3: 查看结果
%% ========================================

% 所有生成的图像保存在:
% outputs_matlab/figures/

% 在MATLAB中打开图像目录:
% cd outputs_matlab/figures/

% 或在文件资源管理器中打开:
% winopen('outputs_matlab/figures/')

%% ========================================
%% 常见命令参考
%% ========================================

% 列出所有可用任务
% main_visualization('list')

% 查看帮助
% help main_visualization
% help vis_artifact_probability
% help vis_masking_strategy
% help vis_denoising_results

%% ========================================
%% 故障排除
%% ========================================

% 问题: 找不到数据文件
% 解决: 检查配置文件中的路径
% config = config_visualization();
% config.REAL_DATA_FILE
% config.MODEL_PREDICTION_FILE

% 问题: 没有预测结果文件
% 解决: 先运行Python预测脚本生成预测结果

% 问题: 索引超出范围
% 解决: 检查样本数和通道数
% load(config.REAL_DATA_FILE);
% size(eeg_data)  % 查看 [样本数, 通道数, 信号长度]

%% ========================================
%% 完整示例工作流
%% ========================================

% 1. 【在Python中】运行模型预测
%    python test_real_dataset.py

% 2. 【在MATLAB中】测试配置
%    test_visualization

% 3. 【在MATLAB中】运行可视化
%    main_visualization('all', 1, 1)

% 4. 【在MATLAB中】查看结果
%    cd outputs_matlab/figures/

fprintf('\n');
fprintf('========================================\n');
fprintf('执行指南已显示完成!\n');
fprintf('========================================\n');
fprintf('\n请按照上述步骤执行可视化\n');
fprintf('取消注释相应的代码行来运行\n');
fprintf('========================================\n\n');
