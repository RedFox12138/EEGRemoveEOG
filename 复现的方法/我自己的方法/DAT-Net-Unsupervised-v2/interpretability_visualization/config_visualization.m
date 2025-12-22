function config = config_visualization()
% CONFIG_VISUALIZATION 可解释性可视化配置函数
% 包含所有可视化任务的参数设置,针对真实数据集
%
% 输出:
%   config - 配置结构体
%
% 用法:
%   cfg = config_visualization();

    %% ==================== 路径配置 ====================
    config.BASE_DIR = fileparts(mfilename('fullpath'));
    config.PROJECT_ROOT = fileparts(config.BASE_DIR);
    
    % 真实数据集路径
    config.REAL_DATA_DIR = 'D:\Pycharm_Projects\EOG Remove\真实数据集';
    config.REAL_DATA_FILE = fullfile(config.REAL_DATA_DIR, 'eog_dataset.mat');
    
    % 模型预测结果路径
    config.RESULTS_DIR = 'D:\Pycharm_Projects\EOG Remove\复现的方法\我自己的方法\DAT-Net-Unsupervised-v2\无监督真实数据集\results';
    config.MODEL_PREDICTION_FILE = fullfile(config.RESULTS_DIR, 'real_data_predictions.mat');
    
    % 输出目录
    config.OUTPUT_DIR = fullfile(config.BASE_DIR, 'outputs_matlab');
    config.FIGURE_DIR = fullfile(config.OUTPUT_DIR, 'figures');
    config.DATA_OUTPUT_DIR = fullfile(config.OUTPUT_DIR, 'data');
    
    % 创建输出目录
    if ~exist(config.OUTPUT_DIR, 'dir'), mkdir(config.OUTPUT_DIR); end
    if ~exist(config.FIGURE_DIR, 'dir'), mkdir(config.FIGURE_DIR); end
    if ~exist(config.DATA_OUTPUT_DIR, 'dir'), mkdir(config.DATA_OUTPUT_DIR); end
    
    %% ==================== 数据配置 ====================
    config.DATA.fs = 250;  % 真实数据集的采样率
    config.DATA.num_channels = 1;  % 单通道数据
    config.DATA.channels_to_visualize = 1;  % 可视化的通道（单通道固定为1）
    
    %% ==================== 模型配置 ====================
    config.MODEL.in_channels = 1;
    config.MODEL.base_channels = 32;
    
    %% ==================== 无监督损失函数配置 ====================
    config.LOSS.mask_base = 0.1857;
    config.LOSS.boost_scale = 0.2341;
    config.LOSS.lambda_rec = 0.8121;
    config.LOSS.lambda_con = 1.2613;
    config.LOSS.lambda_teacher = 0.4036;
    config.LOSS.lambda_n2v = 0.2440;
    config.LOSS.lambda_band = 0.0553;
    config.LOSS.lambda_low = 0.0577;
    config.LOSS.lambda_decor = 0.3153;
    config.LOSS.lambda_content = 0.6763;
    config.LOSS.gamma_art_weight = 1.0;
    config.LOSS.artifact_win_size = 64;
    config.LOSS.mask_neighborhood = 5;
    config.LOSS.teacher_cutoff = 8.0;
    config.LOSS.lowpass_cutoff = 4.0;
    config.LOSS.teacher_threshold = 0.7;
    
    %% ==================== 可视化配置 ====================
    config.VIS.dpi = 300;  % MATLAB中使用更高的分辨率
    config.VIS.figsize_single = [1200, 800];  % 单图尺寸[宽, 高]
    config.VIS.figsize_multi = [1600, 1200];  % 多图尺寸
    config.VIS.font_size = 10;
    config.VIS.title_size = 14;
    config.VIS.line_width = 1.5;
    
    % 样本选择
    config.VIS.default_sample_idx = 1;  % 默认样本索引(MATLAB从1开始)
    config.VIS.default_channel_idx = 1;  % 默认通道索引（单通道固定为1）
    config.VIS.num_samples_to_show = 5;
    
    % 颜色方案 - 使用RGB值
    config.VIS.colors.original = [46, 134, 171] / 255;      % 蓝色
    config.VIS.colors.clean = [6, 167, 125] / 255;          % 绿色
    config.VIS.colors.artifact = [214, 34, 70] / 255;       % 红色
    config.VIS.colors.target = [247, 127, 0] / 255;         % 橙色
    config.VIS.colors.masked = [160, 3, 111] / 255;         % 紫色
    config.VIS.colors.branch_a = [17, 138, 178] / 255;      % 深蓝
    config.VIS.colors.branch_b = [239, 71, 111] / 255;      % 粉红
    config.VIS.colors.eog = [255, 107, 53] / 255;          % EOG橙色
    
    %% ==================== 任务特定配置 ====================
    
    % 伪影概率计算
    config.ARTIFACT_PROB.show_intermediate_steps = true;
    config.ARTIFACT_PROB.win_size = 64;
    config.ARTIFACT_PROB.lowpass_cutoff = 4.0;
    
    % 掩蔽策略
    config.MASKING.compare_random = true;
    config.MASKING.num_comparisons = 3;
    
    % 去噪效果
    config.DENOISING.sample_selection = 'auto';
    config.DENOISING.num_good = 2;
    config.DENOISING.num_medium = 2;
    config.DENOISING.num_bad = 2;
    
    %% ==================== 导出格式 ====================
    config.EXPORT.save_png = false;  % 不保存PNG，直接显示
    config.EXPORT.save_pdf = false;
    config.EXPORT.save_eps = false;
    config.EXPORT.save_fig = false;  % 不保存MATLAB .fig格式
    config.EXPORT.save_data = false;  % 不保存数据
    
    %% ==================== 调试配置 ====================
    config.DEBUG.verbose = true;
    config.DEBUG.save_intermediate = true;
    
end
