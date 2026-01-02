function config = getDatasetConfig(varargin)
% GETDATASETCONFIG - 获取数据集配置信息
%
% 用法:
%   config = getDatasetConfig()  % 使用默认数据集(fully_simulated)
%   config = getDatasetConfig('semi_simulated')  % 使用半模拟数据集
%   config = getDatasetConfig('fully_simulated')  % 使用全模拟数据集
%
% 输出:
%   config - 包含以下字段的结构体:
%       .name - 数据集名称
%       .fs - 采样率(Hz)
%       .windowSize - 窗口大小(样本数)
%       .dataDir - 数据目录路径
%       .trainContaminated - 训练集污染数据文件名
%       .trainPure - 训练集纯净数据文件名
%       .valContaminated - 验证集污染数据文件名
%       .valPure - 验证集纯净数据文件名
%       .testContaminated - 测试集污染数据文件名
%       .testPure - 测试集纯净数据文件名
%       .dataKey - MAT文件中的数据变量名
%
% 示例:
%   % 使用全模拟数据集
%   cfg = getDatasetConfig('fully_simulated');
%   data = load(fullfile(cfg.dataDir, cfg.trainContaminated));
%   trainX = data.(cfg.dataKey);
%
% 作者: Auto-generated
% 日期: 2025-12-02

    % 默认使用全模拟数据集
    if nargin == 0
        datasetName = 'semi_simulated';
    else
        datasetName = varargin{1};
    end
    
    % 获取项目根目录
    % 假设此函数在 "复现的方法" 目录下
    scriptPath = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(scriptPath);
    
    % 定义数据集配置
    switch lower(datasetName)
        case 'semi_simulated'
            config.name = '半模拟数据集';
            config.fs = 200;  % Hz
            config.windowSize = 1200;  % 样本数 (200Hz * 6s)
            config.dataDir = fullfile(projectRoot, '生成半模拟数据', '已经生成好的数据', 'multi_snr');
            config.description = '基于真实EEG数据生成的半模拟数据集，包含多SNR级别';
            
            % 多SNR测试集配置
            config.testSnrLevels = [-8,-6, -4, -2, 0, 2,4];
            config.hasMultiSnrTest = true;
            
        case 'fully_simulated'
            config.name = '全模拟数据集';
            config.fs = 250;  % Hz  
            config.windowSize = 1500;  % 样本数 (250Hz * 6s)
            config.dataDir = fullfile(projectRoot, '生成全模拟数据', '已经生成好的数据', 'Multi_SNR_Merged');
            config.description = '完全模拟生成的数据集,7个SNR级别,格式[n_samples, 1500]';
            
            % 多SNR测试集配置
            config.testSnrLevels = [0, -2, -4, -8, -12, -14, -16];
            config.hasMultiSnrTest = true;
            
        otherwise
            error('未知的数据集: %s. 可用选项: semi_simulated, fully_simulated', datasetName);
    end
    
    % 通用配置(所有数据集共有)
    config.datasetName = datasetName;
    config.trainContaminated = 'Train_Contaminated.mat';
    config.trainPure = 'Train_Pure.mat';
    config.valContaminated = 'Val_Contaminated.mat';
    config.valPure = 'Val_Pure.mat';
    config.dataKey = 'data';
    
    % 生成完整路径
    config.trainContaminatedPath = fullfile(config.dataDir, config.trainContaminated);
    config.trainPurePath = fullfile(config.dataDir, config.trainPure);
    config.valContaminatedPath = fullfile(config.dataDir, config.valContaminated);
    config.valPurePath = fullfile(config.dataDir, config.valPure);
    
    % 处理测试集路径 - 支持多SNR级别
    if isfield(config, 'hasMultiSnrTest') && config.hasMultiSnrTest
        % 多SNR测试集
        config.testSnrPaths = struct('contaminated', {}, 'pure', {});
        for i = 1:length(config.testSnrLevels)
            snr = config.testSnrLevels(i);
            config.testSnrPaths(i).contaminated = fullfile(config.dataDir, ...
                sprintf('Test_Contaminated_SNR%ddB.mat', snr));
            config.testSnrPaths(i).pure = fullfile(config.dataDir, ...
                sprintf('Test_Pure_SNR%ddB.mat', snr));
        end
    else
        % 单一测试集（向后兼容）
        config.testContaminated = 'Test_Contaminated.mat';
        config.testPure = 'Test_Pure.mat';
        config.testContaminatedPath = fullfile(config.dataDir, config.testContaminated);
        config.testPurePath = fullfile(config.dataDir, config.testPure);
    end
    
    % 验证数据目录是否存在
    if ~exist(config.dataDir, 'dir')
        warning('数据目录不存在: %s', config.dataDir);
    end
    
end


function printDatasetInfo(datasetName)
% PRINTDATASETINFO - 打印数据集配置信息
%
% 用法:
%   printDatasetInfo()
%   printDatasetInfo('semi_simulated')

    if nargin == 0
        datasetName = 'fully_simulated';
    end
    
    config = getDatasetConfig(datasetName);
    
    fprintf('\n');
    fprintf('========================================\n');
    fprintf('数据集配置信息\n');
    fprintf('========================================\n');
    fprintf('数据集名称: %s (%s)\n', config.name, config.datasetName);
    fprintf('描述: %s\n', config.description);
    fprintf('采样率: %d Hz\n', config.fs);
    fprintf('窗口大小: %d 样本\n', config.windowSize);
    fprintf('数据目录: %s\n', config.dataDir);
    fprintf('\n数据文件:\n');
    fprintf('  训练集(污染): %s\n', config.trainContaminatedPath);
    fprintf('  训练集(纯净): %s\n', config.trainPurePath);
    fprintf('  验证集(污染): %s\n', config.valContaminatedPath);
    fprintf('  验证集(纯净): %s\n', config.valPurePath);
    fprintf('  测试集(污染): %s\n', config.testContaminatedPath);
    fprintf('  测试集(纯净): %s\n', config.testPurePath);
    fprintf('数据键名: %s\n', config.dataKey);
    fprintf('========================================\n');
    fprintf('\n');
end
