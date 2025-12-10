%% ACMD 真实数据集测试脚本
% 使用ACMD方法对真实数据集进行EOG伪影去除
% 输出统一格式的.mat文件

clear; clc;

%% 配置
% 数据路径
REAL_DATA_PATH = 'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat';
DATA_KEY = 'eog_dataset';

% 采样率
SAMPLING_RATE = 250; % Hz

% 结果保存路径
RESULTS_DIR = 'results';
PREDICTION_SAVE_PATH = fullfile(RESULTS_DIR, 'ACMD_real_data_predictions.mat');

% 总结果保存路径
FINAL_RESULTS_DIR = fullfile('..', '..', '..', 'results');
FINAL_PREDICTION_PATH = fullfile(FINAL_RESULTS_DIR, 'ACMD_real_data_predictions.mat');

% 添加必要的路径
addpath(fullfile('..', '..'));  % 添加"复现的方法"目录以访问loadRealDatasetSplit

%% 加载真实数据集
fprintf('================================================================================\n');
fprintf('ACMD 真实数据集测试\n');
fprintf('================================================================================\n\n');

% 使用统一的数据划分函数（只加载测试集）
data = loadRealDatasetSplit(REAL_DATA_PATH, DATA_KEY);

%% 处理所有样本
n_samples = size(data, 1);
n_timepoints = size(data, 2);

fprintf('\n开始处理 %d 个样本...\n', n_samples);

% 初始化结果矩阵
cleaned_eeg = zeros(size(data));

tic; % 开始计时

for i = 1:n_samples
    % 获取当前样本
    signal = data(i, :);
    
    % 使用ACMD去除EOG伪影
    try
        % 调用ACMD去除函数
        cleaned_signal = oa_remove_acmd(signal, SAMPLING_RATE);
        cleaned_eeg(i, :) = cleaned_signal;
    catch ME
        fprintf('  ⚠️ 警告: 样本 %d 处理失败: %s\n', i, ME.message);
        % 如果处理失败，保留原始信号
        cleaned_eeg(i, :) = signal;
    end
    
    % 进度显示（每10个样本）
    if mod(i, 10) == 0
        elapsed = toc;
        avg_time = elapsed / i;
        remaining = avg_time * (n_samples - i);
        fprintf('  已处理 %d/%d 样本 (平均 %.3f秒/样本, 预计剩余 %.1f分钟)\n', ...
                i, n_samples, avg_time, remaining/60);
    end
end

total_time = toc;
time_per_sample = total_time / n_samples;

fprintf('\n处理完成!\n');
fprintf('  总耗时: %.1f秒\n', total_time);
fprintf('  单样本平均时间: %.3f秒\n', time_per_sample);

%% 计算伪影和统计信息
fprintf('\n计算统计信息...\n');

% 提取的EOG伪影
extracted_eog = data - cleaned_eeg;

% 验证解耦一致性
reconstructed = cleaned_eeg + extracted_eog;
consistency_error = mean((reconstructed - data).^2, 'all');
fprintf('  重建一致性 MSE: %.6f\n', consistency_error);

% 统计信息
original_std = std(data(:));
cleaned_std = std(cleaned_eeg(:));
artifact_std = std(extracted_eog(:));

fprintf('  原始信号标准差: %.4f\n', original_std);
fprintf('  去噪信号标准差: %.4f\n', cleaned_std);
fprintf('  伪影标准差: %.4f\n', artifact_std);

% 功率降低
power_reduction = (mean(data(:).^2) - mean(cleaned_eeg(:).^2)) / mean(data(:).^2);
fprintf('  平均功率降低: %.2f%%\n', power_reduction * 100);

%% 保存结果
fprintf('\n保存结果...\n');

% 创建结果目录
if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR);
end

% 保存到本地目录
original = data; % 重命名以匹配输出格式
save(PREDICTION_SAVE_PATH, ...
     'cleaned_eeg', 'extracted_eog', 'original', 'time_per_sample', ...
     'consistency_error', 'power_reduction', 'SAMPLING_RATE', 'n_timepoints', ...
     '-v7.3');

fprintf('  ✓ 本地结果已保存: %s\n', PREDICTION_SAVE_PATH);

% 保存到总结果目录
if ~exist(FINAL_RESULTS_DIR, 'dir')
    mkdir(FINAL_RESULTS_DIR);
end

% 重命名变量以匹配Python版本
original = data;
sampling_rate = SAMPLING_RATE;
window_size = n_timepoints;

save(FINAL_PREDICTION_PATH, ...
     'cleaned_eeg', 'extracted_eog', 'original', 'time_per_sample', ...
     'consistency_error', 'power_reduction', 'sampling_rate', 'window_size', ...
     '-v7.3');

fprintf('  ✓ 总结果已保存: %s\n', FINAL_PREDICTION_PATH);
fprintf('    - 去噪 EEG 形状: [%d, %d]\n', size(cleaned_eeg, 1), size(cleaned_eeg, 2));
fprintf('    - 提取 EOG 形状: [%d, %d]\n', size(extracted_eog, 1), size(extracted_eog, 2));

fprintf('\n================================================================================\n');
fprintf('测试完成！\n');
fprintf('================================================================================\n');
