%% ACMD方法统一测试脚本
% 在测试集上运行ACMD,保存预测结果和计算指标
%
% 使用前提:
%   1. 先运行 convert_to_mat.py 将数据转换为.mat格式
%   2. 确保 oa_remove_acmd.m 和 compute_eog_metrics.m 在路径中
%
% 输出:
%   - results/ACMD_predictions.mat: 预测结果
%   - results/ACMD_metrics.mat: 评价指标
%
% 作者: GitHub Copilot
% 日期: 2025-11-03

clear; clc; close all;

%% 添加路径
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\ACMD');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法');  % 通用metrics函数

%% 加载测试数据
fprintf('==============================================\n');
fprintf('           ACMD 测试脚本\n');
fprintf('==============================================\n\n');

fprintf('加载数据...\n');

% 加载.mat格式数据
data_contaminated = load('D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Contaminated.mat');
data_clean = load('D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Pure_Data.mat');

contaminated = data_contaminated.contaminated;
clean = data_clean.pure;

% 数据拆分: 80% 训练, 10% 验证, 10% 测试
num_samples = size(contaminated, 1);
verify_end = floor(num_samples * 0.9);

% 提取测试集
test_contaminated = contaminated(verify_end+1:end, :);
test_clean = clean(verify_end+1:end, :);

num_test = size(test_contaminated, 1);
fprintf('测试集样本数: %d\n', num_test);
fprintf('信号长度: %d\n\n', size(test_contaminated, 2));

%% 运行ACMD去噪
fprintf('开始ACMD去噪...\n');

fs = 200;  % 采样率
predictions = zeros(size(test_contaminated));

% 记录时间
tic;

for i = 1:num_test
    try
        % 调用ACMD去噪函数
        predictions(i, :) = oa_remove_acmd(test_contaminated(i, :), fs);
        
        % 显示进度
        if mod(i, 10) == 0
            fprintf('  已处理 %d/%d 样本 (%.1f%%)\n', i, num_test, i/num_test*100);
        end
    catch ME
        warning('样本 %d 处理失败: %s', i, ME.message);
        predictions(i, :) = test_contaminated(i, :);  % 失败时返回原始信号
    end
end

% 计算总时间
total_time = toc;
time_per_sample = total_time / num_test;

fprintf('\nACMD去噪完成!\n');
fprintf('总耗时: %.2f 秒\n', total_time);
fprintf('单样本处理时间: %.3f ms\n\n', time_per_sample * 1000);

%% 保存预测结果
output_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

pred_save_path = fullfile(output_dir, 'ACMD_predictions.mat');
save(pred_save_path, 'predictions');

fprintf('预测结果已保存: %s\n\n', pred_save_path);

%% 计算评价指标
fprintf('==============================================\n');
fprintf('           计算评价指标\n');
fprintf('==============================================\n\n');

% 调用通用评价指标函数
metrics = compute_eog_metrics(test_clean, predictions, fs);
metrics.time_per_sample = time_per_sample;

% 保存指标
metrics_save_path = fullfile(output_dir, 'ACMD_metrics.mat');
save(metrics_save_path, 'metrics');

fprintf('\n指标结果已保存至: %s\n\n', metrics_save_path);

%% 打印汇总
fprintf('==============================================\n');
fprintf('           测试完成汇总\n');
fprintf('==============================================\n');
fprintf('测试样本数:         %d\n', num_test);
fprintf('单样本处理时间:     %.3f ms\n', time_per_sample * 1000);
fprintf('----------------------------------------------\n');
fprintf('RRMSE:              %.4f ± %.4f\n', metrics.RRMSE_mean, metrics.RRMSE_std);
fprintf('CC:                 %.4f ± %.4f\n', metrics.CC_mean, metrics.CC_std);
fprintf('RRMSE_PSD:          %.4f ± %.4f\n', metrics.RRMSE_PSD_mean, metrics.RRMSE_PSD_std);
fprintf('MI:                 %.4f ± %.4f\n', metrics.MI_mean, metrics.MI_std);
fprintf('==============================================\n');
