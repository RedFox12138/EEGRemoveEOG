%% SSA方法统一测试脚本
% 在测试集上运行SSA,保存预测结果和计算指标
%
% 使用前提:
%   1. 先运行 convert_to_mat.py 将数据转换为.mat格式
%   2. 确保 ci_ssa_eog_removal.m 和 compute_eog_metrics.m 在路径中
%
% 输出:
%   - results/SSA_predictions.mat: 预测结果
%   - results/SSA_metrics.mat: 评价指标
%
% 作者: GitHub Copilot
% 日期: 2025-11-03

% clear; clc;

% 加载数据集配置
addpath('..');
cfg = getDatasetConfig(); 
close all;

%% 添加路径
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\SSA');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法');

%% 加载测试数据
fprintf('==============================================\n');
fprintf('           SSA 测试脚本\n');
fprintf('==============================================\n\n');

fprintf('加载数据...\n');

data_contaminated = load(cfg.testContaminatedPath);
data_clean = load(cfg.testPurePath);

test_contaminated = data_contaminated.(cfg.dataKey);
test_clean = data_clean.(cfg.dataKey);

num_test = size(test_contaminated, 1);
fprintf('测试集样本数: %d\n', num_test);
fprintf('信号长度: %d\n\n', size(test_contaminated, 2));

%% 运行SSA去噪
fprintf('开始SSA去噪...\n');

fs = cfg.fs;
predictions = zeros(size(test_contaminated));

tic;

for i = 1:num_test
    try
        predictions(i, :) = ci_ssa_eog_removal(test_contaminated(i, :), 'fs', fs, ...
            'mode', 'paper', 'visualize', false, 'verbose', false);
        
        if mod(i, 10) == 0
            fprintf('  已处理 %d/%d 样本 (%.1f%%)\n', i, num_test, i/num_test*100);
        end
    catch ME
        warning('样本 %d 处理失败: %s', i, ME.message);
        predictions(i, :) = test_contaminated(i, :);
    end
end

total_time = toc;
time_per_sample = total_time / num_test;

fprintf('\nSSA去噪完成!\n');
fprintf('总耗时: %.2f 秒\n', total_time);
fprintf('单样本处理时间: %.3f ms\n\n', time_per_sample * 1000);

%% 保存预测结果
output_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

pred_save_path = fullfile(output_dir, 'SSA_predictions.mat');
save(pred_save_path, 'predictions', 'time_per_sample');

fprintf('预测结果已保存: %s\n', pred_save_path);

%% 打印汇总
fprintf('==============================================\n');
fprintf('           测试完成汇总\n');
fprintf('==============================================\n');
fprintf('测试样本数:         %d\n', num_test);
fprintf('单样本处理时间:     %.3f ms\n', time_per_sample * 1000);
fprintf('==============================================\n');
fprintf('\n✓ 完成！请运行统一指标计算脚本来评估所有方法。\n');
