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

%% 确定测试模式
fprintf('==============================================\n');
fprintf('           SSA 测试脚本\n');
fprintf('==============================================\n\n');

% 检查是否是多SNR测试配置
if cfg.hasMultiSnrTest
    snr_levels = cfg.testSnrLevels;
    fprintf('检测到多SNR测试配置\n');
    fprintf('SNR级别: [%s] dB\n\n', join(string(snr_levels), ', '));
else
    snr_levels = nan;  % 单一测试集
    fprintf('使用单一测试集\n\n');
end

%% 参数设置
fs = cfg.fs;

%% 输出目录
output_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

%% 循环处理每个SNR级别（或单一测试集）
for snr_idx = 1:length(snr_levels)
    if ~isnan(snr_levels(1))
        % 多SNR模式
        current_snr = snr_levels(snr_idx);
        fprintf('\n========== 处理 SNR = %d dB ==========\n', current_snr);
        
        % 获取当前SNR的测试集路径 - 使用索引访问
        snr_idx_in_list = find(cfg.testSnrLevels == current_snr);
        test_contaminated_path = cfg.testSnrPaths(snr_idx_in_list).contaminated;
        test_pure_path = cfg.testSnrPaths(snr_idx_in_list).pure;
    else
        % 单一测试集模式
        fprintf('\n========== 处理测试集 ==========\n');
        test_contaminated_path = cfg.testContaminatedPath;
        test_pure_path = cfg.testPurePath;
    end
    
    %% 加载测试数据
    fprintf('加载数据...\n');
    
    data_contaminated = load(test_contaminated_path);
    data_clean = load(test_pure_path);
    
    test_contaminated = data_contaminated.(cfg.dataKey);
    test_clean = data_clean.(cfg.dataKey);
    
    num_test = size(test_contaminated, 1);
    fprintf('测试集样本数: %d\n', num_test);
    fprintf('信号长度: %d\n\n', size(test_contaminated, 2));

%% 运行SSA去噪
fprintf('开始SSA去噪...\n');

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
    if ~isnan(snr_levels(1))
        % 多SNR模式: 保存带SNR标识的文件
        pred_save_path = fullfile(output_dir, sprintf('SSA_predictions_SNR%ddB.mat', current_snr));
    else
        % 单一测试集模式
        pred_save_path = fullfile(output_dir, 'SSA_predictions.mat');
    end
    
    save(pred_save_path, 'predictions', 'time_per_sample');
    fprintf('预测结果已保存: %s\n', pred_save_path);

end  % SNR循环结束

%% 打印汇总
fprintf('\n==============================================\n');
fprintf('           测试完成汇总\n');
fprintf('==============================================\n');
if ~isnan(snr_levels(1))
    fprintf('多SNR测试模式\n');
    fprintf('SNR级别:            [%s] dB\n', join(string(snr_levels), ', '));
else
    fprintf('单一测试集模式\n');
end
fprintf('==============================================\n');
fprintf('\n✓ 完成！请运行统一指标计算脚本来评估所有方法。\n');
