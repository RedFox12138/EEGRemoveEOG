%% VME-EFD方法统一测试脚本
% 在测试集上运行VME-EFD,保存预测结果和计算指标
%
% 使用前提:
%   1. 先运行 convert_to_mat.py 将数据转换为.mat格式
%   2. 确保 vme_efd_denoise.m 和 compute_eog_metrics.m 在路径中
%
% 输出:
%   - results/VME_EFD_predictions.mat: 预测结果
%   - results/VME_EFD_metrics.mat: 评价指标
%
% 作者: GitHub Copilot
% 日期: 2025-11-03

% clear; clc; 
close all;

%% 添加路径
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\VME_EFD');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\VME_EFD\VME');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法');

%% 加载测试数据
fprintf('==============================================\n');
fprintf('           VME-EFD 测试脚本\n');
fprintf('==============================================\n\n');

fprintf('加载数据...\n');

data_contaminated = load('D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Test_Contaminated.mat');
data_clean = load('D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据\Test_Pure.mat');

test_contaminated = data_contaminated.data;
test_clean = data_clean.data;

num_test = size(test_contaminated, 1);
fprintf('测试集样本数: %d\n', num_test);
fprintf('信号长度: %d\n\n', size(test_contaminated, 2));

%% 运行VME-EFD去噪
fprintf('开始VME-EFD去噪...\n');

fs = 200;
predictions = zeros(size(test_contaminated));

% 使用固定的最优参数
params = struct();
params.alpha = 3000;
params.omega0 = 2.8;
params.lfCut = 3;
params.nArtifacts = 2;
params.artifactGain = 0.75;
params.verbose = false;

fprintf('参数设置:\n');
fprintf('  alpha = %.0f\n', params.alpha);
fprintf('  omega0 = %.1f Hz\n', params.omega0);
fprintf('  lfCut = %.0f Hz\n', params.lfCut);
fprintf('  nArtifacts = %d\n', params.nArtifacts);
fprintf('  artifactGain = %.2f\n\n', params.artifactGain);

tic;

for i = 1:num_test
    try
        predictions(i, :) = vme_efd_denoise(test_contaminated(i, :), fs, params);
        
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

fprintf('\nVME-EFD去噪完成!\n');
fprintf('总耗时: %.2f 秒\n', total_time);
fprintf('单样本处理时间: %.3f ms\n\n', time_per_sample * 1000);

%% 保存预测结果
output_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

pred_save_path = fullfile(output_dir, 'VME_EFD_predictions.mat');
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
