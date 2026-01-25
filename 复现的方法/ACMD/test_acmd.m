%% ACMD方法统一测试脚本 - 支持多SNR测试集
% 在测试集上运行ACMD,保存预测结果和计算指标
%
% 使用前提:
%   1. 先运行 convert_to_mat.py 将数据转换为.mat格式
%   2. 确保 oa_remove_acmd.m 和 compute_eog_metrics.m 在路径中
%
% 输出:
%   - results/ACMD_predictions_SNR{snr}dB.mat: 各SNR级别的预测结果
%
% 作者: GitHub Copilot
% 日期: 2025-12-24

clear; clc; close all;

% 加载数据集配置
addpath('..');
cfg = getDatasetConfig();

%% 添加路径
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\ACMD');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法');  % 通用metrics函数

%% 主测试流程
fprintf('==============================================\n');
fprintf('     ACMD 测试脚本 - 多SNR测试集\n');
fprintf('==============================================\n\n');

% 检查是否有多SNR测试集
if cfg.hasMultiSnrTest
    snr_levels = cfg.testSnrLevels;
    fprintf('检测到多SNR测试集，SNR级别: ');
    fprintf('%d ', snr_levels);
    fprintf('dB\n\n');
else
    snr_levels = [0];  % 单一测试集
    fprintf('使用单一测试集\n\n');
end

%% 估计阈值ξ (使用训练集，只需计算一次)
fprintf('估计阈值ξ...\n');

% 加载训练集用于阈值估计
data_train_contaminated = load(cfg.trainContaminatedPath);
data_train_clean = load(cfg.trainPurePath);

train_contaminated = data_train_contaminated.(cfg.dataKey);
train_clean = data_train_clean.(cfg.pureKey);  % 使用pureKey加载纯净数据

% 采样一部分训练数据计算阈值(避免过慢)
n_samples_for_threshold = min(50, size(train_contaminated, 1));
psi_clean_list = zeros(n_samples_for_threshold, 1);
psi_contaminated_list = zeros(n_samples_for_threshold, 1);

fs = cfg.fs;  % 采样率

fprintf('  从训练集采样%d个样本估计阈值...\n', n_samples_for_threshold);
for i = 1:n_samples_for_threshold
    % 对干净信号提取第一模态并计算峰计数
    [~, info_clean] = oa_remove_acmd(train_clean(i, :), fs, struct('returnAll', true));
    psi_clean_list(i) = info_clean.psi;
    
    % 对污染信号提取第一模态并计算峰计数
    [~, info_cont] = oa_remove_acmd(train_contaminated(i, :), fs, struct('returnAll', true));
    psi_contaminated_list(i) = info_cont.psi;
    
    if mod(i, 10) == 0
        fprintf('    已处理 %d/%d\n', i, n_samples_for_threshold);
    end
end

% 根据论文,阈值ξ = (mean(psi_clean) + mean(psi_contaminated)) / 2
mean_psi_clean = mean(psi_clean_list);
mean_psi_contaminated = mean(psi_contaminated_list);
threshold_xi = (mean_psi_clean + mean_psi_contaminated) / 2;

fprintf('  干净信号平均峰计数: %.2f\n', mean_psi_clean);
fprintf('  污染信号平均峰计数: %.2f\n', mean_psi_contaminated);
fprintf('  估计阈值ξ = %.2f\n\n', threshold_xi);

%% 配置ACMD参数(严格按照论文)
opts = struct();
opts.threshold = threshold_xi;        % 使用估计的阈值
opts.autoTune = true;                 % 开启自动调优(论文Fig.3)
opts.tuneBW = 0.5:0.5:3;             % 扫描带宽范围
opts.tuneFmax = 8:2:14;              % 扫描最大频率范围
opts.returnAll = true;

%% 对每个SNR级别进行测试
output_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

for snr_idx = 1:length(snr_levels)
    snr = snr_levels(snr_idx);
    
    if length(snr_levels) > 1
        fprintf('\n==============================================\n');
        fprintf('          测试 SNR = %d dB\n', snr);
        fprintf('==============================================\n\n');
    end
    
    % 加载该SNR级别的测试数据
    fprintf('加载测试数据...\n');
    if cfg.hasMultiSnrTest
        % 多SNR测试集 - 查找对应SNR的索引
        snr_idx_in_list = find(cfg.testSnrLevels == snr);
        test_contaminated_path = cfg.testSnrPaths(snr_idx_in_list).contaminated;
        test_pure_path = cfg.testSnrPaths(snr_idx_in_list).pure;
    else
        % 单一测试集
        test_contaminated_path = cfg.testContaminatedPath;
        test_pure_path = cfg.testPurePath;
    end
    
    data_contaminated = load(test_contaminated_path);
    data_clean = load(test_pure_path);
    
    test_contaminated = data_contaminated.(cfg.dataKey);
    test_clean = data_clean.(cfg.pureKey);  % 使用pureKey加载纯净数据
    
    num_test = size(test_contaminated, 1);
    fprintf('SNR=%ddB 测试集样本数: %d\n', snr, num_test);
    fprintf('信号长度: %d\n\n', size(test_contaminated, 2));
    
    %% 运行ACMD去噪
    fprintf('开始ACMD去噪(使用阈值ξ=%.2f)...\n', threshold_xi);
    
    predictions = zeros(size(test_contaminated));
    
    % 记录时间
    tic;
    
    for i = 1:num_test
        try
            % 调用ACMD去噪函数,传入opts
            predictions(i, :) = oa_remove_acmd(test_contaminated(i, :), fs, opts);
            
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
    if length(snr_levels) > 1
        pred_save_path = fullfile(output_dir, sprintf('ACMD_predictions_SNR%ddB.mat', snr));
    else
        pred_save_path = fullfile(output_dir, 'ACMD_predictions.mat');
    end
    
    save(pred_save_path, 'predictions', 'time_per_sample');
    fprintf('✓ 结果已保存: %s\n', pred_save_path);
end

%% 打印汇总
fprintf('\n==============================================\n');
fprintf('        所有SNR级别测试完成！\n');
fprintf('==============================================\n');
fprintf('\n✓ 完成！请运行统一指标计算脚本来评估所有方法。\n');

