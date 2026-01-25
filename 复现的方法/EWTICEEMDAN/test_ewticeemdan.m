%% EWT-ICEEMDAN方法统一测试脚本
% 在测试集上运行EWT-ICEEMDAN,保存预测结果和计算指标
%
% 使用前提:
%   1. 先运行 convert_to_mat.py 将数据转换为.mat格式
%   2. 确保相关函数在路径中
%
% 输出:
%   - results/EWTICEEMDAN_predictions.mat: 预测结果
%   - results/EWTICEEMDAN_metrics.mat: 评价指标
%
% 作者: GitHub Copilot
% 日期: 2025-11-03

% clear; clc;

% 加载数据集配置
addpath('..');
cfg = getDatasetConfig(); 
close all;

%% 添加路径
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\EWTICEEMDAN');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法\EWTICEEMDAN\ICEEMDAN_wp');
addpath('D:\Pycharm_Projects\EOG Remove\复现的方法');

%% 确定测试模式
fprintf('==============================================\n');
fprintf('       EWT-ICEEMDAN 测试脚本\n');
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

%% 确定测试模式
fprintf('==============================================\n');
fprintf('       EWT-ICEEMDAN 测试脚本\n');
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
cutoff_freq = 4;  % EWT截止频率
sample_entropy_threshold = 0.4;  % 样本熵阈值
Nstd = 0.2;  % 噪声标准差
NR = 100;    % 迭代次数
MaxIter = 10; % 最大迭代次数

fprintf('参数设置:\n');
fprintf('  cutoff_freq = %.0f Hz\n', cutoff_freq);
fprintf('  sample_entropy_threshold = %.2f\n', sample_entropy_threshold);
fprintf('  ICEEMDAN: Nstd=%.1f, NR=%d, MaxIter=%d\n\n', Nstd, NR, MaxIter);

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
    test_clean = data_clean.(cfg.pureKey);
    
    num_test = size(test_contaminated, 1);
    fprintf('测试集样本数: %d\n', num_test);
    fprintf('信号长度: %d\n\n', size(test_contaminated, 2));

%% 运行EWT-ICEEMDAN去噪
fprintf('开始EWT-ICEEMDAN去噪...\n');
fprintf('⚠️  警告: ICEEMDAN分解较慢,可能需要数小时!\n\n');

fs = cfg.fs;
predictions = zeros(size(test_contaminated));

% 参数设置
cutoff_freq = 4;  % EWT截止频率
sample_entropy_threshold = 0.4;  % 样本熵阈值
Nstd = 0.2;  % 噪声标准差
NR = 100;    % 迭代次数
MaxIter = 10; % 最大迭代次数

fprintf('参数设置:\n');
fprintf('  cutoff_freq = %.0f Hz\n', cutoff_freq);
fprintf('  sample_entropy_threshold = %.2f\n', sample_entropy_threshold);
fprintf('  ICEEMDAN: Nstd=%.1f, NR=%d, MaxIter=%d\n\n', Nstd, NR, MaxIter);

    %% 运行EWT-ICEEMDAN去噪
    fprintf('开始EWT-ICEEMDAN去噪...\n');
    fprintf('⚠️  警告: ICEEMDAN分解较慢,可能需要数小时!\n\n');
    
    predictions = zeros(size(test_contaminated));
    
    tic;

for i = 1:num_test
    close all
    try
        signal = test_contaminated(i, :);
        
        % 1. EWT分离 (低频/高频)
        [ewt_low, ewt_high] = ewt_custom_boundary(signal, fs, cutoff_freq);
        
        % 确保是列向量
        ewt_low = ewt_low(:);
        ewt_high = ewt_high(:);
        
        % 2. 对低频分量进行ICEEMDAN分解
        % pICEEMDAN(data, FsOrT, Nstd, NE, MaxIter)
        % FsOrT可以是采样频率(标量)或时间向量
        modes_low = pICEEMDAN(ewt_low', fs, Nstd, NR, MaxIter);  % 转为行向量输入
        
        % 3. 样本熵筛选 (去除样本熵<阈值的IMF)
        n_imf = size(modes_low, 1);
        selected_modes = [];
        
        for j = 1:n_imf
            try
                % 计算样本熵 (m=2, r=0.2*std)
                % SampEn函数使用name-value pairs参数
                Samp = SampEn(modes_low(j,:), 'm', 2, 'r', 0.2*std(modes_low(j,:)));
                se = Samp(end);  % 取最后一个值（m=2的样本熵）
                
                % 保留样本熵>=阈值的分量
                if se >= sample_entropy_threshold
                    selected_modes = [selected_modes; modes_low(j,:)];
                end
            catch
                % 如果计算失败,保留该分量
                selected_modes = [selected_modes; modes_low(j,:)];
            end
        end
        
        % 4. 重构信号 (确保输出为行向量)
        if ~isempty(selected_modes)
            reconstructed_low = sum(selected_modes, 1);  % 行向量
        else
            reconstructed_low = zeros(1, length(ewt_low));  % 行向量
        end
        
        % 确保所有信号都是行向量后再相加
        predictions(i, :) = reconstructed_low + ewt_high';
        
        if mod(i, 5) == 0
            fprintf('  已处理 %d/%d 样本 (%.1f%%) - 预计剩余: %.1f 分钟\n', ...
                i, num_test, i/num_test*100, toc/i*(num_test-i)/60);
        end
        
    catch ME
        warning('样本 %d 处理失败: %s', i, ME.message);
        predictions(i, :) = test_contaminated(i, :);
    end
end

total_time = toc;
time_per_sample = total_time / num_test;

fprintf('\nEWT-ICEEMDAN去噪完成!\n');
fprintf('总耗时: %.2f 秒 (%.1f 分钟)\n', total_time, total_time/60);
fprintf('单样本处理时间: %.3f ms\n\n', time_per_sample * 1000);

%% 保存预测结果
    if ~isnan(snr_levels(1))
        % 多SNR模式: 保存带SNR标识的文件
        pred_save_path = fullfile(output_dir, sprintf('EWTICEEMDAN_predictions_SNR%ddB.mat', current_snr));
    else
        % 单一测试集模式
        pred_save_path = fullfile(output_dir, 'EWTICEEMDAN_predictions.mat');
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
