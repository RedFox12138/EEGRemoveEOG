% 计算基于阈值方法的评估指标
% 读取predictions，计算metrics并保存

clear; clc;

fprintf('========================================\n');
fprintf('基于阈值方法 - 评估指标计算\n');
fprintf('========================================\n');

%% 1. 加载预测结果
results_dir = fullfile(pwd,'复现的方法', 'results');
pred_file = fullfile(results_dir, 'Threshold_predictions.mat');

if ~isfile(pred_file)
    error('未找到预测结果文件: %s\n请先运行 test_threshold.py', pred_file);
end

fprintf('\n加载预测结果...\n');
pred_data = load(pred_file);
predictions = pred_data.predictions;
fprintf('  预测结果形状: [%s]\n', num2str(size(predictions)));

%% 2. 加载测试数据（真值）
data_dir = 'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据';

% 加载测试集数据
pure_mat = fullfile(data_dir, 'Test_Pure.mat');
cont_mat = fullfile(data_dir, 'Test_Contaminated.mat');

if isfile(pure_mat) && isfile(cont_mat)
    fprintf('加载MATLAB格式测试数据...\n');
    pure_data_full = load(pure_mat);
    cont_data_full = load(cont_mat);
    
    % 提取数据（处理可能的结构体）
    if isstruct(pure_data_full)
        fn = fieldnames(pure_data_full);
        test_pure = pure_data_full.(fn{1});
    end
    if isstruct(cont_data_full)
        fn = fieldnames(cont_data_full);
        test_contaminated = cont_data_full.(fn{1});
    end
else
    error('无法找到测试集文件。请确保存在 Test_Pure.mat 和 Test_Contaminated.mat');
end

fprintf('  测试集形状: [%s]\n', num2str(size(test_pure)));

%% 3. 计算评估指标
fprintf('\n计算评估指标...\n');

% 调用compute_eog_metrics函数（在上级目录）
addpath(fullfile(pwd, '..'));

% 采样率
fs = 200;

try
    % 参数顺序: (真实纯净信号, 预测去噪信号, 采样率)
    metrics = compute_eog_metrics(test_pure, predictions, fs);
    
    fprintf('\n评估结果:\n');
    fprintf('  RRMSE: %.4f ± %.4f\n', metrics.RRMSE_mean, metrics.RRMSE_std);
    fprintf('  CC:    %.4f ± %.4f\n', metrics.CC_mean, metrics.CC_std);
    fprintf('  RRMSE_PSD: %.4f ± %.4f\n', metrics.RRMSE_PSD_mean, metrics.RRMSE_PSD_std);
    fprintf('  MI:    %.4f ± %.4f\n', metrics.MI_mean, metrics.MI_std);
    
    %% 4. 保存结果
    metrics_file = fullfile(results_dir, 'Threshold_metrics.mat');
    save(metrics_file, 'metrics');
    fprintf('\n指标已保存到:\n  %s\n', metrics_file);
    
    fprintf('\n========================================\n');
    fprintf('计算完成！可以运行 evaluate_all_methods.py 进行对比\n');
    fprintf('========================================\n');
    
catch ME
    fprintf('错误: %s\n', ME.message);
    fprintf('请检查 compute_eog_metrics.m 是否存在\n');
    rethrow(ME);
end
