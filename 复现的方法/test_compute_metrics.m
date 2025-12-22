%% 快速测试脚本 - test_compute_metrics.m
% 用于验证compute_all_metrics.m的功能
%
% 运行此脚本可以：
% 1. 检查环境配置是否正确
% 2. 验证数据文件是否存在
% 3. 测试单个方法的指标计算
% 4. 快速预览结果

clear; clc; close all;

fprintf('========================================\n');
fprintf('   compute_all_metrics.m 功能测试\n');
fprintf('========================================\n\n');

%% 1. 检查工作目录
fprintf('[1/6] 检查工作目录...\n');
current_dir = pwd;
fprintf('当前目录: %s\n', current_dir);

% 确保在正确的目录
expected_folder = '复现的方法';
if ~contains(current_dir, expected_folder)
    warning('当前不在 "%s" 目录！', expected_folder);
    fprintf('正在切换目录...\n');
    try
        cd(fullfile(fileparts(pwd), expected_folder));
        fprintf('✓ 已切换到: %s\n\n', pwd);
    catch
        error('无法切换到正确目录，请手动cd到 "复现的方法" 目录');
    end
else
    fprintf('✓ 目录正确\n\n');
end

%% 2. 检查必要文件
fprintf('[2/6] 检查必要文件...\n');
required_files = {
    'compute_all_metrics.m',
    'getDatasetConfig.m',
    'results'
};

all_exist = true;
for i = 1:length(required_files)
    if exist(required_files{i}, 'file') || exist(required_files{i}, 'dir')
        fprintf('  ✓ %s\n', required_files{i});
    else
        fprintf('  ✗ %s (缺失)\n', required_files{i});
        all_exist = false;
    end
end

if ~all_exist
    error('缺少必要文件，请检查');
end
fprintf('✓ 所有必要文件存在\n\n');

%% 3. 检查数据集配置
fprintf('[3/6] 检查数据集配置...\n');
try
    config = getDatasetConfig();
    fprintf('  数据集: %s\n', config.name);
    fprintf('  采样率: %d Hz\n', config.fs);
    fprintf('  窗口大小: %d 样本\n', config.windowSize);
    
    % 检查测试集文件
    if exist(config.test_pure_path, 'file')
        fprintf('  ✓ 测试集文件存在: %s\n', config.testPure);
        % 加载测试数据
        test_data = load(config.test_pure_path);
        test_signals = test_data.(config.data_key);
        fprintf('    数据维度: %s\n', mat2str(size(test_signals)));
    else
        warning('测试集文件不存在: %s', config.test_pure_path);
    end
    fprintf('✓ 数据集配置正确\n\n');
catch ME
    fprintf('✗ 数据集配置错误: %s\n\n', ME.message);
    rethrow(ME);
end

%% 4. 扫描results目录
fprintf('[4/6] 扫描results目录...\n');
results_dir = fullfile(pwd, 'results');
if ~exist(results_dir, 'dir')
    error('results目录不存在: %s', results_dir);
end

mat_files = dir(fullfile(results_dir, '*.mat'));
fprintf('  发现 %d 个.mat文件:\n', length(mat_files));

method_count = 0;
for i = 1:length(mat_files)
    file_name = mat_files(i).name;
    if contains(file_name, 'Test_Pure')
        fprintf('    - %s (跳过，测试集)\n', file_name);
    else
        method_count = method_count + 1;
        % 提取方法名
        underscore_pos = strfind(file_name, '_');
        if ~isempty(underscore_pos)
            method_name = file_name(1:underscore_pos(1)-1);
        else
            [~, method_name, ~] = fileparts(file_name);
        end
        
        % 检查文件大小
        file_info = dir(fullfile(results_dir, file_name));
        file_size_mb = file_info.bytes / 1024 / 1024;
        
        fprintf('    %d. [%s] - %s (%.2f MB)\n', ...
                method_count, method_name, file_name, file_size_mb);
    end
end

if method_count == 0
    error('未找到有效的预测结果文件！');
end
fprintf('✓ 找到 %d 个有效方法\n\n', method_count);

%% 5. 测试单个指标计算
fprintf('[5/6] 测试指标计算函数...\n');
fprintf('  生成测试信号...\n');

% 生成简单的测试信号
fs = 200;
t = 0:1/fs:6-1/fs;
true_sig = sin(2*pi*1*t) + 0.5*sin(2*pi*5*t);  % 1Hz + 5Hz
pred_sig = true_sig + 0.1*randn(size(true_sig));  % 加入小噪声

% 测试各个指标函数
fprintf('  测试RRMSE计算...\n');
rrmse_val = compute_rrmse(true_sig, pred_sig);
fprintf('    RRMSE = %.4f\n', rrmse_val);

fprintf('  测试CC计算...\n');
cc_val = compute_cc(true_sig, pred_sig);
fprintf('    CC = %.4f\n', cc_val);

fprintf('  测试RRMSE_PSD计算...\n');
rrmse_psd_val = compute_rrmse_psd(true_sig, pred_sig, fs);
fprintf('    RRMSE_PSD = %.4f\n', rrmse_psd_val);

fprintf('  测试MI计算...\n');
mi_val = compute_mi(true_sig, pred_sig);
fprintf('    MI = %.4f\n', mi_val);

fprintf('✓ 所有指标计算正常\n\n');

%% 6. 询问是否运行完整测试
fprintf('[6/6] 准备运行完整测试\n');
fprintf('========================================\n\n');

response = input('是否运行完整的compute_all_metrics？(y/n): ', 's');

if strcmpi(response, 'y') || strcmpi(response, 'yes')
    fprintf('\n正在运行完整测试...\n');
    fprintf('========================================\n\n');
    
    try
        % 运行主程序
        compute_all_metrics();
        
        fprintf('\n========================================\n');
        fprintf('✓ 测试完成！\n');
        fprintf('========================================\n\n');
        
        % 显示生成的文件
        fprintf('生成的文件:\n');
        fprintf('----------------------------------------\n');
        
        % CSV文件
        csv_file = fullfile(results_dir, 'all_metrics.csv');
        if exist(csv_file, 'file')
            fprintf('✓ %s\n', csv_file);
            fprintf('\n前5行内容:\n');
            T = readtable(csv_file, 'Encoding', 'UTF-8');
            disp(T(1:min(5, height(T)), :));
        end
        
        % 图像文件
        img_files = {
            'metrics_comparison.png',
            'metrics_comparison.eps',
            'time_comparison.png',
            'time_comparison.eps'
        };
        
        fprintf('\n图像文件:\n');
        for i = 1:length(img_files)
            img_path = fullfile(results_dir, img_files{i});
            if exist(img_path, 'file')
                fprintf('✓ %s\n', img_path);
            else
                fprintf('✗ %s (未生成)\n', img_path);
            end
        end
        
        % 打开生成的PNG图片
        fprintf('\n正在打开生成的图片...\n');
        png_metrics = fullfile(results_dir, 'metrics_comparison.png');
        png_time = fullfile(results_dir, 'time_comparison.png');
        
        if exist(png_metrics, 'file')
            figure('Name', '性能对比图', 'NumberTitle', 'off');
            imshow(png_metrics);
        end
        
        if exist(png_time, 'file')
            figure('Name', '时间对比图', 'NumberTitle', 'off');
            imshow(png_time);
        end
        
        fprintf('\n✓ 测试成功完成！所有功能正常！\n');
        
    catch ME
        fprintf('\n✗ 运行出错:\n');
        fprintf('错误信息: %s\n', ME.message);
        fprintf('错误位置: %s (第%d行)\n', ME.stack(1).name, ME.stack(1).line);
        rethrow(ME);
    end
else
    fprintf('\n跳过完整测试。\n');
    fprintf('如需运行完整测试，请执行: compute_all_metrics\n');
end

fprintf('\n========================================\n');
fprintf('   测试脚本执行完毕\n');
fprintf('========================================\n');

%% ===== 辅助函数（与主程序相同） =====

function rrmse = compute_rrmse(true_signal, pred_signal)
    mse = mean((true_signal(:) - pred_signal(:)).^2);
    true_power = mean(true_signal(:).^2);
    if true_power == 0
        rrmse = Inf;
    else
        rrmse = sqrt(mse / true_power);
    end
end

function cc = compute_cc(true_signal, pred_signal)
    true_flat = true_signal(:);
    pred_flat = pred_signal(:);
    corr_matrix = corrcoef(true_flat, pred_flat);
    cc = corr_matrix(1, 2);
end

function rrmse_psd = compute_rrmse_psd(true_signal, pred_signal, fs)
    if nargin < 3, fs = 200; end
    nperseg = min(256, length(true_signal));
    noverlap = floor(nperseg / 2);
    [psd_true, ~] = pwelch(true_signal, nperseg, noverlap, [], fs);
    [psd_pred, ~] = pwelch(pred_signal, nperseg, noverlap, [], fs);
    mse_psd = mean((psd_true - psd_pred).^2);
    true_psd_power = mean(psd_true.^2);
    if true_psd_power == 0
        rrmse_psd = Inf;
    else
        rrmse_psd = sqrt(mse_psd / true_psd_power);
    end
end

function mi = compute_mi(true_signal, pred_signal, bins)
    if nargin < 3, bins = 50; end
    true_flat = true_signal(:);
    pred_flat = pred_signal(:);
    [hist_2d, ~, ~] = histcounts2(true_flat, pred_flat, bins);
    pxy = hist_2d / sum(hist_2d(:));
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    px_py = px * py;
    nonzero_mask = (pxy > 0) & (px_py > 0);
    if sum(nonzero_mask(:)) == 0
        mi = 0.0;
    else
        pxy_nz = pxy(nonzero_mask);
        px_py_nz = px_py(nonzero_mask);
        mi = sum(pxy_nz .* log(pxy_nz ./ px_py_nz));
    end
end
