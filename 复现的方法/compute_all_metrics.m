function compute_all_metrics()
% COMPUTE_ALL_METRICS 统一指标计算脚本 - MATLAB实现
%
% 该脚本会：
% 1. 加载测试集的真实纯净信号（Test_Pure.mat）
% 2. 自动扫描 results 文件夹下所有 .mat 文件
% 3. 提取文件名第一个 "_" 之前的内容作为方法名
% 4. 计算 RRMSE, CC, RRMSE_PSD, MI 指标
% 5. 保存结果到CSV：使用 "Mean ± Std" 格式，保留3位小数
% 6. 生成性能对比图（4个子图：RRMSE, CC, RRMSE_PSD, MI）
% 7. 生成时间对比图
%
% 作者: GitHub Copilot
% 日期: 2025-12-15
% 基于原Python脚本改写

    fprintf('================================================================================\n');
    fprintf('%s\n', center_text('自动化指标对比脚本', 80));
    fprintf('================================================================================\n');
    
    % ===================== 配置路径 =====================
    % 获取数据集配置（使用半模拟数据集）
    config = getDatasetConfig('semi_simulated');
    
    % 结果目录
    results_dir = fullfile(pwd, 'results');
    % ==================================================
    
    if ~exist(results_dir, 'dir')
        error('错误: 结果目录不存在 -> %s', results_dir);
    end
    
    % 1. 检测是否为多SNR配置
    if isfield(config, 'hasMultiSnrTest') && config.hasMultiSnrTest
        fprintf('\n检测到多SNR测试集配置\n');
        fprintf('SNR级别: %s\n', mat2str(config.testSnrLevels));

        % 初始化收集所有SNR结果的结构
        results_all_snr = struct();
        snr_levels = config.testSnrLevels;

        % 为每个SNR级别计算指标并收集结果（process_snr_level 返回该SNR的 results_dict）
        for snr_idx = 1:length(config.testSnrLevels)
            current_snr = config.testSnrLevels(snr_idx);
            fprintf('\n================================================================================\n');
            fprintf('%s\n', center_text(sprintf('处理 SNR = %d dB 的测试集', current_snr), 80));
            fprintf('================================================================================\n');

            % 获取该SNR级别的测试集路径
            test_pure_path = config.testSnrPaths(snr_idx).pure;

            % 加载测试集
            fprintf('\n正在加载测试集: %s\n', test_pure_path);
            if ~exist(test_pure_path, 'file')
                warning('找不到测试集文件: %s, 跳过', test_pure_path);
                continue;
            end

            data_struct = load(test_pure_path);
            true_signals = data_struct.(config.dataKey);
            fprintf('✓ 已加载基准测试集: %s\n', mat2str(size(true_signals)));

            % 处理该SNR级别的所有方法，并获取结果
            results_snr = process_snr_level(config, current_snr, true_signals, results_dir);

            % 合并到总结果中
            methods_in_snr = fieldnames(results_snr);
            for m_idx = 1:length(methods_in_snr)
                method = methods_in_snr{m_idx};
                if ~isfield(results_all_snr, method)
                    results_all_snr.(method) = struct();
                    results_all_snr.(method).display_name = results_snr.(method).display_name;
                    results_all_snr.(method).snr_metrics = struct();
                end
                snr_key = snr_field(current_snr);
                results_all_snr.(method).snr_metrics.(snr_key) = results_snr.(method);
            end
        end

        fprintf('\n================================================================================\n');
        fprintf('%s\n', center_text('全部SNR级别处理完成！', 80));
        fprintf('================================================================================\n');

        % 生成跨SNR的两张汇总图：指标随SNR变化折线图 & 平均时间柱状图
        plot_snr_comparison(results_all_snr, snr_levels, results_dir);
        plot_avg_time(results_all_snr, results_dir);

        return;
    end
    
    % 单一测试集配置（向后兼容）
    fprintf('\n检测到单一测试集配置\n');
    test_pure_path = config.testPurePath;
    
    % 加载测试集
    fprintf('\n正在加载测试集...\n');
    if ~exist(test_pure_path, 'file')
        error('找不到测试集文件: %s', test_pure_path);
    end
    
    data_struct = load(test_pure_path);
    true_signals = data_struct.(config.dataKey);
    fprintf('✓ 已加载基准测试集 (Test_Pure.mat): %s\n', mat2str(size(true_signals)));
    
    % 2. 扫描文件
    fprintf('\n[扫描目录] %s\n', results_dir);
    mat_files = dir(fullfile(results_dir, '*.mat'));
    
    if isempty(mat_files)
        error('该目录下没有找到 .mat 文件！');
    end
    
    fprintf('发现 %d 个数据文件，开始处理...\n', length(mat_files));
    
    % 存储结果
    results_dict = struct();
    method_names = {};
    
    % 3. 遍历计算
    for i = 1:length(mat_files)
        file_name = mat_files(i).name;
        
        % 跳过测试集本身
        if contains(file_name, 'Test_Pure')
            continue;
        end
        
        % 提取方法名：取第一个 "_" 之前
        underscore_pos = strfind(file_name, '_');
        if ~isempty(underscore_pos)
            method_name_display = file_name(1:underscore_pos(1)-1);
        else
            [~, method_name_display, ~] = fileparts(file_name);
        end
        
        % 创建有效的字段名（移除特殊字符）
        method_name_field = strrep(method_name_display, '-', '_');
        method_name_field = strrep(method_name_field, '%', 'percent');
        method_name_field = strrep(method_name_field, ' ', '_');
        method_name_field = matlab.lang.makeValidName(method_name_field);
        
        file_path = fullfile(results_dir, file_name);
        fprintf('\n>>> 处理方法: [%s] (文件: %s)\n', method_name_display, file_name);
        
        % 加载预测文件
        [predictions, time_per_sample] = load_prediction_file(file_path, file_name);
        if isempty(predictions)
            continue;
        end
        
        % 显示预测数据信息
        fprintf('    数据维度: %s', mat2str(size(predictions)));
        if time_per_sample > 0
            fprintf(' | 单样本耗时: %.3f ms\n', time_per_sample * 1000);
        else
            fprintf(' | 耗时信息: 未提供\n');
        end
        
        % 检查维度
        if ~isequal(size(predictions), size(true_signals))
            fprintf('  ⚠ 警告: 维度不匹配! 预测:%s vs 真实:%s, 跳过\n', ...
                    mat2str(size(predictions)), mat2str(size(true_signals)));
            continue;
        end
        
        % 计算指标
        metrics = compute_metrics_for_method(predictions, true_signals, config.fs);
        metrics.time_per_sample = time_per_sample;
        metrics.display_name = method_name_display;  % 保存显示名称
        
        % 存储结果（使用有效字段名）
        results_dict.(method_name_field) = metrics;
        method_names{end+1} = method_name_field;
        
        fprintf('  ✓ %s 计算完毕 (RRMSE=%.3f, CC=%.3f, Time=%.1fms)\n', ...
                method_name_display, metrics.RRMSE_mean, metrics.CC_mean, time_per_sample * 1000);
    end
    
    if isempty(method_names)
        error('没有成功计算任何方法的指标。');
    end
    
    % 4. 保存结果
    output_csv = fullfile(results_dir, 'all_metrics.csv');
    [results_table, sorted_methods] = save_results(results_dict, method_names, output_csv);
    
    % 5. 打印最终结果
    fprintf('\n================================================================================\n');
    fprintf('%s\n', center_text('最终结果排行', 80));
    fprintf('================================================================================\n');
    disp(results_table);
    
    % 6. 生成图表
    plot_comparison(results_dict, sorted_methods, results_dir);
    plot_time_comparison(results_dict, sorted_methods, results_dir);
    
    fprintf('\n================================================================================\n');
    fprintf('%s\n', center_text('全部完成！', 80));
    fprintf('================================================================================\n');
end


%% ===================== 指标计算函数 =====================

function rrmse = compute_rrmse(true_signal, pred_signal)
% 计算相对均方根误差 (RRMSE)
    mse = mean((true_signal(:) - pred_signal(:)).^2);
    true_power = mean(true_signal(:).^2);
    if true_power == 0
        rrmse = Inf;
    else
        rrmse = sqrt(mse / true_power);
    end
end


function cc = compute_cc(true_signal, pred_signal)
% 计算相关系数 (CC)
    true_flat = true_signal(:);
    pred_flat = pred_signal(:);
    corr_matrix = corrcoef(true_flat, pred_flat);
    cc = corr_matrix(1, 2);
end


function rrmse_psd = compute_rrmse_psd(true_signal, pred_signal, fs)
% 计算功率谱相对误差 (RRMSE_PSD)
    if nargin < 3
        fs = 200;  % 默认采样率
    end
    
    nperseg = min(256, length(true_signal));
    noverlap = floor(nperseg / 2);
    
    % 使用pwelch计算功率谱密度
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
% 计算互信息 (MI)
    if nargin < 3
        bins = 50;
    end
    
    true_flat = true_signal(:);
    pred_flat = pred_signal(:);
    
    % 使用histcounts2计算2D直方图
    [hist_2d, ~, ~] = histcounts2(true_flat, pred_flat, bins);
    
    pxy = hist_2d / sum(hist_2d(:));
    px = sum(pxy, 2);
    py = sum(pxy, 1);
    px_py = px * py;
    
    % 避免log(0)
    nonzero_mask = (pxy > 0) & (px_py > 0);
    
    if sum(nonzero_mask(:)) == 0
        mi = 0.0;
    else
        pxy_nz = pxy(nonzero_mask);
        px_py_nz = px_py(nonzero_mask);
        mi = sum(pxy_nz .* log(pxy_nz ./ px_py_nz));
    end
end


function metrics = compute_metrics_for_method(predictions, true_signals, fs)
% 为单个方法计算所有指标
    if nargin < 3
        fs = 200;
    end
    
    n_samples = size(predictions, 1);
    
    rrmse_list = zeros(n_samples, 1);
    cc_list = zeros(n_samples, 1);
    rrmse_psd_list = zeros(n_samples, 1);
    mi_list = zeros(n_samples, 1);
    
    fprintf('    正在计算 %d 个样本...', n_samples);
    
    for i = 1:n_samples
        true_sig = true_signals(i, :);
        pred_sig = predictions(i, :);
        
        rrmse_list(i) = compute_rrmse(true_sig, pred_sig);
        cc_list(i) = compute_cc(true_sig, pred_sig);
        rrmse_psd_list(i) = compute_rrmse_psd(true_sig, pred_sig, fs);
        mi_list(i) = compute_mi(true_sig, pred_sig);
    end
    
    fprintf(' 完成\n');
    
    % 计算均值和标准差
    metrics.RRMSE_mean = mean(rrmse_list);
    metrics.RRMSE_std = std(rrmse_list);
    metrics.CC_mean = mean(cc_list);
    metrics.CC_std = std(cc_list);
    metrics.RRMSE_PSD_mean = mean(rrmse_psd_list);
    metrics.RRMSE_PSD_std = std(rrmse_psd_list);
    metrics.MI_mean = mean(mi_list);
    metrics.MI_std = std(mi_list);
end


%% ===================== 文件加载函数 =====================

function [predictions, time_per_sample] = load_prediction_file(file_path, file_name)
% 加载指定路径的预测文件
    predictions = [];
    time_per_sample = 0.0;
    
    try
        data = load(file_path);
    catch ME
        fprintf('  ❌ 文件损坏或无法读取: %s\n', file_name);
        return;
    end
    
    % 查找数据变量
    if isfield(data, 'predictions')
        predictions = data.predictions;
    elseif isfield(data, 'data')
        predictions = data.data;
    elseif isfield(data, 'clean_data')
        predictions = data.clean_data;
    else
        % 尝试找到最大的非meta变量
        fields = fieldnames(data);
        max_size = 0;
        for j = 1:length(fields)
            field_name = fields{j};
            if ~startsWith(field_name, '__') && isnumeric(data.(field_name))
                if numel(data.(field_name)) > max_size
                    predictions = data.(field_name);
                    max_size = numel(data.(field_name));
                end
            end
        end
        if ~isempty(predictions)
            fprintf('  ⚠ 警告: 未找到标准变量名，已自动选择最大变量作为数据。\n');
        end
    end
    
    if isempty(predictions)
        fprintf('  ❌ 错误: 在 %s 中未找到有效数据变量\n', file_name);
        return;
    end
    
    % 提取时间信息
    if isfield(data, 'time_per_sample')
        time_arr = data.time_per_sample;
        if ~isempty(time_arr)
            time_per_sample = double(time_arr(1));
        end
    end
end


%% ===================== 结果保存函数 =====================

function [results_table, sorted_methods] = save_results(results_dict, method_names, output_path)
% 格式化并保存结果
    
    % 创建数据结构
    n_methods = length(method_names);
    Method = cell(n_methods, 1);
    RRMSE_mean = zeros(n_methods, 1);
    RRMSE_std = zeros(n_methods, 1);
    CC_mean = zeros(n_methods, 1);
    CC_std = zeros(n_methods, 1);
    RRMSE_PSD_mean = zeros(n_methods, 1);
    RRMSE_PSD_std = zeros(n_methods, 1);
    MI_mean = zeros(n_methods, 1);
    MI_std = zeros(n_methods, 1);
    Time_ms = zeros(n_methods, 1);
    
    for i = 1:n_methods
        method = method_names{i};
        metrics = results_dict.(method);
        
        % 使用显示名称（如果有），否则使用字段名
        if isfield(metrics, 'display_name')
            Method{i} = metrics.display_name;
        else
            Method{i} = method;
        end
        RRMSE_mean(i) = metrics.RRMSE_mean;
        RRMSE_std(i) = metrics.RRMSE_std;
        CC_mean(i) = metrics.CC_mean;
        CC_std(i) = metrics.CC_std;
        RRMSE_PSD_mean(i) = metrics.RRMSE_PSD_mean;
        RRMSE_PSD_std(i) = metrics.RRMSE_PSD_std;
        MI_mean(i) = metrics.MI_mean;
        MI_std(i) = metrics.MI_std;
        Time_ms(i) = metrics.time_per_sample * 1000;
    end
    
    % 同时保存字段名用于访问results_dict
    Method_field = cell(n_methods, 1);
    for i = 1:n_methods
        Method_field{i} = method_names{i};
    end
    
    % 创建表格
    T = table(Method, Method_field, RRMSE_mean, RRMSE_std, CC_mean, CC_std, ...
              RRMSE_PSD_mean, RRMSE_PSD_std, MI_mean, MI_std, Time_ms);
    
    % 按RRMSE排序
    T = sortrows(T, 'RRMSE_mean');
    sorted_methods = T.Method_field;  % 返回字段名列表
    
    % 创建格式化表格用于显示和保存
    n = height(T);
    Method_disp = cell(n, 1);
    RRMSE = cell(n, 1);
    CC = cell(n, 1);
    RRMSE_PSD = cell(n, 1);
    MI = cell(n, 1);
    Time = cell(n, 1);
    
    for i = 1:n
        Method_disp{i} = char(T.Method{i});
        RRMSE{i} = sprintf('%.3f ± %.3f', T.RRMSE_mean(i), T.RRMSE_std(i));
        CC{i} = sprintf('%.3f ± %.3f', T.CC_mean(i), T.CC_std(i));
        RRMSE_PSD{i} = sprintf('%.3f ± %.3f', T.RRMSE_PSD_mean(i), T.RRMSE_PSD_std(i));
        MI{i} = sprintf('%.3f ± %.3f', T.MI_mean(i), T.MI_std(i));
        Time{i} = sprintf('%.3f', T.Time_ms(i));
    end
    
    results_table = table(Method_disp, RRMSE, CC, RRMSE_PSD, MI, Time, ...
                         'VariableNames', {'Method', 'RRMSE', 'CC', 'RRMSE_PSD', 'MI', 'Time_ms'});
    
    % 保存CSV
    writetable(results_table, output_path, 'Encoding', 'UTF-8');
    fprintf('\n✓ 结果已保存到: %s\n', output_path);
end


%% ===================== 绘图函数 =====================
function plot_snr_comparison(results_all_snr, snr_levels, output_dir)
% 生成跨SNR的折线对比图 - 4个子图，每个指标一个子图，随着SNR从大到小
    snr_sorted = sort(snr_levels, 'descend');
    n_snr = length(snr_sorted);
    methods = fieldnames(results_all_snr);
    n_methods = length(methods);

    % 使用不同颜色
    colors = lines(n_methods);

    % 创建图形
    figure('Position', [100, 100, 1600, 1200], 'Color', 'w');

    metrics_names = {'RRMSE', 'CC', 'RRMSE_PSD', 'MI'};
    subplot_labels = {'(a)', '(b)', '(c)', '(d)'};

    for m = 1:4
        subplot(2, 2, m);
        hold on;
        metric_name = metrics_names{m};

        for i = 1:n_methods
            method = methods{i};
            if isfield(results_all_snr.(method), 'display_name')
                display_name = results_all_snr.(method).display_name;
            else
                display_name = method;
            end
            y_vals = nan(n_snr, 1);

            for j = 1:n_snr
                snr = snr_sorted(j);
                snr_key = snr_field(snr);
                if isfield(results_all_snr.(method).snr_metrics, snr_key)
                    metrics_entry = results_all_snr.(method).snr_metrics.(snr_key);
                    field_name = sprintf('%s_mean', metric_name);
                    if isfield(metrics_entry, field_name)
                        y_vals(j) = metrics_entry.(field_name);
                    end
                end
            end

            plot(snr_sorted, y_vals, 'o-', 'Color', colors(i,:), 'LineWidth', 2, 'MarkerSize', 8, 'DisplayName', display_name);
        end

        xlabel('SNR (dB)', 'FontSize', 14, 'FontWeight', 'bold');
        ylabel(metric_name, 'FontSize', 14, 'FontWeight', 'bold');
        title(subplot_labels{m}, 'FontSize', 16, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 10);
        grid on;
        set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
        box on;
        hold off;
    end

    % 添加总标题
    sgtitle('各方法性能指标随SNR变化', 'FontSize', 20, 'FontWeight', 'bold');
    fprintf('✓ 跨SNR性能对比图已显示\n');
end


function plot_avg_time(results_all_snr, output_dir)
% 生成平均运行时间的柱状图（对所有SNR取平均）
    methods = fieldnames(results_all_snr);
    n_methods = length(methods);
    avg_times = zeros(n_methods, 1);
    display_names = cell(n_methods, 1);

    for i = 1:n_methods
        method = methods{i};
        if isfield(results_all_snr.(method), 'display_name')
            display_names{i} = results_all_snr.(method).display_name;
        else
            display_names{i} = method;
        end
        snr_keys = fieldnames(results_all_snr.(method).snr_metrics);
        times = [];
        for k = 1:length(snr_keys)
            entry = results_all_snr.(method).snr_metrics.(snr_keys{k});
            if isfield(entry, 'time_per_sample')
                times(end+1) = entry.time_per_sample;
            end
        end
        if isempty(times)
            avg_times(i) = NaN;
        else
            avg_times(i) = mean(times) * 1000; % ms
        end
    end

    % 创建图形（水平柱状图，按时间从小到大排序）
    [avg_times_sorted, sort_idx] = sort(avg_times, 'ascend');
    display_names_sorted = display_names(sort_idx);

    figure('Position', [100, 100, 1200, 600], 'Color', 'w');
    h = barh(1:n_methods, avg_times_sorted, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.2);
    gray_values = linspace(0.3, 0.8, n_methods)';
    colors = repmat(gray_values, 1, 3);
    h.CData = colors;

    % 设置Y轴为方法名
    set(gca, 'YTick', 1:n_methods, 'YTickLabel', display_names_sorted, 'FontSize', 13);
    xlabel('平均运行时间 (ms)', 'FontSize', 16, 'FontWeight', 'bold');
    title('各方法平均运行时间（跨SNR平均）', 'FontSize', 18, 'FontWeight', 'bold');

    % 处理数值范围差异：当最大/最小(非零) > 50 时使用对数刻度
    nonzero = avg_times_sorted(avg_times_sorted > 0);
    if ~isempty(nonzero) && (max(nonzero) / min(nonzero) > 50)
        set(gca, 'XScale', 'log');
        % 对数刻度时，添加小的 offset 显示为标签
    end

    % 添加数值标签（在条形右侧）
    for i = 1:n_methods
        val = avg_times_sorted(i);
        if ~isnan(val)
            if strcmp(get(gca, 'XScale'), 'log')
                text(max(nonzero)*0.01, i, sprintf('%.2f', val), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 11, 'FontWeight', 'bold');
            else
                text(val, i, sprintf('  %.2f', val), 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'FontSize', 11, 'FontWeight', 'bold');
            end
        end
    end

    grid on; box on;
    fprintf('✓ 平均运行时间对比图已显示\n');
end

function plot_comparison(results_dict, method_names, output_dir, varargin)
% 生成对比图表 - 4个子图，带(a)(b)(c)(d)标识
% varargin{1}: 可选的文件名后缀 (例如 '_SNR-6dB')
    
    % 获取可选的后缀
    if nargin >= 4
        suffix = varargin{1};
    else
        suffix = '';
    end
    
    n_methods = length(method_names);
    
    % 提取数据和显示名称
    RRMSE_mean = zeros(n_methods, 1);
    RRMSE_std = zeros(n_methods, 1);
    CC_mean = zeros(n_methods, 1);
    CC_std = zeros(n_methods, 1);
    RRMSE_PSD_mean = zeros(n_methods, 1);
    RRMSE_PSD_std = zeros(n_methods, 1);
    MI_mean = zeros(n_methods, 1);
    MI_std = zeros(n_methods, 1);
    display_names = cell(n_methods, 1);
    
    for i = 1:n_methods
        method_field = method_names{i};  % 字段名
        metrics = results_dict.(method_field);
        
        % 获取显示名称
        if isfield(metrics, 'display_name')
            display_names{i} = metrics.display_name;
        else
            display_names{i} = method_field;
        end
        RRMSE_mean(i) = metrics.RRMSE_mean;
        RRMSE_std(i) = metrics.RRMSE_std;
        CC_mean(i) = metrics.CC_mean;
        CC_std(i) = metrics.CC_std;
        RRMSE_PSD_mean(i) = metrics.RRMSE_PSD_mean;
        RRMSE_PSD_std(i) = metrics.RRMSE_PSD_std;
        MI_mean(i) = metrics.MI_mean;
        MI_std(i) = metrics.MI_std;
    end
    
    % 创建图形
    fig = figure('Position', [100, 100, 1600, 1200], 'Color', 'w');
    
    % 灰度颜色映射 - 从浅灰到深灰
    gray_values = linspace(0.3, 0.8, n_methods)';
    colors = repmat(gray_values, 1, 3);
    
    % 子图1: RRMSE
    subplot(2, 2, 1);
    hold on;
    b1 = bar(1:n_methods, RRMSE_mean, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
    b1.CData = colors;
    
    % 在对数刺度下，误差棒需要特殊处理
    % 计算上下边界，确保不为负
    err_lower = min(RRMSE_std, RRMSE_mean * 0.99);  % 下边界不能超过均值
    err_upper = RRMSE_std;
    errorbar(1:n_methods, RRMSE_mean, err_lower, err_upper, 'k.', 'LineWidth', 2, 'CapSize', 10);
    
    % 使用对数刻度处理离群值
    set(gca, 'YScale', 'log');
    
    % 添加数值标签（显示在误差棒上方）
    for i = 1:n_methods
        if RRMSE_mean(i) > 0
            label_y = RRMSE_mean(i) + err_upper(i);  % 误差棒顶端
            text(i, label_y, sprintf('%.3f', RRMSE_mean(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                 'FontSize', 11, 'FontWeight', 'bold');
        end
    end
    
    ylabel('RRMSE', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('(a)', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', display_names, ...
             'XTickLabelRotation', 45, 'FontSize', 13);
    grid on;
    set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    box on;
    hold off;
    
    % 子图2: CC
    subplot(2, 2, 2);
    hold on;
    b2 = bar(1:n_methods, CC_mean, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
    b2.CData = colors;
    
    % CC应该在[0,1]范围内，计算非对称误差
    err_lower = min(CC_std, CC_mean);  % 下边界不能小于0
    err_upper = min(CC_std, 1 - CC_mean);  % 上边界不能大于1
    errorbar(1:n_methods, CC_mean, err_lower, err_upper, 'k.', 'LineWidth', 2, 'CapSize', 10);
    
    % 使用线性刻度，但可能需要调整y轴范围
    ylim_min = max(0, min(CC_mean - CC_std) - 0.1);
    ylim_max = min(1, max(CC_mean + CC_std) + 0.1);
    if ylim_max - ylim_min > 0.01
        ylim([ylim_min, ylim_max]);
    end
    
    for i = 1:n_methods
        label_y = CC_mean(i) + err_upper(i);  % 误差棒顶端
        text(i, label_y, sprintf('%.3f', CC_mean(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 11, 'FontWeight', 'bold');
    end
    
    ylabel('CC', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('(b)', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', display_names, ...
             'XTickLabelRotation', 45, 'FontSize', 13);
    grid on;
    set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    box on;
    hold off;
    
    % 子图3: RRMSE_PSD
    subplot(2, 2, 3);
    hold on;
    b3 = bar(1:n_methods, RRMSE_PSD_mean, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
    b3.CData = colors;
    
    % 在对数刺度下，误差棒需要特殊处理
    err_lower = min(RRMSE_PSD_std, RRMSE_PSD_mean * 0.99);
    err_upper = RRMSE_PSD_std;
    errorbar(1:n_methods, RRMSE_PSD_mean, err_lower, err_upper, 'k.', 'LineWidth', 2, 'CapSize', 10);
    
    % 使用对数刺度处理离群值
    set(gca, 'YScale', 'log');
    
    for i = 1:n_methods
        if RRMSE_PSD_mean(i) > 0
            label_y = RRMSE_PSD_mean(i) + err_upper(i);  % 误差棒顶端
            text(i, label_y, sprintf('%.3f', RRMSE_PSD_mean(i)), ...
                 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
                 'FontSize', 11, 'FontWeight', 'bold');
        end
    end
    
    ylabel('RRMSE_{PSD}', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('(c)', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', display_names, ...
             'XTickLabelRotation', 45, 'FontSize', 13);
    grid on;
    set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    box on;
    hold off;
    
    % 子图4: MI
    subplot(2, 2, 4);
    hold on;
    b4 = bar(1:n_methods, MI_mean, 'FaceColor', 'flat', 'EdgeColor', 'k', 'LineWidth', 1.5);
    b4.CData = colors;
    
    % 计算非对称误差棒，确保下边界不为负
    err_lower = min(MI_std, MI_mean);  % 下边界不能小于0
    err_upper = MI_std;
    errorbar(1:n_methods, MI_mean, err_lower, err_upper, 'k.', 'LineWidth', 2, 'CapSize', 10);
    
    % 智能调整y轴范围
    mi_min = min(MI_mean - MI_std);
    mi_max = max(MI_mean + MI_std);
    mi_range = mi_max - mi_min;
    if mi_range > 0
        ylim([max(0, mi_min - 0.1*mi_range), mi_max + 0.2*mi_range]);
    end
    
    for i = 1:n_methods
        label_y = MI_mean(i) + err_upper(i);  % 误差棒顶端
        text(i, label_y, sprintf('%.3f', MI_mean(i)), ...
             'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
             'FontSize', 11, 'FontWeight', 'bold');
    end
    
    ylabel('MI', 'FontSize', 16, 'FontWeight', 'bold');
    xlabel('(d)', 'FontSize', 16, 'FontWeight', 'bold');
    set(gca, 'XTick', 1:n_methods, 'XTickLabel', display_names, ...
             'XTickLabelRotation', 45, 'FontSize', 13);
    grid on;
    set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    box on;
    hold off;
    
    % 添加总标题
    sgtitle('各方法去噪性能对比', 'FontSize', 20, 'FontWeight', 'bold');
    
    fprintf('✓ 性能对比图已显示\n');
end


function plot_time_comparison(results_dict, method_field_names, results_dir, varargin)
    % 绘制运行时间对比图（按时间排序）
    % varargin{1}: 可选的文件名后缀 (例如 '_SNR-6dB')
    
    % 获取可选的后缀
    if nargin >= 4
        suffix = varargin{1};
    else
        suffix = '';
    end
    
    n_methods = length(method_field_names);
    times = zeros(n_methods, 1);
    display_names = cell(n_methods, 1);
    
    % 提取时间数据
    for i = 1:n_methods
        field_name = method_field_names{i};
        metrics = results_dict.(field_name);
        times(i) = metrics.time_per_sample * 1000;  % 转换为毫秒
        if isfield(metrics, 'display_name')
            display_names{i} = metrics.display_name;
        else
            display_names{i} = field_name;
        end
    end
    
    % 按时间排序（从小到大）
    [times_sorted, sort_idx] = sort(times);
    display_names_sorted = display_names(sort_idx);
    
    % 创建图形
    figure('Position', [100, 100, 1200, 600], 'Color', 'w');
    
    % 生成灰度颜色
    gray_colors = repmat(linspace(0.4, 0.7, n_methods)', 1, 3);
    
    % 绘制水平柱状图
    barh(1:n_methods, times_sorted, 'FaceColor', 'flat', 'CData', gray_colors);
    
    % 设置Y轴标签为方法名
    set(gca, 'YTick', 1:n_methods, 'YTickLabel', display_names_sorted, ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    % 设置标题和标签
    xlabel('运行时间/毫秒', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('去噪方法', 'FontSize', 16, 'FontWeight', 'bold');
    % 简化网格
    grid on;
    set(gca, 'XGrid', 'on', 'YGrid', 'off', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    set(gca, 'Layer', 'top');
    
    % 判断是否使用对数刻度
    if max(times_sorted) / min(times_sorted(times_sorted > 0)) > 10
        set(gca, 'XScale', 'log');
    end
    
    % 添加数值标签
    for i = 1:n_methods
        text(times_sorted(i), i, sprintf('  %.2f', times_sorted(i)), ...
            'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', ...
            'FontSize', 12, 'FontWeight', 'bold');
    end
    
    % 调整布局
    set(gca, 'Box', 'on');
    
    fprintf('  ✓ 时间对比图生成完毕\n');
end


%% ===================== 辅助函数 =====================

function results_out = process_snr_level(config, current_snr, true_signals, results_dir)
% 处理单个SNR级别的所有方法预测结果
    
    % 扫描该SNR级别对应的结果文件
    fprintf('\n[扫描目录] %s\n', results_dir);
    snr_pattern = sprintf('*_SNR%ddB.mat', current_snr);
    mat_files = dir(fullfile(results_dir, snr_pattern));
    
    if isempty(mat_files)
        fprintf('该SNR级别没有找到结果文件 (模式: %s)\n', snr_pattern);
        return;
    end
    
    fprintf('发现 %d 个数据文件，开始处理...\n', length(mat_files));
    
    % 存储结果
    results_dict = struct();
    method_names = {};
    
    % 遍历计算
    for i = 1:length(mat_files)
        file_name = mat_files(i).name;
        
        % 提取方法名：取第一个 "_" 之前
        underscore_pos = strfind(file_name, '_');
        if ~isempty(underscore_pos)
            method_name_display = file_name(1:underscore_pos(1)-1);
        else
            [~, method_name_display, ~] = fileparts(file_name);
        end
        
        % 创建有效的字段名（移除特殊字符）
        method_name_field = strrep(method_name_display, '-', '_');
        method_name_field = strrep(method_name_field, '%', 'percent');
        method_name_field = strrep(method_name_field, ' ', '_');
        method_name_field = matlab.lang.makeValidName(method_name_field);
        
        file_path = fullfile(results_dir, file_name);
        fprintf('\n>>> 处理方法: [%s] (文件: %s)\n', method_name_display, file_name);
        
        % 加载预测文件
        [predictions, time_per_sample] = load_prediction_file(file_path, file_name);
        if isempty(predictions)
            continue;
        end
        
        % 显示预测数据信息
        fprintf('    数据维度: %s', mat2str(size(predictions)));
        if time_per_sample > 0
            fprintf(' | 单样本耗时: %.3f ms\n', time_per_sample * 1000);
        else
            fprintf(' | 耗时信息: 未提供\n');
        end
        
        % 检查维度
        if ~isequal(size(predictions), size(true_signals))
            fprintf('  ⚠ 警告: 维度不匹配! 预测:%s vs 真实:%s, 跳过\n', ...
                    mat2str(size(predictions)), mat2str(size(true_signals)));
            continue;
        end
        
        % 计算指标
        metrics = compute_metrics_for_method(predictions, true_signals, config.fs);
        metrics.time_per_sample = time_per_sample;
        metrics.display_name = method_name_display;  % 保存显示名称
        
        % 存储结果（使用有效字段名）
        results_dict.(method_name_field) = metrics;
        method_names{end+1} = method_name_field;
        
        fprintf('  ✓ %s 计算完毕 (RRMSE=%.3f, CC=%.3f, Time=%.1fms)\n', ...
                method_name_display, metrics.RRMSE_mean, metrics.CC_mean, time_per_sample * 1000);
    end
    
    if isempty(method_names)
        fprintf('该SNR级别没有成功计算任何方法的指标。\n');
        return;
    end
    
    % 保存结果
    output_csv = fullfile(results_dir, sprintf('all_metrics_SNR%ddB.csv', current_snr));
    [results_table, sorted_methods] = save_results(results_dict, method_names, output_csv);

    % 打印最终结果
    fprintf('\n================================================================================\n');
    fprintf('%s\n', center_text(sprintf('SNR=%ddB 最终结果排行', current_snr), 80));
    fprintf('================================================================================\n');
    disp(results_table);

    % 返回结果字典（由上层聚合），不再在此处绘图
    results_out = results_dict;
end

function text_out = center_text(text_in, width)
% 将文本居中对齐到指定宽度
    text_len = length(text_in);
    if text_len >= width
        text_out = text_in;
    else
        padding = floor((width - text_len) / 2);
        text_out = [repmat(' ', 1, padding), text_in];
    end
end

function key = snr_field(snr)
% 生成合法的结构字段名用于表示SNR（负号替换为 m）
    s = num2str(snr);
    s = strrep(s, '-', 'm');
    s = strrep(s, ' ', '');
    key = sprintf('snr_%s', s);
end
