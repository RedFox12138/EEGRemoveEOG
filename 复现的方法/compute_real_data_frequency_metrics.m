function [T_formatted, T_numeric] = compute_real_data_frequency_metrics()
% 计算真实数据集的频域评价指标，并生成表格文件
%
% 指标说明（按样本先计算后取均值±标准差）：
% - ΔER_δ / %: δ频段(0.5-4 Hz)能量比变化百分比，越大越好（清除δ的能量比例）
% - MAE_δ: δ频段(0.5-4 Hz)功率谱密度平均绝对误差，越小越好
% - MAE_θ: θ频段(4-8 Hz)功率谱密度平均绝对误差，越小越好
% - MAE_α: α频段(8-13 Hz)功率谱密度平均绝对误差，越小越好
% - MAE_β: β频段(13-30 Hz)功率谱密度平均绝对误差，越小越好
% - Time_ms: 单样本平均处理时间(ms)
%
% 输入: 无（内部扫描结果目录）
% 输出:
%   T_formatted: 含"均值 ± 标准差"字符串的表（便于展示/导出）
%   T_numeric: 纯数值表（便于进一步作图分析）
%
% 结果保存：
%   - real_data_metrics_formatted.csv
%   - real_data_metrics_numeric.csv
%
% 作者: GitHub Copilot
% 日期: 2025-12-21

    fprintf('================================================================================\n');
    fprintf('%s\n', center_text('真实数据集频域指标计算', 80));
    fprintf('================================================================================\n');

    % 结果目录优先级：专用真实数据集结果目录 > 全局results目录
    primary_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\训练完的模型和数据\真实数据集\结果';
    fallback_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\results';

    search_dirs = {};
    if exist(primary_dir, 'dir'), search_dirs{end+1} = primary_dir; end
    if exist(fallback_dir, 'dir'), search_dirs{end+1} = fallback_dir; end

    if isempty(search_dirs)
        error('未找到任何结果目录。请先运行各方法的真实数据集测试脚本。');
    end

    % 收集所有文件（去重：按方法名唯一）
    file_map = containers.Map();
    for d = 1:numel(search_dirs)
        dir_now = search_dirs{d};
        files = dir(fullfile(dir_now, '*_real_data_predictions.mat'));
        for i = 1:numel(files)
            method_name = strrep(files(i).name, '_real_data_predictions.mat', '');
            % 优先保留primary_dir中的文件
            if ~isKey(file_map, method_name) || strcmp(dir_now, primary_dir)
                file_map(method_name) = fullfile(dir_now, files(i).name);
            end
        end
    end

    if file_map.Count == 0
        error('未在结果目录中找到 *_real_data_predictions.mat 文件');
    end

    method_names = file_map.keys;
    fprintf('发现 %d 个方法结果，开始计算...\n', numel(method_names));

    % 指标容器
    numeric_rows = struct('Method', {}, ...
                          'DeltaER_mean', {}, 'DeltaER_std', {}, ...
                          'MAE_delta_mean', {}, 'MAE_delta_std', {}, ...
                          'MAE_theta_mean', {}, 'MAE_theta_std', {}, ...
                          'MAE_alpha_mean', {}, 'MAE_alpha_std', {}, ...
                          'MAE_beta_mean', {}, 'MAE_beta_std', {}, ...
                          'Time_ms_mean', {}, 'Time_ms_std', {});

    for k = 1:numel(method_names)
        method_disp = method_names{k};
        file_path = file_map(method_disp);
        fprintf('\n>>> 处理方法: [%s]\n', method_disp);

        try
            data = load(file_path);
        catch
            fprintf('  ❌ 无法读取: %s，跳过\n', file_path);
            continue;
        end

        % 读取字段
        cleaned = read_field(data, {'cleaned_eeg', 'predictions', 'clean_data', 'data'});
        extracted = read_field(data, {'extracted_eog'});
        original = read_field(data, {'original'});
        time_ps = read_field(data, {'time_per_sample'});
        fs = read_field(data, {'sampling_rate'});
        if isempty(fs), fs = 250; end

        if isempty(cleaned)
            fprintf('  ❌ 缺少 cleaned_eeg/预测数据，跳过\n');
            continue;
        end

        % 如果没有original，尝试用 cleaned + extracted 重建
        if isempty(original)
            if ~isempty(extracted)
                original = cleaned + extracted;
                fprintf('  ⚠ 未找到original字段，已用 cleaned+extracted 重建\n');
            else
                fprintf('  ❌ 缺少 original 且无法重建（没有extracted_eog），跳过\n');
                continue;
            end
        end

        [n_samples, sig_len] = size(cleaned);
        fprintf('  数据维度: %s, fs=%g Hz\n', mat2str(size(cleaned)), fs);

        % 每样本时间
        time_vec = [];
        if ~isempty(time_ps)
            t = double(time_ps);
            if isscalar(t)
                time_vec = repmat(t, n_samples, 1);
            else
                time_vec = t(:);
                if numel(time_vec) ~= n_samples
                    time_vec = repmat(mean(time_vec), n_samples, 1);
                end
            end
        end

        % 频段定义（Hz）
        delta_band = [0.5, 4];
        theta_band = [4, 8];
        alpha_band = [8, 13];
        beta_band  = [13, 30];

        DeltaER_list = nan(n_samples, 1);
        MAE_delta_list = nan(n_samples, 1);
        MAE_theta_list = nan(n_samples, 1);
        MAE_alpha_list = nan(n_samples, 1);
        MAE_beta_list = nan(n_samples, 1);
        Time_ms_list = nan(n_samples, 1);

        for i = 1:n_samples
            x_orig = original(i, :);
            x_clean = cleaned(i, :);

            % 计算PSD
            nperseg = min(512, sig_len);
            noverlap = floor(nperseg/2);
            [pxx_orig, f] = pwelch(x_orig, nperseg, noverlap, [], fs);
            [pxx_clean, ~] = pwelch(x_clean, nperseg, noverlap, [], fs);

            df = mean(diff(f));

            % 计算各频段的能量和MAE
            % δ 频段
            idx_delta = (f >= delta_band(1)) & (f < delta_band(2));
            E_orig_delta = sum(pxx_orig(idx_delta)) * df;
            E_clean_delta = sum(pxx_clean(idx_delta)) * df;
            E_orig_total = sum(pxx_orig) * df;

            if E_orig_delta > 0
                DeltaER_list(i) = 100 * (E_orig_delta - E_clean_delta) / E_orig_delta;
            else
                DeltaER_list(i) = NaN;
            end

            if sum(idx_delta) > 0
                MAE_delta_list(i) = mean(abs(pxx_orig(idx_delta) - pxx_clean(idx_delta)));
            else
                MAE_delta_list(i) = NaN;
            end

            % θ 频段
            idx_theta = (f >= theta_band(1)) & (f < theta_band(2));
            if sum(idx_theta) > 0
                MAE_theta_list(i) = mean(abs(pxx_orig(idx_theta) - pxx_clean(idx_theta)));
            else
                MAE_theta_list(i) = NaN;
            end

            % α 频段
            idx_alpha = (f >= alpha_band(1)) & (f < alpha_band(2));
            if sum(idx_alpha) > 0
                MAE_alpha_list(i) = mean(abs(pxx_orig(idx_alpha) - pxx_clean(idx_alpha)));
            else
                MAE_alpha_list(i) = NaN;
            end

            % β 频段
            idx_beta = (f >= beta_band(1)) & (f < beta_band(2));
            if sum(idx_beta) > 0
                MAE_beta_list(i) = mean(abs(pxx_orig(idx_beta) - pxx_clean(idx_beta)));
            else
                MAE_beta_list(i) = NaN;
            end

            % 时间
            if ~isempty(time_vec)
                Time_ms_list(i) = time_vec(i) * 1000;
            end
        end

        % 聚合指标
        row.Method = method_disp;
        row.DeltaER_mean = mean(DeltaER_list, 'omitnan');
        row.DeltaER_std = std(DeltaER_list, 0, 'omitnan');
        row.MAE_delta_mean = mean(MAE_delta_list, 'omitnan');
        row.MAE_delta_std = std(MAE_delta_list, 0, 'omitnan');
        row.MAE_theta_mean = mean(MAE_theta_list, 'omitnan');
        row.MAE_theta_std = std(MAE_theta_list, 0, 'omitnan');
        row.MAE_alpha_mean = mean(MAE_alpha_list, 'omitnan');
        row.MAE_alpha_std = std(MAE_alpha_list, 0, 'omitnan');
        row.MAE_beta_mean = mean(MAE_beta_list, 'omitnan');
        row.MAE_beta_std = std(MAE_beta_list, 0, 'omitnan');
        row.Time_ms_mean = mean(Time_ms_list, 'omitnan');
        row.Time_ms_std = std(Time_ms_list, 0, 'omitnan');

        numeric_rows(end+1) = row; %#ok<AGROW>
        fprintf('  ✓ 完成 (ΔER_δ=%.2f%%, MAE_δ=%.4f, MAE_θ=%.4f, MAE_α=%.4f, MAE_β=%.4f)\n', ...
            row.DeltaER_mean, row.MAE_delta_mean, row.MAE_theta_mean, row.MAE_alpha_mean, row.MAE_beta_mean);
    end

    if isempty(numeric_rows)
        error('没有成功计算任何方法的指标');
    end

    % 组装数值表
    T_numeric = struct2table(numeric_rows);
    % 按 DeltaER_mean 降序（越大越好）
    try
        T_numeric = sortrows(T_numeric, 'DeltaER_mean', 'descend');
    catch
        % 若某列不存在则不排序
    end

    % 生成格式化表
    fmt_pm = @(m, s) sprintf('%.3f ± %.3f', m, s);
    n = height(T_numeric);
    Method = T_numeric.Method;
    DeltaER = cell(n, 1);
    MAE_d = cell(n, 1);
    MAE_t = cell(n, 1);
    MAE_a = cell(n, 1);
    MAE_b = cell(n, 1);
    Time_ms = cell(n, 1);

    for i = 1:n
        DeltaER{i} = fmt_pm(T_numeric.DeltaER_mean(i), T_numeric.DeltaER_std(i));
        MAE_d{i} = fmt_pm(T_numeric.MAE_delta_mean(i), T_numeric.MAE_delta_std(i));
        MAE_t{i} = fmt_pm(T_numeric.MAE_theta_mean(i), T_numeric.MAE_theta_std(i));
        MAE_a{i} = fmt_pm(T_numeric.MAE_alpha_mean(i), T_numeric.MAE_alpha_std(i));
        MAE_b{i} = fmt_pm(T_numeric.MAE_beta_mean(i), T_numeric.MAE_beta_std(i));
        Time_ms{i} = sprintf('%.3f', T_numeric.Time_ms_mean(i));
    end

    T_formatted = table(Method, DeltaER, MAE_d, MAE_t, MAE_a, MAE_b, Time_ms, ...
        'VariableNames', {'Method', 'DeltaER_delta', 'MAE_delta', 'MAE_theta', 'MAE_alpha', 'MAE_beta', 'Time_ms'});

    % 保存CSV到首选目录
    save_dir = search_dirs{1};
    out_fmt = fullfile(save_dir, 'real_data_metrics_formatted.csv');
    out_num = fullfile(save_dir, 'real_data_metrics_numeric.csv');
    try
        writetable(T_formatted, out_fmt, 'Encoding', 'UTF-8');
        writetable(T_numeric, out_num, 'Encoding', 'UTF-8');
        fprintf('\n✓ 指标表已保存:\n  - %s\n  - %s\n', out_fmt, out_num);
    catch ME
        warning('保存CSV失败: %s', ME.message);
    end

    % 打印格式化结果
    fprintf('\n================================================================================\n');
    fprintf('%s\n', center_text('真实数据集频域指标 (均值 ± 标准差)', 80));
    fprintf('================================================================================\n');
    disp(T_formatted);
    
    % 生成表格式图表
    fprintf('\n正在生成表格式图表...\n');
    plot_table_figure(T_numeric, save_dir);
end


function plot_table_figure(T_numeric, save_dir)
% 生成表格式的图表展示（黑白论文风格，不含Time列）
    
    n_methods = height(T_numeric);
    
    % 准备表格数据（不包含Time列）
    rowNames = T_numeric.Method;
    
    % 格式化数据为"均值±标准差"
    fmt = @(m, s) sprintf('%.3f ± %.3f', m, s);
    
    colData = cell(n_methods, 5);
    for i = 1:n_methods
        colData{i, 1} = fmt(T_numeric.DeltaER_mean(i), T_numeric.DeltaER_std(i));
        colData{i, 2} = fmt(T_numeric.MAE_delta_mean(i), T_numeric.MAE_delta_std(i));
        colData{i, 3} = fmt(T_numeric.MAE_theta_mean(i), T_numeric.MAE_theta_std(i));
        colData{i, 4} = fmt(T_numeric.MAE_alpha_mean(i), T_numeric.MAE_alpha_std(i));
        colData{i, 5} = fmt(T_numeric.MAE_beta_mean(i), T_numeric.MAE_beta_std(i));
    end
    
    % 列标题
    colNames = {'ΔER_δ / %', 'MAE_δ', 'MAE_θ', 'MAE_α', 'MAE_β'};
    
    % 创建图形
    fig = figure('Position', [100, 100, 1200, 80 + 40*n_methods], 'Color', 'w');
    
    % 创建uitable
    t = uitable('Parent', fig, ...
                'Data', colData, ...
                'ColumnName', colNames, ...
                'RowName', rowNames, ...
                'Units', 'normalized', ...
                'Position', [0.05 0.1 0.9 0.85], ...
                'FontSize', 11, ...
                'FontName', 'Arial');
    
    % 设置列宽
    t.ColumnWidth = {100, 100, 100, 100, 100};
    
    % 添加标题
    annotation('textbox', [0.05 0.92 0.9 0.05], ...
               'String', '真实数据集频域指标对比 (均值 ± 标准差)', ...
               'EdgeColor', 'none', ...
               'FontSize', 14, ...
               'FontWeight', 'bold', ...
               'HorizontalAlignment', 'center');
    
    % 保存图表
    try
        out_png = fullfile(save_dir, 'real_data_metrics_table.png');
        out_eps = fullfile(save_dir, 'real_data_metrics_table.eps');
        
        saveas(fig, out_png);
        print(fig, out_eps, '-depsc', '-r300');
        
        fprintf('  ✓ 表格式图表已保存:\n');
        fprintf('    - PNG: %s\n', out_png);
        fprintf('    - EPS: %s (矢量图，适合论文)\n', out_eps);
    catch ME
        warning('保存图表失败: %s', ME.message);
    end
end


function val = read_field(S, names)
% 从结构体中按候选列表读取第一个存在的字段
    val = [];
    for i = 1:numel(names)
        if isfield(S, names{i})
            val = S.(names{i});
            return;
        end
    end
end


function text_out = center_text(text_in, width)
% 文本居中显示
    text_len = length(text_in);
    if text_len >= width
        text_out = text_in;
    else
        padding = floor((width - text_len) / 2);
        text_out = [repmat(' ', 1, padding), text_in];
    end
end
