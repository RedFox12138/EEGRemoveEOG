function visualize_real_data_results()
% VISUALIZE_REAL_DATA_RESULTS 可视化真实数据集上不同方法的去噪结果
%
% 该脚本会：
% 1. 加载真实数据集（污染的EEG信号）
% 2. 加载所有方法的去噪结果
% 3. 在一张图中展示多个子图对比
% 4. 使用灰度配色，适合论文发表
%
% 使用方法：
%   visualize_real_data_results()  % 使用默认参数
%
% 作者: GitHub Copilot
% 日期: 2025-12-17

    fprintf('================================================================================\n');
    fprintf('%s\n', center_text('真实数据集去噪结果可视化', 80));
    fprintf('================================================================================\n\n');
    
    % ===================== 配置路径 =====================
    % 真实数据集目录
    real_data_dir = 'D:\Pycharm_Projects\EOG Remove\真实数据集';
    real_data_file = fullfile(real_data_dir, 'eog_dataset.mat');
    
    % 结果目录
    results_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\训练完的模型和数据\真实数据集\结果';
    % ==================================================
    
    % 1. 加载真实数据集测试集（污染信号）
    fprintf('[1/4] 加载真实数据集测试集...\n');
    if ~exist(real_data_file, 'file')
        error('找不到真实数据集文件: %s', real_data_file);
    end
    
    % 使用统一的数据划分函数，只加载测试集
    contaminated_signals = loadRealDatasetSplit(real_data_file, 'eog_dataset', 0.9, 42);
    
    fprintf('  ✓ 加载完成，测试集维度: %s\n', mat2str(size(contaminated_signals)));
    fprintf('  ⚠️  注意：只使用测试集数据（10%%），确保与各方法预测结果一致\n');
    [n_samples, signal_length] = size(contaminated_signals);
    
    % 2. 扫描并加载所有方法的结果
    fprintf('\n[2/4] 扫描结果目录...\n');
    if ~exist(results_dir, 'dir')
        error('结果目录不存在: %s', results_dir);
    end
    
    mat_files = dir(fullfile(results_dir, '*_real_data_predictions.mat'));
    fprintf('  发现 %d 个结果文件\n', length(mat_files));
    
    % 存储所有方法的结果
    methods = struct();
    method_names = {};
    
    for i = 1:length(mat_files)
        file_name = mat_files(i).name;
        
        % 提取方法名
        method_name = strrep(file_name, '_real_data_predictions.mat', '');
        
        file_path = fullfile(results_dir, file_name);
        fprintf('  [%d/%d] 加载: %s\n', i, length(mat_files), method_name);
        
        % 加载结果
        data = load(file_path);
        
        % 查找预测结果变量
        if isfield(data, 'predictions')
            predictions = data.predictions;
        elseif isfield(data, 'data')
            predictions = data.data;
        elseif isfield(data, 'clean_data')
            predictions = data.clean_data;
        else
            % 尝试找到最大的数据变量
            fields = fieldnames(data);
            max_size = 0;
            predictions = [];
            for j = 1:length(fields)
                field_name = fields{j};
                if ~startsWith(field_name, '__') && isnumeric(data.(field_name))
                    if numel(data.(field_name)) > max_size
                        predictions = data.(field_name);
                        max_size = numel(data.(field_name));
                    end
                end
            end
        end
        
        if isempty(predictions)
            fprintf('    ⚠ 警告: 未找到有效数据，跳过\n');
            continue;
        end
        
        % 检查维度
        if ~isequal(size(predictions), size(contaminated_signals))
            fprintf('    ⚠ 警告: 维度不匹配，跳过 (预测:%s vs 原始:%s)\n', ...
                    mat2str(size(predictions)), mat2str(size(contaminated_signals)));
            continue;
        end
        
        % 创建有效字段名
        field_name = matlab.lang.makeValidName(method_name);
        methods.(field_name).predictions = predictions;
        methods.(field_name).display_name = method_name;
        method_names{end+1} = field_name;
        
        fprintf('    ✓ 加载成功\n');
    end
    
    n_methods = length(method_names);
    if n_methods == 0
        error('没有成功加载任何方法的结果');
    end
    
    fprintf('  ✓ 成功加载 %d 个方法的结果\n', n_methods);
    
    % 3. 调整方法顺序：将DAT-Net相关方法放到最前面
    fprintf('\n[3/4] 调整方法顺序...\n');
    method_names = reorder_methods(method_names);
    fprintf('  排序后的方法顺序:\n');
    for i = 1:length(method_names)
        display_name = methods.(method_names{i}).display_name;
        fprintf('    %d. %s\n', i, display_name);
    end
    
    % 4. 选择要展示的样本
    fprintf('\n[4/5] 准备可视化...\n');
    
    % 默认展示第2个样本
    sample_idx = 2;
    fprintf('  选择样本: %d/%d\n', sample_idx, n_samples);
    
    % 5. 创建对比图
    fprintf('\n[5/5] 生成对比图...\n');
    plot_comparison_vertical(contaminated_signals(sample_idx, :), methods, method_names, sample_idx);
    
    fprintf('\n================================================================================\n');
    fprintf('%s\n', center_text('可视化完成！', 80));
    fprintf('================================================================================\n');
end


function sorted_names = reorder_methods(method_names)
% 将DAT-Net相关方法放到最前面
    dat_net_methods = {};
    other_methods = {};
    
    for i = 1:length(method_names)
        name = method_names{i};
        % 检查是否包含DAT-Net、DAT_Net等变体
        if contains(lower(name), 'dat') && contains(lower(name), 'net')
            dat_net_methods{end+1} = name;
        else
            other_methods{end+1} = name;
        end
    end
    
    % DAT-Net方法在前，其他方法在后
    sorted_names = [dat_net_methods, other_methods];
end


function plot_comparison_vertical(contaminated_signal, methods, method_names, sample_idx)
% 绘制n行1列的对比图，每个子图同时显示原始信号和去噪结果
    
    n_methods = length(method_names);
    signal_length = length(contaminated_signal);
    
    % 创建大图 (n行1列)
    fig = figure('Position', [100, 50, 1400, 300*n_methods], 'Color', 'w');
    sgtitle(sprintf('真实数据集去噪结果对比 (样本 #%d)', sample_idx), ...
            'FontSize', 20, 'FontWeight', 'bold');
    
    % 时间轴（假设采样率250Hz）
    fs = 250;
    time = (0:signal_length-1) / fs;
    
    % 计算全局y轴范围
    all_signals = contaminated_signal;
    for i = 1:n_methods
        field_name = method_names{i};
        all_signals = [all_signals; methods.(field_name).predictions(1, :)];
    end
    y_min = min(all_signals(:));
    y_max = max(all_signals(:));
    y_range = y_max - y_min;
    y_lim = [y_min - 0.1*y_range, y_max + 0.1*y_range];
    
    % 定义灰度颜色和线型
    color_original = [0.5, 0.5, 0.5];  % 中灰色 - 原始信号
    color_denoised = [0.1, 0.1, 0.1];  % 深灰色 - 去噪信号
    linestyle_original = '--';          % 虚线 - 原始信号
    linestyle_denoised = '-';           % 实线 - 去噪信号
    
    % 各方法的对比子图
    for i = 1:n_methods
        subplot(n_methods, 1, i);
        
        field_name = method_names{i};
        denoised_signal = methods.(field_name).predictions(1, :);
        display_name = methods.(field_name).display_name;
        
        % 同时绘制原始信号和去噪信号（使用不同线型）
        hold on;
        plot(time, contaminated_signal, 'LineStyle', linestyle_original, 'Color', color_original, 'LineWidth', 1.5, 'DisplayName', '原始信号(含EOG)');
        plot(time, denoised_signal, 'LineStyle', linestyle_denoised, 'Color', color_denoised, 'LineWidth', 1.5, 'DisplayName', '去噪信号');
        hold off;
        
        % 设置标题和标签
        title(display_name, 'FontSize', 14, 'FontWeight', 'bold', 'Interpreter', 'none');
        ylabel('幅值 (μV)', 'FontSize', 12);
        ylim(y_lim);
        
        % 添加图例（只在第一个子图显示）
        if i == 1
            legend('Location', 'northeast', 'FontSize', 10);
        end
        
        % 网格和样式
        grid on;
        set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
        box on;
        
        % 只在最后一个子图显示x轴标签
        if i < n_methods
            set(gca, 'XTickLabel', []);
        else
            xlabel('时间 (s)', 'FontSize', 12, 'FontWeight', 'bold');
        end
    end
    
    fprintf('  ✓ 对比图已生成 (n行1列布局)\n');
    fprintf('  提示: 可以使用 saveas(gcf, ''filename.png'') 保存图片\n');
end


function plot_detailed_comparison(contaminated_signal, methods, method_names, sample_idx, target_methods)
% 绘制详细对比图（只对比几个选定的方法）
% 
% 参数:
%   contaminated_signal - 原始污染信号
%   methods - 所有方法结构体
%   method_names - 所有方法名列表
%   sample_idx - 样本索引
%   target_methods - 要对比的方法名cell数组
    
    if nargin < 5
        % 默认对比前4个方法
        target_methods = method_names(1:min(4, length(method_names)));
    end
    
    n_target = length(target_methods);
    signal_length = length(contaminated_signal);
    
    % 创建图形
    fig = figure('Position', [100, 100, 1400, 800], 'Color', 'w');
    
    % 时间轴
    fs = 200;
    time = (0:signal_length-1) / fs;
    
    % 计算y轴范围
    all_signals = contaminated_signal;
    for i = 1:n_target
        field_name = target_methods{i};
        all_signals = [all_signals; methods.(field_name).predictions(1, :)];
    end
    y_min = min(all_signals(:));
    y_max = max(all_signals(:));
    y_range = y_max - y_min;
    y_lim = [y_min - 0.1*y_range, y_max + 0.1*y_range];
    
    % 原始信号
    subplot(n_target+1, 1, 1);
    plot(time, contaminated_signal, 'Color', [0.2, 0.2, 0.2], 'LineWidth', 2);
    title(sprintf('原始信号（含EOG伪迹） - 样本 #%d', sample_idx), ...
          'FontSize', 16, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 14, 'FontWeight', 'bold');
    ylim(y_lim);
    grid on;
    set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
    box on;
    set(gca, 'XTickLabel', []);
    
    % 各方法结果
    for i = 1:n_target
        subplot(n_target+1, 1, i+1);
        
        field_name = target_methods{i};
        denoised_signal = methods.(field_name).predictions(1, :);
        display_name = methods.(field_name).display_name;
        
        % 灰度渐变
        gray_level = 0.25 + 0.5 * (i / n_target);
        plot(time, denoised_signal, 'Color', [gray_level, gray_level, gray_level], 'LineWidth', 2);
        
        title(display_name, 'FontSize', 16, 'FontWeight', 'bold', 'Interpreter', 'none');
        ylabel('幅值 (μV)', 'FontSize', 14, 'FontWeight', 'bold');
        ylim(y_lim);
        grid on;
        set(gca, 'XGrid', 'off', 'YGrid', 'on', 'GridAlpha', 0.3, 'GridLineStyle', '--');
        box on;
        
        if i < n_target
            set(gca, 'XTickLabel', []);
        else
            xlabel('时间 (s)', 'FontSize', 14, 'FontWeight', 'bold');
        end
    end
    
    fprintf('  ✓ 详细对比图已生成\n');
end


function text_out = center_text(text_in, width)
% 将文本居中对齐
    text_len = length(text_in);
    if text_len >= width
        text_out = text_in;
    else
        padding = floor((width - text_len) / 2);
        text_out = [repmat(' ', 1, padding), text_in];
    end
end
