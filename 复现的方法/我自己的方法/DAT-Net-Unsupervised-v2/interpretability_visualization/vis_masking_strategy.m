function vis_masking_strategy(sample_idx, channel_idx, config)
% VIS_MASKING_STRATEGY 可视化掩蔽策略
%
% 展示掩蔽生成过程:
% - 原始信号
% - 伪影概率
% - 加权掩蔽概率
% - 实际掩蔽位置
% - 掩蔽后的信号
% - 对比: 随机掩蔽 vs 伪影感知掩蔽
%
% 输入:
%   sample_idx - 样本索引 (可选)
%   channel_idx - 通道索引 (可选)
%   config - 配置结构体 (可选)
%
% 用法:
%   vis_masking_strategy();
%   vis_masking_strategy(5, 10);

    % 默认参数
    if nargin < 1 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 2 || isempty(channel_idx)
        channel_idx = 1;
    end
    if nargin < 3 || isempty(config)
        config = config_visualization();
    end
    
    fprintf('\n========================================\n');
    fprintf('掩蔽策略可视化\n');
    fprintf('========================================\n');
    fprintf('样本: %d, 通道: %d\n', sample_idx, channel_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    data = load_real_dataset(sample_idx, channel_idx, config);
    
    %% 计算掩蔽策略
    fprintf('正在计算掩蔽策略...\n');
    
    x = data.contaminated;
    fs = data.fs;
    mask_base = config.LOSS.mask_base;
    boost_scale = config.LOSS.boost_scale;
    neighborhood = config.LOSS.mask_neighborhood;
    
    % 1. 计算伪影概率
    p_art = compute_artifact_prob(x, fs);
    
    % 2. 生成伪影感知掩蔽
    [x_masked_artifact, mask_artifact, p_mask] = generate_artifact_aware_mask(...
        x, p_art, mask_base, boost_scale, neighborhood);
    
    % 3. 生成随机掩蔽(用于对比)
    [x_masked_random, mask_random] = generate_random_mask(...
        x, mask_base, neighborhood);
    
    %% 创建可视化
    fprintf('正在生成可视化...\n');
    
    fig = figure('Position', [100, 100, 1800, 1400], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('掩蔽策略 - 样本%d', sample_idx), 'NumberTitle', 'off');
    
    % 颜色配置
    col_original = config.VIS.colors.original;
    col_artifact = config.VIS.colors.artifact;
    col_masked = config.VIS.colors.masked;
    
    time = data.time;
    
    %% 第1行: 原始信号和伪影概率
    subplot(5, 2, [1, 2]);
    yyaxis left
    plot(time, x, 'Color', col_original, 'LineWidth', 1.5, 'DisplayName', '受污染信号');
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    
    yyaxis right
    area(time, p_art, 'FaceColor', col_artifact, 'FaceAlpha', 0.3, ...
         'EdgeColor', col_artifact, 'LineWidth', 1.5);
    hold on;
    % 如果有模型预测的伪影，也显示出来
    if ~all(data.artifact_pred == 0)
        % 归一化以便对比
        artifact_normalized = data.artifact_pred / max(abs(data.artifact_pred));
        plot(time, artifact_normalized, 'Color', [1 0.4 0.2], 'LineWidth', 1.2, ...
             'LineStyle', '--', 'DisplayName', '真实EOG (归一化)');
    end
    ylabel('伪影概率', 'FontSize', config.VIS.font_size);
    ylim([0, 1]);
    
    title('步骤 1: 原始信号 vs 伪影概率 & 真实EOG', 'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    ax = gca;
    ax.YAxis(1).Color = col_original;
    ax.YAxis(2).Color = col_artifact;
    
    %% 第2行: 加权掩蔽概率
    subplot(5, 2, [3, 4]);
    area(time, p_mask, 'FaceColor', [155 89 182]/255, 'FaceAlpha', 0.5, ...
         'EdgeColor', [155 89 182]/255, 'LineWidth', 1.5);
    hold on;
    yline(mask_base, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1.5, ...
          'DisplayName', sprintf('基础概率 = %.3f', mask_base));
    yline(mask_base + boost_scale, '--r', 'LineWidth', 1.5, ...
          'DisplayName', sprintf('最大概率 = %.3f', mask_base + boost_scale));
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('掩蔽概率', 'FontSize', config.VIS.font_size);
    ylim([0, 1]);
    title(sprintf('步骤 2: 加权掩蔽概率 (base=%.3f, boost=%.3f)', mask_base, boost_scale), ...
          'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第3行左: 伪影感知掩蔽
    subplot(5, 2, 5);
    plot(time, x, 'Color', [0.8 0.8 0.8], 'LineWidth', 1, 'DisplayName', '原始信号');
    hold on;
    plot(time, x_masked_artifact, 'Color', col_masked, 'LineWidth', 1.5, ...
         'DisplayName', '掩蔽后信号');
    
    % 标记掩蔽区域 - 使用patch更高效
    y_min = min(x);
    y_max = max(x);
    mask_indices = find(mask_artifact > 0);
    if ~isempty(mask_indices)
        % 确保索引有效
        mask_indices = mask_indices(mask_indices > 0 & mask_indices <= length(time));
        % 使用patch绘制所有掩蔽区域
        for i = 1:length(mask_indices)
            idx = mask_indices(i);
            patch([time(idx)-0.01 time(idx)+0.01 time(idx)+0.01 time(idx)-0.01], ...
                  [y_min y_min y_max y_max], 'y', 'FaceAlpha', 0.3, ...
                  'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
    end
    
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('步骤 3a: 伪影感知掩蔽', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第3行右: 随机掩蔽(对比)
    subplot(5, 2, 6);
    plot(time, x, 'Color', [0.8 0.8 0.8], 'LineWidth', 1, 'DisplayName', '原始信号');
    hold on;
    plot(time, x_masked_random, 'Color', [0.4 0.6 0.8], 'LineWidth', 1.5, ...
         'DisplayName', '掩蔽后信号');
    
    % 标记掩蔽区域 - 使用patch更高效
    mask_indices = find(mask_random > 0);
    if ~isempty(mask_indices)
        % 确保索引有效
        mask_indices = mask_indices(mask_indices > 0 & mask_indices <= length(time));
        for i = 1:length(mask_indices)
            idx = mask_indices(i);
            patch([time(idx)-0.01 time(idx)+0.01 time(idx)+0.01 time(idx)-0.01], ...
                  [y_min y_min y_max y_max], 'y', 'FaceAlpha', 0.3, ...
                  'EdgeColor', 'none', 'HandleVisibility', 'off');
        end
    end
    
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('步骤 3b: 随机掩蔽 (对比)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第4行左: 掩蔽位置对比
    subplot(5, 2, 7);
    plot(time, mask_artifact, 'Color', col_masked, 'LineWidth', 2, ...
         'DisplayName', '伪影感知掩蔽');
    hold on;
    plot(time, mask_random, 'Color', [0.4 0.6 0.8], 'LineWidth', 2, ...
         'DisplayName', '随机掩蔽');
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('掩蔽标记', 'FontSize', config.VIS.font_size);
    ylim([-0.1, 1.1]);
    title('步骤 4: 掩蔽位置对比', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第4行右: 掩蔽统计信息
    subplot(5, 2, 8);
    
    % 计算统计信息
    total_points = length(x);
    artifact_masked_count = sum(mask_artifact > 0);
    random_masked_count = sum(mask_random > 0);
    artifact_masked_ratio = artifact_masked_count / total_points;
    random_masked_ratio = random_masked_count / total_points;
    
    % 计算掩蔽与真实伪影的重叠
    high_artifact_mask = p_art > 0.5;
    artifact_overlap = sum(mask_artifact > 0 & high_artifact_mask) / max(sum(high_artifact_mask), 1);
    random_overlap = sum(mask_random > 0 & high_artifact_mask) / max(sum(high_artifact_mask), 1);
    
    % 绘制柱状图
    categories = {'掩蔽比例', '与高伪影重叠'};
    artifact_values = [artifact_masked_ratio * 100, artifact_overlap * 100];
    random_values = [random_masked_ratio * 100, random_overlap * 100];
    
    x_pos = 1:length(categories);
    bar_width = 0.35;
    bar(x_pos - bar_width/2, artifact_values, bar_width, 'FaceColor', col_masked, ...
        'DisplayName', '伪影感知');
    hold on;
    bar(x_pos + bar_width/2, random_values, bar_width, 'FaceColor', [0.4 0.6 0.8], ...
        'DisplayName', '随机掩蔽');
    
    set(gca, 'XTick', x_pos, 'XTickLabel', categories);
    ylabel('百分比 (%)', 'FontSize', config.VIS.font_size);
    title('掩蔽策略统计对比', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    ylim([0, 100]);
    
    % 添加数值标签
    for i = 1:length(categories)
        text(i - bar_width/2, artifact_values(i) + 2, sprintf('%.1f%%', artifact_values(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8);
        text(i + bar_width/2, random_values(i) + 2, sprintf('%.1f%%', random_values(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 8);
    end
    
    %% 第5行: 掩蔽效果对比 (放大视图)
    % 选择一个伪影区域进行放大显示
    [~, max_idx] = max(p_art);
    window_size = min(fs * 2, floor(length(x)/4));  % 2秒窗口或1/4信号长度
    start_idx = max(1, max_idx - window_size/2);
    end_idx = min(length(x), start_idx + window_size - 1);
    zoom_indices = start_idx:end_idx;
    time_zoom = time(zoom_indices);
    
    subplot(5, 2, 9);
    plot(time_zoom, x(zoom_indices), 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
    hold on;
    plot(time_zoom, x_masked_artifact(zoom_indices), 'Color', col_masked, 'LineWidth', 2);
    area(time_zoom, p_art(zoom_indices), 'FaceColor', col_artifact, 'FaceAlpha', 0.2, ...
         'EdgeColor', 'none');
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('步骤 5a: 伪影感知掩蔽 (局部放大)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('原始', '掩蔽后', '伪影概率', 'Location', 'best');
    grid on; grid minor;
    
    subplot(5, 2, 10);
    plot(time_zoom, x(zoom_indices), 'Color', [0.8 0.8 0.8], 'LineWidth', 1);
    hold on;
    plot(time_zoom, x_masked_random(zoom_indices), 'Color', [0.4 0.6 0.8], 'LineWidth', 2);
    area(time_zoom, p_art(zoom_indices), 'FaceColor', col_artifact, 'FaceAlpha', 0.2, ...
         'EdgeColor', 'none');
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('步骤 5b: 随机掩蔽 (局部放大)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('原始', '掩蔽后', '伪影概率', 'Location', 'best');
    grid on; grid minor;
    
    %% 添加总标题
    sgtitle(sprintf('掩蔽策略可视化 - 样本 %d, 通道 %d', sample_idx, channel_idx), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    %% 显示图像
    drawnow;  % 强制刷新显示
    figure(fig);  % 将figure带到前台
    fprintf('图像已显示\n');
    
    % 如果需要保存，可以手动保存或修改config
    if config.EXPORT.save_png || config.EXPORT.save_pdf || config.EXPORT.save_fig
        fprintf('正在保存图像...\n');
        save_figure(fig, sprintf('masking_strategy_sample%d_ch%d', sample_idx, channel_idx), config);
    end
    
    fprintf('✓ 掩蔽策略可视化完成!\n\n');
end


%% ==================== 辅助函数 ====================

function p = compute_artifact_prob(x, fs)
% 简化版伪影概率计算
    win_size = 64;
    lowpass_cutoff = 4.0;
    eps_val = 1e-8;
    
    x = x(:)';
    
    % 局部幅度
    amp = moving_average(abs(x), win_size);
    
    % 局部变化
    diff_sig = [abs(diff(x)), 0];
    diff_sig = moving_average(diff_sig, win_size);
    
    % 低频能量
    x_low = fft_lowpass(x, fs, lowpass_cutoff);
    power_low = moving_average(x_low .^ 2, win_size);
    power_total = moving_average(x .^ 2, win_size);
    r = power_low ./ (power_total + eps_val);
    
    % 归一化
    amp_n = mad_normalize(amp);
    diff_n = mad_normalize(diff_sig);
    r_n = mad_normalize(r);
    
    % 综合分数
    s = amp_n + diff_n + r_n;
    tau = quantile(s, 0.7);
    alpha = 10.0;
    p = 1 ./ (1 + exp(-alpha * (s - tau)));
    p = min(max(p, 0), 1);
end


function [x_masked, mask, p_mask] = generate_artifact_aware_mask(x, p_art, mask_base, boost_scale, neighborhood)
% 生成伪影感知掩蔽（与Python版本完全一致）
% 使用邻域随机采样替换被掩蔽的点
    x = x(:)';
    L = length(x);
    
    % 加权掩蔽概率
    p_mask = mask_base + boost_scale * p_art;
    p_mask = min(p_mask, 1.0);
    
    % 根据概率生成掩蔽（Bernoulli采样）
    mask = double(rand(1, L) < p_mask);
    
    % 如果neighborhood == 0，直接置零
    neighborhood = round(neighborhood);  % 确保neighborhood是整数
    if neighborhood <= 0
        x_masked = x;
        x_masked(mask > 0) = 0;
        return;
    end
    
    % 生成随机偏移，范围 [-neighborhood, neighborhood]，排除0
    offsets = randi([-neighborhood, neighborhood], 1, L);
    offsets(offsets == 0) = 1;  % 将0替换为1，保证不选择自身
    
    % 构造索引并收集邻域值
    base_idx = 1:L;
    gather_idx = base_idx + offsets;
    gather_idx = max(1, min(gather_idx, L));  % 边界处理
    gather_idx = round(gather_idx);  % 确保索引是整数
    
    % 收集邻域值
    gathered = x(gather_idx);
    
    % 应用掩蔽：被掩蔽的点用邻域随机采样值替换
    x_masked = x;
    x_masked(mask > 0) = gathered(mask > 0);
end


function [x_masked, mask] = generate_random_mask(x, mask_ratio, neighborhood)
% 生成随机掩蔽
    x = x(:)';
    L = length(x);
    
    % 随机选择掩蔽位置
    num_mask = round(L * mask_ratio);
    if num_mask > L
        num_mask = L;
    end
    if num_mask < 1
        num_mask = 1;
    end
    
    mask_indices = randperm(L, num_mask);
    
    mask = zeros(1, L);
    
    % 扩展邻域 - 确保索引为整数
    neighborhood = round(neighborhood);  % 确保neighborhood是整数
    for idx = mask_indices
        start_idx = max(1, round(idx - floor(neighborhood/2)));
        end_idx = min(L, round(idx + floor(neighborhood/2)));
        mask(start_idx:end_idx) = 1;
    end
    
    % 应用掩蔽
    x_masked = x;
    x_masked(mask > 0) = 0;
end


% 复用之前定义的辅助函数
function y = moving_average(x, win_size)
    if win_size <= 1
        y = x;
        return;
    end
    kernel = ones(1, win_size) / win_size;
    y = conv(x, kernel, 'same');
end


function x_low = fft_lowpass(x, fs, cutoff)
    L = length(x);
    X = fft(x);
    freqs = (0:L-1) * fs / L;
    mask = freqs <= cutoff | freqs >= (fs - cutoff);
    X_low = X .* mask;
    x_low = real(ifft(X_low));
end


function x_n = mad_normalize(x)
    eps_val = 1e-8;
    med = median(x);
    mad_val = median(abs(x - med));
    mad_val = max(mad_val, eps_val);
    x_n = (x - med) / mad_val;
end


function save_figure(fig, filename, config)
    output_dir = config.FIGURE_DIR;
    
    if config.EXPORT.save_png
        png_path = fullfile(output_dir, [filename, '.png']);
        print(fig, png_path, '-dpng', sprintf('-r%d', config.VIS.dpi));
        fprintf('已保存: %s\n', png_path);
    end
    
    if config.EXPORT.save_pdf
        pdf_path = fullfile(output_dir, [filename, '.pdf']);
        print(fig, pdf_path, '-dpdf', '-vector');
        fprintf('已保存: %s\n', pdf_path);
    end
    
    if config.EXPORT.save_fig
        fig_path = fullfile(output_dir, [filename, '.fig']);
        savefig(fig, fig_path);
        fprintf('已保存: %s\n', fig_path);
    end
end
