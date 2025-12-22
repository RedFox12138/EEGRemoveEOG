function vis_artifact_probability(sample_idx, channel_idx, config)
% VIS_ARTIFACT_PROBABILITY 可视化伪影概率计算过程
%
% 展示compute_artifact_prob函数中的各个步骤:
% - 局部幅度
% - 局部变化速度
% - 低频能量占比
% - 归一化后的特征
% - 最终的伪影概率分布
%
% 输入:
%   sample_idx - 样本索引 (可选)
%   channel_idx - 通道索引 (可选)
%   config - 配置结构体 (可选)
%
% 用法:
%   vis_artifact_probability();  % 使用默认参数
%   vis_artifact_probability(5, 10);  % 指定样本5,通道10

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
    fprintf('伪影概率计算可视化\n');
    fprintf('========================================\n');
    fprintf('样本: %d, 通道: %d\n', sample_idx, channel_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    data = load_real_dataset(sample_idx, channel_idx, config);
    
    %% 计算伪影概率及中间步骤
    fprintf('正在计算伪影概率...\n');
    
    x = data.contaminated;
    fs = data.fs;
    win_size = config.ARTIFACT_PROB.win_size;
    lowpass_cutoff = config.ARTIFACT_PROB.lowpass_cutoff;
    
    % 计算所有中间步骤
    intermediates = compute_artifact_prob_with_intermediates(x, fs, win_size, lowpass_cutoff);
    
    %% 创建可视化
    fprintf('正在生成可视化...\n');
    
    fig = figure('Position', [100, 100, 1800, 1400], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('伪影概率计算 - 样本%d', sample_idx), 'NumberTitle', 'off');
    
    % 颜色配置
    col_original = config.VIS.colors.original;
    col_artifact = config.VIS.colors.artifact;
    col_eog = config.VIS.colors.eog;
    
    time = data.time;
    
    %% 第1行: 原始信号和伪影概率
    subplot(5, 2, [1, 2]);
    yyaxis left
    plot(time, intermediates.original, 'Color', col_original, 'LineWidth', 1.5);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    
    % 标记高伪影概率区域
    hold on;
    high_prob_mask = intermediates.p > 0.5;
    fill_x = [time, fliplr(time)];
    y_min = min(intermediates.original);
    y_max = max(intermediates.original);
    fill_y = [y_min * ones(1, length(time)), ...
              intermediates.original(end:-1:1) .* high_prob_mask(end:-1:1) + ...
              y_min * ~high_prob_mask(end:-1:1)];
    fill(fill_x, fill_y, col_artifact, 'FaceAlpha', 0.2, 'EdgeColor', 'none', ...
         'DisplayName', '检测到的伪影 (p>0.5)');
    
    yyaxis right
    % 如果有真实的artifact预测,显示它
    if ~all(data.artifact_pred == 0)
        plot(time, data.artifact_pred, 'Color', col_artifact, 'LineWidth', 1.2, ...
             'LineStyle', '--', 'DisplayName', '模型预测伪影');
        ylabel('EOG 幅值 (μV)', 'FontSize', config.VIS.font_size);
    end
    
    title('步骤 0: 原始 EEG 信号与 EOG 伪影', 'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on;
    grid minor;
    ax = gca;
    ax.YAxis(1).Color = col_original;
    ax.YAxis(2).Color = col_artifact;
    
    %% 第2行左: 局部幅度
    subplot(5, 2, 3);
    plot(time, intermediates.amp, 'Color', [0.2 0.6 0.8], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('局部幅度', 'FontSize', config.VIS.font_size);
    title('步骤 1a: 局部幅度 (滑动窗口平均)', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; grid minor;
    
    %% 第2行右: 归一化局部幅度
    subplot(5, 2, 4);
    plot(time, intermediates.amp_n, 'Color', [0.2 0.6 0.8], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('归一化幅度', 'FontSize', config.VIS.font_size);
    title('步骤 1b: MAD归一化后的幅度', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; grid minor;
    
    %% 第3行左: 局部变化速度
    subplot(5, 2, 5);
    plot(time, intermediates.diff, 'Color', [0.8 0.4 0.2], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('变化速度', 'FontSize', config.VIS.font_size);
    title('步骤 2a: 局部变化速度 (一阶差分)', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; grid minor;
    
    %% 第3行右: 归一化变化速度
    subplot(5, 2, 6);
    plot(time, intermediates.diff_n, 'Color', [0.8 0.4 0.2], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('归一化变化速度', 'FontSize', config.VIS.font_size);
    title('步骤 2b: MAD归一化后的变化速度', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; grid minor;
    
    %% 第4行左: 低频能量占比
    subplot(5, 2, 7);
    plot(time, intermediates.r, 'Color', [0.4 0.6 0.3], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('能量占比', 'FontSize', config.VIS.font_size);
    title(sprintf('步骤 3a: 低频能量占比 (<%.1f Hz)', lowpass_cutoff), ...
          'FontSize', 11, 'FontWeight', 'bold');
    ylim([0, 1]);
    grid on; grid minor;
    
    %% 第4行右: 归一化低频能量占比
    subplot(5, 2, 8);
    plot(time, intermediates.r_n, 'Color', [0.4 0.6 0.3], 'LineWidth', 1.2);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('归一化能量占比', 'FontSize', config.VIS.font_size);
    title('步骤 3b: MAD归一化后的能量占比', 'FontSize', 11, 'FontWeight', 'bold');
    grid on; grid minor;
    
    %% 第5行左: 综合分数
    subplot(5, 2, 9);
    plot(time, intermediates.s, 'Color', [0.6 0.3 0.7], 'LineWidth', 1.5);
    hold on;
    yline(intermediates.tau, '--r', 'LineWidth', 2, 'DisplayName', ...
          sprintf('阈值 τ = %.2f', intermediates.tau));
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('综合分数 s', 'FontSize', config.VIS.font_size);
    title('步骤 4: 综合分数 (amp_n + diff_n + r_n)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第5行右: 最终伪影概率
    subplot(5, 2, 10);
    area(time, intermediates.p, 'FaceColor', col_artifact, 'FaceAlpha', 0.5, 'EdgeColor', col_artifact, 'LineWidth', 1.5);
    hold on;
    yline(0.5, '--k', 'LineWidth', 1.5, 'DisplayName', '检测阈值 = 0.5');
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('伪影概率 p', 'FontSize', config.VIS.font_size);
    title('步骤 5: 最终伪影概率 (Sigmoid映射)', 'FontSize', 11, 'FontWeight', 'bold');
    ylim([0, 1]);
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 添加总标题
    sgtitle(sprintf('伪影概率计算可视化 - 样本 %d, 通道 %d', sample_idx, channel_idx), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    %% 显示图像
    drawnow;  % 强制刷新显示
    figure(fig);  % 将figure带到前台
    fprintf('图像已显示\n');
    
    % 如果需要保存，可以手动保存或修改config
    if config.EXPORT.save_png || config.EXPORT.save_pdf || config.EXPORT.save_fig
        fprintf('正在保存图像...\n');
        save_figure(fig, sprintf('artifact_probability_sample%d_ch%d', sample_idx, channel_idx), config);
    end
    
    fprintf('✓ 伪影概率计算可视化完成!\n\n');
end


%% ==================== 辅助函数 ====================

function intermediates = compute_artifact_prob_with_intermediates(x, fs, win_size, lowpass_cutoff)
% 计算伪影概率并返回所有中间步骤
%
% 输入:
%   x - 输入信号 (1, L)
%   fs - 采样率
%   win_size - 窗口大小
%   lowpass_cutoff - 低通截止频率
%
% 输出:
%   intermediates - 包含所有中间步骤的结构体

    eps_val = 1e-8;
    
    % 确保是行向量
    x = x(:)';
    L = length(x);
    
    % 1) 局部幅度
    amp = moving_average(abs(x), win_size);
    
    % 2) 局部变化速度
    diff_sig = [abs(diff(x)), 0];  % 补零保持长度
    diff_sig = moving_average(diff_sig, win_size);
    
    % 3) 低频能量占比
    x_low = fft_lowpass(x, fs, lowpass_cutoff);
    power_low = moving_average(x_low .^ 2, win_size);
    power_total = moving_average(x .^ 2, win_size);
    r = power_low ./ (power_total + eps_val);
    r = min(r, 1.0);  % 限制在[0,1]
    
    % 4) MAD归一化
    amp_n = mad_normalize(amp);
    diff_n = mad_normalize(diff_sig);
    r_n = mad_normalize(r);
    
    % 5) 综合分数
    s = amp_n + diff_n + r_n;
    
    % 6) 阈值和sigmoid映射
    tau = quantile(s, 0.7);
    alpha = 10.0;
    p = 1 ./ (1 + exp(-alpha * (s - tau)));
    p = min(max(p, 0), 1);  % 限制在[0,1]
    
    % 返回所有中间结果
    intermediates.original = x;
    intermediates.amp = amp;
    intermediates.diff = diff_sig;
    intermediates.r = r;
    intermediates.x_low = x_low;
    intermediates.power_low = power_low;
    intermediates.power_total = power_total;
    intermediates.amp_n = amp_n;
    intermediates.diff_n = diff_n;
    intermediates.r_n = r_n;
    intermediates.s = s;
    intermediates.tau = tau;
    intermediates.p = p;
end


function y = moving_average(x, win_size)
% 滑动窗口平均
    if win_size <= 1
        y = x;
        return;
    end
    
    % 使用卷积实现滑动平均
    kernel = ones(1, win_size) / win_size;
    y = conv(x, kernel, 'same');
end


function x_low = fft_lowpass(x, fs, cutoff)
% FFT低通滤波
    L = length(x);
    
    % FFT
    X = fft(x);
    freqs = (0:L-1) * fs / L;
    
    % 低通滤波
    mask = freqs <= cutoff | freqs >= (fs - cutoff);
    X_low = X .* mask;
    
    % IFFT
    x_low = real(ifft(X_low));
end


function x_n = mad_normalize(x)
% MAD归一化
    eps_val = 1e-8;
    
    med = median(x);
    mad_val = median(abs(x - med));
    mad_val = max(mad_val, eps_val);
    
    x_n = (x - med) / mad_val;
end


function save_figure(fig, filename, config)
% 保存图像
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
