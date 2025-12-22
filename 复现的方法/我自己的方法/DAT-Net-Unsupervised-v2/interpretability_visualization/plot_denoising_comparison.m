function plot_denoising_comparison(sample_idx, config)
% PLOT_DENOISING_COMPARISON 去噪结果对比图（用于论文）
%
% 展示内容：
% 1. 预处理信号 X_process
% 2. 纯净脑电分量 Y_clean
% 3. 眼电伪影分量 Y_artifact
%
% 输入:
%   sample_idx - 样本索引 (可选, 默认为1)
%   config - 配置结构体 (可选)
%
% 用法:
%   plot_denoising_comparison(1);  % 样本1

    % 默认参数
    if nargin < 1 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 2 || isempty(config)
        config = config_visualization();
    end
    
    fprintf('\n========================================\n');
    fprintf('论文图: 去噪结果对比\n');
    fprintf('样本: %d\n', sample_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    channel_idx = 1;  % 单通道固定为1
    data = load_real_dataset(sample_idx, channel_idx, config);
    
    %% 提取信号
    X_process = data.contaminated;      % 预处理信号
    Y_clean = data.clean_pred;          % 纯净脑电分量
    Y_artifact = data.artifact_pred;    % 眼电伪影分量
    fs = data.fs;
    time = data.time;
    
    %% 创建论文图像
    fprintf('正在生成论文图像...\n');
    
    % 创建高质量figure (适合论文)
    fig = figure('Position', [100, 100, 1400, 900], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('论文图-去噪结果对比-样本%d', sample_idx), 'NumberTitle', 'off');
    
    % 颜色配置
    col_process = config.VIS.colors.original;
    col_clean = config.VIS.colors.clean;
    col_artifact = config.VIS.colors.artifact;
    
    % 计算统一的Y轴范围（以预处理信号为准）
    y_min = min(X_process) * 1.1;
    y_max = max(X_process) * 1.1;
    
    %% 子图1: 预处理信号 X_process
    subplot(3, 1, 1);
    plot(time, X_process, 'k-', 'LineWidth', 1.5);
    ylabel('幅值 (μV)', 'FontSize', 18, 'FontWeight', 'bold');
    title('(a) 预处理信号 X_{process} (含眼电伪影)', ...
          'FontSize', 20, 'FontWeight', 'bold');
    grid on;
    xlim([time(1), time(end)]);
    ylim([y_min, y_max]);
    set(gca, 'FontSize', 16);
    ax = gca;
    ax.YAxis.TickLabelGapOffset = 2;
    
    % 添加统计信息
    text(0.02, 0.95, sprintf('RMS: %.4f μV', rms(X_process)), ...
         'Units', 'normalized', 'FontSize', 15, 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'w', 'EdgeColor', 'k');
    
    %% 子图2: 纯净脑电分量 Y_clean
    subplot(3, 1, 2);
    plot(time, Y_clean, 'Color', [0.3 0.3 0.3], 'LineWidth', 1.5);
    ylabel('幅值 (μV)', 'FontSize', 18, 'FontWeight', 'bold');
    title('(b) 纯净脑电分量 Y_{clean} (去噪后)', ...
          'FontSize', 20, 'FontWeight', 'bold');
    grid on;
    xlim([time(1), time(end)]);
    ylim([y_min, y_max]);
    set(gca, 'FontSize', 16);
    ax = gca;
    ax.YAxis.TickLabelGapOffset = 2;
    
    % 添加统计信息
    text(0.02, 0.95, sprintf('RMS: %.4f μV', rms(Y_clean)), ...
         'Units', 'normalized', 'FontSize', 15, 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'w', 'EdgeColor', 'k');
    
    %% 子图3: 眼电伪影分量 Y_artifact
    subplot(3, 1, 3);
    plot(time, Y_artifact, 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5);
    xlabel('时间 (秒)', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 18, 'FontWeight', 'bold');
    title('(c) 提取的眼电伪影分量 Y_{artifact}', ...
          'FontSize', 20, 'FontWeight', 'bold');
    grid on;
    xlim([time(1), time(end)]);
    ylim([y_min, y_max]);
    set(gca, 'FontSize', 16);
    ax = gca;
    ax.YAxis.TickLabelGapOffset = 2;
    
    % 添加统计信息
    text(0.02, 0.95, sprintf('RMS: %.4f μV', rms(Y_artifact)), ...
         'Units', 'normalized', 'FontSize', 15, 'VerticalAlignment', 'top', ...
         'BackgroundColor', 'w', 'EdgeColor', 'k');
    
    %% 添加总标题
    sgtitle(sprintf('去噪结果对比 (样本 %d)', sample_idx), ...
            'FontSize', 22, 'FontWeight', 'bold');
    
    %% 显示图像
    drawnow;
    figure(fig);
    fprintf('✓ 论文图像已显示!\n\n');
    
    %% 打印质量评估
    fprintf('信号质量评估:\n');
    fprintf('  预处理信号 RMS: %.4f μV\n', rms(X_process));
    fprintf('  纯净脑电 RMS: %.4f μV\n', rms(Y_clean));
    fprintf('  伪影分量 RMS: %.4f μV\n', rms(Y_artifact));
    fprintf('  能量保持率: %.2f%%\n', (sum(Y_clean.^2) / sum(X_process.^2)) * 100);
    fprintf('  伪影占比: %.2f%%\n', (sum(Y_artifact.^2) / sum(X_process.^2)) * 100);
    fprintf('\n');
    
    fprintf('提示: 如需保存图像，在Figure窗口点击"文件→另存为"\n');
    fprintf('推荐格式: EPS (矢量图) 或 PNG (高分辨率)\n\n');
end
