function vis_denoising_results(sample_idx, channel_idx, config)
% VIS_DENOISING_RESULTS 可视化去噪效果
%
% 展示典型样本的处理结果:
% - 原始受污染信号
% - 去噪后的干净信号
% - 提取的伪影信号
% - 评价指标(由于真实数据集没有ground truth,使用其他指标)
% - 频谱分析对比
%
% 输入:
%   sample_idx - 样本索引 (可选)
%   channel_idx - 通道索引 (可选)
%   config - 配置结构体 (可选)
%
% 用法:
%   vis_denoising_results();
%   vis_denoising_results(5, 10);

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
    fprintf('去噪效果可视化\n');
    fprintf('========================================\n');
    fprintf('样本: %d, 通道: %d\n', sample_idx, channel_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    data = load_real_dataset(sample_idx, channel_idx, config);
    
    %% 提取信号
    contaminated = data.contaminated;
    clean_pred = data.clean_pred;
    artifact_pred = data.artifact_pred;
    fs = data.fs;
    time = data.time;
    
    %% 计算评价指标
    fprintf('正在计算评价指标...\n');
    
    % 由于真实数据集没有ground truth, 我们使用以下指标:
    % 1. 信号能量减少比例
    % 2. 高频成分减少
    % 3. 伪影能量占比
    % 4. 信号平滑度改善
    
    metrics = struct();
    
    % 能量指标
    energy_contaminated = sum(contaminated.^2);
    energy_clean = sum(clean_pred.^2);
    energy_artifact = sum(artifact_pred.^2);
    metrics.energy_reduction = (energy_contaminated - energy_clean) / energy_contaminated * 100;
    metrics.artifact_ratio = energy_artifact / energy_contaminated * 100;
    
    % 高频能量 (>30 Hz)
    [freq_cont, psd_cont] = compute_psd(contaminated, fs);
    [freq_clean, psd_clean] = compute_psd(clean_pred, fs);
    high_freq_mask = freq_cont > 30;
    hf_energy_cont = sum(psd_cont(high_freq_mask));
    hf_energy_clean = sum(psd_clean(high_freq_mask));
    metrics.high_freq_reduction = (hf_energy_cont - hf_energy_clean) / hf_energy_cont * 100;
    
    % 信号平滑度 (通过一阶差分的标准差衡量)
    smoothness_cont = std(diff(contaminated));
    smoothness_clean = std(diff(clean_pred));
    metrics.smoothness_improvement = (smoothness_cont - smoothness_clean) / smoothness_cont * 100;
    
    % RMS值
    metrics.rms_contaminated = sqrt(mean(contaminated.^2));
    metrics.rms_clean = sqrt(mean(clean_pred.^2));
    metrics.rms_artifact = sqrt(mean(artifact_pred.^2));
    
    % 峰值
    metrics.peak_contaminated = max(abs(contaminated));
    metrics.peak_clean = max(abs(clean_pred));
    metrics.peak_artifact = max(abs(artifact_pred));
    
    % 打印指标
    fprintf('\n去噪效果评估:\n');
    fprintf('  能量减少: %.2f%%\n', metrics.energy_reduction);
    fprintf('  伪影能量占比: %.2f%%\n', metrics.artifact_ratio);
    fprintf('  高频能量减少: %.2f%%\n', metrics.high_freq_reduction);
    fprintf('  平滑度改善: %.2f%%\n', metrics.smoothness_improvement);
    fprintf('\nRMS值:\n');
    fprintf('  受污染信号: %.4f μV\n', metrics.rms_contaminated);
    fprintf('  干净信号: %.4f μV\n', metrics.rms_clean);
    fprintf('  伪影信号: %.4f μV\n', metrics.rms_artifact);
    fprintf('\n峰值:\n');
    fprintf('  受污染信号: %.4f μV\n', metrics.peak_contaminated);
    fprintf('  干净信号: %.4f μV\n', metrics.peak_clean);
    fprintf('  伪影信号: %.4f μV\n', metrics.peak_artifact);
    
    %% 创建可视化
    fprintf('\n正在生成可视化...\n');
    
    fig = figure('Position', [100, 100, 1800, 1400], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('去噪效果 - 样本%d', sample_idx), 'NumberTitle', 'off');
    
    % 颜色配置
    col_contaminated = config.VIS.colors.original;
    col_clean = config.VIS.colors.clean;
    col_artifact = config.VIS.colors.artifact;
    
    %% 第1行: 原始受污染信号
    subplot(5, 2, [1, 2]);
    plot(time, contaminated, 'Color', col_contaminated, 'LineWidth', 1.5);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('原始受污染的 EEG 信号', 'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    grid on; grid minor;
    xlim([time(1), time(end)]);
    
    %% 第2行: 去噪后的干净信号
    subplot(5, 2, [3, 4]);
    plot(time, clean_pred, 'Color', col_clean, 'LineWidth', 1.5);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('去噪后的干净 EEG 信号', 'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    grid on; grid minor;
    xlim([time(1), time(end)]);
    
    %% 第3行: 提取的伪影信号
    subplot(5, 2, [5, 6]);
    plot(time, artifact_pred, 'Color', col_artifact, 'LineWidth', 1.5);
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('提取的 EOG 伪影信号', 'FontSize', config.VIS.title_size, 'FontWeight', 'bold');
    grid on; grid minor;
    xlim([time(1), time(end)]);
    
    %% 第4行左: 信号对比 (局部放大)
    subplot(5, 2, 7);
    % 选择中间部分进行放大
    window_size = min(fs * 3, floor(length(contaminated)/3));  % 3秒窗口
    start_idx = floor((length(contaminated) - window_size) / 2);
    end_idx = start_idx + window_size - 1;
    zoom_indices = start_idx:end_idx;
    time_zoom = time(zoom_indices);
    
    plot(time_zoom, contaminated(zoom_indices), 'Color', col_contaminated, ...
         'LineWidth', 1.5, 'DisplayName', '受污染');
    hold on;
    plot(time_zoom, clean_pred(zoom_indices), 'Color', col_clean, ...
         'LineWidth', 1.5, 'DisplayName', '去噪后');
    plot(time_zoom, artifact_pred(zoom_indices), 'Color', col_artifact, ...
         'LineWidth', 1.5, 'DisplayName', 'EOG伪影');
    xlabel('时间 (秒)', 'FontSize', config.VIS.font_size);
    ylabel('幅值 (μV)', 'FontSize', config.VIS.font_size);
    title('信号对比 (局部放大)', 'FontSize', 11, 'FontWeight', 'bold');
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第4行右: 频谱对比
    subplot(5, 2, 8);
    plot(freq_cont, 10*log10(psd_cont), 'Color', col_contaminated, ...
         'LineWidth', 1.5, 'DisplayName', '受污染');
    hold on;
    plot(freq_clean, 10*log10(psd_clean), 'Color', col_clean, ...
         'LineWidth', 1.5, 'DisplayName', '去噪后');
    xlabel('频率 (Hz)', 'FontSize', config.VIS.font_size);
    ylabel('功率谱密度 (dB/Hz)', 'FontSize', config.VIS.font_size);
    title('频谱对比', 'FontSize', 11, 'FontWeight', 'bold');
    xlim([0, min(50, fs/2)]);  % 显示0-50Hz
    legend('Location', 'best');
    grid on; grid minor;
    
    %% 第5行左: 评价指标柱状图
    subplot(5, 2, 9);
    categories = {'能量减少', '高频减少', '平滑度改善'};
    values = [metrics.energy_reduction, metrics.high_freq_reduction, metrics.smoothness_improvement];
    
    bar(values, 'FaceColor', [0.3 0.6 0.8]);
    set(gca, 'XTickLabel', categories);
    ylabel('改善百分比 (%)', 'FontSize', config.VIS.font_size);
    title('去噪效果指标', 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    % 添加数值标签
    for i = 1:length(values)
        text(i, values(i) + max(values)*0.02, sprintf('%.1f%%', values(i)), ...
             'HorizontalAlignment', 'center', 'FontSize', 9, 'FontWeight', 'bold');
    end
    
    %% 第5行右: 信号能量分布
    subplot(5, 2, 10);
    labels = {'受污染', '干净信号', 'EOG伪影'};
    energies = [energy_contaminated, energy_clean, energy_artifact];
    
    pie(energies, labels);
    title('信号能量分布', 'FontSize', 11, 'FontWeight', 'bold');
    colormap([col_contaminated; col_clean; col_artifact]);
    
    %% 添加总标题和文本信息
    sgtitle(sprintf('去噪效果可视化 - 样本 %d, 通道 %d', sample_idx, channel_idx), ...
            'FontSize', 16, 'FontWeight', 'bold');
    
    %% 添加文本说明框
    annotation('textbox', [0.02, 0.02, 0.2, 0.12], ...
               'String', sprintf(['去噪统计:\n' ...
                                 '能量减少: %.1f%%\n' ...
                                 '伪影占比: %.1f%%\n' ...
                                 '高频减少: %.1f%%\n' ...
                                 '平滑改善: %.1f%%'], ...
                                metrics.energy_reduction, ...
                                metrics.artifact_ratio, ...
                                metrics.high_freq_reduction, ...
                                metrics.smoothness_improvement), ...
               'FontSize', 9, 'BackgroundColor', 'w', ...
               'EdgeColor', [0.3 0.3 0.3], 'LineWidth', 1);
    
    %% 显示图像
    drawnow;  % 强制刷新显示
    figure(fig);  % 将figure带到前台
    fprintf('图像已显示\n');
    
    % 如果需要保存，可以手动保存或修改config
    if config.EXPORT.save_png || config.EXPORT.save_pdf || config.EXPORT.save_fig
        fprintf('正在保存图像...\n');
        save_figure(fig, sprintf('denoising_results_sample%d_ch%d', sample_idx, channel_idx), config);
    end
    
    %% 保存数据
    if config.EXPORT.save_data
        output_data = struct();
        output_data.sample_idx = sample_idx;
        output_data.channel_idx = channel_idx;
        output_data.contaminated = contaminated;
        output_data.clean_pred = clean_pred;
        output_data.artifact_pred = artifact_pred;
        output_data.metrics = metrics;
        output_data.time = time;
        output_data.fs = fs;
        
        data_path = fullfile(config.DATA_OUTPUT_DIR, ...
                            sprintf('denoising_data_sample%d_ch%d.mat', sample_idx, channel_idx));
        save(data_path, '-struct', 'output_data');
        fprintf('已保存数据: %s\n', data_path);
    end
    
    fprintf('✓ 去噪效果可视化完成!\n\n');
end


%% ==================== 辅助函数 ====================

function [freq, psd] = compute_psd(signal, fs)
% 计算功率谱密度
%
% 输入:
%   signal - 输入信号
%   fs - 采样率
%
% 输出:
%   freq - 频率轴
%   psd - 功率谱密度

    % 使用Welch方法计算PSD
    window = hamming(min(512, length(signal)));
    noverlap = floor(length(window) / 2);
    nfft = max(512, 2^nextpow2(length(signal)));
    
    [psd, freq] = pwelch(signal, window, noverlap, nfft, fs);
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
