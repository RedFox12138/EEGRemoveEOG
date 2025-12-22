function plot_masking_visualization(sample_idx, config)
% PLOT_MASKING_VISUALIZATION 伪影先验概率估计示意图（用于论文）
%
% 展示内容：
% 1. 掩蔽可视化
% 2. 掩蔽后的信号图
%
% 输入:
%   sample_idx - 样本索引 (可选, 默认为1)
%   config - 配置结构体 (可选)
%
% 用法:
%   plot_masking_visualization(1);  % 样本1

    % 默认参数
    if nargin < 1 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 2 || isempty(config)
        config = config_visualization();
    end
    
    fprintf('\n========================================\n');
    fprintf('论文图: 伪影先验概率估计示意图\n');
    fprintf('样本: %d\n', sample_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    channel_idx = 1;  % 单通道固定为1
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
    
    %% 调试信息：检查掩蔽效果
    num_masked = sum(mask_artifact > 0);
    mask_ratio = num_masked / length(x) * 100;
    diff_before_after = abs(x - x_masked_artifact);
    max_diff = max(diff_before_after);
    mean_diff = mean(diff_before_after(mask_artifact > 0));
    
    fprintf('掩蔽统计信息:\n');
    fprintf('  掩蔽点数: %d / %d (%.2f%%)\n', num_masked, length(x), mask_ratio);
    fprintf('  最大差异: %.4f μV\n', max_diff);
    if num_masked > 0
        fprintf('  掩蔽点平均差异: %.4f μV\n', mean_diff);
    end
    fprintf('\n');
    
    %% 创建论文图像
    fprintf('正在生成论文图像...\n');
    
    % 创建高质量figure (适合论文)
    fig = figure('Position', [100, 100, 1400, 800], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('论文图-伪影先验概率估计-样本%d', sample_idx), 'NumberTitle', 'off');
    
    % 颜色配置
    col_original = config.VIS.colors.original;
    col_artifact = config.VIS.colors.artifact;
    col_masked = config.VIS.colors.masked;
    
    time = data.time;
    
    %% 子图1: 原始信号和伪影先验概率
    subplot(2, 1, 1);
    yyaxis left
    plot(time, x, 'k-', 'LineWidth', 2);
    ylabel('幅值 (μV)', 'FontSize', 18, 'FontWeight', 'bold');
    ylim([min(x)*1.1, max(x)*1.1]);
    
    yyaxis right
    area(time, p_art, 'FaceColor', [0.7 0.7 0.7], 'FaceAlpha', 0.5, ...
         'EdgeColor', [0.5 0.5 0.5], 'LineWidth', 1.5);
    ylabel('伪影概率', 'FontSize', 18, 'FontWeight', 'bold');
    ylim([0, 1]);
    
    title('(a) 原始信号和伪影先验概率', ...
          'FontSize', 20, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 16);
    ax = gca;
    ax.YAxis(1).Color = 'k';
    ax.YAxis(2).Color = [0.5 0.5 0.5];
    ax.YAxis(1).TickLabelGapOffset = 2;
    ax.YAxis(2).TickLabelGapOffset = 2;
    
    %% 子图2: 掩蔽位置和掩蔽后信号对比
    subplot(2, 1, 2);
    
    plot(time, x, 'k-', 'LineWidth', 2, ...
         'DisplayName', '原始信号');
    hold on;
    plot(time, x_masked_artifact, '-', 'Color', [0.5 0.5 0.5], 'LineWidth', 2.5, ...
         'DisplayName', '掩蔽后信号');
    
    xlabel('时间 (秒)', 'FontSize', 18, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 18, 'FontWeight', 'bold');
    title(sprintf('(c) 掩蔽效果 (掩蔽率: %.2f%%, 掩蔽点数: %d)', mask_ratio, num_masked), ...
          'FontSize', 20, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 15);
    grid on;
    set(gca, 'FontSize', 16);
    ax = gca;
    ax.YAxis.TickLabelGapOffset = 2;
    
    %% 添加总标题
    sgtitle(sprintf('伪影先验概率估计和掩蔽策略 (样本 %d)', sample_idx), ...
            'FontSize', 22, 'FontWeight', 'bold');
    
    %% 显示图像
    drawnow;
    figure(fig);
    fprintf('✓ 论文图像已显示!\n\n');
    
    fprintf('提示: 如需保存图像，在Figure窗口点击"文件→另存为"\n');
    fprintf('推荐格式: EPS (矢量图) 或 PNG (高分辨率)\n\n');
end


%% ==================== 辅助函数 ====================

function p = compute_artifact_prob(x, fs)
% 计算伪影概率
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
    
    % 收集邻域值
    gathered = x(gather_idx);
    
    % 应用掩蔽：被掩蔽的点用邻域随机采样值替换
    x_masked = x;
    x_masked(mask > 0) = gathered(mask > 0);
end


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
