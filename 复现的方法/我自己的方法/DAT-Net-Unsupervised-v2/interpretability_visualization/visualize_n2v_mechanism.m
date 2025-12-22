function visualize_n2v_mechanism(sample_idx, config)
% VISUALIZE_N2V_MECHANISM 可视化N2V掩蔽机制的细节
%
% 展示为什么掩蔽后信号"看起来几乎一样"是合理的
%
% 输入:
%   sample_idx - 样本索引 (可选, 默认为1)
%   config - 配置结构体 (可选)

    if nargin < 1 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 2 || isempty(config)
        config = config_visualization();
    end
    
    fprintf('\n========================================\n');
    fprintf('N2V掩蔽机制详细可视化\n');
    fprintf('样本: %d\n', sample_idx);
    fprintf('========================================\n\n');
    
    %% 加载数据
    channel_idx = 1;
    data = load_real_dataset(sample_idx, channel_idx, config);
    
    x = data.contaminated;
    fs = data.fs;
    mask_base = config.LOSS.mask_base;
    boost_scale = config.LOSS.boost_scale;
    neighborhood = config.LOSS.mask_neighborhood;
    
    %% 计算掩蔽
    p_art = compute_artifact_prob(x, fs);
    [x_masked, mask, p_mask] = generate_artifact_aware_mask(...
        x, p_art, mask_base, boost_scale, neighborhood);
    
    time = data.time;
    
    %% 找一个有代表性的掩蔽区域
    mask_indices = find(mask > 0);
    if isempty(mask_indices)
        error('没有找到掩蔽点，请调整mask_base或boost_scale参数');
    end
    
    % 选择中间的掩蔽点
    center_idx = mask_indices(round(length(mask_indices)/2));
    
    % 局部窗口：前后50个点
    window_start = max(1, center_idx - 50);
    window_end = min(length(x), center_idx + 50);
    window_range = window_start:window_end;
    
    %% 创建可视化
    fig = figure('Position', [100, 100, 1600, 1000], 'Color', 'w', 'Visible', 'on');
    set(fig, 'Name', sprintf('N2V掩蔽机制详解-样本%d', sample_idx), 'NumberTitle', 'off');
    
    %% 子图1: 全局视图
    subplot(3, 2, [1, 2]);
    plot(time, x, 'b-', 'LineWidth', 1.5, 'DisplayName', '原始信号');
    hold on;
    plot(time, x_masked, 'r-', 'LineWidth', 1.5, 'DisplayName', '掩蔽后信号');
    
    % 标记选定的局部区域
    xregion([time(window_start), time(window_end)], 'FaceColor', 'y', ...
            'FaceAlpha', 0.2, 'DisplayName', '局部放大区域');
    
    xlabel('时间 (秒)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 16, 'FontWeight', 'bold');
    title('全局视图: 掩蔽后信号看起来几乎相同', 'FontSize', 18, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 14);
    grid on;
    set(gca, 'FontSize', 14);
    
    %% 子图2: 局部放大 - 显示掩蔽细节
    subplot(3, 2, 3);
    time_local = time(window_range);
    x_local = x(window_range);
    x_masked_local = x_masked(window_range);
    mask_local = mask(window_range);
    
    plot(time_local, x_local, 'b-o', 'LineWidth', 2, 'MarkerSize', 4, ...
         'DisplayName', '原始信号');
    hold on;
    plot(time_local, x_masked_local, 'r-s', 'LineWidth', 2, 'MarkerSize', 4, ...
         'DisplayName', '掩蔽后信号');
    
    % 高亮掩蔽点
    mask_local_idx = find(mask_local > 0);
    if ~isempty(mask_local_idx)
        scatter(time_local(mask_local_idx), x_local(mask_local_idx), 100, 'g', ...
                'filled', 'DisplayName', '掩蔽位置(原始值)', 'MarkerEdgeColor', 'k');
        scatter(time_local(mask_local_idx), x_masked_local(mask_local_idx), 100, 'm', ...
                'filled', 'DisplayName', '掩蔽位置(替换值)', 'MarkerEdgeColor', 'k');
        
        % 用箭头连接原始值和替换值
        for i = 1:length(mask_local_idx)
            idx = mask_local_idx(i);
            quiver(time_local(idx), x_local(idx), 0, ...
                   x_masked_local(idx) - x_local(idx), 0, ...
                   'Color', 'k', 'LineWidth', 1.5, 'MaxHeadSize', 0.5);
        end
    end
    
    xlabel('时间 (秒)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 16, 'FontWeight', 'bold');
    title('局部放大: 邻域采样导致值接近但不相同', 'FontSize', 18, 'FontWeight', 'bold');
    legend('Location', 'best', 'FontSize', 12);
    grid on;
    set(gca, 'FontSize', 14);
    
    %% 子图3: 掩蔽差异分布
    subplot(3, 2, 4);
    diff_all = abs(x - x_masked);
    diff_masked = diff_all(mask > 0);
    
    histogram(diff_masked, 30, 'FaceColor', 'r', 'EdgeColor', 'k');
    xlabel('掩蔽点的差异 (μV)', 'FontSize', 16, 'FontWeight', 'bold');
    ylabel('频数', 'FontSize', 16, 'FontWeight', 'bold');
    title(sprintf('掩蔽点差异分布 (平均: %.4f μV)', mean(diff_masked)), ...
          'FontSize', 18, 'FontWeight', 'bold');
    grid on;
    set(gca, 'FontSize', 14);
    
    %% 子图4: N2V原理示意图
    subplot(3, 2, 5);
    
    % 选择一个具体的掩蔽点进行详细说明
    if ~isempty(mask_indices)
        demo_idx = mask_indices(1);
        demo_neighborhood = max(1, demo_idx-10):min(length(x), demo_idx+10);
        
        stem(demo_neighborhood - demo_idx, x(demo_neighborhood), 'b', 'LineWidth', 2, ...
             'DisplayName', '邻域点值');
        hold on;
        plot(0, x(demo_idx), 'go', 'MarkerSize', 15, 'LineWidth', 3, ...
             'DisplayName', sprintf('原始值=%.3f', x(demo_idx)));
        plot(0, x_masked(demo_idx), 'ms', 'MarkerSize', 15, 'LineWidth', 3, ...
             'DisplayName', sprintf('替换值=%.3f', x_masked(demo_idx)));
        
        xlabel('相对位置', 'FontSize', 16, 'FontWeight', 'bold');
        ylabel('幅值 (μV)', 'FontSize', 16, 'FontWeight', 'bold');
        title(sprintf('N2V原理: 位置%d用邻域±%d范围随机采样', demo_idx, neighborhood), ...
              'FontSize', 18, 'FontWeight', 'bold');
        legend('Location', 'best', 'FontSize', 12);
        grid on;
        set(gca, 'FontSize', 14);
    end
    
    %% 子图5: 为什么"几乎一样"是合理的
    subplot(3, 2, 6);
    axis off;
    
    text_content = {
        '\bf为什么掩蔽后信号"几乎一样"是合理的：'
        ''
        '\bf1. N2V的目的：'
        '   • 不是让信号看起来"很不同"'
        '   • 而是防止信息泄露（盲点机制）'
        ''
        '\bf2. 邻域采样的特点：'
        sprintf('   • neighborhood=%d: 用附近±%d点的随机值替换', neighborhood, neighborhood)
        '   • 对于平滑信号，相邻点值本来就接近'
        '   • 所以替换后差异小是正常的'
        ''
        '\bf3. 关键在于"盲点"：'
        '   • 模型预测位置i时，看不到i的真实值'
        '   • 只能看到邻域采样值（来自i±offset）'
        '   • 损失 = (预测值 - 原始值)[掩蔽位置]'
        ''
        '\bf4. 统计信息：'
        sprintf('   • 总点数: %d', length(x))
        sprintf('   • 掩蔽点数: %d (%.2f%%)', sum(mask>0), sum(mask>0)/length(x)*100)
        sprintf('   • 平均差异: %.4f μV', mean(diff_all(mask>0)))
        sprintf('   • 最大差异: %.4f μV', max(diff_all(mask>0)))
    };
    
    text(0.05, 0.95, text_content, 'FontSize', 13, ...
         'VerticalAlignment', 'top', 'Interpreter', 'tex');
    
    %% 总标题
    sgtitle(sprintf('N2V掩蔽机制详解 (样本 %d) - "几乎一样"是预期行为', sample_idx), ...
            'FontSize', 20, 'FontWeight', 'bold');
    
    drawnow;
    figure(fig);
    
    fprintf('\n✓ N2V机制可视化完成!\n');
    fprintf('关键要点: 掩蔽后信号看起来"几乎一样"是N2V的正常行为\n');
    fprintf('重要的是"盲点"机制，而不是信号的视觉差异\n\n');
end


%% ==================== 辅助函数 ====================

function p = compute_artifact_prob(x, fs)
    win_size = 64;
    lowpass_cutoff = 4.0;
    eps_val = 1e-8;
    
    x = x(:)';
    
    amp = moving_average(abs(x), win_size);
    diff_sig = [abs(diff(x)), 0];
    diff_sig = moving_average(diff_sig, win_size);
    
    x_low = fft_lowpass(x, fs, lowpass_cutoff);
    power_low = moving_average(x_low .^ 2, win_size);
    power_total = moving_average(x .^ 2, win_size);
    r = power_low ./ (power_total + eps_val);
    
    amp_n = mad_normalize(amp);
    diff_n = mad_normalize(diff_sig);
    r_n = mad_normalize(r);
    
    s = amp_n + diff_n + r_n;
    tau = quantile(s, 0.7);
    alpha = 10.0;
    p = 1 ./ (1 + exp(-alpha * (s - tau)));
    p = min(max(p, 0), 1);
end


function [x_masked, mask, p_mask] = generate_artifact_aware_mask(x, p_art, mask_base, boost_scale, neighborhood)
    x = x(:)';
    L = length(x);
    
    p_mask = mask_base + boost_scale * p_art;
    p_mask = min(p_mask, 1.0);
    
    mask = double(rand(1, L) < p_mask);
    
    if neighborhood <= 0
        x_masked = x;
        x_masked(mask > 0) = 0;
        return;
    end
    
    offsets = randi([-neighborhood, neighborhood], 1, L);
    offsets(offsets == 0) = 1;
    
    base_idx = 1:L;
    gather_idx = base_idx + offsets;
    gather_idx = max(1, min(gather_idx, L));
    
    gathered = x(gather_idx);
    
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
