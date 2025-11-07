function metrics = compute_eog_metrics(true_signals, pred_signals, fs)
%COMPUTE_EOG_METRICS 计算EOG去除方法的评价指标
%
% 输入:
%   true_signals - 真实纯净EEG信号矩阵 (n_samples × seq_len)
%   pred_signals - 预测去噪信号矩阵 (n_samples × seq_len)
%   fs           - 采样率 (Hz), 默认200
%
% 输出:
%   metrics - 结构体,包含4个指标的均值和标准差:
%             - RRMSE_mean, RRMSE_std
%             - CC_mean, CC_std
%             - RRMSE_PSD_mean, RRMSE_PSD_std
%             - MI_mean, MI_std
%
% 作者: GitHub Copilot
% 日期: 2025-11-03

    if nargin < 3
        fs = 200;
    end
    
    num_samples = size(true_signals, 1);
    
    rrmse_list = zeros(num_samples, 1);
    cc_list = zeros(num_samples, 1);
    rrmse_psd_list = zeros(num_samples, 1);
    mi_list = zeros(num_samples, 1);
    
    fprintf('计算评价指标 (共%d个样本)...\n', num_samples);
    
    for i = 1:num_samples
        s_true = true_signals(i, :);
        s_pred = pred_signals(i, :);
        
        % 1. RRMSE
        rrmse_list(i) = compute_rrmse(s_true, s_pred);
        
        % 2. CC
        cc_list(i) = compute_cc(s_true, s_pred);
        
        % 3. RRMSE_PSD
        rrmse_psd_list(i) = compute_rrmse_psd(s_true, s_pred, fs);
        
        % 4. MI
        mi_list(i) = compute_mi(s_true, s_pred);
        
        if mod(i, 20) == 0
            fprintf('  已计算 %d/%d 样本\n', i, num_samples);
        end
    end
    
    % 计算均值和标准差
    metrics.RRMSE_mean = mean(rrmse_list);
    metrics.RRMSE_std = std(rrmse_list);
    metrics.CC_mean = mean(cc_list);
    metrics.CC_std = std(cc_list);
    metrics.RRMSE_PSD_mean = mean(rrmse_psd_list);
    metrics.RRMSE_PSD_std = std(rrmse_psd_list);
    metrics.MI_mean = mean(mi_list);
    metrics.MI_std = std(mi_list);
    
    % 保存原始列表
    metrics.RRMSE_list = rrmse_list;
    metrics.CC_list = cc_list;
    metrics.RRMSE_PSD_list = rrmse_psd_list;
    metrics.MI_list = mi_list;
end

%% ========== 子函数 ==========

function rrmse_val = compute_rrmse(s_true, s_pred)
    % RRMSE = RMS(s_true - s_pred) / RMS(s_true)
    numerator = sqrt(mean((s_true - s_pred).^2));
    denominator = sqrt(mean(s_true.^2));
    
    if denominator == 0
        rrmse_val = inf;
    else
        rrmse_val = numerator / denominator;
    end
end

function cc_val = compute_cc(s_true, s_pred)
    % CC = corr(s_true, s_pred)
    cc_val = corr(s_true(:), s_pred(:));
    
    if isnan(cc_val)
        cc_val = 0;
    end
end

function rrmse_psd_val = compute_rrmse_psd(s_true, s_pred, fs)
    % RRMSE_PSD = RMS(PSD_true - PSD_pred) / RMS(PSD_true)
    
    % 使用pwelch计算功率谱密度
    nperseg = min(256, length(s_true));
    [psd_true, ~] = pwelch(s_true, hamming(nperseg), [], [], fs);
    [psd_pred, ~] = pwelch(s_pred, hamming(nperseg), [], [], fs);
    
    numerator = sqrt(mean((psd_true - psd_pred).^2));
    denominator = sqrt(mean(psd_true.^2));
    
    if denominator == 0
        rrmse_psd_val = inf;
    else
        rrmse_psd_val = numerator / denominator;
    end
end

function mi_val = compute_mi(s_true, s_pred)
    % 互信息 (使用离散化方法)
    
    bins = 50;
    
    % 处理特殊情况
    if all(s_true == s_true(1)) || all(s_pred == s_pred(1))
        mi_val = 0;
        return;
    end
    
    % 离散化
    edges_true = linspace(min(s_true), max(s_true), bins+1);
    edges_pred = linspace(min(s_pred), max(s_pred), bins+1);
    
    s_true_binned = discretize(s_true, edges_true);
    s_pred_binned = discretize(s_pred, edges_pred);
    
    % 计算联合概率分布
    joint_hist = zeros(bins, bins);
    valid_count = 0;
    
    for i = 1:length(s_true_binned)
        if ~isnan(s_true_binned(i)) && ~isnan(s_pred_binned(i))
            joint_hist(s_true_binned(i), s_pred_binned(i)) = ...
                joint_hist(s_true_binned(i), s_pred_binned(i)) + 1;
            valid_count = valid_count + 1;
        end
    end
    
    if valid_count == 0
        mi_val = 0;
        return;
    end
    
    joint_prob = joint_hist / valid_count;
    
    % 边缘概率
    prob_true = sum(joint_prob, 2);
    prob_pred = sum(joint_prob, 1);
    
    % 计算互信息
    mi_val = 0;
    for i = 1:bins
        for j = 1:bins
            if joint_prob(i,j) > 0
                mi_val = mi_val + joint_prob(i,j) * ...
                    log(joint_prob(i,j) / (prob_true(i) * prob_pred(j) + eps));
            end
        end
    end
    
    % 确保非负
    mi_val = max(0, mi_val);
end
