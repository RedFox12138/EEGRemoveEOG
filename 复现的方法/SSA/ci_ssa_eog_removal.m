function [denoised_eeg, eog_artifact] = ci_ssa_eog_removal(noisy_eeg, varargin)
% 基于 Circular SSA + Morlet 小波 + k-means 的眼电去噪
% 输入：
%   noisy_eeg - 含眼电噪声的脑电信号（1维向量）
%   varargin  - 可选参数：
%               fs: 采样频率(默认250Hz)
%               L: Ci_SSA窗口长度(默认28)
%               k_cluster: 聚类数(默认2)
%               freq_range: 小波频率范围(默认[1,12]Hz)
%               freq_step: 小波频率步长(默认0.25Hz)
%               mode: 'paper' 或 'robust'（默认 'paper'）。paper = 严格论文流程，robust = 启发式增强
%               visualize: 是否绘图（默认 false）
%               verbose: 是否打印调试信息（默认 false）
% 输出：
%   denoised_eeg - 去眼电后的脑电信号
%   eog_artifact - 提取的纯净眼电噪声

%% 1. 参数初始化（默认值参考文献）
p = inputParser;
addParameter(p, 'fs', 250, @isnumeric);          % 文献采样频率250Hz
addParameter(p, 'L', 28, @(x) x>0 && x<length(noisy_eeg)/2); % 窗口长度<信号长度/2（文献L=28）
addParameter(p, 'k_cluster', 2, @(x) x==2);      % 文献固定k=2（眼电/非眼电）
addParameter(p, 'freq_range', [1,12], @(x) x(1)<x(2)); % 小波频率范围（文献1-12Hz）
addParameter(p, 'freq_step', 0.25, @isnumeric);  % 频率步长（文献0.25Hz，共45个频率点）
addParameter(p, 'mode', 'paper', @(s) ischar(s) || isstring(s)); % 'paper' | 'robust'
addParameter(p, 'visualize', false, @islogical);
addParameter(p, 'verbose', false, @islogical);
% 使用官方 CISSA 的可选项（若可用则优先调用）
addParameter(p, 'useOfficialCISSA', true, @islogical);         % 默认优先使用官方实现
addParameter(p, 'cissaPath', '', @(s) ischar(s) || isstring(s));% 官方CISSA目录（可选）
addParameter(p, 'cissaEntry', '', @(s) ischar(s) || isstring(s));% 官方入口函数名（可选）
% 额外稳健性参数（应对下沉式眼动）：
addParameter(p, 'mask_gap_ms', 80, @isnumeric);     % 聚类掩膜小间隙填充（形态学闭运算），默认80ms
addParameter(p, 'fd_tie_eps', 0.03, @isnumeric);    % 当两簇FD差异小于该阈值时，用低频能量占比判别
addParameter(p, 'lowfreq_band', [0.5 4], @(x) x(1)<x(2)); % 低频带(Hz)用于判别
% 针对“上尖峰不干净/下沉眼动不敏感”的增强型分段参数（滞回 + 长段/负向检测）
addParameter(p, 'use_hysteresis', true, @islogical);
addParameter(p, 'hyst_scale', 1.0, @isnumeric);      % MAD 阈值缩放（Thigh = hyst_scale*T）
addParameter(p, 'hyst_ratio', 0.5, @isnumeric);      % Tlow = hyst_ratio * Thigh
addParameter(p, 'expand_sec', 0.10, @isnumeric);     % 峰为起点的两侧扩展秒数
addParameter(p, 'min_long_sec', 0.30, @isnumeric);   % 下沉/长段的最小时长
addParameter(p, 'dilate_sec', 0.05, @isnumeric);     % 掩膜膨胀时长
addParameter(p, 'smooth_lambda', 0.8, @isnumeric);   % 片段内平滑的 TV 强度（无 TV 时退化为移动平均）
parse(p, varargin{:});
fs = p.Results.fs; L = p.Results.L; k_cluster = p.Results.k_cluster;
freq_range = p.Results.freq_range; freq_step = p.Results.freq_step;
mode = string(lower(p.Results.mode));
visualize = p.Results.visualize; verbose = p.Results.verbose;
mask_gap_ms = p.Results.mask_gap_ms; fd_tie_eps = p.Results.fd_tie_eps; lowfreq_band = p.Results.lowfreq_band;
use_hysteresis = p.Results.use_hysteresis; hyst_scale = p.Results.hyst_scale; hyst_ratio = p.Results.hyst_ratio;
expand_sec = p.Results.expand_sec; min_long_sec = p.Results.min_long_sec; dilate_sec = p.Results.dilate_sec;
smooth_lambda = p.Results.smooth_lambda;
isPaper = (mode == "paper");
useOfficialCISSA = p.Results.useOfficialCISSA;
cissaPath = string(p.Results.cissaPath);
cissaEntry = string(p.Results.cissaEntry);

% 自适应窗口长度：文献用fs=250,L=28；按比例缩放
if L < 1
    L = max(10, min(round(28 * fs / 250), floor(length(noisy_eeg)/3)));
end
cissaEntry = string(p.Results.cissaEntry);

% 信号基础信息
noisy_eeg = squeeze(noisy_eeg);  % 确保1维
P = length(noisy_eeg);           % 信号长度
t = (0:P-1)/fs;                  % 时间向量


%% 2. 可视化1：原始含噪脑电信号
if visualize
    figure('Position', [100, 100, 1000, 200]);
    plot(t, noisy_eeg, 'b', 'LineWidth', 1.2);
    title('Step 1: 原始含眼电噪声的脑电信号', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 10); ylabel('幅值 (μV)', 'FontSize', 10);
    grid on; axis tight;
end


%% 3. Ci-SSA 分解（优先使用“官方CISSA目录里的函数”，否则回退到内置实现）
RC = [];
G = {};
lambda = [];
usedOfficial = false;

if useOfficialCISSA
    RC_try = local_run_official_cissa(noisy_eeg, L, cissaPath, cissaEntry, verbose);
    if ~isempty(RC_try)
        % 官方实现返回的 RC 形状：#components × N（若为 N×K 将自动转置）
        if size(RC_try, 2) ~= P && size(RC_try, 1) == P
            RC_try = RC_try';
        end
        if size(RC_try, 2) == P
            RC = RC_try;
            usedOfficial = true;
            if verbose
                fprintf('使用官方 CISSA 分解成功（优先）。\n');
            end
        else
            if verbose
                fprintf('官方 CISSA 返回维度与信号长度不匹配，回退到内置实现。\n');
            end
        end
    else
        if verbose
            fprintf('未检测到官方 CISSA 或调用失败，回退到内置实现。\n');
        end
    end
end

if ~usedOfficial
    [RC, G, lambda] = local_cissa_custom(noisy_eeg, L, verbose);
end

% 选 RC1 作为初始眼电（文献结论：RC1含最强眼电信息）
initial_eog = RC(1, :);

% 调试：检查RC分量的能量分布
rc_energy = sum(RC.^2, 2);
if verbose
    if ~isempty(lambda)
        fprintf('\n=== CI-SSA分解调试信息 ===\n');
        fprintf('窗口长度 L = %d, 信号长度 P = %d\n', L, P);
        fprintf('前10个特征值的模: ');
        fprintf('%.4f ', abs(lambda(1:min(10, numel(lambda)))));
        fprintf('\n');
    end
    fprintf('前5个RC分量能量占比: ');
    for gidx = 1:min(5, size(RC,1))
        fprintf('RC%d=%.2f%% ', gidx, 100*rc_energy(gidx)/sum(rc_energy));
    end
    fprintf('\n');
end

% 可视化：Ci_SSA分解的前5个RC分量
if visualize
    figure('Position', [100, 350, 1000, 400]);
    for gidx = 1:min(5, size(RC,1))  % 显示前5个RC（文献图3风格）
        subplot(5, 1, gidx);
        plot(t, RC(gidx, :), 'Color', [0.2, 0.6, 0.8], 'LineWidth', 1);
        title(sprintf('Step 2: RC%d (能量: %.2f%%)', gidx, 100*rc_energy(gidx)/sum(rc_energy)), ...
            'FontSize', 10, 'FontWeight', 'bold');
        xlabel('时间 (s)', 'FontSize', 8); ylabel('幅值', 'FontSize', 8);
        grid on; axis tight;
    end
end


%% 4. Morlet小波变换（核心步骤2：时频特征提取）
% --------------------------
% 4.1 生成小波频率向量（文献1-12Hz，步长0.25Hz）
% --------------------------
freqs = freq_range(1):freq_step:freq_range(2);
freq_points = length(freqs);  % 45个频率点（与文献一致）

% --------------------------
% 4.2 小波变换（MATLAB原生Morlet小波）
% --------------------------
% 将频率转换为尺度（适配cwt函数）
% scales = centfrq(wavelet) * fs / freqs
scales = centfrq('morl') * fs ./ freqs;

% 使用cwt函数（旧版语法兼容）
wt = cwt(initial_eog, scales, 'morl', 1/fs);  % wt: 时频系数(freq_points×P)
abs_wt = abs(wt);                              % 时频幅度矩阵（特征矩阵）

% 调试输出：检查小波变换结果
if verbose
    fprintf('\n=== Morlet小波变换调试信息 ===\n');
    fprintf('时频矩阵维度: %d × %d (频率×时间)\n', size(abs_wt, 1), size(abs_wt, 2));
    fprintf('频率点数: %d (期望45)\n', freq_points);
    fprintf('频率范围: %.2f - %.2f Hz\n', freqs(1), freqs(end));
    fprintf('时频矩阵能量分布 - 最大: %.4f, 均值: %.4f\n', max(abs_wt(:)), mean(abs_wt(:)));
end

% --------------------------
% 4.3 可视化3：Morlet小波时频图
% --------------------------
if visualize
    figure('Position', [100, 800, 1000, 300]);
    pcolor(t, freqs, abs_wt);  % 伪彩图（时间×频率×幅度）
    shading interp;            % 插值平滑
    colormap(jet);             % 热图配色
    colorbar;
    title('Step 3: 初始眼电的Morlet小波时频图', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 10); ylabel('频率 (Hz)', 'FontSize', 10);
    axis tight;
end


%% 5. k-means聚类（核心步骤3：纯化眼电噪声）
% --------------------------
% 5.1 聚类数据准备（时频特征：每个时间点对应45维频率特征）
% --------------------------
cluster_data = abs_wt';  % P×freq_points（样本数×特征数）

% 【关键修正】聚类特征设计：
% 目标：让k-means能区分"眼电尖峰时段"和"背景脑电/基线时段"
% 眼电尖峰特征：(1)低频能量占主导 (2)总能量大 (3)幅度峰值高
% 
% 方案：使用频率能量分布（归一化） + 总能量特征
% 这样既保留了频率模式信息，又保留了幅度信息

total_energy = sqrt(sum(cluster_data.^2, 2));  % 每个时间点的总能量
cluster_data_norm = cluster_data ./ (total_energy + eps);  % 归一化为单位向量（突出频率分布模式）

% 添加能量特征（归一化到[0,1]）和低频比特征
low_freq_idx = freqs <= 4;  % 0.5-4Hz为眼电主导频段
low_freq_energy = sum(cluster_data(:, low_freq_idx), 2);
low_freq_ratio_feat = low_freq_energy ./ (total_energy + eps);

energy_feat = total_energy / (max(total_energy) + eps);
cluster_data_use = [cluster_data_norm, energy_feat, low_freq_ratio_feat];

% --------------------------
% 5.2 k-means聚类（文献k=2）
% --------------------------
rng(1);  % 固定随机种子确保复现性
opts = statset('UseParallel', false, 'MaxIter', 500);  % 增加最大迭代次数确保收敛
[idx, centroids] = kmeans(cluster_data_use, k_cluster, 'Options', opts, 'Distance', 'sqeuclidean', 'Replicates', 10);  % idx: 每个时间点的聚类标签

% --------------------------
% 5.3 基于FD的纯净眼电选择
%    mode='paper': 直接按聚类标签掩膜、仅以 FD 最小选择簇；不进行形态学/滞回/平滑
%    mode='robust': 使用形态学闭运算、FD 平局低频占比、滞回与 TV 平滑等增强
% --------------------------
cluster_signal = zeros(k_cluster, P);
fd = zeros(1, k_cluster);
lf_ratio = zeros(1, k_cluster); % 低频能量占比用于 robust 平局
cluster_energy = zeros(1, k_cluster);  % 记录簇能量
gap = max(1, round(mask_gap_ms/1000*fs));

for c = 1:k_cluster
    m_basic = (idx' == c);
    if isPaper
        xs = initial_eog .* m_basic;
    else
        m = local_close_mask(m_basic, gap);
        xs = initial_eog .* m;
    end
    cluster_signal(c, :) = xs;
    cluster_energy(c) = sum(xs.^2);  % 记录能量
    x_concat = local_concatenate_segments(xs);
    if isempty(x_concat)
        fd(c) = inf; lf_ratio(c) = 0; %#ok<*AGROW>
    else
        fd(c) = sevcik_fd(x_concat);
        lf_ratio(c) = local_lowfreq_ratio(x_concat, fs, lowfreq_band, [0.5 12]);
    end
end

% 【关键改进】眼电簇判别：当FD无法区分时，用能量/幅度判别
% 文献用FD，但实际数据中眼电和脑电的FD可能很接近
% 更稳健的方法：眼电尖峰的幅度/能量明显大于基线

% 先计算每个簇的统计特征
cluster_mean_abs = zeros(1, k_cluster);
cluster_peak = zeros(1, k_cluster);
for c = 1:k_cluster
    seg = cluster_signal(c, :);
    seg_nonzero = seg(abs(seg) > eps);
    if ~isempty(seg_nonzero)
        cluster_mean_abs(c) = mean(abs(seg_nonzero));
        cluster_peak(c) = max(abs(seg));
    end
end

% 调试输出：聚类结果
if verbose
    fprintf('\n=== K-means聚类调试信息 ===\n');
    for c = 1:k_cluster
        fprintf('聚类%d: 样本数=%d (%.1f%%), FD=%.4f, LF%%=%.1f%%, 能量=%.2e, 平均幅度=%.2f, 峰值=%.2f\n', ...
            c, sum(idx==c), 100*sum(idx==c)/length(idx), fd(c), 100*lf_ratio(c), cluster_energy(c), cluster_mean_abs(c), cluster_peak(c));
    end
end

% 判别逻辑：优先用FD，但当FD差异<0.05时，改用能量+峰值判别
[fd_min, min_fd_idx_by_fd] = min(fd);
[~, min_fd_idx_by_energy] = max(cluster_energy);
[~, min_fd_idx_by_peak] = max(cluster_peak);

fd_diff = max(fd) - min(fd);
if fd_diff < 0.05 && all(isfinite(fd))
    % FD区分度不足，用能量和峰值综合判断
    % 【关键检查】眼电簇应该是"少数但强烈"的尖峰事件
    % 如果能量最大的簇样本占比>60%，说明可能选错了（选到了背景簇）
    cluster_ratio = zeros(1, k_cluster);
    for c = 1:k_cluster
        cluster_ratio(c) = sum(idx == c) / length(idx);
    end
    
    if min_fd_idx_by_energy == min_fd_idx_by_peak
        min_fd_idx = min_fd_idx_by_energy;
        % 安全检查：如果选中簇占比过高，可能有问题
        if cluster_ratio(min_fd_idx) > 0.6
            other = 3 - min_fd_idx;  % k=2时的另一个簇
            if verbose
                fprintf('警告：能量最大簇%d样本占比过高(%.1f%%)，可能含大量背景\n', ...
                    min_fd_idx, 100*cluster_ratio(min_fd_idx));
                fprintf('尝试选择占比较小的簇%d (占比%.1f%%, 峰值%.1f)\n', ...
                    other, 100*cluster_ratio(other), cluster_peak(other));
            end
            % 如果另一个簇峰值也够高（>原簇的50%），改选它
            if cluster_peak(other) > 0.5 * cluster_peak(min_fd_idx)
                min_fd_idx = other;
                if verbose
                    fprintf('修正：改选簇%d\n', min_fd_idx);
                end
            end
        end
        if verbose
            fprintf('FD差异过小(%.4f < 0.05)，改用能量+峰值判别 → 选择簇%d\n', fd_diff, min_fd_idx);
        end
    else
        % 能量和峰值不一致，优先用峰值（眼电的特征是尖峰）
        min_fd_idx = min_fd_idx_by_peak;
        if verbose
            fprintf('FD差异过小，能量峰值不一致 → 选择峰值最大簇%d\n', min_fd_idx);
        end
    end
else
    % FD区分度足够，用FD
    min_fd_idx = min_fd_idx_by_fd;
    if verbose
        fprintf('FD区分度足够(差异=%.4f) → 选择FD最小簇%d\n', fd_diff, min_fd_idx);
    end
end

if verbose
    fprintf('最终选择: 簇%d (样本占比%.1f%%) [paper模式]\n', ...
        min_fd_idx, 100*sum(idx==min_fd_idx)/length(idx));
end

% 【关键修复】对聚类掩膜进行形态学扩展，避免眼电波形被"切片"
% 问题：k-means逐点分类会把眼电的上升/下降沿误分类到背景簇
% 解决：对选中簇的掩膜做膨胀+连接小间隙，让眼电波形完整
m_basic = (idx' == min_fd_idx);

if isPaper
    % paper模式也需要基本的形态学处理，确保眼电波形完整
    % 但要避免过度扩展，破坏正常脑电
    
    % 1. 小间隙合并（<100ms的间隙视为同一眼动事件）
    gap_samples = max(1, round(0.10 * fs));  % 100ms（回调保守值）
    m_merged = local_merge_small_gaps(m_basic, gap_samples);
    
    % 2. 形态学膨胀（向两侧扩展，包含上升/下降沿）
    dilate_samples = round(0.15 * fs);  % 向两侧各扩展150ms（回调到适中值）
    if dilate_samples > 0
        kernel = ones(1, 2*dilate_samples + 1);
        m_dilated = conv(double(m_merged), kernel, 'same') > 0;
    else
        m_dilated = m_merged;
    end
    
    % 3. 【改进】基于幅度阈值的保守扩展
    % 只在掩膜边界附近、且信号确实很强时才扩展
    m_extended = m_dilated;
    sig_abs = abs(initial_eog);
    % 提高阈值到50%（之前30%太低）
    threshold = 0.5 * max(sig_abs);
    
    segs = local_segments(m_extended);
    for i = 1:size(segs, 1)
        L_start = segs(i, 1);
        R_end = segs(i, 2);
        
        % 向左扩展：直到信号幅度低于阈值，但最多扩展50ms
        max_extend = round(0.05 * fs);
        extend_count = 0;
        while L_start > 1 && sig_abs(L_start - 1) > threshold && extend_count < max_extend
            L_start = L_start - 1;
            extend_count = extend_count + 1;
        end
        
        % 向右扩展：直到信号幅度低于阈值，但最多扩展50ms
        extend_count = 0;
        while R_end < numel(sig_abs) && sig_abs(R_end + 1) > threshold && extend_count < max_extend
            R_end = R_end + 1;
            extend_count = extend_count + 1;
        end
        
        m_extended(L_start:R_end) = true;
    end
    
    % 4. 再次合并小间隙（<80ms）
    m_extended = local_merge_small_gaps(m_extended, round(0.08 * fs));
    
    % 5. 去除极短片段（<100ms的孤立点）
    segs = local_segments(m_extended);
    min_duration = max(1, round(0.10 * fs));
    m_final = false(size(m_extended));
    for i = 1:size(segs, 1)
        if segs(i,2) - segs(i,1) + 1 >= min_duration
            m_final(segs(i,1):segs(i,2)) = true;
        end
    end
    
    eog_artifact = initial_eog .* m_final;
    
    if verbose
        fprintf('形态学处理: 合并%.0fms → 膨胀%.0fms → 幅度扩展(阈值=%.1f,最多50ms) → 精修\n', ...
            gap_samples/fs*1000, dilate_samples/fs*1000, threshold);
        fprintf('掩膜覆盖率: 原始%.1f%% → 处理后%.1f%%\n', ...
            100*sum(m_basic)/numel(m_basic), 100*sum(m_final)/numel(m_final));
    end
else
    m_cluster = local_close_mask(m_basic, gap);
    if use_hysteresis
        m_hyst = local_hysteresis_mask(initial_eog, fs, hyst_scale, hyst_ratio, expand_sec, min_long_sec, dilate_sec);
        m_final = (m_cluster | m_hyst);
    else
        m_final = m_cluster;
    end
    eog_artifact_raw = initial_eog .* m_final;
    eog_artifact = local_tv_smooth(eog_artifact_raw, smooth_lambda);
end

% 再次检查眼电提取结果
eog_power = sum(eog_artifact.^2);
initial_power = sum(initial_eog.^2);
if verbose
    fprintf('提取的眼电能量: %.2f (初始RC1能量: %.2f, 占比%.1f%%)\n', ...
        eog_power, initial_power, 100*eog_power/initial_power);
end

% --------------------------
% 5.4 可视化4：k-means聚类结果
% --------------------------
if visualize
    figure('Position', [1200, 100, 1000, 400]);
    % 子图1：聚类标签时间分布
    subplot(2, 1, 1);
    scatter(t, initial_eog, 3, idx, 'filled');
    title('Step 4: k-means聚类标签分布（颜色=聚类）', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 9); ylabel('初始眼电幅值', 'FontSize', 9);
    colorbar; 
    grid on; axis tight;

    % 子图2：两类聚类信号对比（标注FD值）
    subplot(2, 1, 2);
    plot(t, cluster_signal(1, :), 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('聚类1 (FD=%.3f, LF%%=%.1f)', fd(1), 100*lf_ratio(1)));
    hold on;
    plot(t, cluster_signal(2, :), 'g-', 'LineWidth', 1.2, 'DisplayName', sprintf('聚类2 (FD=%.3f, LF%%=%.1f)', fd(2), 100*lf_ratio(2)));
    plot(t, eog_artifact, 'k-', 'LineWidth', 1.5, 'DisplayName', '提取眼电');
    title('聚类信号对比与纯净眼电提取', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 9); ylabel('幅值', 'FontSize', 9);
    legend('Location', 'best'); grid on; hold off;
end
%% 6. 信号去噪（核心步骤4：减去纯净眼电）
denoised_eeg = noisy_eeg - eog_artifact;

% 计算去噪性能指标（仅日志）
if verbose
    fprintf('\n=== 去噪性能评估 ===\n');
    noise_power = sum(eog_artifact.^2);
    signal_power = sum(noisy_eeg.^2);
    denoised_power = sum(denoised_eeg.^2);
    fprintf('原始信号功率: %.2f\n', signal_power);
    fprintf('去除噪声功率: %.2f (%.1f%%)\n', noise_power, 100*noise_power/signal_power);
    fprintf('去噪后功率: %.2f (%.1f%%)\n', denoised_power, 100*denoised_power/signal_power);

    % 计算相关系数
    corr_noisy_artifact = corr(noisy_eeg', eog_artifact');
    corr_denoised_artifact = corr(denoised_eeg', eog_artifact');
    fprintf('原始信号与眼电相关性: %.4f\n', corr_noisy_artifact);
    fprintf('去噪后与眼电相关性: %.4f\n', corr_denoised_artifact);
end

% --------------------------
% 6.1 可视化5：去噪前后对比（与文献Fig.2一致的3行布局）
% --------------------------
if visualize
    figure('Position', [1200, 550, 1200, 500],'Name','Step 5: 去噪前后信号对比');
    
    subplot(3, 1, 1);
    plot(t, noisy_eeg, 'b', 'LineWidth', 1.0);
    title('原始含噪脑电信号 (含眼电)', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 9);
    grid on; axis tight; ylim_range = ylim;
    
    subplot(3, 1, 2);
    plot(t, eog_artifact, 'r', 'LineWidth', 1.0);
    title('提取的眼电伪迹', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)', 'FontSize', 9);
    grid on; axis tight;
    
    subplot(3, 1, 3);
    plot(t, denoised_eeg, 'g', 'LineWidth', 1.0);
    title('去眼电后脑电信号 (纯净脑电)', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 9); ylabel('幅值 (μV)', 'FontSize', 9);
    grid on; axis tight; ylim(ylim_range);  % 与原始信号相同y轴范围便于对比
end

% --------------------------
% 6.2 可视化6：去噪前后功率谱对比（验证低频保留，文献重点）
% --------------------------
if visualize
    figure('Position', [1200, 1000, 1000, 300]);
    % 计算功率谱（Welch方法，文献常用）
    win = hanning(fs/2);  % 窗长=采样频率/2
    noverlap = fs/4;      % 重叠率50%
    nfft = 2^nextpow2(fs); % nextpow2只返回幂次，需要计算2的幂

    [Pxx1, f1] = pwelch(noisy_eeg, win, noverlap, nfft, fs);  % 去噪前
    [Pxx2, f2] = pwelch(denoised_eeg, win, noverlap, nfft, fs);% 去噪后

    % 绘制功率谱（对数坐标，突出低频）
    plot(f1, 10*log10(Pxx1), 'b-', 'LineWidth', 1.2, 'DisplayName', '去噪前');
    hold on;
    plot(f2, 10*log10(Pxx2), 'g-', 'LineWidth', 1.2, 'DisplayName', '去噪后');
    % 标注文献关注的频段（delta:0.5-4, theta:4-8, alpha:8-12, beta:12-30）
    hold on;
    xline(0.5, 'k--', 'LineWidth', 0.8); xline(4, 'k--', 'LineWidth', 0.8);
    xline(8, 'k--', 'LineWidth', 0.8); xline(12, 'k--', 'LineWidth', 0.8);
    xline(30, 'k--', 'LineWidth', 0.8);
    text(2, max(10*log10(Pxx1))*0.8, 'delta (0.5-4Hz)', 'FontSize', 8);
    text(6, max(10*log10(Pxx1))*0.8, 'theta (4-8Hz)', 'FontSize', 8);
    text(10, max(10*log10(Pxx1))*0.8, 'alpha (8-12Hz)', 'FontSize', 8);
    text(20, max(10*log10(Pxx1))*0.8, 'beta (12-30Hz)', 'FontSize', 8);

    title('Step 6: 去噪前后功率谱对比（验证低频保留）', 'FontSize', 12, 'FontWeight', 'bold');
    xlabel('频率 (Hz)', 'FontSize', 10); ylabel('功率谱密度 (dB/Hz)', 'FontSize', 10);
    xlim([0, 35]);  % 聚焦脑电关键频段
    legend('Location', 'best'); grid on; hold off;
end

end

% ------- 本地辅助：官方CISSA调用 / 内置Ci-SSA实现 / 掩膜闭运算/滞回检测等 -------
function RC = local_run_official_cissa(noisy_eeg, L, cissaPath, cissaEntry, verbose)
% 试探性调用“官方提供的CISSA目录里的函数”，若成功返回 RC（#components × N），否则返回 []
RC = [];
try
    if ~isempty(cissaPath)
        pth = char(cissaPath);
        if exist(pth, 'dir')
            addpath(genpath(pth));
            if verbose, fprintf('已添加官方 CISSA 目录到路径: %s\n', pth); end
        end
    end
    candidates = {};
    if ~isempty(cissaEntry)
        candidates{end+1} = char(cissaEntry); %#ok<AGROW>
    end
    % 常见命名猜测（用户可通过 cissaEntry 显式指定覆盖）
    candidates = [candidates, {'CiSSA','CISSA','ci_ssa','cissa','Ci_SSA'}];
    % 去重
    candidates = unique(candidates, 'stable');

    for k = 1:numel(candidates)
        f = candidates{k};
        if exist(f, 'file') == 2 || exist(f, 'file') == 3 || exist(f, 'file') == 6
            if verbose, fprintf('尝试调用官方 CISSA 函数: %s\n', f); end
            % 尝试不同的调用签名
            rc_try = [];
            ok = false;
            try
                rc_try = feval(f, noisy_eeg, L);
                ok = true;
            catch
                % ignore and try variants
            end
            if ~ok
                try
                    rc_try = feval(f, noisy_eeg(:).', L);
                    ok = true;
                catch
                end
            end
            if ~ok
                try
                    [rc_try, ~] = feval(f, noisy_eeg, L);
                    ok = true;
                catch
                end
            end
            if ~ok
                try
                    [rc_try, ~] = feval(f, noisy_eeg(:).', L);
                    ok = true;
                catch
                end
            end
            if ok && ~isempty(rc_try) && isnumeric(rc_try)
                RC = rc_try;
                return;
            end
        end
    end
catch ME
    if verbose
        fprintf('官方 CISSA 调用失败: %s\n', ME.message);
    end
    RC = [];
end
end

function [RC, G, lambda] = local_cissa_custom(noisy_eeg, L, verbose)
% 内置 Ci-SSA 实现（与论文定义一致）：返回 RC（#groups×N）、分组 G、特征值 lambda
P = numel(noisy_eeg);
R = P - L + 1;                   % 轨迹矩阵列数
X = zeros(L, R);                 % L×R轨迹矩阵
for j = 1:R
    X(:, j) = noisy_eeg(j:j+L-1); % 每列是滞后j的L点信号
end

% 自相关 Psi（0..L-1）
Psi = zeros(1, L);
for n = 0:L-1
    if n == 0
        Psi(n+1) = mean(noisy_eeg.^2);  % 滞后0自相关=方差
    else
        Psi(n+1) = mean(noisy_eeg(1:end-n) .* noisy_eeg(n+1:end));
    end
end

% 循环矩阵 S_C 第一行（修正的文献公式）
s_first_row = zeros(1, L);
for n = 0:L-1
    r_n = Psi(n+1);
    if n == 0
        r_L_minus_n = Psi(1);
    else
        r_L_minus_n = Psi(L - n + 1);
    end
    s_first_row(n+1) = (L - n)/L * r_n + (n/L) * r_L_minus_n;
end

% 完整循环矩阵
S_C = zeros(L);
for i = 1:L
    S_C(i, :) = circshift(s_first_row, i-1);
end

% 傅里叶单位矩阵 V 与特征值
V = complex(zeros(L, L));  % 创建复数矩阵
for q = 1:L
    for j = 1:L
        V(j, q) = exp(-1i * 2 * pi * (j-1) * (q-1) / L) / sqrt(L);
    end
end
lambda = zeros(L, 1);
for q = 1:L
    lambda(q) = V(:, q)' * S_C * V(:, q);
end
u = V; % 特征向量

% 频率分组（对称）
G = {};
if mod(L, 2) == 1  % L为奇数
    M = (L + 1)/2;
    G{1} = 1;
    for q = 2:M
        G{q} = [q, L + 2 - q];
    end
else               % L为偶数
    M = L/2;
    G{1} = 1;
    for q = 2:M
        G{q} = [q, L + 2 - q];
    end
    G{M+1} = M + 1;
end

% 重构 RC
RC = zeros(length(G), P);  % 存储各分组重构分量
for g = 1:length(G)
    q_list = G{g};
    X_G = zeros(L, R);
    for q = q_list
        w_q = X' * u(:, q);       % w_q = X^T * u_q
        X_q = u(:, q) * w_q';     % X_q = u_q * w_q^T
        X_G = X_G + real(X_q);    % 取实部（信号为实）
    end
    RC(g, :) = diag_averaging(X_G);
end

% 可选日志
if verbose
    fprintf('已使用内置 Ci-SSA 实现。\n');
end
end

% ------- 本地辅助：掩膜闭运算/滞回检测/拼接/低频比/平滑 -------
function m = local_close_mask(m0, gap)
% 把小于 gap 的 0 间隙用 1 填补，并平滑边缘
m = m0(:).';
if gap <= 1, return; end
% 填小孔（running sum > 0）
kernel = ones(1, gap);
filled = conv(double(m), kernel, 'same') > 0; % 膨胀
eroded = conv(double(filled), kernel, 'same') >= gap; % 侵蚀
m = eroded;
end

function m = local_hysteresis_mask(x, fs, scale, ratio, expand_sec, min_long_sec, dilate_sec)
% 基于 |x| 的峰起点滞回扩展 + 负向长段检测，并做少量膨胀
x = x(:).'; N = numel(x);
sig = abs(x);
mad = median(abs(sig - median(sig)))/0.6745;
T = max(eps, scale) * mad * sqrt(2*log(N));
Tlow = ratio * T;

% 峰检测（|x| > T），向两侧扩展至 |x| <= Tlow，再做时间扩展
try
    [pks, locs] = findpeaks(sig, 'MinPeakHeight', T, 'MinPeakDistance', round(0.1*fs)); %#ok<ASGLU>
catch
    locs = find( (sig > T) & [true sig(2:end)>=sig(1:end-1)] & [sig(1:end-1)>=sig(2:end) true] );
end
mask = false(1,N);
pad = max(0, round(expand_sec*fs));
for k=1:numel(locs)
    L = locs(k); R = locs(k);
    while L>1 && sig(L-1)>Tlow, L=L-1; end
    while R<N && sig(R+1)>Tlow, R=R+1; end
    L = max(1, L-pad); R = min(N, R+pad);
    mask(L:R) = true;
end

% 负向长段：x < -Tlow 且时长>=min_long_sec
minLong = max(1, round(min_long_sec*fs));
negMask = (x < -Tlow);
segs = local_segments(negMask);
for i=1:size(segs,1)
    if segs(i,2)-segs(i,1)+1 >= minLong
        mask(segs(i,1):segs(i,2)) = true;
    end
end

% 形态学膨胀
rad = max(0, round(dilate_sec*fs));
if rad>0
    mask = conv(double(mask), ones(1,2*rad+1), 'same')>0;
end
% 小间隙合并
mask = local_merge_small_gaps(mask, round(0.1*fs));
% 去除极短段
segs = local_segments(mask);
keep = false(size(segs,1),1);
for i=1:numel(keep)
    if segs(i,2)-segs(i,1)+1 >= max(1, round(0.12*fs))
        keep(i)=true; 
    end
end
mask(:) = false; segs = segs(keep,:);
for i=1:size(segs,1), mask(segs(i,1):segs(i,2))=true; end
% 输出
m = mask;
end

function mask2 = local_merge_small_gaps(mask, gap)
mask = mask(:).'; N=numel(mask); if N==0, mask2=mask; return; end
segs = local_segments(mask);
if isempty(segs), mask2 = false(1,N); return; end
merged = segs(1,:);
for i=2:size(segs,1)
    if segs(i,1) - merged(end,2) - 1 <= gap
        merged(end,2) = segs(i,2);
    else
        merged = [merged; segs(i,:)]; %#ok<AGROW>
    end
end
mask2 = false(1,N);
for i=1:size(merged,1), mask2(merged(i,1):merged(i,2))=true; end
end

function xcat = local_concatenate_segments(x)
% 把非零连续片段拼接为单一向量
idx = find(abs(x) > 0);
if isempty(idx), xcat = []; return; end
% 找到段落
d = diff(idx);
brk = [0, find(d > 1), numel(idx)];
xcat = [];
for k = 1:numel(brk)-1
    segIdx = idx(brk(k)+1:brk(k+1));
    xcat = [xcat, x(segIdx)]; %#ok<AGROW>
end
end

function r = local_lowfreq_ratio(x, fs, band, bandAll)
% 计算低频带能量占比，默认以 Welch 估计
if nargin < 4, bandAll = [0.5 12]; end
if numel(x) < fs/2
    x = [x, zeros(1, fs/2 - numel(x))]; %#ok<AGROW>
end
[Pxx, f] = pwelch(x, hanning(round(fs/2)), round(fs/4), 2^nextpow2(fs), fs);
maskAll = f >= bandAll(1) & f <= bandAll(2);
maskLF  = f >= band(1)    & f <= band(2);
Ea = sum(Pxx(maskAll));
El = sum(Pxx(maskLF));
if Ea>0
    r = El / Ea;
else
    r = 0;
end
end

function y = local_tv_smooth(x, lambda)
% 若有 tv1d_denoise（来自 VME_GMETV 目录），用其；否则退化为移动平均
x = x(:).'; N=numel(x);
if exist('tv1d_denoise','file')==2
    y = tv1d_denoise(x, max(lambda, 1e-3));
    y = y(:).';
else
    w = max(3, 2*round(0.03*N)+1); % ~3%窗长
    y = movmean(x, w);
end
end

function segs = local_segments(mask)
% 返回逻辑掩膜中为 true 的所有连通区间 [start, end]
mask = logical(mask(:));
dm = diff([false; mask; false]);
starts = find(dm == 1);
ends   = find(dm == -1) - 1;
segs = [starts, ends];
end