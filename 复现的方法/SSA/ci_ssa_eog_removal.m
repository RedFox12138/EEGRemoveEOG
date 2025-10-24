function [denoised_eeg, eog_artifact] = ci_ssa_eog_removal(noisy_eeg, varargin)
% 基于Circular SSA+Morlet小波+k-means的眼电去噪（复现文献算法）
% 输入：
%   noisy_eeg - 含眼电噪声的脑电信号（1维向量）
%   varargin  - 可选参数：
%               fs: 采样频率(默认250Hz)、L: Ci_SSA窗口长度(默认28)、k_cluster: 聚类数(默认2)
%               freq_range: 小波频率范围(默认[1,12]Hz)、freq_step: 小波频率步长(默认0.25Hz)
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
mask_gap_ms = p.Results.mask_gap_ms; fd_tie_eps = p.Results.fd_tie_eps; lowfreq_band = p.Results.lowfreq_band;
use_hysteresis = p.Results.use_hysteresis; hyst_scale = p.Results.hyst_scale; hyst_ratio = p.Results.hyst_ratio;
expand_sec = p.Results.expand_sec; min_long_sec = p.Results.min_long_sec; dilate_sec = p.Results.dilate_sec;
smooth_lambda = p.Results.smooth_lambda;

% 信号基础信息
noisy_eeg = squeeze(noisy_eeg);  % 确保1维
P = length(noisy_eeg);           % 信号长度
t = (0:P-1)/fs;                  % 时间向量


%% 2. 可视化1：原始含噪脑电信号
figure('Position', [100, 100, 1000, 200]);
plot(t, noisy_eeg, 'b', 'LineWidth', 1.2);
title('Step 1: 原始含眼电噪声的脑电信号', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 10); ylabel('幅值 (μV)', 'FontSize', 10);
grid on; axis tight;


%% 3. Circular SSA (Ci_SSA) 分解（核心步骤1：提取初始眼电）
% --------------------------
% 3.1 嵌入阶段：构建轨迹矩阵
% --------------------------
R = P - L + 1;                   % 轨迹矩阵列数
X = zeros(L, R);                 % L×R轨迹矩阵
for j = 1:R
    X(:, j) = noisy_eeg(j:j+L-1); % 每列是滞后j的L点信号
end

% --------------------------
% 3.2 分解阶段：循环矩阵+特征分解
% --------------------------
% 3.2.1 计算自相关函数Ψ (文献公式方程2)
% 注意：只需要计算0到L-1的滞后
Psi = zeros(1, L);
for n = 0:L-1
    if n == 0
        Psi(n+1) = mean(noisy_eeg.^2);  % 滞后0自相关=方差
    else
        Psi(n+1) = mean(noisy_eeg(1:end-n) .* noisy_eeg(n+1:end));
    end
end

% 3.2.2 构建循环矩阵S_C第一行（文献方程2）
% 正确公式：s_n = (L-n)/L * \hat{\phi}_n + (n/L) * \hat{\phi}_{L-n}, n=0,...,L-1
% 注意：第二项索引应为 L-n（而非 P-n），确保与Ci_SSA定义一致
s_first_row = zeros(1, L);
for n = 0:L-1
    r_n = Psi(n+1);
    % \hat{\phi}_{L-n}: 当 n=0 时按定义取 \hat{\phi}_0
    if n == 0
        r_L_minus_n = Psi(1);
    else
        r_L_minus_n = Psi(L - n + 1);
    end
    % 文献公式（修正）
    s_first_row(n+1) = (L - n)/L * r_n + (n/L) * r_L_minus_n;
end

% 3.2.3 构建完整循环矩阵S_C（每行是第一行的循环位移）
S_C = zeros(L);
for i = 1:L
    S_C(i, :) = circshift(s_first_row, i-1);
end

% 3.2.4 傅里叶单位矩阵V与特征分解（文献方程3）
% V的每一列是一个特征向量
V = complex(zeros(L, L));  % 创建复数矩阵
for q = 1:L
    for j = 1:L
        % 文献公式：v_{q,j} = exp(-i*2*pi*(j-1)*(q-1)/L)
        V(j, q) = exp(-1i * 2 * pi * (j-1) * (q-1) / L) / sqrt(L);
    end
end

% 计算特征值：lambda_q = V_q^H * S_C * V_q
lambda = zeros(L, 1);
for q = 1:L
    lambda(q) = V(:, q)' * S_C * V(:, q);
end

% 特征向量矩阵（每列是一个特征向量）
u = V;

% --------------------------
% 3.3 频率分组（文献对称分组规则）
% --------------------------
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

% --------------------------
% 3.4 重构阶段：生成RC分量（文献对角平均）
% --------------------------
RC = zeros(length(G), P);  % 存储各分组重构分量
for g = 1:length(G)
    q_list = G{g};
    X_G = zeros(L, R);
    
    % 对于每个频率组，累加对应的X_q分量
    for q = q_list
        % 文献方程(4): w_q = X^T * u_q
        w_q = X' * u(:, q);
        % 文献方程(4): X_q = u_q * w_q^T
        X_q = u(:, q) * w_q';
        
        % 如果特征向量是复数，需要取实部（因为信号是实数）
        X_G = X_G + real(X_q);
    end
    
    RC(g, :) = diag_averaging(X_G);  % 对角平均重构（子函数）
end

% 调试输出：检查特征值
fprintf('\n=== CI-SSA分解调试信息 ===\n');
fprintf('窗口长度 L = %d, 信号长度 P = %d\n', L, P);
fprintf('前10个特征值的模: ');
fprintf('%.4f ', abs(lambda(1:min(10,L))));
fprintf('\n');

% 选RC1作为初始眼电（文献结论：RC1含最强眼电信息）
initial_eog = RC(1, :);

% 调试：检查RC分量的能量分布
rc_energy = sum(RC.^2, 2);
fprintf('前5个RC分量能量占比: ');
for g = 1:min(5, length(G))
    fprintf('RC%d=%.2f%% ', g, 100*rc_energy(g)/sum(rc_energy));
end
fprintf('\n');

% --------------------------
% 3.5 可视化2：Ci_SSA分解的前5个RC分量
% --------------------------
figure('Position', [100, 350, 1000, 400]);
for g = 1:min(5, length(G))  % 显示前5个RC（文献图3风格）
    subplot(5, 1, g);
    plot(t, RC(g, :), 'Color', [0.2, 0.6, 0.8], 'LineWidth', 1);
    title(sprintf('Step 2: RC%d (能量: %.2f%%)', g, 100*rc_energy(g)/sum(rc_energy)), ...
        'FontSize', 10, 'FontWeight', 'bold');
    xlabel('时间 (s)', 'FontSize', 8); ylabel('幅值', 'FontSize', 8);
    grid on; axis tight;
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
fprintf('\n=== Morlet小波变换调试信息 ===\n');
fprintf('时频矩阵维度: %d × %d (频率×时间)\n', size(abs_wt, 1), size(abs_wt, 2));
fprintf('频率点数: %d (期望45)\n', freq_points);
fprintf('频率范围: %.2f - %.2f Hz\n', freqs(1), freqs(end));
fprintf('时频矩阵能量分布 - 最大: %.4f, 均值: %.4f\n', max(abs_wt(:)), mean(abs_wt(:)));

% --------------------------
% 4.3 可视化3：Morlet小波时频图
% --------------------------
figure('Position', [100, 800, 1000, 300]);
pcolor(t, freqs, abs_wt);  % 伪彩图（时间×频率×幅度）
shading interp;            % 插值平滑
colormap(jet);             % 热图配色
colorbar;
title('Step 3: 初始眼电的Morlet小波时频图', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 10); ylabel('频率 (Hz)', 'FontSize', 10);
axis tight;


%% 5. k-means聚类（核心步骤3：纯化眼电噪声）
% --------------------------
% 5.1 聚类数据准备（时频特征：每个时间点对应45维频率特征）
% --------------------------
cluster_data = abs_wt';  % P×freq_points（样本数×特征数）

% 不做归一化，保持原始幅度信息（眼电和脑电的幅度差异是关键特征）

% --------------------------
% 5.2 k-means聚类（文献k=2）
% --------------------------
rng(1);  % 固定随机种子确保复现性
opts = statset('UseParallel', false);
[idx, ~] = kmeans(cluster_data, k_cluster, 'Options', opts);  % idx: 每个时间点的聚类标签

% --------------------------
% 5.3 基于FD的纯净眼电选择（改进：在“连接片段”上计算FD，避免掩膜断点抬高FD）
% --------------------------
cluster_signal = zeros(k_cluster, P);
fd = zeros(1, k_cluster);
lf_ratio = zeros(1, k_cluster); % 低频能量占比作为平局判据
gap = max(1, round(mask_gap_ms/1000*fs));

for c = 1:k_cluster
    % 初始二值掩膜（每个时间点的类别）
    m0 = (idx' == c);
    % 形态学“闭运算”：先填小间隙再平滑，增强连续性以适配下沉式长时间眼动
    m = local_close_mask(m0, gap);
    % 用掩膜提取该簇的时间域分量
    xs = initial_eog .* m;
    cluster_signal(c, :) = xs;
    % 把非零段拼接起来作为FD的输入，避免零间隙引入的尖角
    x_concat = local_concatenate_segments(xs);
    if isempty(x_concat)
        fd(c) = inf; lf_ratio(c) = 0; %#ok<*AGROW>
    else
        fd(c) = sevcik_fd(x_concat);
        lf_ratio(c) = local_lowfreq_ratio(x_concat, fs, lowfreq_band, [0.5 12]);
    end
end

% 先按FD最小选择；如两簇FD相差很小（<=fd_tie_eps），选择低频能量占比较大的那簇
[fd_min, min_fd_idx] = min(fd);
if k_cluster == 2
    other = 3 - min_fd_idx;
    if abs(fd_min - fd(other)) <= fd_tie_eps
        if lf_ratio(other) > lf_ratio(min_fd_idx)
            min_fd_idx = other;
        end
    end
end
% 初始基于聚类的掩膜
m_cluster = local_close_mask((idx' == min_fd_idx), gap);

% 可选：叠加滞回+长段负向检测的掩膜，增强对“下沉/长段”的敏感性
if use_hysteresis
    m_hyst = local_hysteresis_mask(initial_eog, fs, hyst_scale, hyst_ratio, expand_sec, min_long_sec, dilate_sec);
    m_final = (m_cluster | m_hyst);
else
    m_final = m_cluster;
end

% 生成纯净眼电：在最终掩膜内保留 initial_eog，并做轻度平滑避免尖角/断裂
eog_artifact_raw = initial_eog .* m_final;
eog_artifact = local_tv_smooth(eog_artifact_raw, smooth_lambda);

% 调试输出：聚类结果
fprintf('\n=== K-means聚类调试信息 ===\n');
for c = 1:k_cluster
    fprintf('聚类%d: 样本数=%d (%.1f%%), FD=%.4f, LF%%=%.1f%%\n', ...
        c, sum(idx==c), 100*sum(idx==c)/length(idx), fd(c), 100*lf_ratio(c));
end
fprintf('选择的聚类: %d (FD最小)\n', min_fd_idx);
fprintf('提取的眼电能量: %.2f (原信号能量: %.2f)\n', ...
    sum(eog_artifact.^2), sum(initial_eog.^2));

% --------------------------
% 5.4 可视化4：k-means聚类结果
% --------------------------
figure('Position', [1200, 100, 1000, 400]);
% 子图1：聚类标签时间分布
subplot(2, 1, 1);
scatter(t, initial_eog, 3, idx, 'filled');
title('Step 4: k-means聚类标签分布（颜色=聚类）', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 9); ylabel('初始眼电幅值', 'FontSize', 9);
colorbar; legend({'聚类1', '聚类2'}, 'Location', 'best');
grid on; axis tight;

% 子图2：两类聚类信号对比（标注FD值）
subplot(2, 1, 2);
plot(t, cluster_signal(1, :), 'r--', 'LineWidth', 1.2, 'DisplayName', sprintf('聚类1 (FD=%.3f, LF%%=%.1f)', fd(1), 100*lf_ratio(1)));
hold on;
plot(t, cluster_signal(2, :), 'g-', 'LineWidth', 1.2, 'DisplayName', sprintf('聚类2 (FD=%.3f, LF%%=%.1f)', fd(2), 100*lf_ratio(2)));
plot(t, eog_artifact, 'k-', 'LineWidth', 1.5, 'DisplayName', '纯净眼电（融合滞回+平滑）');
title('聚类信号对比与纯净眼电提取', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 9); ylabel('幅值', 'FontSize', 9);
legend('Location', 'best'); grid on; hold off;


%% 6. 信号去噪（核心步骤4：减去纯净眼电）
denoised_eeg = noisy_eeg - eog_artifact;

% 计算去噪性能指标
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

% --------------------------
% 6.1 可视化5：去噪前后对比
% --------------------------
figure('Position', [1200, 550, 1000, 400]);
subplot(3, 1, 1);
plot(t, noisy_eeg, 'b', 'LineWidth', 1.2);
title('Step 5: 去噪前后信号对比', 'FontSize', 12, 'FontWeight', 'bold');
xlabel('时间 (s)', 'FontSize', 10); ylabel('原始含噪脑电', 'FontSize', 10);
grid on; axis tight;

subplot(3, 1, 2);
plot(t, eog_artifact, 'r', 'LineWidth', 1.2);
xlabel('时间 (s)', 'FontSize', 10); ylabel('提取的眼电噪声', 'FontSize', 10);
grid on; axis tight;

subplot(3, 1, 3);
plot(t, denoised_eeg, 'g', 'LineWidth', 1.2);
xlabel('时间 (s)', 'FontSize', 10); ylabel('去眼电后脑电', 'FontSize', 10);
grid on; axis tight;

% --------------------------
% 6.2 可视化6：去噪前后功率谱对比（验证低频保留，文献重点）
% --------------------------
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