%% EWT + ICEEMDAN 眼电伪迹去除算法
% 功能: 
%   1. 使用 EWT 自定义边界将脑电信号分离为低频(≤8Hz)和高频(>8Hz)两个频段
%   2. 对低频分量进行 ICEEMDAN 分解,得到多个 IMF 分量
%   3. 计算各 IMF 的样本熵,筛选并去除眼电伪迹(样本熵<0.4的分量)
%   4. 重构纯净脑电信号,并绘制时域和频域对比图
% 
% 算法流程:
%   原始信号 → EWT分离 → 低频ICEEMDAN分解 → 样本熵筛选 → 信号重构
%
% 作者: GitHub Copilot
% 日期: 2025-10-20
%
% 使用说明:
%   1. 修改 data_path 变量为你的数据路径
%   2. 设置正确的采样率 fs 和截止频率 cutoff_freq
%   3. 确保 ewt_custom_boundary.m, iceemdan.m, SampEn.m 在同一目录
%   4. 运行脚本

% clear; clc; close all;
%% ========== 参数设置 ==========
% TODO: 修改为你的数据路径
data_path = 'D:\Pycharm_Projects\ADHD-master\data\额头信号去眼电\1014 XY额头躲避游戏2_processed.txt';  % 改成你的 txt 文件路径

% 采样率设置
fs = 200;  % Hz, 请根据你的实际采样率修改

% EWT 分离参数
cutoff_freq = 4;  % 截止频率 8Hz (低频<=8Hz, 高频>8Hz)

% 备用滤波器参数(当 EWT 失败时使用)
filter_order = 6;  % Butterworth 滤波器阶数

%% ========== 数据加载 ==========
fprintf('正在加载数据...\n');

% 尝试加载数据
if exist(data_path, 'file')
    try
        % 加载 txt 文件
        % 支持多种格式: 逗号分隔、空格分隔、制表符分隔等
%         signal = load(data_path);
        signal = sim10_con';  % 使用 load 函数读取数值型 txt
        
        fprintf('成功加载数据: %s\n', data_path);
        fprintf('数据维度: %s\n', mat2str(size(signal)));
        
    catch ME
        % 如果 load 失败,尝试其他读取方法
        try
            fprintf('尝试使用 readmatrix 读取...\n');
            signal = readmatrix(data_path);
            fprintf('成功加载数据: %s\n', data_path);
            fprintf('数据维度: %s\n', mat2str(size(signal)));
        catch ME2
            warning('无法加载指定数据: %s', ME2.message);
            fprintf('使用模拟数据代替...\n');
            signal = generate_demo_signal(fs);
        end
    end
else
    warning('数据文件不存在: %s', data_path);
    fprintf('使用模拟数据代替...\n');
    signal = generate_demo_signal(fs);
end

% 数据预处理: 如果是多通道,取第一个通道
if size(signal, 2) > 1
    fprintf('检测到多通道数据,使用第一个通道\n');
    signal = signal(:, 1);
else
    signal = signal(:);  % 确保是列向量
end

% 取一段数据进行分析(避免数据过长)
max_samples = 10 * fs;  % 最多使用10秒数据
if length(signal) > max_samples
    signal = signal(1:max_samples);
    fprintf('数据过长,截取前 %.1f 秒进行分析\n', max_samples/fs);
end

fprintf('信号长度: %d 采样点 (%.2f 秒)\n', length(signal), length(signal)/fs);

%% ========== 使用 EWT 自定义边界进行频域分离 ==========
fprintf('\n正在使用 EWT 自定义边界进行频域分离...\n');
fprintf('截止频率: %.1f Hz\n', cutoff_freq);

% 设置自定义边界 (归一化频率: cycles/sample)
% boundaries = [cutoff_freq / fs];  % 单个边界点,分成2个频段
% 注意: ewt_custom_boundary 需要的是归一化的角频率 (0 到 pi)
custom_boundaries = [cutoff_freq / fs * 2 * pi];  % 转换为弧度

fprintf('正在执行 EWT 自定义边界分解...\n');
fprintf('自定义边界 (归一化): %.4f (%.2f Hz)\n', cutoff_freq/fs, cutoff_freq);

% 执行 EWT 自定义边界分解
try
    [MRA_ewt, ~, ~, INFO_ewt] = ewt_custom_boundary(signal, custom_boundaries, fs);
    
    % 获取低频和高频分量
    % EWT 返回的 MRA 是按频率降序排列的
    % 第一列是最低频(0-8Hz), 第二列是高频(8Hz以上)
    signal_low = MRA_ewt(:, end);     % 最后一列是最低频段 (≤8Hz)
    signal_high = MRA_ewt(:, 1);      % 第一列是高频段 (>8Hz)
    
    fprintf('EWT 自定义边界分解完成!\n');
    fprintf('  得到 %d 个频段分量\n', size(MRA_ewt, 2));
    fprintf('  低频分量 (≤%.1fHz): %.2f%% 能量\n', cutoff_freq, 100*sum(signal_low.^2)/sum(signal.^2));
    fprintf('  高频分量 (>%.1fHz): %.2f%% 能量\n', cutoff_freq, 100*sum(signal_high.^2)/sum(signal.^2));
    
    % 显示实际的频率边界
    if isfield(INFO_ewt, 'FilterBank')
        passbands_hz = INFO_ewt.FilterBank.Passbands * fs;
        fprintf('\n  实际频段划分:\n');
        for i = 1:size(passbands_hz, 1)
            fprintf('    分量 %d: [%.2f - %.2f] Hz\n', i, passbands_hz(i,1), passbands_hz(i,2));
        end
    end
    
    ewt_success = true;
    
catch ME
    warning('EWT 自定义边界分解失败: %s', ME.message);
    fprintf('回退到 Butterworth 滤波器方法...\n');
    
    % 回退到原来的 Butterworth 滤波器方法
    wn = cutoff_freq / (fs/2);
    [b_low, a_low] = butter(filter_order, wn, 'low');
    [b_high, a_high] = butter(filter_order, wn, 'high');
    signal_low = filtfilt(b_low, a_low, signal);
    signal_high = filtfilt(b_high, a_high, signal);
    
    fprintf('Butterworth 滤波完成!\n');
    fprintf('  低频分量 (≤8Hz): %.2f%% 能量\n', 100*sum(signal_low.^2)/sum(signal.^2));
    fprintf('  高频分量 (>8Hz): %.2f%% 能量\n', 100*sum(signal_high.^2)/sum(signal.^2));
    
    ewt_success = false;
end

%% ========== 设计双边滤波器 ==========
fprintf('\n正在设计 Butterworth 滤波器...\n');
fprintf('截止频率: %.1f Hz\n', cutoff_freq);
fprintf('滤波器阶数: %d\n', filter_order);

% 归一化截止频率 (相对于奈奎斯特频率)
wn = cutoff_freq / (fs/2);

% 设计低通滤波器 (保留 8Hz 以内)
[b_low, a_low] = butter(filter_order, wn, 'low');

% 设计高通滤波器 (保留 8Hz 以上)
[b_high, a_high] = butter(filter_order, wn, 'high');

fprintf('滤波器设计完成!\n');

%% ========== 执行滤波 ==========
fprintf('\n正在执行滤波分离...\n');

% 使用 filtfilt 进行零相位滤波(避免相位失真)
signal_low = filtfilt(b_low, a_low, signal);   % 8Hz 以内
signal_high = filtfilt(b_high, a_high, signal); % 8Hz 以上

fprintf('滤波完成!\n');
fprintf('  低频分量 (≤8Hz): %.2f%% 能量\n', 100*sum(signal_low.^2)/sum(signal.^2));
fprintf('  高频分量 (>8Hz): %.2f%% 能量\n', 100*sum(signal_high.^2)/sum(signal.^2));

%% ========== 对低频分量进行 ICEEMDAN 分解 ==========
fprintf('\n正在对低频分量(≤8Hz)进行 ICEEMDAN 分解...\n');

% ICEEMDAN 参数设置
Nstd = 0.2;           % 噪声标准差
NR = 100;             % 集成次数(实现次数)
MaxIter = 5000;       % 最大迭代次数
SNRFlag = 1;          % SNR标志: 1-每级递增, 2-所有级相同

% 执行 ICEEMDAN 分解
try
    IMFs = pICEEMDAN(signal,t,Nstd,NR,MaxIter);
    num_imfs = size(IMFs, 1);
    fprintf('ICEEMDAN 分解完成! 共得到 %d 个 IMF 分量\n', num_imfs);
    
    % 显示各 IMF 的能量占比
    fprintf('\n各 IMF 分量能量分布:\n');
    for i = 1:num_imfs
        energy_percent = 100 * sum(IMFs(i,:).^2) / sum(signal_low.^2);
        if i < num_imfs
            fprintf('  IMF %d: %.2f%% 能量\n', i, energy_percent);
        else
            fprintf('  残差 (Residual): %.2f%% 能量\n', energy_percent);
        end
    end
    
    iceemdan_success = true;
catch ME
    warning('ICEEMDAN 分解失败: %s', ME.message);
    fprintf('将仅显示滤波结果\n');
    IMFs = signal_low;
    num_imfs = 1;
    iceemdan_success = false;
end

% 存储所有分量信息
if iceemdan_success
    % IMF分量 + 高频分量
    all_components = [IMFs; signal_high'];
    num_total_components = num_imfs + 1;
    
    % 构建分量名称
    component_names = cell(1, num_total_components);
    for i = 1:num_imfs-1
        component_names{i} = sprintf('IMF%d (低频分解)', i);
    end
    component_names{num_imfs} = '残差 (低频)';
    component_names{num_total_components} = '>8Hz (高频)';
else
    % 只有两个频段
    all_components = [signal_low'; signal_high'];
    num_total_components = 2;
    component_names = {'≤8Hz (低频)', '>8Hz (高频)'};
end

%% ========== 显示频率信息 ==========
fprintf('\n========== 完整分量信息 ==========\n');
fprintf('第一步: Butterworth 滤波 - 分为 ≤8Hz 和 >8Hz\n');
fprintf('第二步: ICEEMDAN 分解低频 - 得到 %d 个 IMF 分量\n', num_imfs);
fprintf('总分量数: %d\n', num_total_components);

%% ========== 计算样本熵并筛选 IMF 分量 ==========
if iceemdan_success
    fprintf('\n========== 样本熵计算与筛选 ==========\n');
    
    % 样本熵阈值
    sampen_threshold = 0.4;
    
    % 计算每个 IMF 的样本熵
    sampen_values = zeros(num_imfs, 1);
    for i = 1:num_imfs
        try
            Samp = SampEn(IMFs(i, :));
            sampen_values(i) = Samp(3);  % 取第3个值
        catch ME
            warning('IMF%d 样本熵计算失败: %s', i, ME.message);
            sampen_values(i) = 0;  % 计算失败则设为0
        end
    end
    
    % 显示样本熵结果
    fprintf('各 IMF 分量样本熵:\n');
    for i = 1:num_imfs
        if i < num_imfs
            fprintf('  IMF%d: %.4f', i, sampen_values(i));
        else
            fprintf('  残差: %.4f', sampen_values(i));
        end
        
        if sampen_values(i) < sampen_threshold
            fprintf(' → 低于阈值 %.2f,判定为眼电伪迹,将被去除\n', sampen_threshold);
        else
            fprintf(' → 高于阈值 %.2f,保留\n', sampen_threshold);
        end
    end
    
    % 筛选 IMF 分量: 保留样本熵 >= 阈值的分量
    imf_keep_mask = sampen_values >= sampen_threshold;
    imf_remove_mask = sampen_values < sampen_threshold;
    num_kept = sum(imf_keep_mask);
    num_removed = num_imfs - num_kept;
    
    fprintf('\n筛选结果:\n');
    fprintf('  保留分量: %d 个\n', num_kept);
    fprintf('  去除分量: %d 个 (判定为眼电伪迹)\n', num_removed);
    
    % 重构低频信号: 仅使用保留的 IMF
    signal_low_reconstructed = sum(IMFs(imf_keep_mask, :), 1);
    signal_low_reconstructed = signal_low_reconstructed(:);
    
    % 重构最终信号: 低频重构 + 高频分量
    signal_final = signal_low_reconstructed + signal_high;
    
    fprintf('\n========== 信号重构 ==========\n');
    fprintf('低频重构: 使用 %d 个保留的 IMF 分量\n', num_kept);
    fprintf('最终重构: 低频重构 + 高频分量\n');
    
    % 计算去除效果
    removed_signal = sum(IMFs(imf_remove_mask, :), 1);
    energy_removed = 100 * sum(removed_signal.^2) / sum(signal.^2);
    correlation_before_after = corr(signal, signal_final);
    
    fprintf('\n去除效果:\n');
    fprintf('  去除能量占比: %.2f%%\n', energy_removed);
    fprintf('  重构信号与原始信号相关系数: %.4f\n', correlation_before_after);
    
    % 标记哪些 IMF 被保留/去除
    imf_status = cell(1, num_imfs);
    for i = 1:num_imfs
        if imf_keep_mask(i)
            imf_status{i} = '✓ 保留';
        else
            imf_status{i} = '✗ 去除';
        end
    end
else
    % 如果 ICEEMDAN 失败,则不进行样本熵筛选
    signal_final = signal;
    signal_low_reconstructed = signal_low;
    imf_keep_mask = [];
    sampen_values = [];
    num_kept = 0;
    num_removed = 0;
end

%% ========== 时域可视化 ==========
fprintf('\n正在绘制时域图...\n');

t = (0:length(signal)-1) / fs;  % 时间向量(秒)

% 创建时域图窗口 - 显示所有 IMF 分量
fig1 = figure('Name', 'ICEEMDAN 时域分解', 'Position', [50, 50, 1400, 1000]);

% 原始信号
subplot(num_total_components + 1, 1, 1);
plot(t, signal, 'k', 'LineWidth', 1);
title('原始信号', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('幅值');
grid on;
xlim([t(1), t(end)]);

% 各个 IMF 和高频分量
for i = 1:num_total_components
    subplot(num_total_components + 1, 1, i + 1);
    
    % 根据是否保留来设置颜色
    if iceemdan_success && i <= num_imfs
        if imf_keep_mask(i)
            % 保留的 IMF: 蓝色
            plot(t, all_components(i, :), 'b-', 'LineWidth', 1);
        else
            % 去除的 IMF (眼电): 红色
            plot(t, all_components(i, :), 'r-', 'LineWidth', 1);
        end
    else
        % 高频分量或其他: 默认颜色
        plot(t, all_components(i, :), 'LineWidth', 1);
    end
    
    % 添加标题,包含样本熵和状态信息
    if iceemdan_success && i <= num_imfs
        title(sprintf('%s | SampEn=%.4f | %s', component_names{i}, ...
            sampen_values(i), imf_status{i}), 'FontSize', 9);
    else
        title(sprintf('%s', component_names{i}), 'FontSize', 10);
    end
    
    ylabel('幅值');
    grid on;
    xlim([t(1), t(end)]);
    
    % 最后一个子图添加 x 轴标签
    if i == num_total_components
        xlabel('时间 (秒)');
    end
end

sgtitle('EWT + ICEEMDAN 时域分解结果 (红色=去除,蓝色=保留)', 'FontSize', 14, 'FontWeight', 'bold');

%% ========== 频域可视化 ==========
fprintf('正在绘制频域图...\n');

% 计算功率谱
nfft = 2^nextpow2(length(signal));
freq_vec = (0:nfft/2) * fs / nfft;

% 原始信号的功率谱
[pxx_orig, f_orig] = pwelch(signal, hamming(min(length(signal), 512)), ...
    [], nfft, fs);

% 各 IMF 和高频分量的功率谱
pxx_all = zeros(length(f_orig), num_total_components);
for i = 1:num_total_components
    [pxx_all(:, i), ~] = pwelch(all_components(i, :), ...
        hamming(min(length(signal), 512)), [], nfft, fs);
end

% 创建频域图窗口
fig2 = figure('Name', 'ICEEMDAN 频域分析', 'Position', [100, 100, 1400, 1000]);

% 原始信号频谱
subplot(num_total_components + 1, 1, 1);
plot(f_orig, 10*log10(pxx_orig), 'k', 'LineWidth', 1.5);
title('原始信号功率谱密度', 'FontSize', 12, 'FontWeight', 'bold');
ylabel('功率 (dB)');
grid on;
xlim([0, min(fs/2, 50)]);  % 显示到 50Hz 或奈奎斯特频率

% 标记截止频率
hold on;
xline(cutoff_freq, 'r--', sprintf('截止频率 %.1fHz', cutoff_freq), ...
    'LineWidth', 2, 'LabelHorizontalAlignment', 'center', 'FontSize', 10);
hold off;

% 各个 IMF 和高频分量的频谱
for i = 1:num_total_components
    subplot(num_total_components + 1, 1, i + 1);
    plot(f_orig, 10*log10(pxx_all(:, i)), 'LineWidth', 1.5);
    
    % 添加标题
    title(sprintf('%s 功率谱', component_names{i}), 'FontSize', 10);
    
    ylabel('功率 (dB)');
    grid on;
    xlim([0, min(fs/2, 50)]);
    
    % 对于低频 IMF,标记低频范围;对于高频,标记高频范围
    if i < num_total_components  % IMF 分量,标记低频范围
        xline(0, 'g--', 'LineWidth', 1);
        xline(cutoff_freq, 'g--', 'LineWidth', 1);
        yl = ylim;
        patch([0, cutoff_freq, cutoff_freq, 0], ...
            [yl(1), yl(1), yl(2), yl(2)], 'g', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    else  % 高频分量,标记高频范围
        xline(cutoff_freq, 'r--', 'LineWidth', 1);
        xline(fs/2, 'r--', 'LineWidth', 1);
        yl = ylim;
        patch([cutoff_freq, fs/2, fs/2, cutoff_freq], ...
            [yl(1), yl(1), yl(2), yl(2)], 'r', 'FaceAlpha', 0.1, 'EdgeColor', 'none');
    end
    hold off;
    
    % 最后一个子图添加 x 轴标签
    if i == num_total_components
        xlabel('频率 (Hz)');
    end
end

sgtitle('EWT + ICEEMDAN 频域分解结果', 'FontSize', 14, 'FontWeight', 'bold');

%% ========== 综合对比图 ==========
fprintf('正在绘制综合对比图...\n');

fig3 = figure('Name', 'ICEEMDAN 综合分析', 'Position', [150, 150, 1600, 900]);

% 左上: 原始信号时域
subplot(2, 3, 1);
plot(t, signal, 'k', 'LineWidth', 1.2);
title('原始信号 - 时域', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('时间 (秒)');
ylabel('幅值');
grid on;

% 中上: 低频分量(滤波后)时域
subplot(2, 3, 2);
plot(t, signal_low, 'b', 'LineWidth', 1.2);
title('低频分量 (≤8Hz) - 时域', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('时间 (秒)');
ylabel('幅值');
grid on;

% 右上: 高频分量时域
subplot(2, 3, 3);
plot(t, signal_high, 'r', 'LineWidth', 1.2);
title('高频分量 (>8Hz) - 时域', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('时间 (秒)');
ylabel('幅值');
grid on;

% 左下: 原始信号频域
subplot(2, 3, 4);
plot(f_orig, 10*log10(pxx_orig), 'k', 'LineWidth', 1.5);
hold on;
xline(cutoff_freq, 'r--', sprintf('截止 %.1fHz', cutoff_freq), ...
    'LineWidth', 2, 'LabelHorizontalAlignment', 'center');
hold off;
title('原始信号 - 频域', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('频率 (Hz)');
ylabel('功率 (dB)');
grid on;
xlim([0, min(fs/2, 50)]);

% 中下: 所有 IMF 叠加(频域)
subplot(2, 3, 5);
hold on;
for i = 1:num_imfs
    if i < num_imfs
        plot(f_orig, 10*log10(pxx_all(:, i)), 'LineWidth', 1.5, ...
            'DisplayName', sprintf('IMF%d', i));
    else
        plot(f_orig, 10*log10(pxx_all(:, i)), 'LineWidth', 1.5, ...
            'DisplayName', '残差');
    end
end
plot(f_orig, 10*log10(pxx_all(:, end)), 'LineWidth', 2, ...
    'DisplayName', '高频分量');
hold off;
title('所有分量 - 频域叠加', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('频率 (Hz)');
ylabel('功率 (dB)');
grid on;
xlim([0, min(fs/2, 30)]);
legend('Location', 'best', 'FontSize', 8);

% 右下: 能量分布
subplot(2, 3, 6);
energy_all = sum(all_components.^2, 2);
energy_percent_all = 100 * energy_all / sum(signal.^2);

bar(1:num_total_components, energy_percent_all);
title('各分量能量分布', 'FontSize', 11, 'FontWeight', 'bold');
if num_total_components <= 10
    set(gca, 'XTick', 1:num_total_components, 'XTickLabel', component_names, 'FontSize', 8);
    xtickangle(45);
else
    xlabel('分量编号');
end
ylabel('能量占比 (%)');
grid on;
% 添加数值标签(只对能量较大的显示)
for i = 1:num_total_components
    if energy_percent_all(i) > 1  % 只显示能量>1%的
        text(i, energy_percent_all(i), sprintf('%.1f%%', energy_percent_all(i)), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontSize', 8);
    end
end

sgtitle(sprintf('EWT + ICEEMDAN 综合分析 (共 %d 个分量)', num_total_components), ...
    'FontSize', 14, 'FontWeight', 'bold');

%% ========== 眼电去除效果对比图 ==========
if iceemdan_success && num_removed > 0
    fprintf('正在绘制眼电去除效果对比图...\n');
    
    fig4 = figure('Name', '眼电去除效果对比', 'Position', [200, 200, 1600, 900]);
    
    % 左上: 原始信号时域
    subplot(2, 3, 1);
    plot(t, signal, 'k', 'LineWidth', 1.2);
    title('原始污染信号', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (秒)');
    ylabel('幅值');
    grid on;
    
    % 中上: 去除的眼电成分
    subplot(2, 3, 2);
    plot(t, removed_signal, 'r', 'LineWidth', 1.2);
    title(sprintf('去除的眼电伪迹 (能量: %.2f%%)', energy_removed), ...
        'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (秒)');
    ylabel('幅值');
    grid on;
    
    % 右上: 去除后的纯净信号
    subplot(2, 3, 3);
    plot(t, signal_final, 'b', 'LineWidth', 1.2);
    title('去除眼电后的纯净信号', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('时间 (秒)');
    ylabel('幅值');
    grid on;
    
    % 左下: 原始信号频域
    [pxx_final, f_final] = pwelch(signal_final, hamming(min(length(signal), 512)), ...
        [], nfft, fs);
    [pxx_removed, f_removed] = pwelch(removed_signal, hamming(min(length(signal), 512)), ...
        [], nfft, fs);
    
    subplot(2, 3, 4);
    plot(f_orig, 10*log10(pxx_orig), 'k', 'LineWidth', 1.5);
    title('原始信号功率谱', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('频率 (Hz)');
    ylabel('功率 (dB)');
    xlim([0, min(fs/2, 50)]);
    grid on;
    
    % 中下: 去除的眼电成分频域
    subplot(2, 3, 5);
    plot(f_removed, 10*log10(pxx_removed), 'r', 'LineWidth', 1.5);
    hold on;
    xline(cutoff_freq, 'g--', sprintf('%.1fHz', cutoff_freq), 'LineWidth', 1.5);
    hold off;
    title('去除的眼电伪迹功率谱', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('频率 (Hz)');
    ylabel('功率 (dB)');
    xlim([0, min(fs/2, 50)]);
    grid on;
    
    % 右下: 去除后的信号频域
    subplot(2, 3, 6);
    plot(f_final, 10*log10(pxx_final), 'b', 'LineWidth', 1.5);
    hold on;
    plot(f_orig, 10*log10(pxx_orig), 'k--', 'LineWidth', 1, 'DisplayName', '原始');
    hold off;
    title('去除眼电后信号功率谱', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('频率 (Hz)');
    ylabel('功率 (dB)');
    xlim([0, min(fs/2, 50)]);
    legend('去除后', '原始', 'Location', 'best');
    grid on;
    
    sgtitle(sprintf('眼电去除效果对比 (去除 %d 个IMF, 样本熵阈值=%.2f)', ...
        num_removed, sampen_threshold), 'FontSize', 14, 'FontWeight', 'bold');
end

%% ========== 保存结果 ==========
fprintf('\n========== EWT + ICEEMDAN 分析完成! ==========\n');
fprintf('处理流程:\n');
fprintf('  1. EWT 自定义边界分离: ≤%.1fHz 和 >%.1fHz\n', cutoff_freq, cutoff_freq);
fprintf('  2. ICEEMDAN 分解低频: 得到 %d 个 IMF 分量\n', num_imfs);
if iceemdan_success
    fprintf('  3. 样本熵筛选: 保留 %d 个, 去除 %d 个 (阈值=%.2f)\n', ...
        num_kept, num_removed, sampen_threshold);
    fprintf('  4. 信号重构: 保留的IMF + 高频分量\n');
end
fprintf('  总分量数: %d\n', num_total_components);
fprintf('\n各分量能量占比:\n');
for i = 1:num_total_components
    fprintf('  %s: %.2f%%\n', component_names{i}, energy_percent_all(i));
end

if iceemdan_success && num_removed > 0
    fprintf('\n眼电去除结果:\n');
    fprintf('  去除能量占比: %.2f%%\n', energy_removed);
    fprintf('  重构信号相关系数: %.4f\n', correlation_before_after);
    fprintf('  去除的 IMF 编号: ');
    removed_indices = find(~imf_keep_mask);
    fprintf('%s\n', mat2str(removed_indices(:)'));
end

% user_input = input('\n是否保存结果? (输入 y 保存,其他键跳过): ', 's');
% 
% if strcmpi(user_input, 'y')
%     % 保存图像
%     saveas(fig1, 'ICEEMDAN_时域分解.png');
%     saveas(fig2, 'ICEEMDAN_频域分析.png');
%     saveas(fig3, 'ICEEMDAN_综合分析.png');
%     if exist('fig4', 'var')
%         saveas(fig4, 'ICEEMDAN_眼电去除对比.png');
%     end
%     
%     % 保存数据
%     if iceemdan_success
%         save('ICEEMDAN_结果.mat', 'IMFs', 'signal', 'signal_low', 'signal_high', ...
%             'signal_final', 'removed_signal', 'sampen_values', 'imf_keep_mask', ...
%             'all_components', 'component_names', 'fs', 'cutoff_freq', 'sampen_threshold');
%     else
%         save('ICEEMDAN_结果.mat', 'signal', 'signal_low', 'signal_high', ...
%             'fs', 'cutoff_freq');
%     end
%     
%     fprintf('结果已保存!\n');
% end

%% ========== 辅助函数: 生成模拟信号 ==========
function signal = generate_demo_signal(fs)
    % 生成包含多个频率成分的模拟脑电信号
    t = 0:1/fs:10-1/fs;  % 10秒信号
    
    % 各种频率成分
    delta = 0.8 * sin(2*pi*2*t);        % Delta波 (1-4 Hz)
    theta = 0.6 * sin(2*pi*6*t);        % Theta波 (4-8 Hz)
    alpha = 1.0 * sin(2*pi*10*t);       % Alpha波 (8-13 Hz)
    beta = 0.4 * sin(2*pi*20*t);        % Beta波 (13-30 Hz)
    gamma = 0.2 * sin(2*pi*40*t);       % Gamma波 (30-100 Hz)
    
    % 合成信号
    signal = delta + theta + alpha + beta + gamma;
    
    % 添加白噪声
    signal = signal + 0.1 * randn(size(signal));
    
    signal = signal(:);
    
    fprintf('生成模拟信号: %.1f 秒, 采样率 %d Hz\n', length(signal)/fs, fs);
    fprintf('包含频率成分: 2Hz, 6Hz, 10Hz, 20Hz, 40Hz\n');
end
