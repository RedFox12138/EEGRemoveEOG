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

% 创建时间向量用于ICEEMDAN
t = (0:length(signal)-1) / fs;  % 时间向量(秒)

% 执行 ICEEMDAN 分解
try
    % pICEEMDAN(data, FsOrT, Nstd, NE, MaxIter)
    % 这里用fs（采样频率）而不是时间向量t，因为更简单
    IMFs = pICEEMDAN(signal_low, fs, Nstd, NR, MaxIter);
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
    sampen_threshold = 0.3;
    
    % 计算每个 IMF 的样本熵
    sampen_values = zeros(num_imfs, 1);
    for i = 1:num_imfs
        try
            % SampEn函数使用name-value pairs参数
            % 默认参数：m=2, tau=1, r=0.2*std(Sig)
            Samp = SampEn(IMFs(i, :), 'm', 2, 'r', 0.2*std(IMFs(i, :)));
            sampen_values(i) = Samp(end);  % 取最后一个值（m=2的样本熵）
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

%% ========== 绘制眼电去噪结果对比图 ==========
if iceemdan_success
    fprintf('\n正在绘制眼电去噪结果对比图...\n');
    
    % 创建时间向量
    t = (0:length(signal)-1) / fs;
    
    % 创建图形窗口
    figure('Name', 'EWT-ICEEMDAN眼电去噪结果对比', 'Position', [100, 100, 1400, 900]);
    
    % ========== 子图1: 原始含噪信号（时域）==========
    subplot(3, 2, 1);
    plot(t, signal, 'k', 'LineWidth', 1);
    title('原始含眼电伪迹的脑电信号', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)');
    xlabel('时间 (秒)');
    grid on;
    xlim([t(1), t(end)]);
    
    % ========== 子图2: 去噪后信号（时域）==========
    subplot(3, 2, 2);
    plot(t, signal_final, 'b', 'LineWidth', 1);
    title('去眼电后的脑电信号', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)');
    xlabel('时间 (秒)');
    grid on;
    xlim([t(1), t(end)]);
    
    % ========== 子图3: 被去除的眼电成分（时域）==========
    subplot(3, 2, 3);
    removed_signal = sum(IMFs(imf_remove_mask, :), 1);
    plot(t, removed_signal, 'r', 'LineWidth', 1);
    title(sprintf('去除的眼电伪迹成分 (共%d个IMF)', num_removed), 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)');
    xlabel('时间 (秒)');
    grid on;
    xlim([t(1), t(end)]);
    
    % ========== 子图4: 原始与去噪后信号叠加对比 ==========
    subplot(3, 2, 4);
    plot(t, signal, 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', '原始信号');
    hold on;
    plot(t, signal_final, 'b', 'LineWidth', 1, 'DisplayName', '去噪后信号');
    hold off;
    title('原始信号与去噪后信号叠加对比', 'FontSize', 11, 'FontWeight', 'bold');
    ylabel('幅值 (μV)');
    xlabel('时间 (秒)');
    legend('Location', 'best');
    grid on;
    xlim([t(1), t(end)]);
    
    % ========== 子图5: 功率谱对比 ==========
    subplot(3, 2, 5);
    % 计算功率谱
    [pxx_orig, f_orig] = pwelch(signal, hamming(min(length(signal), 512)), ...
        [], [], fs);
    [pxx_final, f_final] = pwelch(signal_final, hamming(min(length(signal_final), 512)), ...
        [], [], fs);
    
    plot(f_orig, 10*log10(pxx_orig), 'Color', [0.7 0.7 0.7], 'LineWidth', 1.5, 'DisplayName', '原始信号');
    hold on;
    plot(f_final, 10*log10(pxx_final), 'b', 'LineWidth', 1.5, 'DisplayName', '去噪后信号');
    hold off;
    title('功率谱对比', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('频率 (Hz)');
    ylabel('功率谱密度 (dB/Hz)');
    xlim([0, min(fs/2, 50)]);
    legend('Location', 'best');
    grid on;
    
    % ========== 子图6: 样本熵分布 ==========
    subplot(3, 2, 6);
    bar(1:num_imfs, sampen_values, 'FaceColor', [0.3 0.6 0.9]);
    hold on;
    % 绘制阈值线
    yline(sampen_threshold, 'r--', 'LineWidth', 2, 'Label', sprintf('阈值=%.2f', sampen_threshold));
    % 标记被去除的IMF
    removed_indices = find(imf_remove_mask);
    if ~isempty(removed_indices)
        bar(removed_indices, sampen_values(removed_indices), 'FaceColor', [1 0.3 0.3]);
    end
    hold off;
    title('各IMF样本熵分布', 'FontSize', 11, 'FontWeight', 'bold');
    xlabel('IMF编号');
    ylabel('样本熵值');
    grid on;
    legend('保留的IMF', sprintf('阈值=%.2f', sampen_threshold), '去除的IMF(眼电)', ...
        'Location', 'best');
    
    % ========== 总标题 ==========
    sgtitle(sprintf('EWT-ICEEMDAN眼电去噪结果  |  去除%d个IMF  |  样本熵阈值=%.2f  |  相关系数=%.4f', ...
        num_removed, sampen_threshold, correlation_before_after), ...
        'FontSize', 13, 'FontWeight', 'bold');
    
    fprintf('眼电去噪结果对比图已生成！\n');
    fprintf('图形说明:\n');
    fprintf('  - 左上: 原始含眼电伪迹的脑电信号\n');
    fprintf('  - 右上: 去除眼电后的干净脑电信号\n');
    fprintf('  - 左中: 被识别并去除的眼电伪迹成分\n');
    fprintf('  - 右中: 原始与去噪后信号的叠加对比\n');
    fprintf('  - 左下: 功率谱对比（频域）\n');
    fprintf('  - 右下: 各IMF的样本熵分布及筛选结果\n');
end

fprintf('\n========== EWT-ICEEMDAN 眼电去除完成！==========\n');