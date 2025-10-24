function [Tfr, t, f, sig_low, sig_high] = swst(signal, fs, wavelet, scales)
    % 输入参数：
    % signal: 输入信号（一维向量）
    % fs: 采样频率（Hz）
    % wavelet: 小波基函数（如 'morl', 'amor', 'db4'）
    % scales: 尺度向量（建议使用对数尺度）
    
    % 输出参数：
    % Tfr: 同步压缩小波变换结果
    % t: 时间向量
    % f: 频率向量
    % sig_low: 1-8Hz频段的时域信号
    % sig_high: 8-40Hz频段的时域信号
    
    % 时间向量
    t = (0:length(signal)-1)/fs;
    
    % 计算连续小波变换（CWT）
    [cfs, freq] = cwt(signal, scales, wavelet, 'SamplingPeriod', 1/fs);
    
    % 估计瞬时频率（基于相位导数）
    phase = angle(cfs);
    dphase = diff(phase, 1, 2);  % 相位差分
    inst_freq = cumsum(dphase, 2) ./ (2*pi*fs);  % 累积积分得到瞬时频率
    
    % 同步压缩操作
    [Tfr, f] = wsstridge(cfs, inst_freq, freq);  % 使用WSST算法压缩能量
    
    % 创建频率范围掩码
    mask_low = (f >= 1) & (f <= 8);
    mask_high = (f >= 8) & (f <= 40);
    
    % 提取特定频率范围的时频表示
    Tfr_low = zeros(size(Tfr));
    Tfr_high = zeros(size(Tfr));
    Tfr_low(mask_low, :) = Tfr(mask_low, :);
    Tfr_high(mask_high, :) = Tfr(mask_high, :);
    
    % 逆同步压缩小波变换，将时频表示转换回时域信号
    % 这里使用cwt的逆变换，利用筛选后的系数
    sig_low = icwt(Tfr_low, scales, wavelet, 'SamplingPeriod', 1/fs);
    sig_high = icwt(Tfr_high, scales, wavelet, 'SamplingPeriod', 1/fs);
    
    % 确保输出信号与输入信号长度一致
    sig_low = sig_low(1:length(signal));
    sig_high = sig_high(1:length(signal));
    
    % 可视化结果
    figure;
    subplot(3,1,1);
    plot(t, signal);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('Original Signal');
    
    subplot(3,1,2);
    plot(t, sig_low);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('1-8Hz Band Signal');
    
    subplot(3,1,3);
    plot(t, sig_high);
    xlabel('Time (s)');
    ylabel('Amplitude');
    title('8-40Hz Band Signal');
    tight_layout;
    
    % 可视化时频分布
    figure;
    subplot(3,1,1);
    imagesc(t, f, abs(Tfr));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('Full Frequency Range');
    colorbar;
    
    subplot(3,1,2);
    imagesc(t, f(mask_low), abs(Tfr(mask_low,:)));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('1-8Hz Frequency Range');
    colorbar;
    
    subplot(3,1,3);
    imagesc(t, f(mask_high), abs(Tfr(mask_high,:)));
    axis xy;
    xlabel('Time (s)');
    ylabel('Frequency (Hz)');
    title('8-40Hz Frequency Range');
    colorbar;
    tight_layout;
end
