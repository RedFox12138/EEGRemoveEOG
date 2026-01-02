function [pureEEG, contaminatedEEG, eogArtifact, blinkArtifact, typeIndices] = ...
    generateSimulatedEEG_MultiSNR(duration_sec, targetSNREOG_dB, targetSNRBlink_dB, numChannels)
% ========================================================================
% 生成多SNR等级的全模拟脑电数据
% ========================================================================
% 参数:
%   duration_sec: 信号持续时间(秒)
%   targetSNREOG_dB: 眼动伪影目标信噪比(dB)
%   targetSNRBlink_dB: 眨眼伪影目标信噪比(dB)
%   numChannels: 要生成的总样本数
%
% 返回:
%   pureEEG: 纯净EEG信号 [numChannels, numSamples]
%   contaminatedEEG: 污染的EEG信号 [numChannels, numSamples]
%   eogArtifact: 眼动伪影 [numChannels, numSamples]
%   blinkArtifact: 眨眼伪影 [numChannels, numSamples]
%   typeIndices: 样本类型索引 [numChannels, 1] (1:无噪声, 2:仅眼动, 3:仅眨眼, 4:混合)
%
% SNR计算公式 (参考generate_multi_snr_dataset.py):
%   SNR_dB = 20 * log10(RMS_signal / RMS_noise)
%   lambda = RMS_signal / (RMS_noise * 10^(SNR_dB / 20))
% ========================================================================

    % 参数设置
    fs = 250;
    t = 0:1/fs:duration_sec-1/fs;
    numSamples = length(t);
    
    % 初始化
    pureEEG = zeros(numChannels, numSamples);
    eogArtifact = zeros(numChannels, numSamples);
    blinkArtifact = zeros(numChannels, numSamples);
    contaminatedEEG = zeros(numChannels, numSamples);
    
    % 计算每种类型的样本数量（确保总和为numChannels）
    quarter = floor(numChannels / 4);
    counts = [quarter, quarter, quarter, numChannels - 3*quarter];
    typeIndices = [];
    for i = 1:4
        typeIndices = [typeIndices; i * ones(counts(i), 1)];
    end
    % 随机打乱样本类型分配
    typeIndices = typeIndices(randperm(numChannels));
    
    %% 为每个通道独立生成数据
    fprintf('  生成 %d 个独立样本...\n', numChannels);
    for ch = 1:numChannels
        % 当前样本类型 (1:无噪声, 2:仅眼动, 3:仅眨眼, 4:混合)
        sampleType = typeIndices(ch);
        
        %% 1. 生成纯净脑电 (每个通道独立)
        % 采用"Aperiodic background + Periodic oscillations"策略
        
        % 1.1 生成 1/f 粉红噪声背景
        % 先生成高斯白噪声
        white_noise = randn(1, numSamples);
        
        % 应用 1/f 滤波器生成粉红噪声
        % 设计一个低通滤波器，通过频域操作实现 1/f 特性
        fft_white = fft(white_noise);
        n_fft = length(fft_white);
        freqs_fft = (0:n_fft-1) * fs / n_fft;
        
        % 创建 1/f 功率谱密度衰减
        pink_filter = ones(1, n_fft);
        for k = 2:n_fft
            freq = freqs_fft(k);
            if freq > 0
                pink_filter(k) = 1 / sqrt(freq); % 1/f^0.5 功率谱
            end
        end
        
        % 应用滤波器
        fft_pink = fft_white .* pink_filter;
        pink_noise = real(ifft(fft_pink));
        
        % 归一化粉红噪声（RMS约为1）
        pink_noise = pink_noise / std(pink_noise);
        
        % 1.2 生成周期性震荡成分
        % 随机选择5-10个震荡频率
        num_oscillations = 8;
        oscillation_signal = zeros(1, numSamples);
        
        for f = 1:num_oscillations
            % 随机频率 4-30 Hz
            freq = 4 + rand() * 26;
            
            % 随机相位
            phase = 2 * pi * rand();
            
            % 频率相关的幅度：低频幅度大于高频
            % Alpha波(8-13Hz)幅度较大，Beta波(13-30Hz)幅度较小
            if freq < 13
                amplitude = 0.8 + 0.7 * rand(); % Alpha及以下：0.8-1.5
            else
                amplitude = 0.4 + 0.6 * rand(); % Beta及以上：0.4-1.0
            end
            
            % 生成正弦波
            oscillation_signal = oscillation_signal + amplitude * sin(2 * pi * freq * t + phase);
        end
        
        % 归一化震荡信号
        if std(oscillation_signal) > 0
            oscillation_signal = oscillation_signal / std(oscillation_signal);
        end
        
        % 1.3 混合背景噪声和震荡成分
        % 混合比例：70%震荡 + 30%粉红噪声
        channelEEG = 0.7 * oscillation_signal + 0.3 * pink_noise;
        
        % 【关键】幅度归一化：统一能量基准
        % 1) 计算RMS值
        rms_eeg = sqrt(mean(channelEEG.^2));
        
        % 2) 标准化为RMS=1
        if rms_eeg > 0
            channelEEG = channelEEG / rms_eeg;
        end
        
        % 3) 乘以20，模拟真实脑电的20-50微伏基线水平
        channelEEG = channelEEG * 20;
        
        pureEEG(ch, :) = channelEEG;
        
        %% 2. 生成眼动伪迹 (根据样本类型决定是否生成)
        eog_single = zeros(1, numSamples);
        if sampleType == 2 || sampleType == 4 % 需要眼动的样本类型
            % 直接随机指定眼动事件次数 (1-3次)
            num_eog_events = randi([1, 3]);
            
            % 用于记录已占用的位置，防止重叠
            occupied_ranges = [];
            min_interval_samples = round(1.5 * fs); % 最小间隔1.5秒
            
            actual_eog_count = 0;
            max_attempts = 20; % 最大尝试次数
            
            % 为每个眼动事件创建短方波
            for i = 1:num_eog_events
                % 方波持续时间 (0.5-2.0秒) - 符合文献要求
                eog_duration = 0.5 + 1.5 * rand();
                eog_samples = round(eog_duration * fs);
                
                % 尝试找到不冲突的位置
                pos_found = false;
                for attempt = 1:max_attempts
                    % 随机选择起始位置（留出足够空间）
                    pos = randi([1, max(1, numSamples - eog_samples)]);
                    end_pos = min(pos + eog_samples - 1, numSamples);
                    
                    % 检查是否与已有伪影冲突
                    conflict = false;
                    for j = 1:size(occupied_ranges, 1)
                        existing_start = occupied_ranges(j, 1);
                        existing_end = occupied_ranges(j, 2);
                        % 检查是否有重叠或间隔过近
                        if (pos <= existing_end + min_interval_samples) && ...
                           (end_pos >= existing_start - min_interval_samples)
                            conflict = true;
                            break;
                        end
                    end
                    
                    if ~conflict
                        pos_found = true;
                        break;
                    end
                end
                
                % 如果找到合适位置，生成伪影
                if pos_found
                    % 创建方波
                    amplitude = 0.6 * (0.8 + 0.4 * rand()); % 幅度随机变化
                    eog_single(pos:end_pos) = amplitude;
                    
                    % 随机选择正负极性
                    if rand() > 0.5
                        eog_single(pos:end_pos) = -eog_single(pos:end_pos);
                    end
                    
                    % 记录占用范围
                    occupied_ranges = [occupied_ranges; pos, end_pos];
                    actual_eog_count = actual_eog_count + 1;
                end
            end
            
            % 【关键】归一化EOG到最大幅值为1
            % 让后续lambda计算完全由SNR公式控制
            if max(abs(eog_single)) > 0
                eog_single = eog_single / max(abs(eog_single));
            end
        end
        eogArtifact(ch, :) = eog_single; % 已归一化，后续根据SNR调整
        
        %% 3. 生成眨眼伪迹 (根据样本类型决定是否生成)
        % 参考文献："Eye blinking is synthesized using random noise band-pass filtered between 1 and 3 Hz"
        blink_oscillation = zeros(1, numSamples);
        if sampleType == 3 || sampleType == 4 % 需要眨眼的样本类型
            % 直接随机指定眨眼事件次数 (1-3次)
            num_blinks = randi([1, 3]);
            
            % 设计1-3 Hz带通滤波器（4阶巴特沃斯）
            low_freq = 1;    % 下限频率 1 Hz
            high_freq = 3;   % 上限频率 3 Hz
            filter_order = 4; % 4阶滤波器
            nyquist_freq = fs / 2;
            [b_blink, a_blink] = butter(filter_order, [low_freq high_freq] / nyquist_freq, 'bandpass');
            
            % 用于记录已占用的位置，防止重叠
            occupied_ranges = [];
            min_interval_samples = round(1.5 * fs); % 最小间隔1.5秒
            
            actual_blink_count = 0;
            max_attempts = 20; % 最大尝试次数
            
            % 为每个眨眼事件生成带通滤波的随机噪声
            for i = 1:num_blinks
                % 随机确定眨眼持续时间(0.5-2.0秒) - 符合文献要求
                blink_duration = 0.5 + 1.5 * rand();
                blink_samples = round(blink_duration * fs);
                
                % 尝试找到不冲突的位置
                pos_found = false;
                for attempt = 1:max_attempts
                    % 随机选择起始位置（留出足够空间）
                    pos = randi([1, max(1, numSamples - blink_samples)]);
                    end_pos = min(pos + blink_samples - 1, numSamples);
                    actual_samples = end_pos - pos + 1;
                    
                    % 检查是否与已有伪影冲突
                    conflict = false;
                    for j = 1:size(occupied_ranges, 1)
                        existing_start = occupied_ranges(j, 1);
                        existing_end = occupied_ranges(j, 2);
                        % 检查是否有重叠或间隔过近
                        if (pos <= existing_end + min_interval_samples) && ...
                           (end_pos >= existing_start - min_interval_samples)
                            conflict = true;
                            break;
                        end
                    end
                    
                    if ~conflict
                        pos_found = true;
                        break;
                    end
                end
                
                % 如果找到合适位置，生成伪影
                if pos_found
                    % 生成高斯白噪声
                    white_noise = randn(1, actual_samples);
                    
                    % 应用1-3 Hz带通滤波器
                    filtered_noise = filtfilt(b_blink, a_blink, white_noise);
                    
                    % 应用包络以产生平滑的眨眼波形
                    envelope = sin(pi * (0:actual_samples-1) / actual_samples);
                    blink_waveform = filtered_noise .* envelope;
                    
                    % 归一化到[-1, 1]范围
                    if max(abs(blink_waveform)) > 0
                        blink_waveform = blink_waveform / max(abs(blink_waveform));
                    end
                    
                    % 添加到基础信号
                    blink_oscillation(pos:end_pos) = blink_oscillation(pos:end_pos) + blink_waveform;
                    
                    % 记录占用范围
                    occupied_ranges = [occupied_ranges; pos, end_pos];
                    actual_blink_count = actual_blink_count + 1;
                end
            end
            
            % 【关键】归一化Blink到最大幅值为1
            % 让后续lambda计算完全由SNR公式控制
            if max(abs(blink_oscillation)) > 0
                blink_oscillation = blink_oscillation / max(abs(blink_oscillation));
            end
        end
        blinkArtifact(ch, :) = blink_oscillation; % 已归一化，后续根据SNR调整
        
        %% 4. 根据目标SNR调整伪影强度并混合信号
        % 使用与generate_multi_snr_dataset.py相同的SNR计算方法
        eog_ch = eogArtifact(ch, :);
        blink_ch = blinkArtifact(ch, :);
        pure_ch = pureEEG(ch, :);

        rms_pure = sqrt(mean(pure_ch.^2));  % 计算纯净信号的RMS
        lambda_eog = 0; 
        lambda_blink = 0;

        % 眼动伪影缩放 (使用SNR公式)
        if (sampleType == 2 || sampleType == 4)
            % 【关键修正】只计算有效伪影部分的RMS，避免大量零值拉低RMS
            % 方法：只计算绝对值超过最大值10%的部分
            eog_abs = abs(eog_ch);
            eog_threshold = 0.1 * max(eog_abs);
            eog_valid_mask = eog_abs > eog_threshold;
            
            if sum(eog_valid_mask) > 0
                % 只用有效部分计算RMS
                rms_eog = sqrt(mean(eog_ch(eog_valid_mask).^2));
                if rms_eog > 0
                    % SNR_dB = 20 * log10(RMS_signal / RMS_noise)
                    % => lambda = RMS_signal / (RMS_noise * 10^(SNR_dB / 20))
                    lambda_eog = rms_pure / (rms_eog * 10^(targetSNREOG_dB / 20));
                else
                    lambda_eog = 0;
                end
            else
                lambda_eog = 0;
            end
        end
        
        % 眨眼伪影缩放 (使用SNR公式)
        if (sampleType == 3 || sampleType == 4)
            % 【关键修正】只计算有效伪影部分的RMS
            blink_abs = abs(blink_ch);
            blink_threshold = 0.1 * max(blink_abs);
            blink_valid_mask = blink_abs > blink_threshold;
            
            if sum(blink_valid_mask) > 0
                % 只用有效部分计算RMS
                rms_blink = sqrt(mean(blink_ch(blink_valid_mask).^2));
                if rms_blink > 0
                    lambda_blink = rms_pure / (rms_blink * 10^(targetSNRBlink_dB / 20));
                else
                    lambda_blink = 0;
                end
            else
                lambda_blink = 0;
            end
        end

        % 应用缩放系数
        eogArtifact(ch, :) = lambda_eog * eog_ch;
        blinkArtifact(ch, :) = lambda_blink * blink_ch;
        
        % 生成污染信号
        contaminatedEEG(ch, :) = pure_ch + eogArtifact(ch, :) + blinkArtifact(ch, :);

        % 进度显示
        if mod(ch, 1000) == 0
            fprintf('    已处理 %d/%d 样本\n', ch, numChannels);
        end
    end
    
    % 显示样本类型分布
    typeCounts = [sum(typeIndices==1), sum(typeIndices==2), sum(typeIndices==3), sum(typeIndices==4)];
    fprintf('  样本类型分布:\n');
    fprintf('    无干扰:        %d (%.1f%%)\n', typeCounts(1), typeCounts(1)/numChannels*100);
    fprintf('    仅眼动:        %d (%.1f%%)\n', typeCounts(2), typeCounts(2)/numChannels*100);
    fprintf('    仅眨眼:        %d (%.1f%%)\n', typeCounts(3), typeCounts(3)/numChannels*100);
    fprintf('    眼动+眨眼:     %d (%.1f%%)\n', typeCounts(4), typeCounts(4)/numChannels*100);
    
end
