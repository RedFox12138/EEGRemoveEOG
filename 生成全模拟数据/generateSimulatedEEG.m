function [pureEEG, contaminatedEEG, eogArtifact, blinkArtifact] = generateSimulatedEEG(duration_sec, targetSNREOG, targetSNRBlink, eogStrength, blinkStrength)
    % 参数设置
    fs = 250;
    numChannels = 4000; % 4000个独立样本
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
    fprintf('Generating independent samples for each channel...\n');
    for ch = 1:numChannels
        % 当前样本类型 (1:无噪声, 2:仅EOG, 3:仅眨眼, 4:混合)
        sampleType = typeIndices(ch);
        
        %% 1. 生成纯净脑电 (每个通道独立)
        freqs = 4 + rand(8, 1) * 18; % 4-22 Hz的脑电频率
        channelEEG = zeros(1, numSamples);
        for f = 1:8
            phase = 2 * pi * rand();
            channelEEG = channelEEG + sin(2 * pi * freqs(f) * t + phase);
        end
        pureEEG(ch, :) = channelEEG;
        
        %% 2. 生成EOG伪迹 (根据样本类型决定是否生成)
        eog_single = zeros(1, numSamples);
        if sampleType == 2 || sampleType == 4 % 需要EOG的样本类型
            % 随机生成EOG事件的位置
            eog_interval = 4 + 4 * rand(); % 间隔在4-8秒之间变化
            num_eog_events = round(duration_sec / eog_interval);
            eog_positions = sort(randperm(round(numSamples * 0.9), num_eog_events));
            
            % 为每个EOG事件创建短方波
            for i = 1:num_eog_events
                pos = eog_positions(i);
                
                % 方波持续时间 (0.6-1.2秒)
                eog_duration = 0.6 + 0.6 * rand();
                eog_samples = round(eog_duration * fs);
                
                % 确保不超过信号长度
                end_pos = min(pos + eog_samples - 1, numSamples);
                
                % 创建方波
                amplitude = 0.6 * (0.8 + 0.4 * rand()); % 幅度随机变化
                eog_single(pos:end_pos) = amplitude;
                
                % 随机选择正负极性
                if rand() > 0.5
                    eog_single(pos:end_pos) = -eog_single(pos:end_pos);
                end
            end
        end
        eogArtifact(ch, :) = eogStrength * eog_single;
        
        %% 3. 生成眨眼伪迹 (根据样本类型决定是否生成)
        blink_oscillation = zeros(1, numSamples);
        if sampleType == 3 || sampleType == 4 % 需要眨眼的样本类型
            % 随机生成眨眼事件的位置
            blink_interval = 2 + 2 * rand(); % 间隔在2-4秒之间变化
            num_blinks = round(duration_sec / blink_interval);
            blink_positions = sort(randperm(round(numSamples * 0.9), num_blinks));
            
            % 为每个眨眼事件创建振荡
            for i = 1:num_blinks
                pos = blink_positions(i);
                
                % 随机选择频率(1-3 Hz)
                blink_freq = 1 + 2 * rand();
                
                % 随机确定振荡持续时间(0.3-0.8秒)
                blink_duration = 0.3 + 0.5 * rand();
                blink_samples = round(blink_duration * fs);
                
                % 确保不超过信号长度
                end_pos = min(pos + blink_samples - 1, numSamples);
                actual_samples = end_pos - pos + 1;
                
                % 创建时间向量
                t_blink = (0:actual_samples-1) / fs;
                
                % 生成振荡信号
                oscillation = sin(2 * pi * blink_freq * t_blink);
                
                % 应用包络
                envelope = sin(pi * (0:actual_samples-1) / actual_samples);
                oscillation = oscillation .* envelope;
                
                % 添加到基础信号
                blink_oscillation(pos:end_pos) = blink_oscillation(pos:end_pos) + oscillation;
            end
            
            % 标准化
            if max(abs(blink_oscillation)) > 0
                blink_oscillation = blink_oscillation / max(abs(blink_oscillation));
            end
        end
        blinkArtifact(ch, :) = blinkStrength * blink_oscillation;
        
        %% 4. 混合信号 (为EOG与眨眼分别匹配目标SNR)
        eog_ch = eogArtifact(ch, :);
        blink_ch = blinkArtifact(ch, :);

        rms_pure = rms(pureEEG(ch, :));
        lambda_eog = 0; lambda_blink = 0;

        % EOG缩放
        if (sampleType == 2 || sampleType == 4)
            rms_eog = rms(eog_ch);
            if rms_eog > 0 && targetSNREOG > 0
                lambda_eog = rms_pure / (targetSNREOG * rms_eog);
            else
                lambda_eog = 0;
            end
        end
        
        % 眨眼缩放
        if (sampleType == 3 || sampleType == 4)
            rms_blink = rms(blink_ch);
            if rms_blink > 0 && targetSNRBlink > 0
                lambda_blink = rms_pure / (targetSNRBlink * rms_blink);
            else
                lambda_blink = 0;
            end
        end

        contaminatedEEG(ch, :) = pureEEG(ch, :) + lambda_eog * eog_ch + lambda_blink * blink_ch;

        % 进度显示
        if mod(ch, 1000) == 0
            fprintf('Processed %d/%d samples\n', ch, numChannels);
        end
    end
    
    % 显示样本类型分布
    typeCounts = [sum(typeIndices==1), sum(typeIndices==2), sum(typeIndices==3), sum(typeIndices==4)];
    fprintf('\nData generation complete! Sample distribution:\n');
    fprintf('  No Noise:      %d (%.1f%%)\n', typeCounts(1), typeCounts(1)/numChannels*100);
    fprintf('  Only EOG:      %d (%.1f%%)\n', typeCounts(2), typeCounts(2)/numChannels*100);
    fprintf('  Only Blink:    %d (%.1f%%)\n', typeCounts(3), typeCounts(3)/numChannels*100);
    fprintf('  EOG+Blink:     %d (%.1f%%)\n', typeCounts(4), typeCounts(4)/numChannels*100);
    
    %% 保存数据
    save('D:\Pycharm_Projects\EOG Remove\生成全模拟数据\已经生成好的数据\Pure_Data.mat', 'pureEEG');
    save('D:\Pycharm_Projects\EOG Remove\生成全模拟数据\已经生成好的数据\Contaminated.mat', 'contaminatedEEG');
    
    %% 绘制结果用于验证 (每种类型显示1个样本)
    figure('Position', [100, 100, 1600, 1200]);
    for type = 1:4
        % 找到该类型的第一个样本
        idx = find(typeIndices == type, 1);
        
        % 绘制EOG信号
        subplot(4, 3, (type-1)*3 + 1);
        plot(t, eogArtifact(idx, :));
        title(sprintf('Type %d: EOG Artifact', type));
        xlabel('Time (s)');
        ylabel('Amplitude');
        xlim([0, 10]);
        grid on;
        
        % 绘制眨眼信号
        subplot(4, 3, (type-1)*3 + 2);
        plot(t, blinkArtifact(idx, :));
        title('Blink Artifact');
        xlabel('Time (s)');
        ylabel('Amplitude');
        xlim([0, 10]);
        grid on;
        
        % 绘制污染后的EEG信号
        subplot(4, 3, (type-1)*3 + 3);
        plot(t, contaminatedEEG(idx, :));
        title('Contaminated EEG');
        xlabel('Time (s)');
        ylabel('Amplitude');
        xlim([0, 10]);
        grid on;
    end
    sgtitle('Sample Signals for Each Type');
end