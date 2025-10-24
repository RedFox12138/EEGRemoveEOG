function artifact_signal = generate_ocular_artifacts(total_duration_s, segment_duration_s, Fs)
% generate_ocular_artifacts: 生成一个由眼动(EOG)和眨眼(Blink)伪迹拼接而成的单通道信号
    num_segments = total_duration_s / segment_duration_s;
    samples_per_segment = segment_duration_s * Fs;
    t_segment = (0:samples_per_segment-1) / Fs;
    artifact_signal = [];

    for i = 1:num_segments
        % 随机选择一种眼电伪迹类型 (1=眼动, 2=眨眼)
        artifact_type = randi(2);
        segment_data = zeros(1, samples_per_segment);
        
        switch artifact_type
            case 1
                % EOG (眼动): 0.2 Hz 的低频方波
                segment_data = 100 * square(2 * pi * 0.2 * t_segment);
                
            case 2
                % Eye blinking (眨眼): 1-3 Hz 的带通滤波噪声
                noise = randn(1, samples_per_segment);
                [b, a] = butter(4, [1 3] / (Fs/2), 'bandpass');
                segment_data = 200 * filter(b, a, noise);
        end
        
        artifact_signal = [artifact_signal, segment_data];
    end
end