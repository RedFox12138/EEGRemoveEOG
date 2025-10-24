function Xp = generate_pure_eeg(num_channels, total_duration_s, segment_duration_s, Fs)
% generate_pure_eeg: 生成多通道纯净模拟EEG信号 (PSEthal)
    num_segments = total_duration_s / segment_duration_s;
    samples_per_segment = segment_duration_s * Fs;
    total_samples = total_duration_s * Fs;
    t_segment = (0:samples_per_segment-1) / Fs;
    Xp = zeros(num_channels, total_samples);
    for i = 1:num_channels
        channel_data = [];
        for j = 1:num_segments
            freqs = 4 + (30 - 4) * rand(4, 1);
            phases = 2 * pi * rand(4, 1);
            amps = 0.5 + rand(4, 1);
            sinusoids = amps .* sin(2 * pi * freqs .* t_segment + phases);
            segment_data = sum(sinusoids, 1);
            channel_data = [channel_data, segment_data];
        end
        Xp(i, :) = channel_data;
    end
end