clear all;
clf;
close all;
% 设置参数
dataLength = 6; % 生成6秒的数据
snrEOG = 1.5;      % EOG目标信噪比（SNR越小，伪迹越强）
snrBlink = 0.5;    % 眨眼目标信噪比（可与EOG不同）
eogStrength = 3;   % EOG强度
blinkStrength = 10; % 眨眼强度
fs = 250;
% 调用修改后的函数生成数据
[pureEEG, contaminatedEEG, eogArtifact, blinkArtifact] = ...
    generateSimulatedEEG(dataLength, snrEOG, snrBlink, eogStrength, blinkStrength);

% 选择要绘图的通道（例如第1通道）
channelToPlot = 1;

% 绘制结果
t = (0:size(pureEEG, 2)-1) / fs; % 时间轴（秒）

figure('Position', [100, 100, 1200, 800]);
subplot(4, 1, 1);
plot(t, pureEEG(channelToPlot, :));
title(sprintf('Channel %d: Pure EEG (PSEEG)', channelToPlot));
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
xlim([0, 10]); % 只显示前10秒以便观察

subplot(4, 1, 2);
plot(t, eogArtifact(channelToPlot, :));
title(sprintf('Channel %d: EOG Artifact', channelToPlot));
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
xlim([0, 10]);

subplot(4, 1, 3);
plot(t, blinkArtifact(channelToPlot, :));
title(sprintf('Channel %d: Blink Artifact (1-3 Hz Low-frequency Events)', channelToPlot));
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
xlim([0, 10]);

subplot(4, 1, 4);
plot(t, contaminatedEEG(channelToPlot, :));
title(sprintf('Channel %d: Contaminated EEG (SNR_EOG=%.2f, SNR_Blink=%.2f)', channelToPlot, snrEOG, snrBlink));
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
xlim([0, 10]);

% 添加频谱分析以验证眨眼信号的频率特性
figure('Position', [100, 100, 1000, 800]);

% 计算并绘制眨眼信号的频谱
subplot(2,1,1);
blink_single_channel = blinkArtifact(channelToPlot, :);
L = length(blink_single_channel);
Y = fft(blink_single_channel);
P2 = abs(Y/L);
P1 = P2(1:L/2+1);
P1(2:end-1) = 2*P1(2:end-1);
f = fs*(0:(L/2))/L;
plot(f, P1);
title('Single-Sided Amplitude Spectrum of Blink Artifact');
xlabel('Frequency (Hz)');
ylabel('|P1(f)|');
xlim([0, 10]); % 重点关注0-10Hz范围
grid on;

% 标记1-3 Hz范围
hold on;
area([1, 3], [max(P1), max(P1)], 'FaceColor', 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
text(2, max(P1)*0.9, '1-3 Hz Range', 'HorizontalAlignment', 'center');

% 绘制纯净EEG和污染EEG的对比
subplot(2,1,2);
plot(t, pureEEG(channelToPlot, :), 'b', 'LineWidth', 1.5);
hold on;
plot(t, contaminatedEEG(channelToPlot, :), 'r', 'LineWidth', 1);
title('Comparison: Pure EEG (Blue) vs Contaminated EEG (Red)');
xlabel('Time (s)');
ylabel('Amplitude (a.u.)');
xlim([0, 10]);
legend('Pure EEG', 'Contaminated EEG');
grid on;