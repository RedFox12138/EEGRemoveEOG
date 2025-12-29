%% 验证全模拟数据集的实际SNR
% 计算已生成数据的真实信噪比
%
% 作者: GitHub Copilot
% 日期: 2025-12-24

clear; clc;

%% 加载数据
fprintf('加载全模拟数据集...\n');
data_dir = 'D:\Pycharm_Projects\EOG Remove\生成全模拟数据\已经生成好的数据';

pure_data = load(fullfile(data_dir, 'Pure_Data.mat'));
contaminated_data = load(fullfile(data_dir, 'Contaminated.mat'));

pureEEG = pure_data.pureEEG;
contaminatedEEG = contaminated_data.contaminatedEEG;

[num_channels, num_samples] = size(pureEEG);
fprintf('数据规模: %d 通道 × %d 样本点\n\n', num_channels, num_samples);

%% 计算每个通道的噪声和SNR
fprintf('计算每个通道的实际SNR...\n');

snr_values = zeros(num_channels, 1);
has_noise = false(num_channels, 1);

for ch = 1:num_channels
    % 提取信号
    pure = pureEEG(ch, :);
    contaminated = contaminatedEEG(ch, :);
    
    % 计算噪声（伪迹）
    noise = contaminated - pure;
    
    % 计算RMS
    rms_pure = rms(pure);
    rms_noise = rms(noise);
    
    % 判断是否有噪声
    if rms_noise > 1e-10  % 阈值避免除零
        has_noise(ch) = true;
        % 标准SNR公式 (dB)
        snr_values(ch) = 20 * log10(rms_pure / rms_noise);
    else
        has_noise(ch) = false;
        snr_values(ch) = inf;  % 无噪声
    end
    
    if mod(ch, 1000) == 0
        fprintf('  已处理 %d/%d 通道\n', ch, num_channels);
    end
end

%% 统计结果
fprintf('\n==============================================\n');
fprintf('              实际SNR统计结果\n');
fprintf('==============================================\n\n');

% 分离有噪声和无噪声的通道
snr_with_noise = snr_values(has_noise);
num_with_noise = sum(has_noise);
num_without_noise = sum(~has_noise);

fprintf('通道分布:\n');
fprintf('  有噪声通道: %d (%.1f%%)\n', num_with_noise, num_with_noise/num_channels*100);
fprintf('  无噪声通道: %d (%.1f%%)\n\n', num_without_noise, num_without_noise/num_channels*100);

if num_with_noise > 0
    fprintf('有噪声通道的SNR统计 (单位: dB):\n');
    fprintf('  平均值:   %.2f dB\n', mean(snr_with_noise));
    fprintf('  中位数:   %.2f dB\n', median(snr_with_noise));
    fprintf('  标准差:   %.2f dB\n', std(snr_with_noise));
    fprintf('  最小值:   %.2f dB\n', min(snr_with_noise));
    fprintf('  最大值:   %.2f dB\n', max(snr_with_noise));
    fprintf('  25分位:   %.2f dB\n', prctile(snr_with_noise, 25));
    fprintf('  75分位:   %.2f dB\n\n', prctile(snr_with_noise, 75));
end

%% 绘制SNR分布直方图
if num_with_noise > 0
    figure('Position', [100, 100, 1200, 600]);
    
    subplot(1, 2, 1);
    histogram(snr_with_noise, 50, 'FaceColor', [0.2 0.6 0.8]);
    xlabel('SNR (dB)');
    ylabel('通道数量');
    title('SNR分布直方图');
    grid on;
    
    % 添加统计线
    hold on;
    yl = ylim;
    plot([mean(snr_with_noise), mean(snr_with_noise)], yl, 'r--', 'LineWidth', 2);
    plot([median(snr_with_noise), median(snr_with_noise)], yl, 'g--', 'LineWidth', 2);
    legend('SNR分布', sprintf('均值 = %.2f dB', mean(snr_with_noise)), ...
           sprintf('中位数 = %.2f dB', median(snr_with_noise)));
    
    subplot(1, 2, 2);
    boxplot(snr_with_noise);
    ylabel('SNR (dB)');
    title('SNR箱线图');
    grid on;
    
    sgtitle('全模拟数据集实际SNR分析');
end

%% 详细分析：尝试区分EOG和眨眼伪迹
fprintf('==============================================\n');
fprintf('       尝试区分不同类型伪迹的SNR\n');
fprintf('==============================================\n\n');

% 根据噪声能量分布尝试识别类型
noise_energies = zeros(num_channels, 1);
for ch = 1:num_channels
    if has_noise(ch)
        noise = contaminatedEEG(ch, :) - pureEEG(ch, :);
        noise_energies(ch) = sum(noise.^2);
    end
end

% 使用K-means聚类（假设有3-4个簇：无噪声、仅EOG、仅眨眼、混合）
valid_energies = noise_energies(has_noise);
valid_snrs = snr_values(has_noise);

if length(valid_energies) > 10
    try
        [idx, C] = kmeans([valid_energies, valid_snrs], 3);
        
        fprintf('聚类分析结果（3个簇）:\n');
        for cluster = 1:3
            cluster_snrs = valid_snrs(idx == cluster);
            fprintf('  簇 %d: %d 通道, SNR均值 = %.2f dB\n', ...
                cluster, sum(idx == cluster), mean(cluster_snrs));
        end
    catch
        fprintf('聚类分析失败（样本可能不足）\n');
    end
end

fprintf('\n✓ 分析完成！\n');
