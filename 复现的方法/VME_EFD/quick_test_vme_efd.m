% QUICK_TEST_VME_EFD - 快速验证 VME-EFD 实现
% 使用合成信号测试基本功能

clear; clc; close all;

fprintf('=== VME-EFD Quick Test ===\n\n');

% 添加路径
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir);
addpath(fullfile(thisDir, 'VME'));

% 生成测试信号
fs = 200;  % Hz
dur = 10;  % 秒
t = (0:1/fs:dur-1/fs)';
N = numel(t);

fprintf('Generating synthetic signal...\n');
fprintf('  Length: %d samples (%s)\n', N, iif(mod(N,2)==0, 'even', 'odd'));
fprintf('  Duration: %.1f s\n', dur);
fprintf('  Sampling rate: %g Hz\n\n', fs);

% 干净 EEG（多频率成分）
clean_eeg = 0.5*sin(2*pi*8*t) + ...      % Alpha 波
            0.3*sin(2*pi*12*t) + ...     % Beta 波
            0.2*randn(size(t));          % 噪声

% EOG 伪迹（低频）
eog = 2*sin(2*pi*1.5*t) + ...            % 主频 1.5 Hz
      1*sin(2*pi*0.8*t);                 % 次频 0.8 Hz

% 污染信号
contaminated = clean_eeg + eog;

% 测试 VME-EFD
fprintf('Running VME-EFD...\n');
params = struct();
params.alpha = 3500;
params.omega0 = 2.0;  % EOG 中心频率附近
params.K = 6;
params.nArtifacts = 2;
params.lfCut = 5;
params.verbose = true;

try
    [denoised, info] = vme_efd_denoise(contaminated, fs, params);
    fprintf('\n✓ VME-EFD completed successfully!\n\n');
    
    % 检查输出长度
    fprintf('Output length: %d samples\n', numel(denoised));
    if numel(denoised) ~= numel(contaminated)
        warning('Output length mismatch!');
    end
    
    % 评价指标
    CC_before = corr(clean_eeg(:), contaminated(:));
    CC_after  = corr(clean_eeg(:), denoised(:));
    
    RRMSE_before = sqrt(sum((clean_eeg-contaminated).^2)/sum(clean_eeg.^2));
    RRMSE_after  = sqrt(sum((clean_eeg-denoised).^2)/sum(clean_eeg.^2));
    
    SNR_before = 10*log10(sum(clean_eeg.^2)/(sum((contaminated-clean_eeg).^2)+eps));
    SNR_after  = 10*log10(sum(clean_eeg.^2)/(sum((denoised-clean_eeg).^2)+eps));
    
    fprintf('\nPerformance:\n');
    fprintf('  CC:    %.4f -> %.4f (%.4f improvement)\n', CC_before, CC_after, CC_after-CC_before);
    fprintf('  RRMSE: %.4f -> %.4f (%.4f reduction)\n', RRMSE_before, RRMSE_after, RRMSE_before-RRMSE_after);
    fprintf('  SNR:   %.2f -> %.2f dB (%.2f dB gain)\n', SNR_before, SNR_after, SNR_after-SNR_before);
    
    % EFD 分层信息
    fprintf('\nEFD Decomposition:\n');
    fprintf('  Artifact layers: %s\n', mat2str(info.artifactIdx.'));
    for k = 1:numel(info.efd)
        fprintf('  Layer %d: [%.2f, %.2f] Hz, Energy=%.4f, Centroid=%.2f Hz%s\n', ...
            k, info.bands(k,1), info.bands(k,2), info.energy(k), info.centroid(k), ...
            iif(ismember(k, info.artifactIdx), ' [REMOVED]', ''));
    end
    
    % 绘图
    figure('Name', 'VME-EFD Quick Test');
    
    subplot(5,1,1);
    plot(t, clean_eeg, 'k');
    grid on; ylabel('Clean EEG');
    title('Ground Truth (Clean EEG)');
    
    subplot(5,1,2);
    plot(t, eog, 'r');
    grid on; ylabel('True EOG');
    title('True EOG Artifact');
    
    subplot(5,1,3);
    plot(t, contaminated, 'b');
    grid on; ylabel('Contaminated');
    title('Contaminated EEG (Clean + EOG)');
    
    subplot(5,1,4);
    plot(t(1:numel(info.xeog)), info.xeog, 'm');
    grid on; ylabel('Estimated EOG');
    title(sprintf('VME Estimated EOG (alpha=%.0f, omega0=%.1f Hz)', info.alpha, info.omega0));
    
    subplot(5,1,5);
    t_out = (0:numel(denoised)-1)/fs;
    plot(t_out, denoised, 'g', 'LineWidth', 1.5);
    hold on;
    plot(t, clean_eeg, ':k', 'LineWidth', 1);
    grid on; ylabel('Denoised');
    xlabel('Time (s)');
    title('Denoised EEG (VME-EFD)');
    legend('Denoised', 'True Clean', 'Location', 'best');
    
    fprintf('\n=== Test completed successfully! ===\n');
    
catch ME
    fprintf('\n✗ Error occurred:\n');
    fprintf('  %s\n', ME.message);
    fprintf('  in %s (line %d)\n', ME.stack(1).name, ME.stack(1).line);
    rethrow(ME);
end

% Helper function for inline if
function out = iif(cond, true_val, false_val)
    if cond
        out = true_val;
    else
        out = false_val;
    end
end
