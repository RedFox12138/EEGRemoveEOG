% compute_SNR_direct_method.m
% 使用直接方法计算SNR（不使用能量相减法）
% 假设：Wink和EyeMove文件中存储的是纯伪影信号，或者至少伪影是主导成分

function compute_SNR_direct_method(varargin)
% 用法:
% compute_SNR_direct_method()                 % 使用当前目录
% compute_SNR_direct_method(folderPath)       % 指定目录

if nargin==0
    baseDir = fileparts(mfilename('fullpath'));
    if isempty(baseDir)
        baseDir = pwd;
    end
else
    baseDir = varargin{1};
end

fprintf('计算目录: %s\n', baseDir);
fprintf('\n===使用直接RMS比值法计算SNR===\n');
fprintf('注意：此方法假设Wink/EyeMove文件中伪影成分是主导的\n\n');

% Helper: load numeric vector from .mat
    function x = load_vector(matfile)
        S = load(matfile);
        if isfield(S,'data')
            x = S.data;
        elseif isfield(S,'seg')
            x = S.seg;
        else
            % pick first numeric array
            f = fieldnames(S);
            x = [];
            for k=1:numel(f)
                v = S.(f{k});
                if isnumeric(v) && ~isempty(v)
                    x = v; break;
                end
            end
        end
        if isempty(x)
            error('在 %s 中未找到数值变量。', matfile);
        end
        x = double(x(:));
    end

% RMS function
    function r = rms_val(x)
        r = sqrt(mean(x.^2));
    end

% 1) 计算Brain文件的平均RMS作为信号强度
brainFiles = dir(fullfile(baseDir,'Brain_*.mat'));
if isempty(brainFiles)
    error('未找到任何 Brain_*.mat 文件于目录 %s', baseDir);
end
RMS_pure_list = zeros(numel(brainFiles),1);
for i=1:numel(brainFiles)
    fn = fullfile(baseDir, brainFiles(i).name);
    try
        x = load_vector(fn);
        RMS_pure_list(i) = rms_val(x);
    catch ME
        warning('跳过 %s: %s', brainFiles(i).name, ME.message);
        RMS_pure_list(i) = NaN;
    end
end

RMS_pure_list = RMS_pure_list(~isnan(RMS_pure_list));
if isempty(RMS_pure_list)
    error('所有 Brain_* 样本均不可用');
end
RMS_Signal_Avg = mean(RMS_pure_list);
fprintf('纯净信号平均RMS = %.6g (来自 %d 个样本)\n', RMS_Signal_Avg, numel(RMS_pure_list));

% 2) 处理Wink文件（直接将其RMS作为伪影强度）
fprintf('\n--- 眨眼伪影 (Wink_*.mat) ---\n');
winkFiles = dir(fullfile(baseDir,'Wink_*.mat'));
if ~isempty(winkFiles)
    wink_SNRs = zeros(numel(winkFiles),1);
    for i=1:numel(winkFiles)
        fn = fullfile(baseDir, winkFiles(i).name);
        try
            x = load_vector(fn);
            RMS_artifact = rms_val(x);
            if RMS_artifact > 0
                SNR_dB = 20*log10(RMS_Signal_Avg / RMS_artifact);
                wink_SNRs(i) = SNR_dB;
                fprintf('  %s: RMS=%.4g, SNR=%.2f dB\n', winkFiles(i).name, RMS_artifact, SNR_dB);
            else
                wink_SNRs(i) = NaN;
                fprintf('  %s: RMS=0, 跳过\n', winkFiles(i).name);
            end
        catch ME
            warning('跳过 %s: %s', winkFiles(i).name, ME.message);
            wink_SNRs(i) = NaN;
        end
    end
    valid_wink = wink_SNRs(~isnan(wink_SNRs));
    if ~isempty(valid_wink)
        fprintf('  眨眼SNR统计: 均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
            mean(valid_wink), median(valid_wink), min(valid_wink), max(valid_wink));
    end
else
    fprintf('  未找到Wink文件\n');
end

% 3) 处理EyeMove文件
fprintf('\n--- 眼动伪影 (EyeMove_*.mat) ---\n');
eyeFiles = dir(fullfile(baseDir,'EyeMove_*.mat'));
if ~isempty(eyeFiles)
    eye_SNRs = zeros(numel(eyeFiles),1);
    for i=1:numel(eyeFiles)
        fn = fullfile(baseDir, eyeFiles(i).name);
        try
            x = load_vector(fn);
            RMS_artifact = rms_val(x);
            if RMS_artifact > 0
                SNR_dB = 20*log10(RMS_Signal_Avg / RMS_artifact);
                eye_SNRs(i) = SNR_dB;
                fprintf('  %s: RMS=%.4g, SNR=%.2f dB\n', eyeFiles(i).name, RMS_artifact, SNR_dB);
            else
                eye_SNRs(i) = NaN;
                fprintf('  %s: RMS=0, 跳过\n', eyeFiles(i).name);
            end
        catch ME
            warning('跳过 %s: %s', eyeFiles(i).name, ME.message);
            eye_SNRs(i) = NaN;
        end
    end
    valid_eye = eye_SNRs(~isnan(eye_SNRs));
    if ~isempty(valid_eye)
        fprintf('  眼动SNR统计: 均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
            mean(valid_eye), median(valid_eye), min(valid_eye), max(valid_eye));
    end
else
    fprintf('  未找到EyeMove文件\n');
end

fprintf('\n===计算完成===\n');
fprintf('\n【重要说明】\n');
fprintf('1. 此方法假设Wink/EyeMove文件中伪影成分占主导地位\n');
fprintf('2. 如果这些文件是"纯净脑电+伪影"的混合，则此结果会低估实际SNR\n');
fprintf('3. 原始能量相减法的问题在于：Brain片段和Wink/EyeMove片段的纯净脑电成分不同\n');
fprintf('4. 建议：检查Wink/EyeMove数据来源，确认是纯伪影还是含伪影的混合信号\n');

end
