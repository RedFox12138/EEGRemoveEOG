% compute_SNR_improved.m
% 改进的SNR估计方法：提供多种估计策略
% 适用场景：Brain=纯净脑电, Wink/EyeMove=伪影+脑电混合信号

function compute_SNR_improved(varargin)
% 用法:
% compute_SNR_improved()                 % 使用当前目录
% compute_SNR_improved(folderPath)       % 指定目录

if nargin==0
    baseDir = fileparts(mfilename('fullpath'));
    if isempty(baseDir)
        baseDir = pwd;
    end
else
    baseDir = varargin{1};
end

fprintf('计算目录: %s\n', baseDir);

% Helper: load numeric vector from .mat
    function x = load_vector(matfile)
        S = load(matfile);
        if isfield(S,'data')
            x = S.data;
        elseif isfield(S,'seg')
            x = S.seg;
        else
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

%% 1) 加载并分析Brain数据
brainFiles = dir(fullfile(baseDir,'Brain_*.mat'));
if isempty(brainFiles)
    error('未找到任何 Brain_*.mat 文件于目录 %s', baseDir);
end
RMS_brain_list = zeros(numel(brainFiles),1);
for i=1:numel(brainFiles)
    fn = fullfile(baseDir, brainFiles(i).name);
    try
        x = load_vector(fn);
        RMS_brain_list(i) = rms_val(x);
    catch ME
        warning('跳过 %s: %s', brainFiles(i).name, ME.message);
        RMS_brain_list(i) = NaN;
    end
end

RMS_brain_list = RMS_brain_list(~isnan(RMS_brain_list));
if isempty(RMS_brain_list)
    error('所有 Brain_* 样本均不可用');
end

RMS_Brain_Mean = mean(RMS_brain_list);
RMS_Brain_Std = std(RMS_brain_list);
RMS_Brain_Median = median(RMS_brain_list);

fprintf('\n=== 纯净脑电统计 (Brain) ===\n');
fprintf('样本数: %d\n', numel(RMS_brain_list));
fprintf('RMS均值: %.4f\n', RMS_Brain_Mean);
fprintf('RMS标准差: %.4f (变异系数: %.1f%%)\n', RMS_Brain_Std, 100*RMS_Brain_Std/RMS_Brain_Mean);
fprintf('RMS中位数: %.4f\n', RMS_Brain_Median);
fprintf('RMS范围: [%.4f, %.4f]\n', min(RMS_brain_list), max(RMS_brain_list));

%% 2) 处理函数：提供多种估计方法
    function res = process_artifacts_multi_method(pattern, artifact_name)
        files = dir(fullfile(baseDir, pattern));
        n = numel(files);
        if n == 0
            fprintf('\n未找到 %s 文件\n', pattern);
            res = [];
            return;
        end
        
        fprintf('\n=== %s ===\n', artifact_name);
        fprintf('找到 %d 个样本\n\n', n);
        
        names = cell(n,1);
        RMS_mix_list = nan(n,1);
        
        % 方法1: 能量相减法（使用均值）
        SNR_method1 = nan(n,1);
        % 方法2: 能量相减法（使用中位数）
        SNR_method2 = nan(n,1);
        % 方法3: 保守估计（下界）
        SNR_method3 = nan(n,1);
        % 方法4: 激进估计（上界）
        SNR_method4 = nan(n,1);
        
        for j=1:n
            names{j} = files(j).name;
            fn = fullfile(baseDir, files(j).name);
            try
                x = load_vector(fn);
            catch ME
                warning('跳过 %s: %s', files(j).name, ME.message);
                continue;
            end
            
            RMS_mix = rms_val(x);
            RMS_mix_list(j) = RMS_mix;
            
            % 方法1: 能量相减法（使用Brain的均值）
            if RMS_mix > RMS_Brain_Mean
                RMS_art1 = sqrt(RMS_mix^2 - RMS_Brain_Mean^2);
                SNR_method1(j) = 20*log10(RMS_Brain_Mean / RMS_art1);
            end
            
            % 方法2: 能量相减法（使用Brain的中位数）
            if RMS_mix > RMS_Brain_Median
                RMS_art2 = sqrt(RMS_mix^2 - RMS_Brain_Median^2);
                SNR_method2(j) = 20*log10(RMS_Brain_Median / RMS_art2);
            end
            
            % 方法3: 保守估计（假设混合信号中脑电功率可能更大）
            % 使用 Brain_Mean + Brain_Std 作为背景脑电
            RMS_brain_upper = RMS_Brain_Mean + RMS_Brain_Std;
            if RMS_mix > RMS_brain_upper
                RMS_art3 = sqrt(RMS_mix^2 - RMS_brain_upper^2);
                SNR_method3(j) = 20*log10(RMS_Brain_Mean / RMS_art3);
            end
            
            % 方法4: 激进估计（假设混合信号中脑电功率较小）
            % 使用 Brain_Mean - Brain_Std 作为背景脑电（但不小于0）
            RMS_brain_lower = max(RMS_Brain_Mean - RMS_Brain_Std, 0.1*RMS_Brain_Mean);
            if RMS_mix > RMS_brain_lower
                RMS_art4 = sqrt(RMS_mix^2 - RMS_brain_lower^2);
                SNR_method4(j) = 20*log10(RMS_Brain_Mean / RMS_art4);
            end
        end
        
        % 打印结果
        fprintf('%-20s %10s %12s %12s %12s %12s\n', ...
            '文件名', 'RMS_mix', '方法1(均值)', '方法2(中位)', '方法3(保守)', '方法4(激进)');
        fprintf('%s\n', repmat('-', 1, 90));
        
        for j=1:n
            if isnan(RMS_mix_list(j))
                continue;
            end
            fprintf('%-20s %10.4f %12.2f %12.2f %12.2f %12.2f\n', ...
                names{j}, RMS_mix_list(j), ...
                SNR_method1(j), SNR_method2(j), SNR_method3(j), SNR_method4(j));
        end
        
        % 统计摘要
        valid1 = SNR_method1(~isnan(SNR_method1));
        valid2 = SNR_method2(~isnan(SNR_method2));
        valid3 = SNR_method3(~isnan(SNR_method3));
        valid4 = SNR_method4(~isnan(SNR_method4));
        
        fprintf('\n--- 统计摘要 ---\n');
        if ~isempty(valid1)
            fprintf('方法1 (能量相减-均值): 均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
                mean(valid1), median(valid1), min(valid1), max(valid1));
        end
        if ~isempty(valid2)
            fprintf('方法2 (能量相减-中位): 均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
                mean(valid2), median(valid2), min(valid2), max(valid2));
        end
        if ~isempty(valid3)
            fprintf('方法3 (保守估计):     均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
                mean(valid3), median(valid3), min(valid3), max(valid3));
        end
        if ~isempty(valid4)
            fprintf('方法4 (激进估计):     均值=%.2f dB, 中位数=%.2f dB, 范围=[%.2f, %.2f] dB\n', ...
                mean(valid4), median(valid4), min(valid4), max(valid4));
        end
        
        res.names = names;
        res.RMS_mix = RMS_mix_list;
        res.SNR_method1 = SNR_method1;
        res.SNR_method2 = SNR_method2;
        res.SNR_method3 = SNR_method3;
        res.SNR_method4 = SNR_method4;
    end

%% 3) 处理Wink和EyeMove
res_blink = process_artifacts_multi_method('Wink_*.mat', '眨眼伪影 (Wink)');
res_eye = process_artifacts_multi_method('EyeMove_*.mat', '眼动伪影 (EyeMove)');

%% 4) 方法说明
fprintf('\n\n=== 方法说明 ===\n');
fprintf('方法1 (能量相减-均值): 使用Brain样本的平均RMS作为背景脑电\n');
fprintf('  - 假设: Wink/EyeMove中的背景脑电与Brain样本统计特性相同\n');
fprintf('  - 公式: RMS_artifact = sqrt(RMS_mix^2 - RMS_brain_mean^2)\n');
fprintf('  - 适用: 数据来自相似状态，脑电功率稳定\n\n');

fprintf('方法2 (能量相减-中位): 使用Brain样本的中位数RMS\n');
fprintf('  - 优点: 对异常值更鲁棒\n\n');

fprintf('方法3 (保守估计): 假设Wink/EyeMove中背景脑电可能更强\n');
fprintf('  - 使用: RMS_brain_mean + RMS_brain_std 作为背景\n');
fprintf('  - 结果: SNR会更低（噪声估计更大）\n');
fprintf('  - 适用: 伪影出现时被试可能更警觉，脑电活动增强\n\n');

fprintf('方法4 (激进估计): 假设Wink/EyeMove中背景脑电可能更弱\n');
fprintf('  - 使用: RMS_brain_mean - RMS_brain_std 作为背景\n');
fprintf('  - 结果: SNR会更高（噪声估计更小）\n');
fprintf('  - 适用: 伪影出现时被试可能更放松\n\n');

fprintf('=== 建议 ===\n');
fprintf('1. 如果Brain样本的标准差较小（变异系数<20%%），方法1和2较可靠\n');
fprintf('2. 如果你观察到当前SNR"虚低"，可能是背景脑电功率波动导致\n');
fprintf('3. 建议使用方法1作为主要估计，方法3和4提供置信区间\n');
fprintf('4. 真实SNR很可能在方法3和方法4之间\n');

end
