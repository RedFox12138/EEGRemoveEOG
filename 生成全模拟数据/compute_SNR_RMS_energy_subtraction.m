% compute_SNR_RMS_energy_subtraction.m
% 目录: 生成全模拟数据
% 使用 RMS 能量相减方法估计眨眼和眼动伪迹的 SNR（dB）
% 要求：目录下包含以 Brain_*.mat, Wink_*.mat, EyeMove_*.mat 命名的单通道向量数据文件
% 变量名优先使用 data，否则自动使用第一个数值变量

function compute_SNR_RMS_energy_subtraction(varargin)
% 用法:
% compute_SNR_RMS_energy_subtraction()                 % 使用当前目录
% compute_SNR_RMS_energy_subtraction(folderPath)       % 指定目录

if nargin==0
    baseDir = fileparts(mfilename('fullpath'));
    if isempty(baseDir)
        baseDir = pwd;
    end
else
    baseDir = varargin{1};
end

fprintf('计算目录: %s\n', baseDir);

% Helper: load numeric vector from .mat (prefer 'data')
    function x = load_vector(matfile)
        S = load(matfile);
        if isfield(S,'data')
            x = S.data;
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
        x = double(x);
        % flatten to vector if possible
        if ismatrix(x) && (size(x,1)==1 || size(x,2)==1)
            x = x(:);
        else
            % if multi-channel, try first column
            if ismatrix(x) && size(x,2) > 1
                x = x(:,1);
            else
                x = x(:);
            end
        end
    end

% RMS function
    function r = rms_val(x)
        r = sqrt(mean(x.^2));
    end

% 1) Baseline energy from Brain_*.mat
brainFiles = dir(fullfile(baseDir,'Brain_*.mat'));
if isempty(brainFiles)
    error('未找到任何 Brain_*.mat 文件于目录 %s', baseDir);
end
RMS_pure_list = zeros(numel(brainFiles),1);
brain_names = cell(numel(brainFiles),1);
for i=1:numel(brainFiles)
    fn = fullfile(baseDir, brainFiles(i).name);
    try
        x = load_vector(fn);
    catch ME
        warning('跳过 %s: %s', brainFiles(i).name, ME.message);
        RMS_pure_list(i) = NaN; brain_names{i}=brainFiles(i).name; continue;
    end
    if isempty(x)
        RMS_pure_list(i) = NaN; brain_names{i}=brainFiles(i).name; continue;
    end
    RMS_pure_list(i) = rms_val(x);
    brain_names{i} = brainFiles(i).name;
end

RMS_pure_list = RMS_pure_list(~isnan(RMS_pure_list));
if isempty(RMS_pure_list)
    error('所有 Brain_* 样本均不可用，无法计算基线 RMS');
end
RMS_Pure_Avg = mean(RMS_pure_list);
fprintf('基线 RMS_Pure_Avg = %.6g (来自 %d 个样本)\n', RMS_Pure_Avg, numel(RMS_pure_list));

% process function for artifact files
    function res = process_artifacts(pattern)
        files = dir(fullfile(baseDir,pattern));
        n = numel(files);
        names = cell(n,1);
        SNRs = nan(n,1);
        RMS_mix_list = nan(n,1);
        RMS_art_list = nan(n,1);
        for j=1:n
            names{j} = files(j).name;
            fn = fullfile(baseDir, files(j).name);
            try
                x = load_vector(fn);
            catch ME
                warning('跳过 %s: %s', files(j).name, ME.message); continue;
            end
            if isempty(x)
                warning('跳过 %s: 数据为空', files(j).name); continue;
            end
            RMS_mix = rms_val(x);
            RMS_mix_list(j) = RMS_mix;
            if RMS_mix < RMS_Pure_Avg
                warning('在 %s 中 RMS_mix < RMS_Pure_Avg (%.4g < %.4g), 跳过该样本', files(j).name, RMS_mix, RMS_Pure_Avg);
                continue;
            end
            % energy subtraction
            a2 = RMS_mix^2 - RMS_Pure_Avg^2;
            if a2 <= 0
                warning('在 %s 中估计到的伪迹能量非正 (%.4g), 跳过', files(j).name, a2);
                continue;
            end
            RMS_art = sqrt(a2);
            RMS_art_list(j) = RMS_art;
            if RMS_art==0
                warning('在 %s 中估计的伪迹 RMS 为 0，跳过', files(j).name); continue;
            end
            SNRs(j) = 20*log10(RMS_Pure_Avg / RMS_art);
        end
        res.names = names;
        res.SNRs = SNRs;
        res.RMS_mix = RMS_mix_list;
        res.RMS_art = RMS_art_list;
    end

% 2) Blink (Wink_*)
fprintf('\n处理眨眼 (Wink_*.mat) ...\n');
res_blink = process_artifacts('Wink_*.mat');

% 3) Eye movement (EyeMove_*)
fprintf('\n处理眼动 (EyeMove_*.mat) ...\n');
res_eye = process_artifacts('EyeMove_*.mat');

% Summary printing helper
    function print_summary(title,res)
        valid = ~isnan(res.SNRs);
        vals = res.SNRs(valid);
        fprintf('\n** %s 汇总 **\n', title);
        if isempty(vals)
            fprintf(' 未获得有效 SNR 样本。\n');
            return;
        end
        fprintf(' 样本数: %d (有效 %d)\n', numel(res.SNRs), numel(vals));
        fprintf(' Mean SNR = %.3f dB, Std = %.3f dB\n', mean(vals), std(vals));
        fprintf(' 各文件 SNR (file : SNR dB):\n');
        for k=1:numel(res.names)
            if ~isnan(res.SNRs(k))
                fprintf('  - %s : %.3f dB\n', res.names{k}, res.SNRs(k));
            else
                fprintf('  - %s : (跳过或无效)\n', res.names{k});
            end
        end
    end

print_summary('眨眼 (Wink)', res_blink);
print_summary('眼动 (EyeMove)', res_eye);

% Save results
out.summarized.RMS_Pure_Avg = RMS_Pure_Avg;
out.blink = res_blink;
out.eyemove = res_eye;
save(fullfile(baseDir,'snr_results.mat'),'out');
fprintf('\n结果保存到 %s\n', fullfile(baseDir,'snr_results.mat'));

% 保存 CSV 简易摘要
try
    rows = {};
    rows{end+1,1} = 'Type'; rows{end,2} = 'File'; rows{end,3} = 'SNR_dB';
    for k=1:numel(res_blink.names)
        rows{end+1,1} = 'Wink'; rows{end,2} = res_blink.names{k}; rows{end,3} = res_blink.SNRs(k);
    end
    for k=1:numel(res_eye.names)
        rows{end+1,1} = 'EyeMove'; rows{end,2} = res_eye.names{k}; rows{end,3} = res_eye.SNRs(k);
    end
    T = cell2table(rows(2:end,:),'VariableNames',rows(1,:));
    writetable(T, fullfile(baseDir,'snr_summary.csv'));
    fprintf('CSV 摘要保存到 %s\n', fullfile(baseDir,'snr_summary.csv'));
catch
    warning('保存 CSV 时出现问题，已跳过');
end

end
