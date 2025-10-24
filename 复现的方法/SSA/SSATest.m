% 示例：将 SSA 的测试数据切换为与 ACMD 相同的数据源（sim10_con / sim10_resampled 或标准 .mat）
% clear; clc; close all;
close all

%% 0. 数据来源设置（与 ACMD/example_synth_acmd.m 对齐）
useWorkspaceVars = true;            % 优先从工作区变量读取（sim10_con / sim10_resampled）
contVar = 'sim10_con';              % 污染信号变量名
cleanVar = 'sim10_resampled';       % 干净参考变量名（可为空）
chanIdx  = 1;                       % 通道索引（若为矩阵）

% 若工作区无变量，则尝试从这些文件读取
matFile  = '';                      % 如单个 .mat 同时包含变量时可设置此项
contFile = 'd:/Pycharm_Projects/EOG Remove/生成全模拟数据/已经生成好的数据/Contaminated.mat';
cleanFile= 'd:/Pycharm_Projects/EOG Remove/生成全模拟数据/已经生成好的数据/Pure_Data.mat';
defaultFs= 200;                     % 默认采样率（若文件中未提供）

% 读取污染与干净信号（若可用）
[xc, fs_c] = local_get_signal(useWorkspaceVars, contVar, chanIdx, matFile, contFile, defaultFs);
[xp, fs_p] = local_get_signal(useWorkspaceVars, cleanVar, chanIdx, matFile, cleanFile, defaultFs);

% 如果以上都没有加载成功，则退回到内置的合成数据，避免阻塞
if isempty(xc)
    warning('未找到 ACMD 示例用数据，回退到合成测试信号。');
    fs = 200;          % 采样频率
    t = 0:1/fs:8;      % 8 秒
    P = length(t);
    eeg_clean = 2*sin(2*pi*10*t) + 1.5*sin(2*pi*6*t);
    eog = zeros(1, P);
    eog_peaks = [0.5, 1.8, 3.2, 4.5, 5.9, 7.3];
    for peak = eog_peaks
        idx = round(peak*fs);
        rng(1);
        eog(max(1,idx-20):min(P,idx+20)) = 8*exp(-(((-20:min(20,P-idx))./10).^2));
    end
    noisy_eeg = eeg_clean + 0.8*eog;
    eeg_clean = eeg_clean(:); noisy_eeg = noisy_eeg(:);
else
    % 使用 ACMD 同源数据
    fs = defaultFs; if ~isempty(fs_c), fs = fs_c; end
    noisy_eeg = xc(:).';  % 统一为行向量
    t = (0:numel(noisy_eeg)-1)/fs;
    if ~isempty(xp)
        eeg_clean = xp(:).';
        eeg_clean = eeg_clean(1:min(numel(eeg_clean), numel(noisy_eeg)));
        noisy_eeg = noisy_eeg(1:numel(eeg_clean));
        t = (0:numel(noisy_eeg)-1)/fs;
    else
        eeg_clean = [];
    end
end

%% 2. 调用去噪函数（默认参数，fs 来自数据或默认 200Hz）
[denoised_eeg, eog_artifact] = ci_ssa_eog_removal(noisy_eeg, 'fs', fs);

% 3. 定量验证（若提供了干净参考：RRMSE+CC）
if ~isempty(eeg_clean)
    eeg_clean = eeg_clean(:).';
    L = min(numel(eeg_clean), numel(denoised_eeg));
    rrmse = rms(eeg_clean(1:L) - denoised_eeg(1:L)) / rms(eeg_clean(1:L));
    C = corrcoef(eeg_clean(1:L), denoised_eeg(1:L));
    cc = C(1,2);
    fprintf('去噪后RRMSE=%.4f，CC=%.4f（文献参考：RRMSE越小越好，CC越接近1越好）\n', rrmse, cc);
else
    fprintf('未提供干净参考，跳过 RRMSE/CC 评估。\n');
end

% =============== 辅助函数（从 ACMD 示例移植的简版）===============
function [x, fs] = local_get_signal(useWS, varName, chanIdx, matFile, oneFile, defaultFs)
x = []; fs = [];
if isempty(varName), return; end
% A) 工作区
try
    if useWS && evalin('base', sprintf('exist(''%s'',''var'')', varName))
        v = evalin('base', varName);
        [x, fs] = local_pick(v, defaultFs);
        x = local_channel(x, chanIdx);
        return;
    end
catch
end
% B) 单文件包含变量
if ~isempty(matFile) && isfile(matFile)
    S = load(matFile);
    if isfield(S, varName)
        [x, fs] = local_pick(S.(varName), defaultFs, S);
        x = local_channel(x, chanIdx); return;
    end
end
% C) 独立文件：尝试常见字段名或首个数值数组
if ~isempty(oneFile) && isfile(oneFile)
    S = load(oneFile);
    cand = {varName, 'contaminatedEEG','Contaminated','data','X','x'};
    for i = 1:numel(cand)
        if isfield(S, cand{i})
            [x, fs] = local_pick(S.(cand{i}), defaultFs, S);
            x = local_channel(x, chanIdx); return;
        end
    end
    fns = fieldnames(S);
    for i = 1:numel(fns)
        v = S.(fns{i});
        if isnumeric(v) && (isvector(v) || (ismatrix(v) && ~isscalar(v)))
            [x, fs] = local_pick(v, defaultFs, S);
            x = local_channel(x, chanIdx); return;
        end
    end
end
end

function [x, fs] = local_pick(v, defaultFs, S)
if nargin < 3, S = struct(); end
cand = {'fs','Fs','samplingRate','sample_rate','srate'}; fs = [];
for i = 1:numel(cand)
    if isfield(S, cand{i}) && isnumeric(S.(cand{i}))
        fs = double(S.(cand{i})(1)); break; end
end
if isempty(fs), fs = defaultFs; end
if isnumeric(v)
    if isvector(v), x = double(v(:)); else, x = double(v); end
else
    x = [];
end
end

function xch = local_channel(x, idx)
if isempty(x), xch = []; return; end
if isvector(x), xch = x(:); return; end
if size(x,1) < size(x,2)
    xch = x(idx, :).';
else
    xch = x(:, idx);
end
end