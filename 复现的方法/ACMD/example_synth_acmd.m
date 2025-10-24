% EXAMPLE_SYNTH_ACMD  Dataset-based ACMD demo (single-channel)
%
% 本脚本演示如何读取你自己的数据（如 sim10_con/sim10_resampled），
% 完成 ACMD 去伪迹并绘图。它会：
%   1) 从工作区变量或 .mat 文件加载污染/干净 EEG；
%   2) 估计阈值 ξ（若无干净参考则给出保守默认值）；
%   3) 按论文规则检测并减去第一模态，输出结果与统计信息。

% clear; clc;

% ================= 用户可配置区域 =================
% 方式A：如果你已在工作区有变量（如 sim10_con、sim10_resampled），保持为 true
useWorkspaceVars = true;
contVar = 'sim10_con';        % 污染信号变量名（二维：channels x samples 或向量）
cleanVar = 'sim10_resampled'; % 干净参考变量名（可为空）
chanIdx  = 1;                 % 使用第几个通道

% 方式B：若需从 .mat 文件加载，把下面文件名改成你的路径，并将 useWorkspaceVars 设为 false
matFile  = '';               % 单个 .mat 同时包含上述两个变量时使用
contFile = '';               % 或指定两个文件：污染
cleanFile= '';               % 和干净

% 采样率（若 .mat 中未包含 fs/Fs/samplingRate 等字段时使用）
defaultFs = 200;
% ==================================================

% ---- 1) 加载数据 ----
[xc, fs_c] = local_get_signal(useWorkspaceVars, contVar, chanIdx, matFile, contFile, defaultFs);
[xp, fs_p] = local_get_signal(useWorkspaceVars, cleanVar, chanIdx, matFile, cleanFile, defaultFs);

fs = defaultFs;
if ~isempty(fs_c), fs = fs_c; end
if ~isempty(fs_p), fs = fs_p; end

if isempty(xc)
        error('未能加载污染信号，请检查变量/文件设置。');
end

hasClean = ~isempty(xp);
if ~hasClean
        warning('未提供干净参考，将使用保守阈值并跳过SNR评估。');
end

dur = numel(xc)/fs;

% ---- 2) 基于 Ψp 估计阈值 ξ ----
optsProbe = struct('threshold', [], 'returnAll', false);
[~, ic] = oa_remove_acmd(xc, fs, optsProbe);
if hasClean
        [~, ip] = oa_remove_acmd(xp, fs, optsProbe);
        xi = 0.5*(ic.psi + ip.psi);
else
        % 无参考时：给出偏紧的阈值（第一模态峰值点的 0.9 倍，至少为 1）
        xi = max(1, round(0.9*ic.psi));
        ip.psi = NaN; % 仅用于打印
end

% ---- 3) 去噪 ----
% 进一步降低残留：启用参数自适应和一轮细化
opts = struct('threshold', xi, 'returnAll', true, 'restoreAmplitude', hasClean, ...
              'autoTune', true, 'refineRounds', 10);
[z, info] = oa_remove_acmd(xc, fs, opts);

% ---- 4) 统计 ----
fprintf('Dataset example (fs=%d Hz, dur=%.1f s)\n', fs, dur);
fprintf('Peak-count (contaminated first mode): %d\n', ic.psi);
if hasClean
        fprintf('Peak-count (clean first mode)       : %d\n', ip.psi);
end
fprintf('Chosen threshold ξ                  : %.2f\n', xi);
fprintf('Detected OA? %s\n', string(info.detected));

if hasClean
        % SNR(x|ref) = 10log10( sum(ref^2) / sum((x-ref)^2) )
        SNR_cont = 10*log10( sum(xp.^2) / sum((xc - xp).^2 + eps) );
        SNR_clean = 10*log10( sum(xp.^2) / sum((z  - xp).^2 + eps) );
        fprintf('SNR before: %.2f dB, after: %.2f dB, gain: %.2f dB\n', ...
                        SNR_cont, SNR_clean, SNR_clean - SNR_cont);
end

% ---- 5) 绘图 ----
L = numel(xc);
t = (0:L-1)/fs;

figure('Name','ACMD-based OA removal (dataset)');
pltRow = 5 + hasClean; % 有参考多一行
idx = 1;
if hasClean
        subplot(pltRow,1,idx); idx=idx+1; plot(t, xp, 'k'); grid on; ylabel('Clean'); title('Clean EEG (reference)');
end
subplot(pltRow,1,idx); idx=idx+1; plot(t, info.baseline, 'c'); grid on; ylabel('BL');  title('Extracted baseline wander (<1 Hz)');
subplot(pltRow,1,idx); idx=idx+1; plot(t, xc, 'b'); grid on; ylabel('Input'); title('Contaminated EEG');
subplot(pltRow,1,idx); idx=idx+1; plot(t, info.mode1, 'm'); grid on; ylabel('Mode1'); title(sprintf('First mode (\\Psi_p=%d, \\xi=%.2f)', info.psi, xi));
subplot(pltRow,1,idx);          plot(t, z, 'g'); hold on; if hasClean, plot(t, xp, ':k'); legend('Denoised','Clean ref'); else, legend('Denoised'); end
grid on; ylabel('Output'); xlabel('Time (s)'); title('Reconstructed OA-free EEG');


% ---------------- helpers ----------------
function [x, fs] = local_get_signal(useWS, varName, chanIdx, matFile, oneFile, defaultFs)
% 从工作区或文件加载指定变量，返回选定通道及采样率
x = [];
fs = [];
if isempty(varName), return; end

if useWS && evalin('base', sprintf('exist(''%s'',''var'')', varName))
        v = evalin('base', varName);
        [x, fs] = local_pick(v, defaultFs);
        x = local_channel(x, chanIdx);
        return;
end

% 单文件包含变量
if ~isempty(matFile) && isfile(matFile)
        S = load(matFile);
        if isfield(S, varName)
                [x, fs] = local_pick(S.(varName), defaultFs, S);
                x = local_channel(x, chanIdx);
                return;
        end
end

% 独立文件（包含该变量或任一向量）
if ~isempty(oneFile) && isfile(oneFile)
        S = load(oneFile);
        if isfield(S, varName)
                [x, fs] = local_pick(S.(varName), defaultFs, S);
        else
                % 自动选择第一个数值向量/矩阵
                fns = fieldnames(S);
                for i = 1:numel(fns)
                        v = S.(fns{i});
                        if isnumeric(v) && (isvector(v) || (ismatrix(v) && ~isscalar(v)))
                                [x, fs] = local_pick(v, defaultFs, S);
                                break;
                        end
                end
        end
        x = local_channel(x, chanIdx);
end
end

function [x, fs] = local_pick(v, defaultFs, S)
% 取出数据和采样率
if nargin < 3, S = struct(); end
% 采样率候选 key
cand = {'fs','Fs','samplingRate','sample_rate','srate'};
fs = [];
for i = 1:numel(cand)
        if isfield(S, cand{i}) && isnumeric(S.(cand{i}))
                fs = double(S.(cand{i})(1));
                break;
        end
end
if isempty(fs), fs = defaultFs; end

if isnumeric(v)
        if isvector(v)
                x = double(v(:));
        else
                x = double(v);
        end
else
        x = [];
end
end

function xch = local_channel(x, idx)
% 取行通道（channels x samples）或列向量
if isempty(x)
        xch = [];
        return;
end
if isvector(x)
        xch = x(:);
        return;
end
% 行为通道的常见约定
if size(x,1) < size(x,2)
        xch = x(idx, :).';
else
        xch = x(:, idx);
end
end
