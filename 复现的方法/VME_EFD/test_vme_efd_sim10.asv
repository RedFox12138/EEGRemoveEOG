% TEST_VME_EFD_SIM10  使用与 ACMD 一致的 sim10 数据测试 VME-EFD
%
% 使用方式（两种任选其一）：
%   A) 先在工作区放入变量 sim10_con（污染）和 sim10_resampled（干净参考，可选），
%      然后直接运行本脚本；
%   B) 修改下面的文件路径，让脚本从 .mat 文件加载上述变量。
%
% 本脚本会：
%   1) 加载数据 -> x_cont（必须）和 x_clean（可选）；
%   2) 调用 VME+EFD 去除眼动伪迹；
%   3) 若有参考，计算 CC、RRMSE、SNR 提升，并做对比图。

close all;

% =============== 用户配置 ===============
useWorkspaceVars = true;     % 若已在工作区有 sim10_con/sim10_resampled 则置 true
contVar = 'sim10_con';
cleanVar = 'sim10_resampled';
chanIdx = 1;                 % 使用第几个通道

% 若从文件读取，请设置 mat 路径（同一个或两个文件均可）
matFile  = '';
contFile = '';
cleanFile= '';

% 默认采样率（若文件中未包含 fs/Fs/srate 字段，将使用该值）
defaultFs = 200;  % 与 ACMD 示例一致

% 固定最优参数（来自自动调参结果）
params = struct();
params.alpha  = 3500;      % 最优 alpha
params.omega0 = 2.8;       % 最优 omega0 (Hz)
params.K = 10;              % 保持当前分层数（可按需改回 6）
params.nArtifacts = 1;     % 最优丢弃层数
params.lfCut = 2.5;          % 最优低频上限 (Hz)
params.tau = 0; params.tol = 1e-6; params.verbose = true;
% 软去除系数：降低对低频细节的“过剔除”，范围[0,1]，1=全删，0.6~0.9更保守
params.artifactGain = 0.9;  % 可按需调到 0.6~0.9
% ========================================

% ---- 1) 读取数据 ----
[xc, fs_c] = local_get_signal(useWorkspaceVars, contVar, chanIdx, matFile, contFile, defaultFs);
[xp, fs_p] = local_get_signal(useWorkspaceVars, cleanVar, chanIdx, matFile, cleanFile, defaultFs);

fs = defaultFs;
if ~isempty(fs_c), fs = fs_c; end
if ~isempty(fs_p), fs = fs_p; end

if isempty(xc)
    error('未能加载污染信号，请检查变量/文件设置。');
end
hasClean = ~isempty(xp);

% 打印数据维度诊断信息
fprintf('=== Data loaded ===\n');
fprintf('Contaminated signal size: %s\n', mat2str(size(xc)));
if hasClean
    fprintf('Clean reference size: %s\n', mat2str(size(xp)));
end
fprintf('Sampling rate: %g Hz\n', fs);
fprintf('===================\n\n');

% ---- 2) 路径：确保能找到 VME 实现 ----
thisDir = fileparts(mfilename('fullpath'));
addpath(thisDir); % 自身
addpath(fullfile(fileparts(thisDir), 'VME_GMETV', 'VME')); % vme.m 所在目录

% ---- 3) 调参与去伪迹 ----
doTuning = false; % 固定最优参数，关闭自动调参

if doTuning
    fprintf('=== Hyper-parameter tuning ===\n');
    % 参数网格（适度小而有效）
    alphaGrid   = unique([params.alpha, 3000, 3500, 5000]);
    omega0Grid  = unique([params.omega0, 2.0, 2.5, 2.8]);
    lfCutGrid   = unique([params.lfCut, 3, 4, 5]);
    nArtGrid    = unique([params.nArtifacts, 1, 2]);
    % 固定丢弃层的候选：严格论文风格或自动选择
    fixedSet    = {[], [3 6]};

    bestScore = -inf; bestParams = params; bestInfo = []; bestY = [];
    tried = 0;
    for a = alphaGrid
        for w0 = omega0Grid
            for lf = lfCutGrid
                for na = nArtGrid
                    for fidx = 1:numel(fixedSet)
                        p = params; p.alpha=a; p.omega0=w0; p.lfCut=lf; p.nArtifacts=na;
                        if ~isempty(fixedSet{fidx})
                            p.artifactIdxFixed = fixedSet{fidx};
                        else
                            if isfield(p,'artifactIdxFixed')
                                p = rmfield(p,'artifactIdxFixed');
                            end
                        end
                        tried = tried+1;
                        try
                            [y_tmp, info_tmp] = vme_efd_denoise(xc, fs, p);
                        catch ME
                            fprintf('  skip (err): alpha=%g, w0=%.2f, lf=%g, nArt=%d, fixed=%s -> %s\n',...
                                a, w0, lf, na, mat2str(fixedSet{fidx}), ME.message);
                            continue;
                        end
                        % 统一长度后评估
                        Ltmp = numel(y_tmp);
                        xc_eval = xc; xp_eval = xp;
                        if numel(xc_eval) ~= Ltmp
                            if numel(xc_eval) > Ltmp, xc_eval = xc_eval(1:Ltmp); else, xc_eval(end+1:Ltmp) = xc_eval(end); end
                        end
                        if hasClean
                            if numel(xp_eval) ~= Ltmp
                                if numel(xp_eval) > Ltmp, xp_eval = xp_eval(1:Ltmp); else, xp_eval(end+1:Ltmp) = xp_eval(end); end
                            end
                            CCa = corr(xp_eval(:), y_tmp(:));
                            RRa = sqrt(sum((xp_eval(:)-y_tmp(:)).^2)/sum(xp_eval(:).^2));
                            SNRg = 10*log10(sum(xp_eval.^2)/(sum((y_tmp-xp_eval).^2)+eps)) - ...
                                   10*log10(sum(xp_eval.^2)/(sum((xc_eval-xp_eval).^2)+eps));
                            % 多目标：优先最大化 CC，其次最小 RRMSE，最后最大 SNR gain
                            score = 100*CCa - 10*RRa + 1*SNRg;
                        else
                            % 无参考：兼顾保真与抑制低频
                            % 低频抑制（0-4 Hz 相对输入）
                            p_in  = bandpow_local(xc_eval, fs, [0 4]);
                            p_out = bandpow_local(y_tmp,   fs, [0 4]);
                            redLF = (p_in - p_out) / (p_in + eps);
                            % α/β 波段保真（8-30 Hz 与输入的相关）
                            y_ab  = bp_filt_local(y_tmp, fs, [8 30]);
                            x_ab  = bp_filt_local(xc_eval, fs, [8 30]);
                            ccAB  = corr(x_ab(:), y_ab(:));
                            score = 50*ccAB + 50*max(0, redLF);
                        end

                        if score > bestScore
                            bestScore = score; bestParams = p; bestInfo = info_tmp; bestY = y_tmp;
                        end
                    end
                end
            end
        end
    end
    fprintf('Tried %d combos; best score=%.3f\n', tried, bestScore);
    fixedStr = '[]';
    if isfield(bestParams,'artifactIdxFixed') && ~isempty(bestParams.artifactIdxFixed)
        fixedStr = mat2str(bestParams.artifactIdxFixed);
    end
    fprintf('Best params: alpha=%g, omega0=%.2f, lfCut=%g, nArtifacts=%d, fixed=%s\n', ...
        bestParams.alpha, bestParams.omega0, bestParams.lfCut, bestParams.nArtifacts, fixedStr);
    params = bestParams; y_denoised = bestY; info = bestInfo;
else
    [y_denoised, info] = vme_efd_denoise(xc, fs, params);
end

% ---- 4) 评价指标 ----
L = numel(y_denoised);  % 使用输出信号的长度
t = (0:L-1)/fs;

% 如果有截断，确保参考信号与输出长度一致
if hasClean && numel(xp) ~= L
    if numel(xp) > L
        xp = xp(1:L);
    else
        xp(end+1:L) = xp(end);
    end
end
if numel(xc) ~= L
    if numel(xc) > L
        xc = xc(1:L);
    else
        xc(end+1:L) = xc(end);
    end
end

if hasClean
    % CC (Pearson correlation)
    CC_before = corr(xp(:), xc(:));
    CC_after  = corr(xp(:), y_denoised(:));

    % RRMSE
    RRMSE_before = sqrt(sum((xp(:)-xc(:)).^2)/sum(xp(:).^2));
    RRMSE_after  = sqrt(sum((xp(:)-y_denoised(:)).^2)/sum(xp(:).^2));

    % SNR 提升
    SNR_cont = 10*log10(sum(xp.^2)/(sum((xc-xp).^2)+eps));
    SNR_clean= 10*log10(sum(xp.^2)/(sum((y_denoised-xp).^2)+eps));
    dSNR = SNR_clean - SNR_cont;

    fprintf('fs=%g Hz, dur=%.1f s\n', fs, L/fs);
    fprintf('CC before=%.4f, after=%.4f\n', CC_before, CC_after);
    fprintf('RRMSE before=%.4f, after=%.4f\n', RRMSE_before, RRMSE_after);
    fprintf('SNR gain: %.2f dB (%.2f -> %.2f dB)\n', dSNR, SNR_cont, SNR_clean);
end

% ---- 5) 绘图 ----
figure('Name','VME-EFD OA removal (sim10)');
rows = 5 + hasClean; r = 1;
if hasClean
    subplot(rows,1,r); r=r+1; plot(t, xp, 'k'); grid on; ylabel('Clean'); title('Clean EEG (reference)');
end
subplot(rows,1,r); r=r+1; plot(t, info.y1, 'c'); grid on; ylabel('y1'); title('Residual after VME: y1 = y - x_{eog}');
subplot(rows,1,r); r=r+1; plot(t, xc, 'b'); grid on; ylabel('Input'); title('Contaminated EEG');
subplot(rows,1,r); r=r+1; plot(t, info.xeog, 'm'); grid on; ylabel('x_{eog}'); title('Estimated EOG segment by VME');
subplot(rows,1,r); 
plot(t, y_denoised, 'g', 'DisplayName','Denoised'); hold on; 
if hasClean, plot(t, xp, ':k', 'DisplayName','Clean ref'); end
grid on; ylabel('Output'); xlabel('Time (s)'); title('Denoised EEG vs Clean'); 
legend('show');

% 单独的对比图（同一张图上叠加展示）
figure('Name','Overlay: Clean vs Denoised');
plot(t, y_denoised, 'g', 'DisplayName','Denoised'); hold on;
if hasClean, plot(t, xp, ':k', 'DisplayName','Clean ref'); end
grid on; xlabel('Time (s)'); ylabel('Amplitude'); title('Clean vs Denoised (Overlay)'); legend('show');

% EFD 频段与能量/偏度报告
fprintf('Selected artifacts (indices): %s\n', mat2str(info.artifactIdx.'));
for k = 1:numel(info.efd)
    fprintf('EFD%02d: band=[%.2f, %.2f] Hz, Energy=%.4f, Skew=%.4f, Centroid=%.2f Hz\n', ...
        k, info.bands(k,1), info.bands(k,2), info.energy(k), info.skew(k), info.centroid(k));
end

% ================= helpers =================
function [x, fs] = local_get_signal(useWS, varName, chanIdx, matFile, oneFile, defaultFs)
x = []; fs = [];
if isempty(varName), return; end
if useWS && evalin('base', sprintf('exist(''%s'',''var'')', varName))
    v = evalin('base', varName);
    [x, fs] = local_pick(v, defaultFs);
    x = local_channel(x, chanIdx); return;
end
if ~isempty(matFile) && isfile(matFile)
    S = load(matFile);
    if isfield(S, varName)
        [x, fs] = local_pick(S.(varName), defaultFs, S);
        x = local_channel(x, chanIdx); return;
    end
end
if ~isempty(oneFile) && isfile(oneFile)
    S = load(oneFile);
    if isfield(S, varName)
        [x, fs] = local_pick(S.(varName), defaultFs, S);
    else
        fns = fieldnames(S);
        for i = 1:numel(fns)
            v = S.(fns{i});
            if isnumeric(v) && (isvector(v) || (ismatrix(v) && ~isscalar(v)))
                [x, fs] = local_pick(v, defaultFs, S); break;
            end
        end
    end
    x = local_channel(x, chanIdx);
end
end

function [x, fs] = local_pick(v, defaultFs, S)
if nargin < 3, S = struct(); end
cand = {'fs','Fs','samplingRate','sample_rate','srate'}; fs = [];
for i = 1:numel(cand)
    if isfield(S, cand{i}) && isnumeric(S.(cand{i}))
        fs = double(S.(cand{i})(1)); break;
    end
end
if isempty(fs), fs = defaultFs; end
if isnumeric(v)
    if isvector(v), x = double(v(:)); else, x = double(v); end
else
    x = [];
end

function p = bandpow_local(x, fs, fr)
% 简易带内功率
x = x(:); N = numel(x);
X = fft(x);
Pxx = abs(X).^2 / N;
f = (0:N-1)*(fs/N);
half = 1:floor(N/2)+1; P = Pxx(half); ff = f(half);
idx = (ff>=fr(1) & ff<=fr(2));
if any(idx), df = mean(diff(ff)); p = sum(P(idx))*df; else, p = 0; end
end

function y = bp_filt_local(x, fs, fr)
% 简单频域带通（零相）
x = x(:); N = numel(x);
Nfft = 2^nextpow2(N);
X = fft(x, Nfft);
f = (0:Nfft-1)*(fs/Nfft);
H = zeros(Nfft,1);
% 正频段
mask = (f>=fr(1) & f<=fr(2)) | (f>=fs-fr(2) & f<=fs-fr(1));
H(mask) = 1;
y = real(ifft(X.*H));
y = y(1:N);
end

function out = ifelse(cond, a, b)
if cond, out = a; else, out = b; end
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
