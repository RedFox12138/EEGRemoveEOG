function [y_denoised, info] = vme_efd_denoise(y, fs, params)
% VME_EFD_DENOISE  EEG 眼动伪迹去除（VME + EFD 实现）
%
%   [y_denoised, info] = vme_efd_denoise(y, fs, params)
%
% Inputs
%   y       : 被污染的单通道 EEG（列向量或行向量）
%   fs      : 采样率 (Hz)
%   params  : 可选结构体，字段：
%       .alpha        - VME 紧致系数（默认 3000）；若给为向量则自动择优
%       .omega0       - VME 初始中心频率 (Hz，默认 2.8)；向量则自动择优
%       .tau          - VME 对偶上升步长（默认 0）
%       .tol          - VME 收敛阈值（默认 1e-6）
%       .K            - EFD 分解层数（默认 6）
%       .nArtifacts   - 认为是伪迹的分量个数（默认 2，对应论文中的 efd3 与 efd6）
%       .lfCut        - 低频上限，用于限制伪迹候选 (Hz，默认 3)
%       .verbose      - 是否打印简要日志（默认 true）
%
% Outputs
%   y_denoised : 去伪迹后的 EEG
%   info       : 结构体，包含中间结果（便于调参/画图/复核）：
%       .xeog, .y1
%       .efd      {Kx1} 每层分量（时域）
%       .bands    [Kx2] 每层频段边界 (Hz)
%       .energy   [Kx1]
%       .skew     [Kx1]
%       .centroid [Kx1] 频谱质心 (Hz)
%       .artifactIdx  被丢弃的层索引
%       .alpha, .omega0
%
% 注：本函数依赖目录 VME_GMETV/VME 中的 vme.m。

if nargin < 2
    error('Usage: vme_efd_denoise(y, fs, [params])');
end
if nargin < 3 || isempty(params), params = struct(); end

% ---- 参数 ----
alpha   = getfield_def(params, 'alpha',   3000);
omega0  = getfield_def(params, 'omega0', 2.8);
tau     = getfield_def(params, 'tau',     0);
tol     = getfield_def(params, 'tol',     1e-6);
K       = getfield_def(params, 'K',       6);
nArtifacts = getfield_def(params, 'nArtifacts', 2);
artifactIdxFixed = [];
if isfield(params, 'artifactIdxFixed') && ~isempty(params.artifactIdxFixed)
    artifactIdxFixed = params.artifactIdxFixed(:)';
end
artifactIdx = [];
lfCut   = getfield_def(params, 'lfCut',   3);
verbose = getfield_def(params, 'verbose', true);
artifactGain = getfield_def(params, 'artifactGain', 1.0); % 软去除因子，1=全删，0=不删

% ---- 统一形状 ----
y = double(y(:));
N_orig = numel(y);
N = N_orig;

% VME 要求信号长度为偶数（用于镜像扩展）
truncated = false;
if mod(N, 2) == 1
    y = y(1:end-1);  % 截断最后一个样本
    N = N - 1;
    truncated = true;
    if verbose
        fprintf('[VME-EFD] Signal length was odd (%d), truncated to %d samples\n', N_orig, N);
    end
end

% ---- 1) 估计 EOG 伪迹段（VME）----
% 支持自动网格搜索（很小的搜索空间，避免过慢）
if numel(alpha) > 1 || numel(omega0) > 1
    if verbose, fprintf('[VME] grid search alpha x omega0 ...\n'); end
    bestScore = -inf; bestX = []; bestAlpha = []; bestOmega0 = [];
    for a = alpha(:)'
        for w0 = omega0(:)'
            try
                [xeog_tmp, ~] = local_call_vme(y, a, w0, fs, tau, tol);
            catch ME
                warning('VME failed for alpha=%.1f, omega0=%.2f: %s', a, w0, ME.message);
                continue;
            end
            y1_tmp = y - xeog_tmp(:);
            % 目标：低频集中且与余量 y1 的重叠最小
            rLF = local_bandpower(xeog_tmp, fs, [0, lfCut]) / (local_bandpower(xeog_tmp, fs, [0, fs/2]) + eps);
            c12 = abs(local_corr(xeog_tmp, y1_tmp));
            score = rLF / (c12 + 1e-6);
            if score > bestScore
                bestScore = score; bestX = xeog_tmp(:); bestAlpha = a; bestOmega0 = w0;
            end
        end
    end
    if isempty(bestX)
        warning('Grid search failed for all combinations. Using first alpha/omega0.');
        alpha = alpha(1); omega0 = omega0(1);
        xeog = local_call_vme(y, alpha, omega0, fs, tau, tol);
    else
        xeog = bestX; alpha = bestAlpha; omega0 = bestOmega0;
    end
else
    xeog = local_call_vme(y, alpha, omega0, fs, tau, tol);
end

if verbose
    fprintf('[VME] chosen alpha=%.1f, omega0=%.2f Hz\n', alpha, omega0);
end

% 残差（期望的 EEG 主体）
y1 = y - xeog(:);

% ---- 2) EFD 分解（对 xeog）----
[efd, bands, Finfo] = efd_decompose(xeog, fs, K);
K = numel(efd);

% ---- 3) 指标与伪迹层选择 ----
energy = zeros(K,1);
sk   = zeros(K,1);
cent = zeros(K,1);
for k = 1:K
    xk = efd{k}(:);
    energy(k) = sum(xk.^2);
    sk(k)     = local_skewness(xk);
    cent(k)   = local_spectral_centroid(xk, fs);
end

% 选择要丢弃的层
if ~isempty(artifactIdxFixed)
    artifactIdx = intersect(unique(round(artifactIdxFixed)), 1:K);
    if isempty(artifactIdx)
        warning('artifactIdxFixed provided but invalid; falling back to automatic selection.');
    end
end
if isempty(artifactIdx)
    % 候选：低频（<= lfCut Hz）优先，再用 Energy×|Skew| 联合评分（更贴近论文表1）
    cand = find(cent <= lfCut);
    scoreES = (energy./(max(energy)+eps)) .* (abs(sk)./(max(abs(sk))+eps));
    if isempty(cand)
        [~, idxSort] = sort(scoreES, 'descend');
        artifactIdx = idxSort(1:min(nArtifacts, K));
    else
        [~, idxRel] = sort(scoreES(cand), 'descend');
        artifactIdx = cand(idxRel(1:min(nArtifacts, numel(cand))));
    end
end

% ---- 4) 重构（支持软去除）----
% 公式：y_denoised = y - sum_k g_k * efd_artifact_k
% 当 g_k=1 时等价于原来的硬删除；当 g_k<1 时为部分减去，保留部分低频细节
if isempty(artifactIdx)
    y_denoised = y1 + sum(cat(2, efd{:}), 2); % 无伪迹层时，等于 y
else
    A = cat(2, efd{artifactIdx});
    if isscalar(artifactGain)
        g = repmat(artifactGain, size(A,2), 1);
    elseif numel(artifactGain) == numel(artifactIdx)
        g = artifactGain(:);
    else
        if verbose
            warning('artifactGain size mismatch, fallback to 1.0');
        end
        g = ones(size(A,2),1);
    end
    y_denoised = y - A * g;
end

% 若原始信号被截断，补齐最后一个样本（简单复制）
if truncated
    y_denoised(end+1) = y_denoised(end);
    xeog(end+1) = xeog(end);
    y1(end+1) = y1(end);
    for k = 1:K
        efd{k}(end+1) = efd{k}(end);
    end
end

% ---- 结果 ----
info = struct();
info.xeog = xeog(:);
info.y1   = y1(:);
info.efd  = efd(:);
info.bands= bands;
info.energy = energy;
info.skew   = sk;
info.centroid = cent;
info.artifactIdx = artifactIdx(:);
info.artifactGain = artifactGain;
info.alpha = alpha; info.omega0 = omega0; info.fs = fs;
info.Finfo = Finfo;
info.truncated = truncated;
info.N_orig = N_orig;

end

% ==================== subroutines ====================
function [efd, bands, Finfo] = efd_decompose(x, fs, K)
% 基于“最低谷”准则的经验傅里叶分解，实现零相位滤波器组
% 返回严格 K 层分量；若可用谷值不足，则用最长区间中点补足

x = x(:);
N = numel(x);
% 零填充到 2 的幂以提升频域分辨率（不改变逆变换长度）
Nfft = 2^nextpow2(max(N, 2048));
X = fftshift(fft(x, Nfft));
mag = abs(X);

% 频率轴（对称 -fs/2..fs/2）与一侧 [0, fs/2]
faxis = linspace(-fs/2, fs/2, Nfft).';
mask_pos = faxis >= 0;
fpos = faxis(mask_pos);
Spos = mag(mask_pos);

% 光滑化幅度谱，抑制毛刺（移动平均窗口约为 1% 频点）
win = max(5, round(0.01 * numel(Spos)));
Ssm = movmean(Spos, win);

% 寻找局部最低谷（排除首尾）
idx = 2:(numel(Ssm)-1);
isMin = Ssm(idx) <= Ssm(idx-1) & Ssm(idx) < Ssm(idx+1);
minIdx = idx(isMin);

% 以谷值幅度从小到大排序
[~, ordMin] = sort(Ssm(minIdx), 'ascend');
minIdx = minIdx(ordMin);

% 迭代选择 K-1 个边界，最小间隔约为 0.5 Hz 或频宽的 1/(4K)
minSepHz = max(0.5, (fs/2)/K * 0.25);
edges = [0, fs/2];
for ii = 1:numel(minIdx)
    if numel(edges) >= K+1, break; end
    f_cand = fpos(minIdx(ii));
    if all(abs(f_cand - edges) > minSepHz)
        edges = sort([edges, f_cand]); %#ok<AGROW>
    end
end

% 若边界不足，继续在最长区间中间插入
while numel(edges) < K+1
    [~, idLongest] = max(diff(edges));
    mid = 0.5 * (edges(idLongest) + edges(idLongest+1));
    edges = sort([edges, mid]);
end

% 若过多，合并最近边界
while numel(edges) > K+1
    [~, idMin] = min(diff(edges));
    edges(idMin+1) = [];
end

bands = [edges(1:end-1).', edges(2:end).'];

% 频域零相位滤波器组（升余弦过渡，随后进行归一化使各通道之和≈1）
twDefault = 0.05;  % 过渡带相对带宽（5%）
Hpos_all = zeros(numel(fpos), K);
for k = 1:K
    fL = bands(k,1); fH = bands(k,2);
    bw = max(fH - fL, 1e-6);
    tw = max(0.1, twDefault * bw);           % 防止过小
    fL1 = max(0, fL - tw); fL2 = fL;
    fH1 = fH;         fH2 = min(fs/2, fH + tw);

    Hk = zeros(size(fpos));
    % 左过渡：0 -> 1（半余弦）
    idxL = fpos >= fL1 & fpos < fL2;
    if any(idxL)
        phi = (fpos(idxL) - fL1) / max(fL2 - fL1, eps);
        Hk(idxL) = 0.5 - 0.5*cos(pi*phi);
    end
    % 平顶：1
    idxM = fpos >= fL2 & fpos <= fH1; Hk(idxM) = 1;
    % 右过渡：1 -> 0（半余弦）
    idxR = fpos > fH1 & fpos <= fH2;
    if any(idxR)
        phi = (fH2 - fpos(idxR)) / max(fH2 - fH1, eps);
        Hk(idxR) = 0.5 - 0.5*cos(pi*phi);
    end

    Hpos_all(:,k) = Hk;
end

% 归一化，避免带通重叠造成能量重复
sumH = sum(Hpos_all, 2);
sumH(sumH==0) = 1; % 防零除
Hpos_all = Hpos_all ./ sumH;

% 对称到负频，得到零相滤波器
efd = cell(K,1);
for k = 1:K
    H = zeros(Nfft,1);
    H(mask_pos) = Hpos_all(:,k);
    H(~mask_pos) = flipud(H(mask_pos));
    Yk = X .* H;
    xk = real(ifft(ifftshift(Yk)));
    efd{k} = xk(1:N);
end

Finfo = struct('faxis', faxis, 'mag', mag, 'pos_f', fpos, 'pos_mag', Spos, 'pos_mag_smooth', Ssm, 'edges', edges);
end

function r = getfield_def(S, name, def)
if isfield(S, name)
    r = S.(name);
else
    r = def;
end
end

function x = local_call_vme(y, alpha, omega0, fs, tau, tol)
% 调用外部 VME 实现（VME_GMETV/VME/vme.m）
[u_d, ~] = vme(y(:).', alpha, omega0, fs, tau, tol); % 注意 vme 接受行向量
% vme 返回行向量，转为列向量
if isempty(u_d)
    error('VME returned empty result');
end
x = u_d(:);
if numel(x) ~= numel(y)
    error('VME output length (%d) != input length (%d)', numel(x), numel(y));
end
end

function c = local_corr(x, y)
% 皮尔逊相关（长度对齐）
L = min(numel(x), numel(y));
x = x(1:L); y = y(1:L);
x = x - mean(x); y = y - mean(y);
c = (x.'*y) / (sqrt(sum(x.^2))*sqrt(sum(y.^2)) + eps);
end

function sk = local_skewness(x)
% 无需统计工具箱的偏度实现
x = x(:) - mean(x);
if all(x==0)
    sk = 0; return;
end
m3 = mean(x.^3);
s3 = (std(x)+eps)^3;
sk = m3 / s3;
end

function c = local_spectral_centroid(x, fs)
N = numel(x);
X = abs(fft(x));
X = X(1:floor(N/2)+1);
f = linspace(0, fs/2, numel(X)).';
num = sum(f .* X);
den = sum(X) + eps;
c = num / den;
end

function p = local_bandpower(x, fs, frange)
% 简易带内功率（Periodogram 积分）
if nargin < 3 || isempty(frange), frange = [0, fs/2]; end
x = x(:);
N = numel(x);
X = fft(x);
Pxx = abs(X).^2 / N;                 % 单边尺度差异对比仅用于比值，不做额外缩放
f = (0:N-1)*(fs/N);
% 折叠到 0..fs/2（近似求和）
half = 1:floor(N/2)+1;
P = Pxx(half); ff = f(half);
idx = (ff >= frange(1)) & (ff <= frange(2));
% 近似积分：矩形法
if any(idx)
    df = mean(diff(ff));
    p = sum(P(idx)) * df;
else
    p = 0;
end
end

function [pks, locs] = local_findpeaks(y)
% 兼容无信号处理工具箱的 findpeaks（简单替代）
y = y(:);
% 朴素寻找局部极大值
locs = find( [false; y(2:end-1) > y(1:end-2) & y(2:end-1) >= y(3:end); false] );
pks  = y(locs);
% 去掉过密峰：按 0.2 Hz 的近邻最小距离（在索引上近似）
% 距离阈值按采样点推断：假定正频率长度对应 fs/2 频宽，此处无法直接转 Hz，
% 因此这里先不施加距离筛选，交由后续“取前K个峰”处理。
end
