function [z, info] = oa_remove_acmd(x, fs, opts)
%OA_REMOVE_ACMD  Ocular artifact (OA) removal using DFT + ACMD + peak-count test.
%
%   [z, info] = oa_remove_acmd(x, fs, opts)
%
%   Implements the method described in the attached paper:
%   "Automated OA removal based on Adaptive Chirp Mode Decomposition (ACMD)".
%   Pipeline:
%     1) Remove baseline wander (<1 Hz) in the DFT domain (Section II-A).
%     2) Amplitude normalization to [-1, 1].
%     3) Decompose the normalized signal with ACMD and take the first mode (Section II-B).
%     4) Detect OA presence in the first mode using peak-point count Ψ_p (Eq. 10–11).
%     5) If OA is detected (Ψ_p < ξ), reconstruct: z = x_norm - x1; else z = x_norm. (Eq. 12)
%
%   Inputs
%     x    - 1-D EEG vector (row or column). Real-valued.
%     fs   - Sampling rate in Hz.
%     opts - (optional) struct with fields:
%              .baselineHz  - frequency in Hz for baseline-wander cutoff (default 1 Hz)
%              .fmaxOA      - max frequency to search for ridges (default 12 Hz)
%              .winSec      - STFT window length in seconds (default 1.0)
%              .ridgeSmooth - smoothing (samples) for IF track (default round(0.25*fs))
%              .bandwidthHz - equivalent baseband low-pass cutoff for ACMD (default 2 Hz)
%              .threshold   - ξ for OA detection (if empty, only returns info and keeps signal)
%              .returnAll   - if true, return intermediate signals in info (default true)
%              .restoreAmplitude - if true, rescale output z back to pre-normalization
%                                   amplitude using max|u| (default false)
%              .autoTune    - if true, auto-tune fmaxOA & bandwidthHz by maximizing
%                              first-mode energy (Fig. 3 analogue) [default false]
%              .tuneBW      - vector of candidate bandwidthHz values [default 0.5:0.5:3]
%              .tuneFmax    - vector of candidate fmaxOA values [default 8:2:14]
%              .refineRounds- iterative subtraction passes of first mode [default 1]
%
%   Outputs
%     z    - OA-suppressed EEG (normalized amplitude as in the paper).
%     info - struct with fields:
%              .baseline, .x_norm, .mode1, .psi, .detected, .threshold,
%              .fi (IF track), .phi (phase), .normScale (max|u|),
%              .z_unorm (if returnAll), .psiAll, .modes, .params (effective parameters)
%
%   Notes
%   - This implementation approximates ACMD by complex-demodulation guided by a
%     time-varying spectral ridge (low-frequency, < fmaxOA). It aims to capture
%     the first ACMD mode which contains OA when present. Although not a verbatim
%     reproduction of the original optimization routine, it follows the same
%     signal-processing rationale and respects detection/reconstruction rules.
%
%   Author: Copilot (MATLAB)
%
%   See also: acmd_extract_first_mode, peak_count

if nargin < 2
    error('oa_remove_acmd:NotEnoughInputs', 'Provide x and fs.');
end
if nargin < 3 || isempty(opts)
    opts = struct();
end

% --- defaults ---
def.baselineHz  = 1;      % <1 Hz considered baseline
% The paper uses 200 Hz after resampling for comparisons; keep general
% ACMD approximation params

def.fmaxOA      = 10;     % search ridge below this frequency (Hz)

def.winSec      = 1.0;    % spectrogram window seconds

def.ridgeSmooth = [];     % set below based on fs

def.bandwidthHz = 1.5;      % baseband LPF cutoff after demod (Hz)

def.threshold   = [];     % if empty, do not subtract automatically

def.returnAll   = true;
def.restoreAmplitude = false; % keep normalized scale by default (faithful to paper)
def.autoTune    = false;
def.tuneBW      = 0.5:0.5:3;
def.tuneFmax    = 8:2:14;
def.refineRounds= 1;

% merge defaults
fn = fieldnames(def);
for i = 1:numel(fn)
    if ~isfield(opts, fn{i}) || isempty(opts.(fn{i}))
        opts.(fn{i}) = def.(fn{i});
    end
end
if isempty(opts.ridgeSmooth)
    opts.ridgeSmooth = max(5, round(0.25*fs));
end

x = x(:);                    % enforce column vector
L = numel(x);

% 1) DFT baseline-wander removal (< baselineHz)
[u, baseline] = local_fft_highpass(x, fs, opts.baselineHz);

% 2) Amplitude normalization
mx = max(abs(u));
if mx < eps
    x_norm = u;  % avoid division by near-zero
else
    x_norm = u / mx;
end

% 3) Optional auto-tune (choose bandwidth/fmax maximizing first-mode energy)
fmaxUse = opts.fmaxOA; bwUse = opts.bandwidthHz;
if opts.autoTune
    [bwUse, fmaxUse] = acmd_autotune(x_norm, fs, opts.tuneBW, opts.tuneFmax, ...
                                     opts.winSec, opts.ridgeSmooth);
end

% 4) ACMD approximation + iterative subtraction
x_work = x_norm;
psiAll = [];
modes  = {};
fi = []; phi = []; mode1 = [];
for r = 1:max(1, opts.refineRounds)
    [m1, fi, phi] = acmd_extract_first_mode(x_work, fs, ...
        'fmaxOA', fmaxUse, 'winSec', opts.winSec, ...
        'ridgeSmooth', opts.ridgeSmooth, 'bandwidthHz', bwUse);
    psi_r = peak_count(m1);
    psiAll(end+1) = psi_r; %#ok<AGROW>
    modes{end+1}  = m1; %#ok<AGROW>
    if ~isempty(opts.threshold) && (psi_r < opts.threshold)
        x_work = x_work - m1; % subtract and continue
    else
        break; % stop if not detected
    end
end
mode1 = modes{1};
z_norm = x_work;

% Optionally restore amplitude to match input scale
if opts.restoreAmplitude
    z = z_norm * mx;
else
    z = z_norm;
end

% info pack
info = struct();
if opts.returnAll
    info.baseline  = baseline;
    info.x_norm    = x_norm;
    info.mode1     = mode1;
    info.fi        = fi;
    info.phi       = phi;
    info.z_unorm   = z_norm * mx; % convenience: output at original scale
    info.psiAll    = psiAll;
    info.modes     = modes;
end
info.psi       = psiAll(1);
info.detected  = ~isempty(opts.threshold) && (psiAll(1) < opts.threshold);
info.threshold = opts.threshold;
info.normScale = mx;
info.params    = opts;

end

function [y, b] = local_fft_highpass(x, fs, fcut)
% Remove frequencies |f| < fcut (Hz) via DFT thresholding as in the paper
x = x(:);
L = numel(x);
X = fft(x);
% frequency vector (0..fs*(L-1)/L)
f = (0:L-1)' * (fs/L);
idxCut = f < fcut | f > (fs - fcut); % include both sides for real signal symmetry
% Build keep mask = NOT idxCut for positive band excluding DC region
keep = ~idxCut;
Xkeep = X .* keep;
y = real(ifft(Xkeep));      % baseline removed signal u(l)
% Reconstruct baseline component as the residual
b = x - y;                  % baseline wander (for diagnostics)
end
