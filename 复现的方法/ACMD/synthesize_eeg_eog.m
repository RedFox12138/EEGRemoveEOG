function sig = synthesize_eeg_eog(fs, durSec, opts)
%SYNTHESIZE_EEG_EOG  Create a synthetic single-channel EEG with ocular artifacts.
%
%   sig = synthesize_eeg_eog(fs, durSec, opts)
%
%   Outputs a struct with fields:
%     .t            - time vector (s)
%     .clean        - clean EEG-like signal (uV, arbitrary units)
%     .eog          - EOG artifact signal (slow blinks/saccades)
%     .baseline     - very low-frequency baseline wander
%     .contaminated - clean + eog + baseline
%     .fs           - sampling rate (Hz)
%
%   Parameters (opts, all optional):
%     .alphaHz      - center alpha freq (default 10 Hz)
%     .alphaVarHz   - slow FM deviation for alpha (default 1 Hz)
%     .betaHz       - beta component (default 18 Hz)
%     .snrCleanDb   - SNR of clean EEG vs white noise (default 10 dB)
%     .blinkPerMin  - blink rate (default 18 blinks/min)
%     .blinkDur     - blink duration (default 0.35 s)
%     .blinkAmp     - blink peak amplitude relative to alpha (default 3.0)
%     .baselineHz   - baseline wander freq (default 0.2 Hz)
%     .baselineAmp  - baseline amplitude relative to alpha (default 0.25)
%     .alphaAmp     - base amplitude (default 1.0)
%
%   This generator is designed to emulate the paper's target scenario where
%   OAs are low-frequency, high-amplitude, slowly varying components.

if nargin < 1 || isempty(fs), fs = 200; end
if nargin < 2 || isempty(durSec), durSec = 30; end
if nargin < 3, opts = struct(); end

% defaults
D.alphaHz     = 10;
D.alphaVarHz  = 1;
D.betaHz      = 18;
D.snrCleanDb  = 10;
D.blinkPerMin = 18;  % typical blink rate
D.blinkDur    = 0.35; % second
D.blinkAmp    = 3.0;  % times of alpha amplitude
D.baselineHz  = 0.2;  % Hz
D.baselineAmp = 0.25; % times of alpha amplitude
D.alphaAmp    = 1.0;

fns = fieldnames(D);
for i = 1:numel(fns)
    if ~isfield(opts, fns{i}) || isempty(opts.(fns{i}))
        opts.(fns{i}) = D.(fns{i});
    end
end

N = round(durSec*fs);
t = (0:N-1)'/fs;

% --- Clean EEG: alpha with slow FM + weak beta + noise ---
alphaIF = opts.alphaHz + opts.alphaVarHz * 0.5 * sin(2*pi*0.2*t); % 0.2 Hz FM
phiA = 2*pi*cumsum(alphaIF)/fs;
alpha = opts.alphaAmp * sin(phiA);

beta = 0.2*opts.alphaAmp * sin(2*pi*opts.betaHz*t + 0.7);

% white noise to reach target SNR
clean_wo_noise = alpha + beta;
P_sig = mean(clean_wo_noise.^2);
P_noise = P_sig / (10^(opts.snrCleanDb/10));
wn = sqrt(P_noise) * randn(N,1);
clean = clean_wo_noise + wn;

% --- Baseline wander ---
baseline = opts.baselineAmp * opts.alphaAmp * sin(2*pi*opts.baselineHz*t + 1.1);

% --- EOG blinks: half-sine lobes with smooth rise/fall ---
blinks = zeros(N,1);
rate = opts.blinkPerMin/60;             % blinks per second
lambda = rate;                          % Poisson rate
% generate Poisson blink onsets
onsets = [];
current = 0;
while current < durSec
    % exponential inter-arrival
    delta = -log(max(eps, 1-rand))/lambda;
    current = current + delta;
    if current < durSec
        onsets(end+1) = current; %#ok<AGROW>
    end
end
blinkN = round(opts.blinkDur*fs);
shape = sin(linspace(0,pi,blinkN))';    % half-sine blink shape
shape = shape.^1.5;                      % slightly sharper peak
shape = opts.blinkAmp*opts.alphaAmp * shape; % scale
for k = 1:numel(onsets)
    idx0 = round(onsets(k)*fs) + 1;
    idx1 = min(N, idx0 + blinkN - 1);
    seg = 1:(idx1-idx0+1);
    blinks(idx0:idx1) = blinks(idx0:idx1) + shape(seg);
end

% --- Compose ---
contaminated = clean + blinks + baseline;

sig = struct('t', t, 'clean', clean, 'eog', blinks, 'baseline', baseline, ...
             'contaminated', contaminated, 'fs', fs);
end
