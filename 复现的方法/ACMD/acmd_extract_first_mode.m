function [mode1, fi, phi] = acmd_extract_first_mode(x, fs, varargin)
%ACMD_EXTRACT_FIRST_MODE  Approximate ACMD first mode via ridge-guided demodulation.
%
%   [mode1, fi, phi] = acmd_extract_first_mode(x, fs, 'Name', value, ...)
%
%   This function approximates the first mode produced by Adaptive Chirp
%   Mode Decomposition (ACMD) by:
%     - computing a low-frequency spectral ridge from the STFT,
%     - demodulating the signal along that instantaneous frequency (IF),
%     - low-pass filtering the baseband (bandwidth set by 'bandwidthHz'), and
%     - remodulating back to the original frequency trajectory.
%
%   Inputs
%     x  - normalized real signal (column vector recommended)
%     fs - sampling rate (Hz)
%
%   Name-value pairs
%     'fmaxOA'      - max freq (Hz) to search for the ridge [default 12]
%     'winSec'      - STFT window length in seconds [default 1.0]
%     'ridgeSmooth' - smoothing window for IF track in samples [default round(0.25*fs)]
%     'bandwidthHz' - baseband low-pass cutoff (Hz) after demodulation [default 2]
%
%   Outputs
%     mode1 - extracted first mode (real-valued)
%     fi    - instantaneous frequency track used for demodulation (Hz)
%     phi   - phase track (rad)
%
%   Note: this is an engineering approximation faithful to the paper's
%   intention (extracting the dominant low-frequency chirp-like OA mode).
%
%   Author: Copilot (MATLAB)

p = inputParser;
p.addParameter('fmaxOA', 12, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('winSec', 1.0, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.addParameter('ridgeSmooth', [], @(v)isnumeric(v)&&isscalar(v)&&v>=1);
p.addParameter('bandwidthHz', 2, @(v)isnumeric(v)&&isscalar(v)&&v>0);
p.parse(varargin{:});
pr = p.Results;

x = x(:);
L = numel(x);
if isempty(pr.ridgeSmooth)
    pr.ridgeSmooth = max(5, round(0.25*fs));
end

% --- STFT and ridge extraction ---
winLen = max(32, round(pr.winSec*fs));
win = hann(winLen, 'periodic');
nover = min(winLen-1, round(0.9*winLen));
nfft = 2^nextpow2(max(winLen, round(2*fs))); % frequency resolution ~0.5 Hz or better
[S,F,T] = spectrogram(x, win, nover, nfft, fs);

% limit to low-frequency band (<= fmaxOA)
maskF = F <= pr.fmaxOA & F >= 0.1; % exclude ~0 Hz remnants
Smag = abs(S(maskF, :));
Flow = F(maskF);

% for each time frame, take the frequency index of maximum magnitude
[~, idx] = max(Smag, [], 1);
fi_frame = Flow(idx);                     % Hz at STFT frames

% interpolate to sample grid
if numel(T) == 1
    fi = repmat(fi_frame, L, 1); % degenerate case
else
    tFrames = round(T*fs) + 1;                % approximate frame center indices
    tFrames = max(1, min(L, tFrames));
    % Ensure monotonic time indices for interp1
    [tFramesU, iu] = unique(tFrames);
    fi_frame_u = fi_frame(iu);
    fi = interp1(tFramesU, fi_frame_u, (1:L)', 'linear', 'extrap');
end

% Smooth IF track to reduce jitter
fi = movmedian(fi, pr.ridgeSmooth);
fi(fi < 0.1) = 0.1;                        % keep strictly positive

% --- Complex demodulation ---
phi = 2*pi*cumsum(fi)/fs;                  % instantaneous phase (rad)
expNeg = exp(-1j*phi);
xd = x .* expNeg;                          % demodulated to baseband

% one-pole low-pass (toolbox-free)
fc = min(pr.bandwidthHz, fs/2 - 1);        % safety bound
alpha = exp(-2*pi*fc/fs);                  % 0<alpha<1
yd = onepole_lowpass(xd, alpha);

% remodulate back
mode1_c = yd .* conj(expNeg);              % shift back to original path
mode1 = real(mode1_c);

end

function y = onepole_lowpass(x, alpha)
%ONEPOLE_LOWPASS  y[n] = alpha*y[n-1] + (1-alpha)*x[n]
% Works for real or complex x.
L = numel(x);
y = zeros(size(x));
if L == 0
    return;
end
b = (1 - alpha);
y(1) = b * x(1);
for n = 2:L
    y(n) = alpha * y(n-1) + b * x(n);
end
end
