function [bestBW, bestFmax, gridE] = acmd_autotune(x_norm, fs, candBW, candFmax, winSec, ridgeSmooth)
%ACMD_AUTOTUNE  Select bandwidth and fmax by maximizing first-mode energy.
%
%   [bestBW, bestFmax, gridE] = acmd_autotune(x_norm, fs, candBW, candFmax, winSec, ridgeSmooth)
%
%   Returns the pair (bandwidthHz, fmaxOA) that maximizes E1 = sum(mode1.^2).
%   This mimics Fig. 3 in the paper where the maximum energy in the first
%   decomposed mode indicates parameters capturing OAs in mode 1.
%
%   gridE(i,j) corresponds to BW=candBW(i), Fmax=candFmax(j).

if nargin < 3 || isempty(candBW),  candBW = 0.5:0.5:3; end
if nargin < 4 || isempty(candFmax),candFmax = 8:2:14; end
if nargin < 5 || isempty(winSec),   winSec = 1.0; end
if nargin < 6 || isempty(ridgeSmooth), ridgeSmooth = max(5, round(0.25*fs)); end

x_norm = x_norm(:);
nb = numel(candBW); nf = numel(candFmax);
gridE = zeros(nb, nf);

bestE = -Inf; bestBW = candBW(1); bestFmax = candFmax(1);

for i = 1:nb
    for j = 1:nf
        try
            m1 = acmd_extract_first_mode(x_norm, fs, 'bandwidthHz', candBW(i), ...
                'fmaxOA', candFmax(j), 'winSec', winSec, 'ridgeSmooth', ridgeSmooth);
            E1 = sum(m1.^2);
        catch
            E1 = -Inf; % in case of numerical issues
        end
        gridE(i,j) = E1;
        if E1 > bestE
            bestE = E1; bestBW = candBW(i); bestFmax = candFmax(j);
        end
    end
end

end
