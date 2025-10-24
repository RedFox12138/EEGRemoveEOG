function psi = peak_count(x)
%PEAK_COUNT  Peak-point count Ψ_p as defined in Eq. (10)-(11) of the paper.
%
%   psi = peak_count(x)
%
%   Counts the number of indices p such that |x(p-1)| < |x(p)| > |x(p+1)|.
%   Endpoints are ignored. Flat peaks are handled conservatively (strict >).
%
%   Input:  x - real-valued vector (row or column)
%   Output: psi - scalar count of peak points

x = abs(x(:));
N = numel(x);
if N < 3
    psi = 0;
    return;
end

% Use strict comparison as in Eq. (11)
prev = x(1:N-2);
mid  = x(2:N-1);
next = x(3:N);
peaks = (prev < mid) & (mid > next);
psi = sum(peaks);

end
