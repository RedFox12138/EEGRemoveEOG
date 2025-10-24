% 计算RMS函数
function rms_val = RMS(X)
    [N, K] = size(X);
    rms_val = sqrt(1/(N*K) * sum(sum(X.^2)));
end