function fd = sevcik_fd(x)
% Sevcik分形维数计算（Sevcik 1998原始方法）
% 参考：Sevcik, C. (1998). A procedure to estimate the fractal dimension of waveforms.
% 输入：x - 1维信号
% 输出：fd - 分形维数

% 去除零值和NaN
x = x(:);  % 确保列向量
x = x(~isnan(x) & ~isinf(x));

if length(x) < 2
    fd = 1.0;
    return;
end

% 归一化信号到[0,1]
x = x - min(x);
x = x / (max(x) + eps);

N = length(x);

% Sevcik方法：计算信号的累积欧氏距离
% L = 信号总长度（曲线长度）
L = 0;
for i = 1:N-1
    % 每个点到下一个点的欧氏距离
    dx = 1 / (N - 1);  % 时间步长（归一化）
    dy = x(i+1) - x(i);  % 幅值差
    L = L + sqrt(dx^2 + dy^2);
end

% Sevcik公式
% FD = 1 + log(L) / log(2*(N-1))
if L > 0
    fd = 1 + log(L) / log(2 * (N - 1));
else
    fd = 1.0;  % 常数信号
end

% 确保FD在合理范围内 [1, 2]
fd = max(1.0, min(2.0, fd));

end