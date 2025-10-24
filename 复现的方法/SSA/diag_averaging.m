function x_rec = diag_averaging(X)
% 对角平均重构（复现文献Vautard 1992方法）
% 输入：X - 轨迹矩阵(L×R)
% 输出：x_rec - 重构信号(长度=L+R-1)

[L, R] = size(X);
P = L + R - 1;  % 重构信号长度=原始信号长度
x_rec = zeros(1, P);

% 遍历所有对角线（从左上到右下）
for k = 1:P
    diag_sum = 0;
    diag_count = 0;
    
    % 遍历对角线上的所有元素
    for i = 1:L
        j = k - i + 1;  % 列索引
        if j >= 1 && j <= R
            diag_sum = diag_sum + X(i, j);
            diag_count = diag_count + 1;
        end
    end
    
    x_rec(k) = diag_sum / diag_count;
end

end