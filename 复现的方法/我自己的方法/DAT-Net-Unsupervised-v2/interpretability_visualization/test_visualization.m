% test_visualization.m
% 快速测试脚本 - 验证MATLAB可视化系统是否正常工作
%
% 用法:
%   运行此脚本来测试所有功能是否正常

fprintf('\n');
fprintf('========================================\n');
fprintf('DAT-Net 可视化系统测试脚本\n');
fprintf('========================================\n\n');

%% 测试1: 加载配置
fprintf('测试1: 加载配置文件...\n');
try
    config = config_visualization();
    fprintf('✓ 配置加载成功\n');
    fprintf('  输出目录: %s\n', config.OUTPUT_DIR);
    fprintf('  采样率: %d Hz\n', config.DATA.fs);
catch ME
    fprintf('❌ 配置加载失败: %s\n', ME.message);
    return;
end

%% 测试2: 检查数据文件
fprintf('\n测试2: 检查数据文件...\n');

% 检查真实数据集
if exist(config.REAL_DATA_FILE, 'file')
    fprintf('✓ 真实数据集文件存在: %s\n', config.REAL_DATA_FILE);
else
    fprintf('⚠️  真实数据集文件不存在: %s\n', config.REAL_DATA_FILE);
    fprintf('   请在config_visualization.m中更新路径\n');
end

% 检查模型预测结果
if exist(config.MODEL_PREDICTION_FILE, 'file')
    fprintf('✓ 模型预测文件存在: %s\n', config.MODEL_PREDICTION_FILE);
else
    fprintf('⚠️  模型预测文件不存在: %s\n', config.MODEL_PREDICTION_FILE);
    fprintf('   将使用零填充作为预测结果(仅用于测试)\n');
end

%% 测试3: 尝试加载数据
fprintf('\n测试3: 尝试加载数据...\n');
try
    [data, ~] = load_real_dataset(1, 1, config);
    fprintf('✓ 数据加载成功\n');
    fprintf('  样本索引: %d\n', data.sample_idx);
    fprintf('  通道索引: %d\n', data.channel_idx);
    fprintf('  信号长度: %d\n', data.signal_length);
    fprintf('  采样率: %d Hz\n', data.fs);
catch ME
    fprintf('❌ 数据加载失败: %s\n', ME.message);
    fprintf('   请检查数据文件路径和格式\n');
    return;
end

%% 测试4: 测试辅助函数
fprintf('\n测试4: 测试辅助函数...\n');

% 测试滑动平均
test_signal = randn(1, 1000);
try
    result = moving_average_test(test_signal, 10);
    fprintf('✓ 滑动平均函数正常\n');
catch ME
    fprintf('❌ 滑动平均函数失败: %s\n', ME.message);
end

% 测试FFT低通滤波
try
    result = fft_lowpass_test(test_signal, 200, 10);
    fprintf('✓ FFT低通滤波函数正常\n');
catch ME
    fprintf('❌ FFT低通滤波函数失败: %s\n', ME.message);
end

% 测试MAD归一化
try
    result = mad_normalize_test(test_signal);
    fprintf('✓ MAD归一化函数正常\n');
catch ME
    fprintf('❌ MAD归一化函数失败: %s\n', ME.message);
end

%% 测试5: 创建测试图像
fprintf('\n测试5: 创建测试图像...\n');
try
    fig = figure('Position', [100, 100, 800, 600], 'Color', 'w');
    plot(1:100, randn(1, 100), 'LineWidth', 1.5);
    title('测试图像', 'FontSize', 14);
    xlabel('时间', 'FontSize', 10);
    ylabel('幅值', 'FontSize', 10);
    grid on;
    
    % 尝试保存
    test_filename = fullfile(config.FIGURE_DIR, 'test_figure.png');
    print(fig, test_filename, '-dpng', '-r150');
    
    if exist(test_filename, 'file')
        fprintf('✓ 图像创建和保存成功\n');
        fprintf('  保存位置: %s\n', test_filename);
        delete(test_filename);  % 删除测试文件
    else
        fprintf('⚠️  图像保存失败\n');
    end
    
    close(fig);
catch ME
    fprintf('❌ 图像创建失败: %s\n', ME.message);
end

%% 测试6: 检查可视化函数
fprintf('\n测试6: 检查可视化函数...\n');

functions_to_check = {
    'vis_artifact_probability', '伪影概率计算可视化';
    'vis_masking_strategy', '掩蔽策略可视化';
    'vis_denoising_results', '去噪效果可视化';
    'main_visualization', '主程序入口'
};

all_exist = true;
for i = 1:size(functions_to_check, 1)
    func_name = functions_to_check{i, 1};
    func_desc = functions_to_check{i, 2};
    
    if exist(func_name, 'file')
        fprintf('✓ %s (%s)\n', func_desc, func_name);
    else
        fprintf('❌ %s 不存在 (%s)\n', func_desc, func_name);
        all_exist = false;
    end
end

%% 总结
fprintf('\n========================================\n');
fprintf('测试总结\n');
fprintf('========================================\n');

if all_exist && exist(config.REAL_DATA_FILE, 'file')
    fprintf('✓ 所有测试通过! 系统可以正常使用。\n\n');
    fprintf('快速开始:\n');
    fprintf('  1. 运行所有任务: main_visualization(''all'', 1, 1);\n');
    fprintf('  2. 运行单个任务: main_visualization(''artifact_probability'', 1, 1);\n');
    fprintf('  3. 查看任务列表: main_visualization(''list'');\n');
else
    fprintf('⚠️  部分测试未通过,请检查:\n');
    if ~exist(config.REAL_DATA_FILE, 'file')
        fprintf('  - 真实数据集文件路径\n');
    end
    if ~all_exist
        fprintf('  - 可视化函数文件\n');
    end
end

fprintf('========================================\n\n');


%% ==================== 辅助测试函数 ====================

function y = moving_average_test(x, win_size)
    if win_size <= 1
        y = x;
        return;
    end
    kernel = ones(1, win_size) / win_size;
    y = conv(x, kernel, 'same');
end

function x_low = fft_lowpass_test(x, fs, cutoff)
    L = length(x);
    X = fft(x);
    freqs = (0:L-1) * fs / L;
    mask = freqs <= cutoff | freqs >= (fs - cutoff);
    X_low = X .* mask;
    x_low = real(ifft(X_low));
end

function x_n = mad_normalize_test(x)
    eps_val = 1e-8;
    med = median(x);
    mad_val = median(abs(x - med));
    mad_val = max(mad_val, eps_val);
    x_n = (x - med) / mad_val;
end
