% ========================================================================
% 生成多SNR等级的全模拟数据集（混合所有等级）
% ========================================================================
% 数据集特点:
% 1. 四种类型: 无干扰、仅眨眼、仅眼动、眨眼+眼动
% 2. 七种SNR等级，眨眼和眼动信噪比一一对应:
%    等级1: 眨眼0dB,   眼动6dB
%    等级2: 眨眼-2dB,  眼动4dB
%    等级3: 眨眼-4dB,  眼动2dB
%    等级4: 眨眼-8dB,  眼动-2dB
%    等级5: 眨眼-12dB, 眼动-6dB
%    等级6: 眨眼-14dB, 眼动-8dB
%    等级7: 眨眼-16dB, 眼动-10dB
% 
% 数据划分:
% - 训练集: 80%
% - 验证集: 10%
% - 测试集: 10%
% - 微调集1: 10% (用于无监督方法微调)
% - 微调集2: 20% (用于无监督方法微调)
% 
% 每个等级的数据量相同，每个等级内四种类型的数量也相同
% ========================================================================

clear all;
clf;
close all;

% 定义信噪比等级
snr_blink_levels = [0, -2, -4, -8, -12, -14, -16];  % 眨眼信噪比 (dB)
snr_eog_levels = [6, 4, 2, -2, -6, -8, -10];        % 眼动信噪比 (dB)

% 基本参数
dataLength = 6;              % 每个样本的长度(秒)
samplesPerLevelPerType = 150; % 每个等级每种类型的样本数 (143*4*7=4004)
fs = 250;                    % 采样率

% 创建输出目录
output_dir = 'D:\Pycharm_Projects\EOG Remove\生成全模拟数据\已经生成好的数据\Multi_SNR_Merged';
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

fprintf('========================================\n');
fprintf('开始生成多SNR等级全模拟数据集\n');
fprintf('========================================\n');
fprintf('总等级数: %d\n', length(snr_blink_levels));
fprintf('每个等级每种类型的样本数: %d\n', samplesPerLevelPerType);
fprintf('总样本数: %d\n\n', samplesPerLevelPerType * 4 * length(snr_blink_levels));

% 生成所有等级的数据
[allData, visualizationSamples] = generate_all_levels_data(...
    snr_blink_levels, snr_eog_levels, samplesPerLevelPerType, dataLength, fs);

fprintf('\n========================================\n');
fprintf('开始划分数据集...\n');
fprintf('========================================\n\n');

% 划分数据集
split_and_save_datasets(allData, output_dir, snr_blink_levels, snr_eog_levels);

fprintf('\n========================================\n');
fprintf('生成可视化图例...\n');
fprintf('========================================\n\n');

% 生成可视化
visualize_merged_dataset(visualizationSamples, snr_blink_levels, snr_eog_levels, output_dir, fs, dataLength);

fprintf('\n所有任务完成！\n');
fprintf('数据保存在: %s\n', output_dir);
