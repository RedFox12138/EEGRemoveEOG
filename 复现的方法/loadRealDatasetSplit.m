function [test_data, train_data] = loadRealDatasetSplit(data_path, data_key, train_ratio, random_seed)
% LOADREALDATASETSPLIT 统一的真实数据集加载和划分函数
%
% 功能:
%   - 加载真实数据集
%   - 使用固定随机种子划分训练集和测试集
%   - 确保所有方法使用相同的数据划分
%
% 输入参数:
%   data_path    - 数据文件路径 (默认: 真实数据集路径)
%   data_key     - .mat文件中的数据键名 (默认: 'eog_dataset')
%   train_ratio  - 训练集比例 (默认: 0.9, 即90%训练，10%测试)
%   random_seed  - 随机种子 (默认: 42, 确保可复现)
%
% 输出参数:
%   test_data    - 测试集数据 (10% 的样本)
%   train_data   - 训练集数据 (90% 的样本, 供无监督方法使用)
%
% 使用示例:
%   % 只需要测试集数据 (有监督方法和传统方法)
%   test_data = loadRealDatasetSplit();
%   
%   % 需要训练集和测试集数据 (无监督方法)
%   [test_data, train_data] = loadRealDatasetSplit();
%
% 作者: GitHub Copilot
% 日期: 2025-12-10

%% 参数默认值
if nargin < 1 || isempty(data_path)
    data_path = 'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat';
end

if nargin < 2 || isempty(data_key)
    data_key = 'eog_dataset';
end

if nargin < 3 || isempty(train_ratio)
    train_ratio = 0.9;  % 90% 训练，10% 测试
end

if nargin < 4 || isempty(random_seed)
    random_seed = 42;  % 固定随机种子，确保所有方法数据划分一致
end

%% 加载数据
fprintf('正在加载真实数据集...\n');
fprintf('  数据路径: %s\n', data_path);

% 加载.mat文件
data_struct = load(data_path);

% 尝试不同的可能字段名
if isfield(data_struct, data_key)
    data = data_struct.(data_key);
    fprintf('  ✓ 使用字段: "%s"\n', data_key);
elseif isfield(data_struct, 'data')
    data = data_struct.data;
    fprintf('  ✓ 使用字段: "data"\n');
elseif isfield(data_struct, 'eeg_data')
    data = data_struct.eeg_data;
    fprintf('  ✓ 使用字段: "eeg_data"\n');
else
    error('无法找到数据！请检查.mat文件中的字段名');
end

fprintf('  数据形状: [%d, %d]\n', size(data, 1), size(data, 2));
fprintf('  总样本数量: %d\n', size(data, 1));
fprintf('  样本长度: %d\n', size(data, 2));

%% 随机划分数据集
n_samples = size(data, 1);

% 设置随机种子以确保可复现性
rng(random_seed);

% 生成随机打乱的索引
indices = randperm(n_samples);

% 计算划分点
train_size = floor(n_samples * train_ratio);
test_size = n_samples - train_size;

% 划分数据
train_indices = indices(1:train_size);
test_indices = indices(train_size+1:end);

train_data = data(train_indices, :);
test_data = data(test_indices, :);

%% 输出划分信息
fprintf('\n数据集划分完成 (随机种子=%d):\n', random_seed);
fprintf('  训练集: %d 样本 (%.0f%%)\n', size(train_data, 1), train_ratio * 100);
fprintf('  测试集: %d 样本 (%.0f%%)\n', size(test_data, 1), (1 - train_ratio) * 100);
fprintf('  ⚠️  所有方法应只在测试集上进行评估！\n\n');

end
