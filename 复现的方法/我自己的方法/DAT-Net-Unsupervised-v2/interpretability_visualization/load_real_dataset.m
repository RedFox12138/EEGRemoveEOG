function [data, config] = load_real_dataset(sample_idx, channel_idx, config)
% LOAD_REAL_DATASET 加载真实数据集和模型预测结果
%
% 输入:
%   sample_idx - 样本索引 (可选, 默认为1)
%   channel_idx - 通道索引 (可选, 默认为1)
%   config - 配置结构体 (可选)
%
% 输出:
%   data - 数据结构体,包含:
%       .contaminated - 受污染的EEG信号 (1, L)
%       .clean_pred - 模型预测的干净信号 (1, L)
%       .artifact_pred - 模型预测的伪影信号 (1, L)
%       .fs - 采样率
%       .sample_idx - 样本索引
%       .channel_idx - 通道索引
%   config - 配置结构体
%
% 用法:
%   data = load_real_dataset();  % 使用默认参数
%   data = load_real_dataset(5, 10);  % 指定样本5,通道10

    % 默认参数
    if nargin < 1 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 2 || isempty(channel_idx)
        channel_idx = 1;
    end
    if nargin < 3 || isempty(config)
        config = config_visualization();
    end
    
    fprintf('==================================================\n');
    fprintf('加载真实数据集\n');
    fprintf('==================================================\n');
    fprintf('样本索引: %d\n', sample_idx);
    fprintf('通道索引: %d\n', channel_idx);
    
    %% 加载真实数据集
    if ~exist(config.REAL_DATA_FILE, 'file')
        error('真实数据集文件不存在: %s', config.REAL_DATA_FILE);
    end
    
    fprintf('正在加载真实数据集: %s\n', config.REAL_DATA_FILE);
    real_data = load(config.REAL_DATA_FILE);
    
    % 检查数据集结构 - 自动识别字段名
    field_names = fieldnames(real_data);
    fprintf('数据集包含字段: %s\n', strjoin(field_names, ', '));
    
    if isfield(real_data, 'eog_dataset')
        eeg_contaminated = real_data.eog_dataset;
    elseif isfield(real_data, 'eeg_data')
        eeg_contaminated = real_data.eeg_data;
    elseif isfield(real_data, 'data')
        eeg_contaminated = real_data.data;
    elseif isfield(real_data, 'Test_Contaminated')
        eeg_contaminated = real_data.Test_Contaminated;
    elseif ~isempty(field_names)
        % 自动使用第一个字段
        fprintf('使用第一个字段: %s\n', field_names{1});
        eeg_contaminated = real_data.(field_names{1});
    else
        error('无法在真实数据集中找到EEG数据字段');
    end
    
    fprintf('数据集形状: [%s]\n', num2str(size(eeg_contaminated)));
    
    %% 加载模型预测结果
    if ~exist(config.MODEL_PREDICTION_FILE, 'file')
        warning('模型预测结果文件不存在: %s', config.MODEL_PREDICTION_FILE);
        fprintf('将使用零填充作为预测结果\n');
        
        % 获取数据维度
        data_size = size(eeg_contaminated);
        
        % 处理不同的数据格式
        if length(data_size) == 3
            % 三维数据: [num_samples, num_channels, signal_length]
            [num_samples, num_channels, signal_length] = size(eeg_contaminated);
        elseif length(data_size) == 2
            % 二维数据: [num_samples, signal_length] - 单通道或需要转置
            if data_size(2) > data_size(1)
                % 如果列数远大于行数,可能是 [signal_length, num_samples]，需要转置
                if data_size(2) / data_size(1) > 10
                    eeg_contaminated = eeg_contaminated';
                    data_size = size(eeg_contaminated);
                end
            end
            [num_samples, signal_length] = size(eeg_contaminated);
            num_channels = 1;  % 单通道
            fprintf('检测到单通道数据\n');
        else
            error('不支持的数据维度: %d', length(data_size));
        end
        
        % 检查索引有效性
        if sample_idx > num_samples
            error('样本索引 %d 超出范围 (最大 %d)', sample_idx, num_samples);
        end
        if num_channels > 1 && channel_idx > num_channels
            error('通道索引 %d 超出范围 (最大 %d)', channel_idx, num_channels);
        end
        
        % 提取样本
        if num_channels > 1
            contaminated = squeeze(eeg_contaminated(sample_idx, channel_idx, :));
        else
            contaminated = eeg_contaminated(sample_idx, :);
        end
        
        clean_pred = zeros(size(contaminated));
        artifact_pred = zeros(size(contaminated));
    else
        fprintf('正在加载模型预测结果: %s\n', config.MODEL_PREDICTION_FILE);
        pred_data = load(config.MODEL_PREDICTION_FILE);
        
        % 检查预测数据结构
        pred_field_names = fieldnames(pred_data);
        fprintf('预测结果包含字段: %s\n', strjoin(pred_field_names, ', '));
        
        if isfield(pred_data, 'cleaned_eeg')
            clean_all = pred_data.cleaned_eeg;
            artifact_all = pred_data.extracted_eog;
        elseif isfield(pred_data, 'clean_predictions')
            clean_all = pred_data.clean_predictions;
            artifact_all = pred_data.artifact_predictions;
        elseif isfield(pred_data, 'predictions')
            predictions = pred_data.predictions;
            if isstruct(predictions)
                clean_all = predictions.clean;
                artifact_all = predictions.artifact;
            else
                error('预测数据格式不支持');
            end
        else
            error('无法在预测结果中找到预测数据字段');
        end
        
        fprintf('预测数据形状: [%s]\n', num2str(size(clean_all)));
        
        % 获取数据维度
        data_size = size(eeg_contaminated);
        
        % 处理不同的数据格式
        if length(data_size) == 3
            [num_samples, num_channels, signal_length] = size(eeg_contaminated);
        elseif length(data_size) == 2
            if data_size(2) > data_size(1)
                if data_size(2) / data_size(1) > 10
                    eeg_contaminated = eeg_contaminated';
                    data_size = size(eeg_contaminated);
                end
            end
            [num_samples, signal_length] = size(eeg_contaminated);
            num_channels = 1;
        else
            error('不支持的数据维度');
        end
        
        % 检查索引有效性
        if sample_idx > num_samples
            error('样本索引 %d 超出范围 (最大 %d)', sample_idx, num_samples);
        end
        if num_channels > 1 && channel_idx > num_channels
            error('通道索引 %d 超出范围 (最大 %d)', channel_idx, num_channels);
        end
        
        % 提取样本
        if num_channels > 1
            contaminated = squeeze(eeg_contaminated(sample_idx, channel_idx, :));
            clean_pred = squeeze(clean_all(sample_idx, channel_idx, :));
            artifact_pred = squeeze(artifact_all(sample_idx, channel_idx, :));
        else
            contaminated = eeg_contaminated(sample_idx, :);
            clean_pred = clean_all(sample_idx, :);
            artifact_pred = artifact_all(sample_idx, :);
        end
    end
    
    %% 构建输出数据结构
    data.contaminated = contaminated(:)';  % 确保是行向量
    data.clean_pred = clean_pred(:)';
    data.artifact_pred = artifact_pred(:)';
    data.fs = config.DATA.fs;
    data.sample_idx = sample_idx;
    data.channel_idx = channel_idx;
    data.signal_length = length(contaminated);
    data.time = (0:data.signal_length-1) / data.fs;  % 时间轴(秒)
    
    % 基本统计信息
    data.stats.contaminated_mean = mean(data.contaminated);
    data.stats.contaminated_std = std(data.contaminated);
    data.stats.contaminated_max = max(abs(data.contaminated));
    
    fprintf('\n数据加载完成!\n');
    fprintf('信号长度: %d 个采样点 (%.2f 秒)\n', data.signal_length, data.signal_length/data.fs);
    fprintf('采样率: %d Hz\n', data.fs);
    fprintf('受污染信号统计:\n');
    fprintf('  均值: %.4f μV\n', data.stats.contaminated_mean);
    fprintf('  标准差: %.4f μV\n', data.stats.contaminated_std);
    fprintf('  最大幅值: %.4f μV\n', data.stats.contaminated_max);
    fprintf('==================================================\n\n');
end
