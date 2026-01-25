function diagnose_metrics_difference()
% DIAGNOSE_METRICS_DIFFERENCE - 诊断MATLAB和Python指标计算差异
%
% 该脚本加载同一个.mat预测文件和测试集，使用与Python相同的方式计算指标
% 帮助诊断compute_all_metrics.m和test_unsupervised.py结果不一致的原因
%
% 使用方法:
%   1. 修改下面的文件路径，指向要诊断的预测文件和测试集
%   2. 运行此脚本
%   3. 对比输出的指标与Python脚本的输出

    fprintf('================================================================================\n');
    fprintf('指标计算差异诊断工具\n');
    fprintf('================================================================================\n');
    
    % ========== 配置文件路径 ==========
    % 请根据实际情况修改这些路径
    
    % 获取数据集配置
    config = getDatasetConfig('fully_simulated');
    
    % 选择一个SNR级别进行测试
    test_snr = 0;  % 例如: 0dB
    
    % 预测文件路径（从results目录）
    % 格式: <方法名>_<数据集名>_predictions_SNR<snr>dB.mat
    prediction_file = fullfile(pwd, 'results', ...
        sprintf('DAT-Net-Unsupervised-v2_fully_simulated_predictions_SNR%ddB.mat', test_snr));
    
    % 测试集路径
    snr_idx = find(config.testSnrLevels == test_snr);
    if isempty(snr_idx)
        error('SNR级别 %d dB 不在配置的测试SNR列表中: %s', test_snr, mat2str(config.testSnrLevels));
    end
    test_pure_path = config.testSnrPaths(snr_idx).pure;
    
    % ==================================
    
    fprintf('\n配置信息:\n');
    fprintf('  数据集: %s (%s)\n', config.name, config.datasetName);
    fprintf('  采样率: %d Hz\n', config.fs);
    fprintf('  测试SNR: %d dB\n', test_snr);
    fprintf('  预测文件: %s\n', prediction_file);
    fprintf('  测试集: %s\n', test_pure_path);
    
    % 1. 加载测试集（真实值）
    fprintf('\n[步骤1] 加载测试集真实值...\n');
    if ~exist(test_pure_path, 'file')
        error('测试集文件不存在: %s', test_pure_path);
    end
    
    data_struct = load(test_pure_path);
    fprintf('  测试集文件中的变量: %s\n', strjoin(fieldnames(data_struct), ', '));
    
    true_signals = data_struct.(config.pureKey);
    fprintf('  使用键名: "%s"\n', config.pureKey);
    fprintf('  数据维度: %s\n', mat2str(size(true_signals)));
    fprintf('  数据范围: [%.6f, %.6f]\n', min(true_signals(:)), max(true_signals(:)));
    fprintf('  数据均值: %.6f, 标准差: %.6f\n', mean(true_signals(:)), std(true_signals(:)));
    
    % 2. 加载预测结果
    fprintf('\n[步骤2] 加载预测结果...\n');
    if ~exist(prediction_file, 'file')
        error('预测文件不存在: %s\n请确保已运行过测试脚本并生成了预测结果', prediction_file);
    end
    
    pred_data = load(prediction_file);
    fprintf('  预测文件中的变量: %s\n', strjoin(fieldnames(pred_data), ', '));
    
    % 查找预测数据（按优先级）
    if isfield(pred_data, 'denoised_data')
        predictions = pred_data.denoised_data;
        used_key = 'denoised_data';
    elseif isfield(pred_data, 'predictions')
        predictions = pred_data.predictions;
        used_key = 'predictions';
    elseif isfield(pred_data, 'clean_data')
        predictions = pred_data.clean_data;
        used_key = 'clean_data';
    elseif isfield(pred_data, 'data')
        predictions = pred_data.data;
        used_key = 'data';
    else
        error('预测文件中未找到标准的预测数据变量 (denoised_data, predictions, clean_data, data)');
    end
    
    fprintf('  使用键名: "%s"\n', used_key);
    fprintf('  数据维度: %s\n', mat2str(size(predictions)));
    fprintf('  数据范围: [%.6f, %.6f]\n', min(predictions(:)), max(predictions(:)));
    fprintf('  数据均值: %.6f, 标准差: %.6f\n', mean(predictions(:)), std(predictions(:)));
    
    % 3. 检查维度是否匹配
    fprintf('\n[步骤3] 检查维度匹配...\n');
    if ~isequal(size(predictions), size(true_signals))
        warning('维度不匹配！预测: %s vs 真实: %s', ...
                mat2str(size(predictions)), mat2str(size(true_signals)));
        fprintf('  尝试转置...\n');
        if isequal(size(predictions'), size(true_signals))
            predictions = predictions';
            fprintf('  转置后维度匹配: %s\n', mat2str(size(predictions)));
        else
            error('转置后仍不匹配，无法继续');
        end
    else
        fprintf('  ✓ 维度匹配: %s\n', mat2str(size(predictions)));
    end
    
    % 4. 计算指标（逐样本，与Python一致）
    fprintf('\n[步骤4] 计算评价指标（逐样本，然后平均）...\n');
    n_samples = size(predictions, 1);
    fprintf('  样本数量: %d\n', n_samples);
    
    rrmse_list = zeros(n_samples, 1);
    cc_list = zeros(n_samples, 1);
    rrmse_psd_list = zeros(n_samples, 1);
    mi_list = zeros(n_samples, 1);
    
    fprintf('  正在计算...');
    for i = 1:n_samples
        true_sig = true_signals(i, :);
        pred_sig = predictions(i, :);
        
        % RRMSE
        mse = mean((true_sig - pred_sig).^2);
        true_power = mean(true_sig.^2);
        rrmse_list(i) = sqrt(mse / true_power);
        
        % CC
        corr_matrix = corrcoef(true_sig, pred_sig);
        cc_list(i) = corr_matrix(1, 2);
        
        % RRMSE_PSD
        nperseg = min(256, length(true_sig));
        noverlap = floor(nperseg / 2);
        [psd_true, ~] = pwelch(true_sig, nperseg, noverlap, [], config.fs);
        [psd_pred, ~] = pwelch(pred_sig, nperseg, noverlap, [], config.fs);
        mse_psd = mean((psd_true - psd_pred).^2);
        true_psd_power = mean(psd_true.^2);
        rrmse_psd_list(i) = sqrt(mse_psd / true_psd_power);
        
        % MI
        bins = 50;
        hist_2d = hist3([true_sig(:), pred_sig(:)], [bins, bins]);
        pxy = hist_2d / sum(hist_2d(:));
        px = sum(pxy, 2);
        py = sum(pxy, 1);
        px_py = px * py;
        nonzero_mask = (pxy > 0) & (px_py > 0);
        if sum(nonzero_mask(:)) > 0
            mi_list(i) = sum(pxy(nonzero_mask) .* log(pxy(nonzero_mask) ./ px_py(nonzero_mask)));
        else
            mi_list(i) = 0;
        end
        
        if mod(i, 50) == 0
            fprintf('.');
        end
    end
    fprintf(' 完成\n');
    
    % 5. 显示结果
    fprintf('\n================================================================================\n');
    fprintf('计算结果 (逐样本平均):\n');
    fprintf('================================================================================\n');
    fprintf('  RRMSE:     %.6f (std: %.6f)\n', mean(rrmse_list), std(rrmse_list));
    fprintf('  CC:        %.6f (std: %.6f)\n', mean(cc_list), std(cc_list));
    fprintf('  RRMSE_PSD: %.6f (std: %.6f)\n', mean(rrmse_psd_list), std(rrmse_psd_list));
    fprintf('  MI:        %.6f (std: %.6f)\n', mean(mi_list), std(mi_list));
    fprintf('================================================================================\n');
    
    fprintf('\n对比说明:\n');
    fprintf('  请将上述MATLAB计算结果与Python脚本的输出进行对比。\n');
    fprintf('  如果仍有差异，可能的原因:\n');
    fprintf('    1. Python使用了不同的测试集文件或SNR级别\n');
    fprintf('    2. 预测文件保存时使用了不同的数据处理方式\n');
    fprintf('    3. 采样率配置不一致 (当前: %d Hz)\n', config.fs);
    fprintf('    4. 数值精度差异（通常可忽略）\n');
    fprintf('\n建议:\n');
    fprintf('  1. 检查Python脚本test_unsupervised.py中的DATASET_NAME配置\n');
    fprintf('  2. 确认Python和MATLAB使用的是同一个测试集SNR级别\n');
    fprintf('  3. 检查模型预测文件的生成时间，确保是最新的\n');
    fprintf('================================================================================\n');
end
