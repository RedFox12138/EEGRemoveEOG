function [allData, visualizationSamples] = generate_all_levels_data(...
    snr_blink_levels, snr_eog_levels, samplesPerLevelPerType, dataLength, fs)
% ========================================================================
% 生成所有SNR等级的数据，确保每个等级、每种类型的样本数量相同
% ========================================================================
% 参数:
%   snr_blink_levels: 眨眼信噪比数组
%   snr_eog_levels: 眼动信噪比数组
%   samplesPerLevelPerType: 每个等级每种类型的样本数
%   dataLength: 数据长度(秒)
%   fs: 采样率
%
% 返回:
%   allData: 结构体，包含所有数据
%   visualizationSamples: 用于可视化的样本索引
% ========================================================================

    num_levels = length(snr_blink_levels);
    num_types = 4;  % 四种类型
    samplesPerLevel = samplesPerLevelPerType * num_types;
    totalSamples = samplesPerLevel * num_levels;
    numSamples = round(dataLength * fs);
    
    % 初始化存储数组
    allPureEEG = zeros(totalSamples, numSamples);
    allContaminatedEEG = zeros(totalSamples, numSamples);
    allEOGArtifact = zeros(totalSamples, numSamples);
    allBlinkArtifact = zeros(totalSamples, numSamples);
    allLevelIndices = zeros(totalSamples, 1);  % 记录每个样本属于哪个等级
    allTypeIndices = zeros(totalSamples, 1);   % 记录每个样本属于哪种类型
    
    visualizationSamples = struct();
    currentIdx = 1;
    
    % 为每个等级生成数据
    for level = 1:num_levels
        target_snr_blink = snr_blink_levels(level);
        target_snr_eog = snr_eog_levels(level);
        
        fprintf('正在生成等级 %d/%d: 眨眼SNR=%.0fdB, 眼动SNR=%.0fdB\n', ...
                level, num_levels, target_snr_blink, target_snr_eog);
        
        % 调用生成函数（为当前等级生成固定数量的样本）
        [pureEEG, contaminatedEEG, eogArtifact, blinkArtifact, typeIndices] = ...
            generateSimulatedEEG_MultiSNR(dataLength, target_snr_eog, target_snr_blink, samplesPerLevel);
        
        % 存储数据
        endIdx = currentIdx + samplesPerLevel - 1;
        allPureEEG(currentIdx:endIdx, :) = pureEEG;
        allContaminatedEEG(currentIdx:endIdx, :) = contaminatedEEG;
        allEOGArtifact(currentIdx:endIdx, :) = eogArtifact;
        allBlinkArtifact(currentIdx:endIdx, :) = blinkArtifact;
        allLevelIndices(currentIdx:endIdx) = level;
        allTypeIndices(currentIdx:endIdx) = typeIndices;
        
        % 保存每个等级每种类型的第一个样本索引用于可视化
        for type = 1:num_types
            type_indices_in_level = find(typeIndices == type, 1);
            if ~isempty(type_indices_in_level)
                global_idx = currentIdx + type_indices_in_level - 1;
                visualizationSamples.(sprintf('level%d_type%d', level, type)) = global_idx;
            end
        end
        
        currentIdx = endIdx + 1;
    end
    
    % 将所有数据打包到结构体
    allData = struct();
    allData.pureEEG = allPureEEG;
    allData.contaminatedEEG = allContaminatedEEG;
    allData.eogArtifact = allEOGArtifact;
    allData.blinkArtifact = allBlinkArtifact;
    allData.levelIndices = allLevelIndices;
    allData.typeIndices = allTypeIndices;
    allData.totalSamples = totalSamples;
    allData.samplesPerLevel = samplesPerLevel;
    allData.samplesPerLevelPerType = samplesPerLevelPerType;
    
    fprintf('\n数据生成完成！\n');
    fprintf('总样本数: %d\n', totalSamples);
    fprintf('每个等级样本数: %d\n', samplesPerLevel);
    fprintf('每个等级每种类型样本数: %d\n', samplesPerLevelPerType);
    
    % 验证数据分布
    fprintf('\n验证数据分布:\n');
    for level = 1:num_levels
        level_count = sum(allLevelIndices == level);
        fprintf('  等级 %d: %d 样本\n', level, level_count);
        for type = 1:num_types
            type_count_in_level = sum(allLevelIndices == level & allTypeIndices == type);
            fprintf('    类型 %d: %d 样本\n', type, type_count_in_level);
        end
    end
    
end
