function split_and_save_datasets(allData, output_dir, snr_blink_levels, snr_eog_levels)
% ========================================================================
% 划分数据集为训练集、验证集、测试集和微调集
% ========================================================================
% 参数:
%   allData: 包含所有数据的结构体
%   output_dir: 输出目录
%   snr_blink_levels: 眨眼信噪比数组
%   snr_eog_levels: 眼动信噪比数组
%
% 划分比例:
%   训练集: 80%
%   验证集: 10%
%   测试集: 10%
%   微调集1: 10%
%   微调集2: 20%
%
% 策略: 从每个(等级,类型)组合中均匀采样，确保各数据集的分布一致
% ========================================================================

    num_levels = length(snr_blink_levels);
    num_types = 4;
    samplesPerLevelPerType = allData.samplesPerLevelPerType;
    
    % 计算每个数据集中每个(等级,类型)组合应有的样本数
    train_samples_per_combo = round(samplesPerLevelPerType * 0.8);
    val_samples_per_combo = round(samplesPerLevelPerType * 0.1);
    test_samples_per_combo = samplesPerLevelPerType - train_samples_per_combo - val_samples_per_combo;
    finetune1_samples_per_combo = round(samplesPerLevelPerType * 0.1);
    finetune2_samples_per_combo = round(samplesPerLevelPerType * 0.2);
    
    fprintf('每个(等级,类型)组合的样本划分:\n');
    fprintf('  训练集: %d\n', train_samples_per_combo);
    fprintf('  验证集: %d\n', val_samples_per_combo);
    fprintf('  测试集: %d\n', test_samples_per_combo);
    fprintf('  微调集1(10%%): %d\n', finetune1_samples_per_combo);
    fprintf('  微调集2(20%%): %d\n\n', finetune2_samples_per_combo);
    
    % 初始化索引列表
    train_indices = [];
    val_indices = [];
    test_indices = [];
    finetune1_indices = [];
    finetune2_indices = [];
    
    % 为每个(等级,类型)组合分配样本
    for level = 1:num_levels
        for type = 1:num_types
            % 找到当前(等级,类型)组合的所有样本索引
            combo_indices = find(allData.levelIndices == level & allData.typeIndices == type);
            
            % 随机打乱
            combo_indices = combo_indices(randperm(length(combo_indices)));
            
            % 分配到各个数据集
            idx = 1;
            train_indices = [train_indices; combo_indices(idx:idx+train_samples_per_combo-1)];
            idx = idx + train_samples_per_combo;
            
            val_indices = [val_indices; combo_indices(idx:idx+val_samples_per_combo-1)];
            idx = idx + val_samples_per_combo;
            
            test_indices = [test_indices; combo_indices(idx:idx+test_samples_per_combo-1)];
            
            % 微调集从训练集中随机采样
            train_combo_indices = combo_indices(1:train_samples_per_combo);
            shuffled = train_combo_indices(randperm(length(train_combo_indices)));
            finetune1_indices = [finetune1_indices; shuffled(1:finetune1_samples_per_combo)];
            finetune2_indices = [finetune2_indices; shuffled(1:finetune2_samples_per_combo)];
        end
    end
    
    % 打乱各数据集内部的顺序
    train_indices = train_indices(randperm(length(train_indices)));
    val_indices = val_indices(randperm(length(val_indices)));
    test_indices = test_indices(randperm(length(test_indices)));
    finetune1_indices = finetune1_indices(randperm(length(finetune1_indices)));
    finetune2_indices = finetune2_indices(randperm(length(finetune2_indices)));
    
    fprintf('数据集大小:\n');
    fprintf('  训练集: %d 样本\n', length(train_indices));
    fprintf('  验证集: %d 样本\n', length(val_indices));
    fprintf('  测试集: %d 样本\n', length(test_indices));
    fprintf('  微调集1: %d 样本\n', length(finetune1_indices));
    fprintf('  微调集2: %d 样本\n\n', length(finetune2_indices));
    
    % 保存主数据集（训练、验证）- 主目录下的简化格式
    fprintf('保存主数据集...\n');
    
    % 训练集
    pureEEG = allData.pureEEG(train_indices, :);
    contaminatedEEG = allData.contaminatedEEG(train_indices, :);
    save(fullfile(output_dir, 'Train_Pure.mat'), 'pureEEG', '-v7.3');
    save(fullfile(output_dir, 'Train_Contaminated.mat'), 'contaminatedEEG', '-v7.3');
    fprintf('  已保存: Train_Pure.mat 和 Train_Contaminated.mat\n');
    
    % 验证集
    pureEEG = allData.pureEEG(val_indices, :);
    contaminatedEEG = allData.contaminatedEEG(val_indices, :);
    save(fullfile(output_dir, 'Val_Pure.mat'), 'pureEEG', '-v7.3');
    save(fullfile(output_dir, 'Val_Contaminated.mat'), 'contaminatedEEG', '-v7.3');
    fprintf('  已保存: Val_Pure.mat 和 Val_Contaminated.mat\n\n');
    
    % 测试集 - 按等级分开保存
    fprintf('保存测试集（按等级分开）...\n');
    save_test_by_levels(allData, test_indices, output_dir, snr_blink_levels, snr_eog_levels);
    
    % 保存详细信息到子目录
    fprintf('保存详细信息到 detailed_info/ 目录...\n');
    detail_dir = fullfile(output_dir, 'detailed_info');
    if ~exist(detail_dir, 'dir')
        mkdir(detail_dir);
    end
    
    % 保存训练集详细信息
    save_dataset_details(allData, train_indices, fullfile(detail_dir, 'train'), ...
                         snr_blink_levels, snr_eog_levels, 'train');
    
    % 保存验证集详细信息
    save_dataset_details(allData, val_indices, fullfile(detail_dir, 'val'), ...
                         snr_blink_levels, snr_eog_levels, 'val');
    
    % 保存测试集详细信息
    save_dataset_details(allData, test_indices, fullfile(detail_dir, 'test'), ...
                         snr_blink_levels, snr_eog_levels, 'test');
    
    % 保存微调集1 (10%) - 简化格式
    fprintf('保存微调集1 (10%%)...\n');
    save_finetune_dataset(allData, finetune1_indices, output_dir, '10percent');
    
    % 保存微调集2 (20%) - 简化格式
    fprintf('保存微调集2 (20%%)...\n');
    save_finetune_dataset(allData, finetune2_indices, output_dir, '20percent');
    
    fprintf('\n所有数据集已保存！\n');
end


function save_dataset_details(allData, indices, save_dir, snr_blink_levels, snr_eog_levels, dataset_name)
% 保存数据集的详细信息（伪影、标签等）
    
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % 提取数据
    eogArtifact = allData.eogArtifact(indices, :);
    blinkArtifact = allData.blinkArtifact(indices, :);
    levelIndices = allData.levelIndices(indices);
    typeIndices = allData.typeIndices(indices);
    
    % 保存详细信息
    save(fullfile(save_dir, 'EOG_Artifact.mat'), 'eogArtifact', '-v7.3');
    save(fullfile(save_dir, 'Blink_Artifact.mat'), 'blinkArtifact', '-v7.3');
    save(fullfile(save_dir, 'Level_Indices.mat'), 'levelIndices');
    save(fullfile(save_dir, 'Type_Indices.mat'), 'typeIndices');
    
    % 保存数据集信息
    dataset_info = struct();
    dataset_info.num_samples = length(indices);
    dataset_info.snr_blink_levels = snr_blink_levels;
    dataset_info.snr_eog_levels = snr_eog_levels;
    dataset_info.type_names = {'无干扰', '仅眼动', '仅眨眼', '眼动+眨眼'};
    
    % 统计每个等级和类型的样本数
    num_levels = length(snr_blink_levels);
    for level = 1:num_levels
        level_count = sum(levelIndices == level);
        dataset_info.(sprintf('level_%d_count', level)) = level_count;
        for type = 1:4
            type_count = sum(levelIndices == level & typeIndices == type);
            dataset_info.(sprintf('level_%d_type_%d_count', level, type)) = type_count;
        end
    end
    
    save(fullfile(save_dir, 'dataset_info.mat'), 'dataset_info');
    
    % 保存为CSV格式（方便查看）
    fid = fopen(fullfile(save_dir, 'dataset_info.txt'), 'w');
    fprintf(fid, '数据集信息\n');
    fprintf(fid, '=================================\n');
    fprintf(fid, '总样本数: %d\n\n', dataset_info.num_samples);
    
    fprintf(fid, '各等级样本分布:\n');
    for level = 1:num_levels
        fprintf(fid, '等级 %d (眨眼%.0fdB, 眼动%.0fdB): %d 样本\n', ...
                level, snr_blink_levels(level), snr_eog_levels(level), ...
                dataset_info.(sprintf('level_%d_count', level)));
        for type = 1:4
            fprintf(fid, '  类型 %d (%s): %d 样本\n', ...
                    type, dataset_info.type_names{type}, ...
                    dataset_info.(sprintf('level_%d_type_%d_count', level, type)));
        end
    end
    fclose(fid);
    
    fprintf('  已保存%s集详细信息到: %s\n', dataset_name, save_dir);
end


function save_finetune_dataset(allData, indices, output_dir, percent_name)
% 保存微调数据集的辅助函数 (简化格式，只保存Pure和Contaminated)
    
    % 提取数据
    pureEEG = allData.pureEEG(indices, :);
    contaminatedEEG = allData.contaminatedEEG(indices, :);
    
    % 保存为简化的.mat文件（参考半模拟数据格式）
    save(fullfile(output_dir, sprintf('Finetune_%s_Pure.mat', percent_name)), 'pureEEG', '-v7.3');
    save(fullfile(output_dir, sprintf('Finetune_%s_Contaminated.mat', percent_name)), 'contaminatedEEG', '-v7.3');
    
    fprintf('  已保存: Finetune_%s_Pure.mat 和 Finetune_%s_Contaminated.mat (%d 样本)\n', ...
            percent_name, percent_name, length(indices));
end


function save_test_by_levels(allData, test_indices, output_dir, snr_blink_levels, snr_eog_levels)
% 按等级分开保存测试集
    
    num_levels = length(snr_blink_levels);
    levelIndices_test = allData.levelIndices(test_indices);
    
    % 为每个等级保存测试集
    for level = 1:num_levels
        % 找到当前等级的测试样本
        level_mask = (levelIndices_test == level);
        level_test_indices = test_indices(level_mask);
        
        if isempty(level_test_indices)
            fprintf('  警告: 等级 %d 没有测试样本\n', level);
            continue;
        end
        
        % 提取数据
        pureEEG = allData.pureEEG(level_test_indices, :);
        contaminatedEEG = allData.contaminatedEEG(level_test_indices, :);
        
        % 构造文件名（参考半模拟数据集格式）
        % 眨眼SNR作为主要标识
        snr_blink = snr_blink_levels(level);
        if snr_blink >= 0
            snr_str = sprintf('SNR%ddB', snr_blink);
        else
            snr_str = sprintf('SNR%ddB', snr_blink);  % 负号会自动包含
        end
        
        % 保存文件
        save(fullfile(output_dir, sprintf('Test_Pure_%s.mat', snr_str)), 'pureEEG', '-v7.3');
        save(fullfile(output_dir, sprintf('Test_Contaminated_%s.mat', snr_str)), 'contaminatedEEG', '-v7.3');
        
        fprintf('  已保存: Test_Pure_%s.mat 和 Test_Contaminated_%s.mat (%d 样本, 眨眼%.0fdB/眼动%.0fdB)\n', ...
                snr_str, snr_str, length(level_test_indices), ...
                snr_blink_levels(level), snr_eog_levels(level));
    end
    
    fprintf('\n');
end
