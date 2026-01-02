function visualize_merged_dataset(visualizationSamples, snr_blink_levels, snr_eog_levels, output_dir, fs, dataLength)
% ========================================================================
% 为合并的数据集生成可视化图例
% ========================================================================
% 参数:
%   visualizationSamples: 可视化样本索引结构体
%   snr_blink_levels: 眨眼信噪比数组
%   snr_eog_levels: 眼动信噪比数组
%   output_dir: 输出目录
%   fs: 采样率
%   dataLength: 数据长度(秒)
% ========================================================================

    num_levels = length(snr_blink_levels);
    t = 0:1/fs:dataLength-1/fs;
    
    % 加载合并的数据（从详细信息目录加载）
    detail_dir = fullfile(output_dir, 'detailed_info', 'train');
    fprintf('加载训练集详细数据用于可视化...\n');
    
    % 加载主数据
    train_pure = load(fullfile(output_dir, 'Train_Pure.mat'));
    train_contam = load(fullfile(output_dir, 'Train_Contaminated.mat'));
    pureEEG = train_pure.pureEEG;
    contaminatedEEG = train_contam.contaminatedEEG;
    
    % 加载详细信息
    load(fullfile(detail_dir, 'EOG_Artifact.mat'), 'eogArtifact');
    load(fullfile(detail_dir, 'Blink_Artifact.mat'), 'blinkArtifact');
    load(fullfile(detail_dir, 'Level_Indices.mat'), 'levelIndices');
    load(fullfile(detail_dir, 'Type_Indices.mat'), 'typeIndices');
    
    type_names = {'无干扰', '仅眼动', '仅眨眼', '眼动+眨眼'};
    
    % ====================================================================
    % 图1: 所有等级的完整展示 (7个等级 × 4种类型 = 28个子图)
    % ====================================================================
    fprintf('生成完整展示图...\n');
    fig1 = figure('Position', [50, 50, 2000, 1400]);
    
    for level = 1:num_levels
        for type = 1:4
            % 找到当前(等级,类型)的第一个样本
            sample_idx = find(levelIndices == level & typeIndices == type, 1);
            
            if isempty(sample_idx)
                continue;
            end
            
            % 计算子图位置 (7行4列)
            subplot(num_levels, 4, (level-1)*4 + type);
            plot(t, contaminatedEEG(sample_idx, :), 'k', 'LineWidth', 1);
            
            % 设置标题
            if level == 1
                title(type_names{type}, 'FontSize', 9, 'FontWeight', 'bold');
            end
            
            % 设置y轴标签
            if type == 1
                ylabel(sprintf('Level %d\n(B:%ddB E:%ddB)', ...
                       level, snr_blink_levels(level), snr_eog_levels(level)), ...
                       'FontSize', 8, 'FontWeight', 'bold');
            end
            
            % 设置x轴标签
            if level == num_levels
                xlabel('Time (s)', 'FontSize', 8);
            end
            
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 7);
        end
    end
    
    sgtitle('Fully Simulated Dataset - All Levels and Types', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 保存图1
    fig1_path = fullfile(output_dir, 'All_Levels_All_Types.png');
    saveas(fig1, fig1_path);
    fprintf('Saved: %s\n', fig1_path);
    close(fig1);
    
    % ====================================================================
    % 图2-8: 为每个等级生成详细展示
    % ====================================================================
    fprintf('Generating detailed visualization for all levels...\n');
    
    for level = 1:num_levels
        % 计算噪声/信号比例
        noise_ratio_blink = 10^(-snr_blink_levels(level) / 20);
        noise_ratio_eog = 10^(-snr_eog_levels(level) / 20);
        
        fig2 = figure('Position', [50, 50, 1800, 1200]);
        
        for type = 1:4
            % 找到当前类型的第一个样本
            sample_idx = find(levelIndices == level & typeIndices == type, 1);
            
            if isempty(sample_idx)
                continue;
            end
            
            % 计算RMS用于验证
            rms_pure = sqrt(mean(pureEEG(sample_idx, :).^2));
            rms_eog = sqrt(mean(eogArtifact(sample_idx, :).^2));
            rms_blink = sqrt(mean(blinkArtifact(sample_idx, :).^2));
            rms_contam = sqrt(mean(contaminatedEEG(sample_idx, :).^2));
            
            % 子图1: 纯净EEG
            subplot(4, 4, (type-1)*4 + 1);
            plot(t, pureEEG(sample_idx, :), 'b', 'LineWidth', 1);
            if type == 1
                title('Pure EEG', 'FontSize', 10, 'FontWeight', 'bold');
            end
            ylabel(sprintf('%s\nRMS=%.3f', type_names{type}, rms_pure), ...
                   'FontSize', 9, 'FontWeight', 'bold');
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图2: 眼动伪影
            subplot(4, 4, (type-1)*4 + 2);
            plot(t, eogArtifact(sample_idx, :), 'r', 'LineWidth', 1);
            if type == 1
                title(sprintf('EOG Artifact (%ddB)', snr_eog_levels(level)), ...
                      'FontSize', 10, 'FontWeight', 'bold');
            end
            % 添加验证信息
            if type == 2 || type == 4
                ylabel(sprintf('RMS=%.3f\nTheory:%.3f', rms_eog, noise_ratio_eog), 'FontSize', 8);
            else
                ylabel('None', 'FontSize', 8);
            end
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图3: 眨眼伪影
            subplot(4, 4, (type-1)*4 + 3);
            plot(t, blinkArtifact(sample_idx, :), 'g', 'LineWidth', 1);
            if type == 1
                title(sprintf('Blink Artifact (%ddB)', snr_blink_levels(level)), ...
                      'FontSize', 10, 'FontWeight', 'bold');
            end
            % 添加验证信息
            if type == 3 || type == 4
                ylabel(sprintf('RMS=%.3f\nTheory:%.3f', rms_blink, noise_ratio_blink), 'FontSize', 8);
            else
                ylabel('None', 'FontSize', 8);
            end
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图4: 污染EEG
            subplot(4, 4, (type-1)*4 + 4);
            plot(t, contaminatedEEG(sample_idx, :), 'k', 'LineWidth', 1);
            if type == 1
                title('Contaminated EEG', 'FontSize', 10, 'FontWeight', 'bold');
            end
            ylabel(sprintf('RMS=%.3f', rms_contam), 'FontSize', 8);
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
        end
        
        sgtitle(sprintf('Level %d Detail (Blink %ddB [N/S=%.3f], EOG %ddB [N/S=%.3f])', ...
                level, snr_blink_levels(level), noise_ratio_blink, ...
                snr_eog_levels(level), noise_ratio_eog), ...
                'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图2
        fig2_path = fullfile(output_dir, sprintf('Level_%d_Detailed.png', level));
        saveas(fig2, fig2_path);
        fprintf('Saved: %s\n', fig2_path);
        close(fig2);
    end
    
    % ====================================================================
    % 图9: 所有等级的对比图 (只显示眼动+眨眼类型)
    % ====================================================================
    fprintf('Generating level comparison plot...\n');
    fig3 = figure('Position', [50, 50, 1800, 1200]);
    
    for level = 1:num_levels
        % 找到眼动+眨眼类型的样本
        sample_idx = find(levelIndices == level & typeIndices == 4, 1);
        
        if isempty(sample_idx)
            continue;
        end
        
        subplot(num_levels, 1, level);
        plot(t, contaminatedEEG(sample_idx, :), 'k', 'LineWidth', 1);
        title(sprintf('Level %d (Blink %ddB, EOG %ddB)', ...
              level, snr_blink_levels(level), snr_eog_levels(level)), ...
              'FontSize', 10, 'FontWeight', 'bold');
        ylabel('Amplitude', 'FontSize', 9);
        xlim([0, min(10, dataLength)]);
        grid on;
        set(gca, 'FontSize', 8);
        
        if level == num_levels
            xlabel('Time (s)', 'FontSize', 9);
        end
    end
    
    sgtitle('All Levels Comparison (EOG+Blink Type)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 保存图3
    fig3_path = fullfile(output_dir, 'All_Levels_Comparison.png');
    saveas(fig3, fig3_path);
    fprintf('Saved: %s\n', fig3_path);
    close(fig3);
    
    fprintf('\nAll visualization completed! Generated %d detailed images.\n', num_levels + 2);
    
end
