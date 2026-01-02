function visualize_all_levels(output_dir, snr_blink_levels, snr_eog_levels, all_level_samples, fs, dataLength)
% ========================================================================
% 为每个SNR等级生成可视化图例
% ========================================================================
% 参数:
%   output_dir: 输出目录
%   snr_blink_levels: 眨眼信噪比数组
%   snr_eog_levels: 眼动信噪比数组
%   all_level_samples: 每个等级的示例样本索引
%   fs: 采样率
%   dataLength: 数据长度(秒)
% ========================================================================

    num_levels = length(snr_blink_levels);
    t = 0:1/fs:dataLength-1/fs;
    
    % 为每个等级创建一个图
    for level = 1:num_levels
        target_snr_blink = snr_blink_levels(level);
        target_snr_eog = snr_eog_levels(level);
        
        fprintf('生成等级 %d 的可视化图例...\n', level);
        
        % 加载该等级的数据
        level_dir = fullfile(output_dir, sprintf('Level_%d_Blink%.0fdB_EOG%.0fdB', ...
                             level, target_snr_blink, target_snr_eog));
        
        load(fullfile(level_dir, 'Pure_Data.mat'), 'pureEEG');
        load(fullfile(level_dir, 'Contaminated_Data.mat'), 'contaminatedEEG');
        load(fullfile(level_dir, 'EOG_Artifact.mat'), 'eogArtifact');
        load(fullfile(level_dir, 'Blink_Artifact.mat'), 'blinkArtifact');
        
        sample_indices = all_level_samples{level};
        
        % 创建图形
        fig = figure('Position', [50, 50, 1800, 1200]);
        
        type_names = {'无干扰', '仅眼动', '仅眨眼', '眼动+眨眼'};
        
        for type = 1:4
            idx = sample_indices.(sprintf('type%d', type));
            
            % 子图1: 纯净EEG
            subplot(4, 4, (type-1)*4 + 1);
            plot(t, pureEEG(idx, :), 'b', 'LineWidth', 1);
            if type == 1
                title('纯净EEG', 'FontSize', 10, 'FontWeight', 'bold');
            end
            ylabel(type_names{type}, 'FontSize', 9, 'FontWeight', 'bold');
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图2: 眼动伪影
            subplot(4, 4, (type-1)*4 + 2);
            plot(t, eogArtifact(idx, :), 'r', 'LineWidth', 1);
            if type == 1
                title(sprintf('眼动伪影 (%.0fdB)', target_snr_eog), 'FontSize', 10, 'FontWeight', 'bold');
            end
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图3: 眨眼伪影
            subplot(4, 4, (type-1)*4 + 3);
            plot(t, blinkArtifact(idx, :), 'g', 'LineWidth', 1);
            if type == 1
                title(sprintf('眨眼伪影 (%.0fdB)', target_snr_blink), 'FontSize', 10, 'FontWeight', 'bold');
            end
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
            
            % 子图4: 污染EEG
            subplot(4, 4, (type-1)*4 + 4);
            plot(t, contaminatedEEG(idx, :), 'k', 'LineWidth', 1);
            if type == 1
                title('污染EEG', 'FontSize', 10, 'FontWeight', 'bold');
            end
            xlim([0, min(10, dataLength)]);
            grid on;
            set(gca, 'FontSize', 8);
        end
        
        % 添加总标题
        sgtitle(sprintf('等级 %d: 眨眼SNR=%.0fdB, 眼动SNR=%.0fdB', ...
                level, target_snr_blink, target_snr_eog), ...
                'FontSize', 14, 'FontWeight', 'bold');
        
        % 保存图形
        fig_path = fullfile(level_dir, sprintf('Level_%d_Visualization.png', level));
        saveas(fig, fig_path);
        fprintf('  已保存图例: %s\n', fig_path);
        
        close(fig);
    end
    
    fprintf('\n所有可视化图例生成完成！\n\n');
    
    % 创建一个汇总对比图，显示所有等级的污染EEG（仅眼动+眨眼类型）
    fprintf('生成汇总对比图...\n');
    fig_summary = figure('Position', [50, 50, 1800, 1200]);
    
    for level = 1:num_levels
        target_snr_blink = snr_blink_levels(level);
        target_snr_eog = snr_eog_levels(level);
        
        % 加载该等级的数据
        level_dir = fullfile(output_dir, sprintf('Level_%d_Blink%.0fdB_EOG%.0fdB', ...
                             level, target_snr_blink, target_snr_eog));
        
        load(fullfile(level_dir, 'Contaminated_Data.mat'), 'contaminatedEEG');
        sample_indices = all_level_samples{level};
        
        % 获取眼动+眨眼类型的样本
        idx = sample_indices.type4;
        
        subplot(num_levels, 1, level);
        plot(t, contaminatedEEG(idx, :), 'k', 'LineWidth', 1);
        title(sprintf('等级%d (眨眼%.0fdB, 眼动%.0fdB)', level, target_snr_blink, target_snr_eog), ...
              'FontSize', 10, 'FontWeight', 'bold');
        ylabel('幅值', 'FontSize', 9);
        xlim([0, min(10, dataLength)]);
        grid on;
        set(gca, 'FontSize', 8);
        
        if level == num_levels
            xlabel('时间 (秒)', 'FontSize', 9);
        end
    end
    
    sgtitle('所有等级污染EEG对比 (眼动+眨眼类型)', 'FontSize', 14, 'FontWeight', 'bold');
    
    % 保存汇总图
    summary_path = fullfile(output_dir, 'All_Levels_Summary.png');
    saveas(fig_summary, summary_path);
    fprintf('已保存汇总对比图: %s\n', summary_path);
    
    close(fig_summary);
    
end
