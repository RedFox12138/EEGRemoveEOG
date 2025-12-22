function main_visualization(task, sample_idx, channel_idx)
% MAIN_VISUALIZATION DAT-Net可解释性可视化主程序
%
% 用于毕业设计: 展示模型、无监督过程每一步的可解释性
% 通过绘图方式展示各个模块的直观作用
% 支持真实数据集,可指定样本和通道
%
% 输入:
%   task - 任务名称 (可选):
%       'all' - 运行所有可视化任务
%       'artifact_probability' - 伪影概率计算可视化
%       'masking_strategy' - 掩蔽策略可视化
%       'denoising_results' - 去噪效果可视化
%       'list' - 列出所有可用任务
%   sample_idx - 样本索引 (可选, 默认为1)
%   channel_idx - 通道索引 (可选, 默认为1)
%
% 用法:
%   main_visualization();  % 使用默认参数运行所有任务
%   main_visualization('list');  % 列出所有任务
%   main_visualization('all', 5, 10);  % 运行所有任务,样本5通道10
%   main_visualization('artifact_probability', 3, 8);  % 单个任务
%
% 示例:
%   % 运行所有可视化任务
%   main_visualization('all');
%
%   % 运行单个任务
%   main_visualization('artifact_probability');
%   main_visualization('masking_strategy', 5);
%   main_visualization('denoising_results', 10, 15);
%
% 作者: 毕业设计项目
% 日期: 2025

    %% 默认参数
    if nargin < 1 || isempty(task)
        task = 'all';
    end
    if nargin < 2 || isempty(sample_idx)
        sample_idx = 1;
    end
    if nargin < 3 || isempty(channel_idx)
        channel_idx = 1;
    end
    
    %% 打印欢迎信息
    print_banner();
    
    %% 加载配置
    config = config_visualization();
    
    %% 初始化环境
    fprintf('正在初始化可视化环境...\n');
    fprintf('输出目录: %s\n', config.OUTPUT_DIR);
    fprintf('图像目录: %s\n', config.FIGURE_DIR);
    fprintf('数据目录: %s\n', config.DATA_OUTPUT_DIR);
    fprintf('\n');
    
    %% 任务映射表
    tasks = get_task_mapping();
    
    %% 处理命令
    if strcmpi(task, 'list')
        list_tasks(tasks);
        return;
    end
    
    if strcmpi(task, 'all')
        fprintf('运行所有可视化任务\n');
        fprintf('样本索引: %d, 通道索引: %d\n', sample_idx, channel_idx);
        fprintf('========================================\n\n');
        
        run_all_tasks(sample_idx, channel_idx, config, tasks);
    else
        % 运行单个任务
        run_single_task(task, sample_idx, channel_idx, config, tasks);
    end
    
    fprintf('\n========================================\n');
    fprintf('所有任务完成!\n');
    fprintf('========================================\n');
end


%% ==================== 核心函数 ====================

function tasks = get_task_mapping()
% 获取任务映射表
    tasks = struct();
    
    % 任务1: 伪影概率计算可视化
    tasks.artifact_probability = struct(...
        'id', 1, ...
        'name', '伪影概率计算可视化', ...
        'function', @vis_artifact_probability, ...
        'description', '展示如何计算每个时间点的伪影概率');
    
    % 任务2: 掩蔽策略可视化
    tasks.masking_strategy = struct(...
        'id', 2, ...
        'name', '掩蔽策略可视化', ...
        'function', @vis_masking_strategy, ...
        'description', '对比随机掩蔽vs伪影感知掩蔽');
    
    % 任务3: 去噪效果可视化
    tasks.denoising_results = struct(...
        'id', 3, ...
        'name', '去噪效果可视化', ...
        'function', @vis_denoising_results, ...
        'description', '展示典型样本的处理结果');
end


function list_tasks(tasks)
% 列出所有可用任务
    fprintf('\n========================================\n');
    fprintf('可用的可视化任务\n');
    fprintf('========================================\n\n');
    
    task_names = fieldnames(tasks);
    for i = 1:length(task_names)
        task_name = task_names{i};
        task_info = tasks.(task_name);
        fprintf('%d. %s\n', task_info.id, task_info.name);
        fprintf('   任务名: %s\n', task_name);
        fprintf('   描述: %s\n\n', task_info.description);
    end
    
    fprintf('========================================\n');
    fprintf('使用示例:\n');
    fprintf('  main_visualization(''artifact_probability'', 1, 1);\n');
    fprintf('  main_visualization(''all'', 5, 10);\n');
    fprintf('  main_visualization(''list'');\n');
    fprintf('========================================\n\n');
end


function run_single_task(task_name, sample_idx, channel_idx, config, tasks)
% 运行单个可视化任务
    
    % 检查任务是否存在
    if ~isfield(tasks, task_name)
        fprintf('❌ 错误: 未知任务 ''%s''\n', task_name);
        fprintf('使用 main_visualization(''list'') 查看所有可用任务\n');
        return;
    end
    
    task_info = tasks.(task_name);
    
    fprintf('\n========================================\n');
    fprintf('任务 %d: %s\n', task_info.id, task_info.name);
    fprintf('========================================\n');
    fprintf('描述: %s\n', task_info.description);
    fprintf('样本索引: %d\n', sample_idx);
    fprintf('通道索引: %d\n', channel_idx);
    fprintf('========================================\n');
    
    try
        % 执行任务
        tic;
        task_info.function(sample_idx, channel_idx, config);
        elapsed_time = toc;
        
        fprintf('\n✓ 任务完成: %s (耗时: %.2f 秒)\n', task_info.name, elapsed_time);
    catch ME
        fprintf('\n❌ 任务执行失败: %s\n', ME.message);
        fprintf('错误位置: %s (第 %d 行)\n', ME.stack(1).name, ME.stack(1).line);
        
        % 显示详细错误信息
        fprintf('\n详细错误信息:\n');
        disp(ME);
    end
end


function run_all_tasks(sample_idx, channel_idx, config, tasks)
% 运行所有可视化任务
    
    task_names = fieldnames(tasks);
    num_tasks = length(task_names);
    
    success_count = 0;
    failed_count = 0;
    
    total_start_time = tic;
    
    for i = 1:num_tasks
        task_name = task_names{i};
        task_info = tasks.(task_name);
        
        fprintf('\n========================================\n');
        fprintf('进度: %d/%d\n', i, num_tasks);
        fprintf('任务 %d: %s\n', task_info.id, task_info.name);
        fprintf('========================================\n');
        
        try
            % 执行任务
            tic;
            task_info.function(sample_idx, channel_idx, config);
            elapsed_time = toc;
            
            fprintf('\n✓ 任务完成 (耗时: %.2f 秒)\n', elapsed_time);
            success_count = success_count + 1;
        catch ME
            fprintf('\n❌ 任务失败: %s\n', ME.message);
            failed_count = failed_count + 1;
        end
    end
    
    total_elapsed_time = toc(total_start_time);
    
    %% 打印总结
    fprintf('\n========================================\n');
    fprintf('执行总结\n');
    fprintf('========================================\n');
    fprintf('✓ 成功: %d\n', success_count);
    fprintf('❌ 失败: %d\n', failed_count);
    fprintf('总计: %d\n', num_tasks);
    fprintf('总耗时: %.2f 秒\n', total_elapsed_time);
    fprintf('========================================\n');
end


function print_banner()
% 打印欢迎横幅
    fprintf('\n');
    fprintf('╔═══════════════════════════════════════════════════════════════╗\n');
    fprintf('║                                                               ║\n');
    fprintf('║        DAT-Net 无监督学习可解释性可视化系统                    ║\n');
    fprintf('║        Interpretability Visualization System                  ║\n');
    fprintf('║                                                               ║\n');
    fprintf('║        用于毕业设计: 展示模型每一步的可解释性                  ║\n');
    fprintf('║        使用真实数据集, 支持样本和通道选择                      ║\n');
    fprintf('║                                                               ║\n');
    fprintf('╚═══════════════════════════════════════════════════════════════╝\n');
    fprintf('\n');
end


%% ==================== 便捷函数 ====================

function visualize_sample(sample_idx, channel_idx)
% 便捷函数: 可视化指定样本的所有任务
%
% 用法:
%   visualize_sample(5, 10);  % 可视化样本5通道10的所有任务

    if nargin < 1
        sample_idx = 1;
    end
    if nargin < 2
        channel_idx = 1;
    end
    
    main_visualization('all', sample_idx, channel_idx);
end


function quick_vis(task, sample_idx)
% 便捷函数: 快速可视化(使用默认通道)
%
% 用法:
%   quick_vis('artifact_probability', 5);

    if nargin < 1
        task = 'all';
    end
    if nargin < 2
        sample_idx = 1;
    end
    
    main_visualization(task, sample_idx, 1);
end
