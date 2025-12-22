%% 真实数据集去噪结果可视化 - 使用示例
% 
% 本脚本展示如何使用 visualize_real_data_results 函数
% 可视化真实数据集上不同方法的去噪结果
%
% 作者: GitHub Copilot
% 日期: 2025-12-17

%% 示例1: 基本使用 - 显示所有方法的网格对比图
fprintf('================================================================================\n');
fprintf('示例1: 网格对比图（所有方法）\n');
fprintf('================================================================================\n\n');

visualize_real_data_results();

%% 示例2: 自定义样本和方法对比
fprintf('\n================================================================================\n');
fprintf('示例2: 自定义对比\n');
fprintf('================================================================================\n\n');

% 如果你想要对比特定的几个方法，可以使用以下方式：

% 1. 先运行基本函数加载数据
% 2. 然后手动调用绘图函数

% 注意：需要先运行示例1加载数据到工作空间

%% 示例3: 对比你的方法和其他基线方法
fprintf('\n================================================================================\n');
fprintf('示例3: 重点对比（推荐用于论文）\n');
fprintf('================================================================================\n\n');

% 创建一个简化版本，只对比几个关键方法
% 这个需要你根据实际情况调整方法名

% 示例代码：
% compare_methods({'DAT_Net', 'ACMD', 'EEGIFNet', 'MicroWaveNet'}, sample_idx=1);

%% 示例4: 批量生成多个样本的对比图
fprintf('\n================================================================================\n');
fprintf('示例4: 批量处理\n');
fprintf('================================================================================\n\n');

% 如果你想为多个样本生成对比图，可以使用循环
% 注意：这会生成很多图窗口

% for sample_idx = 1:5
%     fprintf('正在处理样本 %d...\n', sample_idx);
%     visualize_real_data_results_single(sample_idx);
%     % 可以在这里保存图片
%     saveas(gcf, sprintf('sample_%d_comparison.png', sample_idx));
% end

%% 提示信息

fprintf('\n================================================================================\n');
fprintf('                             使用提示                                           \n');
fprintf('================================================================================\n\n');

fprintf('1. 基本使用:\n');
fprintf('   >> visualize_real_data_results()\n\n');

fprintf('2. 保存当前图形:\n');
fprintf('   >> saveas(gcf, ''real_data_comparison.png'')\n');
fprintf('   >> print(gcf, ''real_data_comparison.eps'', ''-depsc'', ''-r300'')  %% 矢量图\n\n');

fprintf('3. 自定义样本:\n');
fprintf('   修改 visualize_real_data_results.m 中的 sample_idx 变量\n\n');

fprintf('4. 选择特定方法对比:\n');
fprintf('   可以编辑脚本，筛选想要对比的方法\n\n');

fprintf('5. 调整图形大小:\n');
fprintf('   修改 figure(''Position'', [...]) 参数\n\n');

fprintf('================================================================================\n\n');

%% 快速测试
fprintf('运行快速测试，检查所有文件是否可访问...\n\n');

% 检查真实数据集
real_data_file = 'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat';
if exist(real_data_file, 'file')
    fprintf('✓ 真实数据集文件存在\n');
else
    fprintf('✗ 真实数据集文件不存在: %s\n', real_data_file);
end

% 检查结果目录
results_dir = 'D:\Pycharm_Projects\EOG Remove\复现的方法\训练完的模型和数据\真实数据集\结果';
if exist(results_dir, 'dir')
    mat_files = dir(fullfile(results_dir, '*_real_data_predictions.mat'));
    fprintf('✓ 结果目录存在，发现 %d 个方法结果\n', length(mat_files));
    
    fprintf('\n可用方法列表:\n');
    for i = 1:length(mat_files)
        method_name = strrep(mat_files(i).name, '_real_data_predictions.mat', '');
        fprintf('  %d. %s\n', i, method_name);
    end
else
    fprintf('✗ 结果目录不存在: %s\n', results_dir);
end

fprintf('\n一切准备就绪！运行 visualize_real_data_results() 开始可视化。\n');

%% 示例5: 计算真实数据集的评价指标
fprintf('\n================================================================================\n');
fprintf('示例5: 计算真实数据集的频域评价指标\n');
fprintf('================================================================================\n\n');

% 计算所有方法的频域指标（ΔER_δ 和 MAE_PSD）
compute_real_data_frequency_metrics();
