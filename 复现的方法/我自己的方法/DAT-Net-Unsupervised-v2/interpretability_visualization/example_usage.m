% example_usage.m
% DAT-Net 可视化系统使用示例
%
% 本脚本展示了如何使用可视化系统的各种功能

%% 示例1: 列出所有可用任务
fprintf('\n========== 示例1: 列出所有可用任务 ==========\n');
main_visualization('list');

%% 示例2: 运行单个可视化任务
fprintf('\n========== 示例2: 运行单个可视化任务 ==========\n');

% 可视化样本1通道1的伪影概率计算
% main_visualization('artifact_probability', 1, 1);

% 可视化样本5通道10的掩蔽策略
% main_visualization('masking_strategy', 5, 10);

% 可视化样本3通道8的去噪效果
% main_visualization('denoising_results', 3, 8);

fprintf('提示: 取消注释上面的代码来运行单个任务\n');

%% 示例3: 运行所有可视化任务
fprintf('\n========== 示例3: 运行所有可视化任务 ==========\n');

% 对样本1通道1运行所有任务
% main_visualization('all', 1, 1);

% 对样本5通道10运行所有任务
% main_visualization('all', 5, 10);

fprintf('提示: 取消注释上面的代码来运行所有任务\n');

%% 示例4: 使用便捷函数
fprintf('\n========== 示例4: 使用便捷函数 ==========\n');

% 快速可视化样本3的所有任务(使用默认通道1)
% quick_vis('all', 3);

% 可视化样本5通道10的所有任务
% visualize_sample(5, 10);

fprintf('提示: 取消注释上面的代码来使用便捷函数\n');

%% 示例5: 批量处理多个样本
fprintf('\n========== 示例5: 批量处理多个样本 ==========\n');

% 为前3个样本生成去噪效果可视化
% for sample_idx = 1:3
%     fprintf('\n处理样本 %d...\n', sample_idx);
%     vis_denoising_results(sample_idx, 1);
% end

fprintf('提示: 取消注释上面的代码来批量处理\n');

%% 示例6: 对比不同通道
fprintf('\n========== 示例6: 对比不同通道 ==========\n');

% 对样本1的多个通道进行可视化
% channels_to_compare = [1, 5, 10, 15, 20];
% for ch = channels_to_compare
%     fprintf('\n处理通道 %d...\n', ch);
%     vis_artifact_probability(1, ch);
% end

fprintf('提示: 取消注释上面的代码来对比不同通道\n');

%% 示例7: 自定义配置
fprintf('\n========== 示例7: 自定义配置 ==========\n');

% 加载配置
config = config_visualization();

% 修改配置
config.VIS.dpi = 150;  % 降低分辨率以加快速度
config.EXPORT.save_pdf = true;  % 同时保存PDF

% 使用自定义配置运行可视化
% vis_artifact_probability(1, 1, config);

fprintf('提示: 取消注释上面的代码来使用自定义配置\n');

%% 示例8: 分析特定样本
fprintf('\n========== 示例8: 分析特定样本(推荐工作流) ==========\n');

% 推荐的工作流程:
% 1. 先查看去噪效果,选择感兴趣的样本
% vis_denoising_results(1, 1);

% 2. 对感兴趣的样本查看伪影概率计算
% vis_artifact_probability(1, 1);

% 3. 查看掩蔽策略如何工作
% vis_masking_strategy(1, 1);

% 4. 如果需要,对所有任务进行完整分析
% main_visualization('all', 1, 1);

fprintf('提示: 按照推荐工作流程进行分析\n');

%% 示例9: 保存多个样本的结果用于论文
fprintf('\n========== 示例9: 为论文准备图像 ==========\n');

% 选择几个代表性样本
% representative_samples = [1, 5, 10];
% 
% for s = representative_samples
%     fprintf('\n生成样本 %d 的所有可视化...\n', s);
%     
%     % 使用高分辨率配置
%     config = config_visualization();
%     config.VIS.dpi = 300;  % 论文质量
%     config.EXPORT.save_pdf = true;  % 保存矢量格式
%     
%     % 运行所有可视化
%     vis_artifact_probability(s, 1, config);
%     vis_masking_strategy(s, 1, config);
%     vis_denoising_results(s, 1, config);
%     
%     fprintf('样本 %d 处理完成\n', s);
% end

fprintf('提示: 取消注释上面的代码来生成论文图像\n');

%% 示例10: 交互式探索
fprintf('\n========== 示例10: 交互式探索 ==========\n');

fprintf('交互式探索步骤:\n');
fprintf('1. 运行 test_visualization 确保系统正常\n');
fprintf('2. 使用 main_visualization(''list'') 查看所有任务\n');
fprintf('3. 尝试不同的样本和通道: main_visualization(''all'', sample_idx, channel_idx)\n');
fprintf('4. 查看生成的图像: outputs_matlab/figures/\n');
fprintf('5. 根据需要调整 config_visualization.m 中的参数\n');

%% 完成
fprintf('\n========================================\n');
fprintf('示例脚本运行完成!\n');
fprintf('\n要运行某个示例,请取消注释相应的代码\n');
fprintf('或者直接在命令窗口中输入相应的命令\n');
fprintf('========================================\n\n');

fprintf('常用命令快速参考:\n');
fprintf('  main_visualization(''list'')                    %% 列出所有任务\n');
fprintf('  main_visualization(''all'', 1, 1)               %% 运行所有任务\n');
fprintf('  vis_artifact_probability(1, 1)                %% 伪影概率可视化\n');
fprintf('  vis_masking_strategy(1, 1)                    %% 掩蔽策略可视化\n');
fprintf('  vis_denoising_results(1, 1)                   %% 去噪效果可视化\n');
fprintf('\n');
