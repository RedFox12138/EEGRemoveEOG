"""
可解释性可视化配置文件
包含所有可视化任务的参数设置
"""
import os

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATA_DIR = r'D:\Pycharm_Projects\EOG Remove\生成半模拟数据\已经生成好的数据'
MODEL_DIR = PROJECT_ROOT

# 输出目录
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
DATA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'data')
REPORT_DIR = os.path.join(OUTPUT_DIR, 'report')

# 创建输出目录
for directory in [OUTPUT_DIR, FIGURE_DIR, DATA_OUTPUT_DIR, REPORT_DIR]:
    os.makedirs(directory, exist_ok=True)

# ==================== 模型配置 ====================
MODEL_CONFIG = {
    'model_path': os.path.join(MODEL_DIR, 'DAT-Net-Unsupervised-v2_best.pth'),
    'model_path_fallback': os.path.join(MODEL_DIR, 'DAT-Net-Unsupervised-v2_final.pth'),
    'in_channels': 1,
    'base_channels': 32,
}

# ==================== 数据配置 ====================
DATA_CONFIG = {
    'test_input_path': os.path.join(DATA_DIR, 'Test_Contaminated.mat'),
    'test_output_path': os.path.join(DATA_DIR, 'Test_Pure.mat'),
    'fs': 200,  # 采样率
}

# ==================== 无监督损失函数配置 ====================
LOSS_CONFIG = {
    'mask_base': 0.1857,
    'boost_scale': 0.2341,
    'lambda_rec': 0.8121,
    'lambda_con': 1.2613,
    'lambda_teacher': 0.4036,
    'lambda_n2v': 0.2440,
    'lambda_band': 0.0553,
    'lambda_low': 0.0577,
    'lambda_decor': 0.3153,
    'lambda_content': 0.6763,
    'gamma_art_weight': 1.0,
    'artifact_win_size': 64,
    'mask_neighborhood': 5,
    'teacher_cutoff': 8.0,
    'lowpass_cutoff': 4.0,
    'teacher_threshold': 0.7,
}

# ==================== 可视化配置 ====================
VIS_CONFIG = {
    # 通用设置
    'dpi': 150,
    'figsize_single': (12, 8),
    'figsize_multi': (16, 12),
    'font_size': 10,
    'title_size': 14,
    'cmap': 'RdYlBu_r',
    
    # 样本选择 - 可以根据效果调整
    'default_sample_idx': 0,  # 默认可视化的样本索引，可修改为效果好的样本编号
    'recommended_samples': [0, 5, 10, 15, 20],  # 推荐的效果较好的样本
    'num_samples_to_show': 5,  # 在多样本可视化中展示的数量
    
    # 时间轴设置
    'time_window': None,  # None表示显示全部，或者 (start, end) 以秒为单位
    
    # 颜色方案
    'colors': {
        'original': '#2E86AB',      # 原始信号 - 蓝色
        'clean': '#06A77D',         # Clean信号 - 绿色
        'artifact': '#D62246',      # Artifact信号 - 红色
        'target': '#F77F00',        # 目标信号 - 橙色
        'masked': '#A4036F',        # 掩蔽信号 - 紫色
        'branch_a': '#118AB2',      # 分支A - 深蓝
        'branch_b': '#EF476F',      # 分支B - 粉红
    }
}

# ==================== 任务特定配置 ====================

# 任务2: 伪影概率计算
ARTIFACT_PROB_CONFIG = {
    'show_intermediate_steps': True,  # 是否显示中间步骤
    'win_size': 64,
    'lowpass_cutoff': 4.0,
}

# 任务3: 掩蔽策略
MASKING_CONFIG = {
    'compare_random': True,  # 是否对比随机掩蔽
    'num_comparisons': 3,    # 对比次数
}

# 任务4: 双分支架构
DUAL_BRANCH_CONFIG = {
    'show_flow_diagram': True,  # 是否显示流程图
}

# 任务5: 编码器特征
ENCODER_FEATURES_CONFIG = {
    'layers_to_visualize': ['down1', 'down2', 'down3', 'bottleneck'],
    'num_channels_to_show': 8,  # 每层显示的通道数
}

# 任务6: 注意力机制
ATTENTION_CONFIG = {
    'visualize_se': True,       # 可视化SE注意力
    'visualize_tcn': True,      # 可视化TCN
}

# 任务7: 频谱分解
SPECTRUM_CONFIG = {
    'freq_range': (0, 50),      # 显示的频率范围 (Hz)
    'use_log_scale': False,     # 是否使用对数刻度
}

# 任务8: Teacher信号
TEACHER_CONFIG = {
    'teacher_cutoff': 8.0,
    'teacher_threshold': 0.7,
}

# 任务9: 损失函数
LOSS_VIS_CONFIG = {
    'show_pie_chart': True,     # 显示饼图
    'show_bar_chart': True,     # 显示柱状图
}

# 任务10-11: 一致性和解耦
CONSISTENCY_CONFIG = {
    'use_heatmap': True,
}

# 任务12: 去噪效果
DENOISING_CONFIG = {
    'sample_selection': 'auto',  # 'auto' 或具体的索引列表
    'num_good': 2,               # 好样本数量
    'num_medium': 2,             # 中等样本数量
    'num_bad': 2,                # 差样本数量
}

# 任务13: 伪影分布
DISTRIBUTION_CONFIG = {
    'use_interactive': True,     # 使用交互式图表
}

# ==================== 设备配置 ====================
DEVICE_CONFIG = {
    'use_cuda': True,
    'device_id': 0,
}

# ==================== 导出格式 ====================
EXPORT_CONFIG = {
    'save_png': True,
    'save_pdf': False,
    'save_svg': False,
    'save_data': True,  # 保存中间数据
}

# ==================== 报告配置 ====================
REPORT_CONFIG = {
    'title': 'DAT-Net 无监督学习可解释性分析报告',
    'author': '毕业设计项目',
    'date': 'auto',  # 'auto' 使用当前日期
    'theme': 'light',  # 'light' 或 'dark'
}

# ==================== 调试配置 ====================
DEBUG_CONFIG = {
    'verbose': True,
    'save_intermediate': True,
}
