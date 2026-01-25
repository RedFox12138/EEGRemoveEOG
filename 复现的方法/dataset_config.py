"""
数据集配置文件
用于统一管理不同数据集的配置,方便切换和扩展

使用方法:
    from dataset_config import get_dataset_config
    config = get_dataset_config('semi_simulated')  # 或 'fully_simulated'
    print(config['sampling_rate'])
    print(config['train_contaminated_path'])
"""

import os

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 数据集配置字典
DATASET_CONFIGS = {
    # 半模拟数据集 (多SNR版本)
    'semi_simulated': {
        'name': '半模拟数据集',
        'sampling_rate': 200.0,  # Hz
        'window_duration': 6,  # seconds
        'window_size': 1200,  # 200 * 6
        'data_dir': os.path.join(PROJECT_ROOT, '生成半模拟数据', '已经生成好的数据', 'multi_snr'),
        'train_contaminated': 'Train_Contaminated.mat',
        'train_pure': 'Train_Pure.mat',
        'val_contaminated': 'Val_Contaminated.mat',
        'val_pure': 'Val_Pure.mat',
        'test_snr_levels': [-8,-6, -4, -2, 0, 2,4],  # 多个SNR级别的测试集
        'test_contaminated_template': 'Test_Contaminated_SNR{}dB.mat',  # 模板，{}会被替换为SNR值
        'test_pure_template': 'Test_Pure_SNR{}dB.mat',
        'data_key': 'data',  # .mat文件中的key
        'description': '基于真实EEG数据生成的半模拟数据集,采样率200Hz,包含5种SNR级别的测试集'
    },
    
    # 全模拟数据集 (新数据集 - 7个SNR等级)
    'fully_simulated': {
        'name': '全模拟数据集',
        'sampling_rate': 250.0,  # Hz
        'window_duration': 6,  # seconds
        'window_size': 1500,  # 250 * 6
        'data_dir': os.path.join(PROJECT_ROOT, '生成全模拟数据', '已经生成好的数据', 'Multi_SNR_Merged'),
        'train_contaminated': 'Train_Contaminated.mat',
        'train_pure': 'Train_Pure.mat',
        'val_contaminated': 'Val_Contaminated.mat',
        'val_pure': 'Val_Pure.mat',
        'test_snr_levels': [4,0, -6, -10, -14, -18, -20, -22],  # 7个SNR级别
        'test_contaminated_template': 'Test_Contaminated_SNR{}dB.mat',
        'test_pure_template': 'Test_Pure_SNR{}dB.mat',
        'data_key': 'contaminatedEEG',  # .mat文件中的key（污染信号）
        'pure_key': 'pureEEG',  # 纯净信号的key
        'description': '完全模拟生成的数据集,采样率250Hz,7个SNR等级,格式[n_samples, 1500]'
    }
}

# 默认使用的数据集 (可以在这里切换)
DEFAULT_DATASET = 'semi_simulated'  # 改为 'semi_simulated' 可切换回半模拟数据


def get_dataset_config(dataset_name=None):
    """
    获取数据集配置
    
    Args:
        dataset_name: 数据集名称 ('semi_simulated' 或 'fully_simulated')
                     如果为None,则使用DEFAULT_DATASET
    
    Returns:
        dict: 数据集配置字典
    """
    if dataset_name is None:
        dataset_name = DEFAULT_DATASET
    
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name].copy()
    
    # 添加完整路径
    config['train_contaminated_path'] = os.path.join(config['data_dir'], config['train_contaminated'])
    config['train_pure_path'] = os.path.join(config['data_dir'], config['train_pure'])
    config['val_contaminated_path'] = os.path.join(config['data_dir'], config['val_contaminated'])
    config['val_pure_path'] = os.path.join(config['data_dir'], config['val_pure'])
    
    # 处理测试集路径 - 支持多SNR级别
    if 'test_snr_levels' in config:
        # 多SNR测试集
        config['test_snr_paths'] = {}
        for snr in config['test_snr_levels']:
            config['test_snr_paths'][snr] = {
                'contaminated': os.path.join(config['data_dir'], 
                                            config['test_contaminated_template'].format(snr)),
                'pure': os.path.join(config['data_dir'], 
                                    config['test_pure_template'].format(snr))
            }
    else:
        # 单一测试集（向后兼容）
        config['test_contaminated_path'] = os.path.join(config['data_dir'], config['test_contaminated'])
        config['test_pure_path'] = os.path.join(config['data_dir'], config['test_pure'])
    
    return config


def list_available_datasets():
    """列出所有可用的数据集"""
    print("可用数据集:")
    print("=" * 80)
    for name, config in DATASET_CONFIGS.items():
        is_default = " [当前默认]" if name == DEFAULT_DATASET else ""
        print(f"\n数据集名称: {name}{is_default}")
        print(f"  描述: {config['description']}")
        print(f"  采样率: {config['sampling_rate']} Hz")
        print(f"  窗口大小: {config['window_size']} 样本 ({config['window_duration']}秒)")
        print(f"  数据目录: {config['data_dir']}")


def print_dataset_info(dataset_name=None):
    """打印数据集详细信息"""
    config = get_dataset_config(dataset_name)
    print(f"\n数据集信息: {config['name']}")
    print("=" * 80)
    print(f"描述: {config['description']}")
    print(f"采样率: {config['sampling_rate']} Hz")
    print(f"窗口大小: {config['window_size']} 样本 ({config['window_duration']}秒)")
    print(f"\n数据文件:")
    print(f"  训练集 (污染): {config['train_contaminated_path']}")
    print(f"  训练集 (纯净): {config['train_pure_path']}")
    print(f"  验证集 (污染): {config['val_contaminated_path']}")
    print(f"  验证集 (纯净): {config['val_pure_path']}")
    print(f"  测试集 (污染): {config['test_contaminated_path']}")
    print(f"  测试集 (纯净): {config['test_pure_path']}")
    print(f"\n数据键名: {config['data_key']}")


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, Rectangle

    # 1. 设置中文字体 (Windows推荐SimHei，Mac推荐Arial Unicode MS)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 2. 创建画布 (16:9)
    fig, ax = plt.subplots(figsize=(16, 9))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')  # 极浅的灰底，突出卡片

    # --- 3. 绘制顶部标题栏 ---
    # 红色装饰条
    ax.add_patch(Rectangle((0, 8.2), 0.6, 0.8, color='#c00000'))
    # 主标题
    plt.text(1.0, 8.6, "三、2026年一季度具体工作措施", fontsize=26, weight='bold', color='#222', va='center')
    # 右上角页码
    ax.add_patch(FancyBboxPatch((15.0, 8.3), 0.8, 0.5, boxstyle="round,pad=0.1", fc='#c00000', ec='none'))
    plt.text(15.4, 8.55, "13", color='white', fontsize=12, weight='bold', ha='center', va='center')

    # 分割线
    ax.add_line(plt.Line2D([0, 16], [8.0, 8.0], color='#c00000', linewidth=2.5))


    # --- 4. 绘制导航 Tab (模拟PPT的导航栏) ---
    def draw_nav_tab(x, width, text, active=False):
        color = '#c00000' if active else '#e0e0e0'
        text_color = 'white' if active else '#666'
        # 绘制矩形Tab
        ax.add_patch(Rectangle((x, 7.2), width, 0.8, color=color))
        plt.text(x + width / 2, 7.6, text, color=text_color, fontsize=14, weight='bold', ha='center', va='center')


    # 绘制三个Tab
    draw_nav_tab(1.5, 4.0, "个人存款", False)
    draw_nav_tab(6.0, 4.0, "非息收入", False)
    draw_nav_tab(10.5, 4.0, "重点客群", True)  # 激活当前页


    # --- 5. 绘制 6 个内容卡片 ---
    # 定义绘制函数
    def draw_card(x, y, w, h, title, points):
        # 卡片背景
        shadow = FancyBboxPatch((x + 0.05, y - 0.05), w, h, boxstyle="round,pad=0.1", fc='#ddd', ec='none', zorder=1)
        ax.add_patch(shadow)
        card = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", fc='white', ec='#e6e6e6', linewidth=1, zorder=2)
        ax.add_patch(card)

        # 标题区域 (左侧红色竖线 + 文字)
        ax.add_patch(
            FancyBboxPatch((x, y + h - 0.7), 0.15, 0.5, boxstyle="round,pad=0", fc='#c00000', ec='none', zorder=3))
        plt.text(x + 0.4, y + h - 0.45, title, fontsize=15, weight='bold', color='#c00000', va='center', zorder=4)

        # 内容列表
        start_text_y = y + h - 1.1
        for i, point in enumerate(points):
            # 项目符号
            plt.text(x + 0.3, start_text_y - i * 0.65, "●", fontsize=10, color='#c00000', va='top', zorder=4)
            # 正文
            plt.text(x + 0.6, start_text_y - i * 0.65, point, fontsize=11.5, color='#444', va='top', ha='left',
                     wrap=True, linespacing=1.5, zorder=4)


    # 准备数据 (从图片中提取并精简)
    content_data = [
        {
            "title": "手机银行",
            "points": [
                "立足厅堂阵地：确保新开卡客户做到“三个百分百”，压降签约漏损率。",
                "全员营销：分享“微煌合伙人”、“冬季财富享好礼”等重点活动，提升占比。"
            ]
        },
        {
            "title": "私人银行",
            "points": [
                "日常管控：每日跟进流失、预晋级及潜力名单，做好客户新增。",
                "产品配置：落实亲见，提升理财覆盖率和信用卡配卡率。",
                "三方联动：加强与中银证券/理财联动，做好托管市值营销。"
            ]
        },
        {
            "title": "社保卡",
            "points": [
                "落实月度管控：按日跟进达成情况。",
                "批量获客：联动新签约代发单位，积极营销社保卡作为代发账户。",
                "厅堂营销：针对到店客户，首推社保卡。"
            ]
        },
        {
            "title": "养老金",
            "points": [
                "目标分解：加强低产能网点的日常管控。",
                "活动宣传：2026年1月起，新开户并缴存50元以上领立减金（保底28元），提升客户质量。"
            ]
        },
        {
            "title": "快捷支付",
            "points": [
                "绑卡管控：提高新开卡绑卡交易力度，做好厅堂流量及外拓存量客户营销。",
                "渠道引导：重点引导抖音、京东平台，发挥“一户获客、多重绑卡”带动效应。"
            ]
        },
        {
            "title": "消费贷款",
            "points": [
                "核心策略：精准外拓、批量获客。",
                "发展路径：做大储备是消费贷快速发展的有效途径和渠道。"
            ]
        }
    ]

    # 布局参数
    col_width = 6.8
    row_height = 2.0
    gap_x = 0.5
    gap_y = 0.2
    start_x_left = 1.0
    start_x_right = 8.2
    start_y = 4.6  # 第一行起始Y坐标

    # 绘制左列 (手机银行, 私人银行, 社保卡) - 注意Y轴是从下往上的，所以要倒着算或者调整坐标
    # 这里我们手动放置以适应不同高度
    # 手机银行 (Top Left)
    draw_card(start_x_left, 4.8, col_width, 2.0, content_data[0]["title"], content_data[0]["points"])

    # 私人银行 (Middle Left) - 内容多，稍微高一点
    draw_card(start_x_left, 2.4, col_width, 2.2, content_data[1]["title"], content_data[1]["points"])

    # 社保卡 (Bottom Left)
    draw_card(start_x_left, 0.2, col_width, 2.0, content_data[2]["title"], content_data[2]["points"])

    # 绘制右列 (养老金, 快捷支付, 消费贷款)
    # 养老金 (Top Right)
    draw_card(start_x_right, 4.8, col_width, 2.0, content_data[3]["title"], content_data[3]["points"])

    # 快捷支付 (Middle Right)
    draw_card(start_x_right, 2.4, col_width, 2.2, content_data[4]["title"], content_data[4]["points"])

    # 消费贷款 (Bottom Right)
    draw_card(start_x_right, 0.8, col_width, 1.4, content_data[5]["title"], content_data[5]["points"])

    plt.tight_layout()
    plt.show()