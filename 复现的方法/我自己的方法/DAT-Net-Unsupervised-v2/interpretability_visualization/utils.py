"""
可解释性可视化工具函数
包含数据加载、模型加载、绘图辅助等功能
"""
import os
import sys
import numpy as np
import scipy.io
import torch
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.font_manager as fm
from typing import Tuple, Optional, List, Dict, Any

# 设置中文字体 - 支持Windows和Linux
def setup_chinese_font_support():
    """配置matplotlib支持中文显示 - 解决PNG保存乱码问题
    
    使用强制方法确保中文正确显示：
    1. 使用英文标签 + Unicode编码
    2. 或使用图片嵌入方式
    """
    # 由于Windows系统matplotlib保存PNG时中文字体嵌入存在问题
    # 采用最稳妥的方案：使用系统默认字体配置
    
    # 方案1：尝试配置中文字体（可能不生效）
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS', 'sans-serif']
    matplotlib.rcParams['axes.unicode_minus'] = False
    matplotlib.rcParams['font.size'] = 10
    
    # 方案2：设置backend参数
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    matplotlib.rcParams['svg.fonttype'] = 'none'
    
    # 方案3：禁用字体缓存（强制重新加载）
    matplotlib.rcParams['font.family'] = 'sans-serif'
    
    return 'SimHei'

# 初始化中文字体支持
_selected_font = setup_chinese_font_support()

# 添加自定义字体属性对象，用于强制指定字体
def get_chinese_font_prop():
    """获取中文字体属性对象，用于text/title等函数的fontproperties参数"""
    font_paths = fm.findSystemFonts(fontpaths=None, fontext='ttf')
    
    for path in font_paths:
        if 'simhei' in path.lower():
            return fm.FontProperties(fname=path, size=10)
        elif 'msyh' in path.lower():
            return fm.FontProperties(fname=path, size=10)
    
    return None

_chinese_font_prop = get_chinese_font_prop()

# 添加路径以导入模型
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
datnet_dir = os.path.join(os.path.dirname(parent_dir), 'DAT-Net')
if os.path.isdir(datnet_dir):
    sys.path.insert(0, datnet_dir)
sys.path.insert(0, parent_dir)

from config import *


def setup_plotting_style():
    """设置绘图风格"""
    # 尝试使用seaborn风格，如果不可用则使用默认风格
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except:
        try:
            plt.style.use('seaborn-darkgrid')
        except:
            plt.style.use('default')
    
    plt.rcParams['figure.dpi'] = VIS_CONFIG['dpi']
    plt.rcParams['font.size'] = VIS_CONFIG['font_size']
    plt.rcParams['axes.titlesize'] = VIS_CONFIG['title_size']
    plt.rcParams['axes.labelsize'] = VIS_CONFIG['font_size']
    plt.rcParams['xtick.labelsize'] = VIS_CONFIG['font_size'] - 1
    plt.rcParams['ytick.labelsize'] = VIS_CONFIG['font_size'] - 1
    plt.rcParams['legend.fontsize'] = VIS_CONFIG['font_size'] - 1


def load_model(verbose=True):
    """
    加载训练好的DAT-Net模型
    
    Returns:
        model: 加载好的模型（包装后返回字典）
        device: 使用的设备
    """
    from model import DATNet
    
    # 设置设备
    if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{DEVICE_CONFIG['device_id']}")
    else:
        device = torch.device('cpu')
    
    if verbose:
        print(f'使用设备: {device}')
    
    # 创建模型
    base_model = DATNet(
        in_channels=MODEL_CONFIG['in_channels'],
        base_channels=MODEL_CONFIG['base_channels']
    ).to(device)
    
    # 加载权重
    model_path = MODEL_CONFIG['model_path']
    if not os.path.exists(model_path):
        model_path = MODEL_CONFIG['model_path_fallback']
        if verbose:
            print(f'主模型未找到，使用备用模型: {model_path}')
    
    if os.path.exists(model_path):
        base_model.load_state_dict(torch.load(model_path, map_location=device))
        if verbose:
            print(f'成功加载模型: {model_path}')
    else:
        if verbose:
            print('⚠️ 警告: 未找到训练好的模型，使用随机初始化权重')
    
    base_model.eval()
    
    # 包装模型使其返回字典格式
    class ModelWrapper:
        def __init__(self, base_model):
            self.base_model = base_model
            # 提供model属性用于访问原始模型（用于hooks等）
            self.model = base_model
            
        def __call__(self, x):
            clean, artifact = self.base_model(x)
            # 返回字典格式，兼容双分支
            return {
                'clean_A': clean,
                'artifact_A': artifact,
                'clean_B': clean,
                'artifact_B': artifact
            }
        
        def eval(self):
            self.base_model.eval()
            return self
        
        def to(self, device):
            self.base_model.to(device)
            return self
    
    wrapped_model = ModelWrapper(base_model)
    return wrapped_model, device


def load_test_data(verbose=True):
    """
    加载测试数据并创建DataLoader
    
    Returns:
        data_loader: PyTorch DataLoader对象
    """
    from torch.utils.data import TensorDataset, DataLoader
    
    test_input = scipy.io.loadmat(DATA_CONFIG['test_input_path'])['data']
    test_target = scipy.io.loadmat(DATA_CONFIG['test_output_path'])['data']
    
    if verbose:
        print(f'测试数据形状: {test_input.shape}')
        print(f'采样率: {DATA_CONFIG["fs"]} Hz')
    
    # 转换为tensor
    test_input_tensor = torch.from_numpy(test_input).float().unsqueeze(1)  # (N, 1, L)
    test_target_tensor = torch.from_numpy(test_target).float().unsqueeze(1)  # (N, 1, L)
    
    # 创建数据集和加载器
    dataset = TensorDataset(test_input_tensor, test_target_tensor)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    return data_loader


def get_sample_by_index(data_loader, idx=0):
    """
    从DataLoader中获取指定索引的样本
    
    Args:
        data_loader: DataLoader对象
        idx: 样本索引
        
    Returns:
        dict: 包含'contaminated'和'target'的字典
    """
    for i, (contaminated, target) in enumerate(data_loader):
        if i == idx:
            return {
                'contaminated': contaminated,  # (1, 1, L)
                'target': target  # (1, 1, L)
            }
    return None


def setup_chinese_font():
    """设置中文字体以避免乱码"""
    # 使用已经配置好的字体
    pass


def get_sample(test_input, test_target, idx=0):
    """
    获取单个样本
    
    Args:
        test_input: 测试输入数据
        test_target: 测试目标数据
        idx: 样本索引
        
    Returns:
        sample_input: (L,) 输入信号
        sample_target: (L,) 目标信号
    """
    return test_input[idx], test_target[idx]


def normalize_signal(signal):
    """
    归一化信号
    
    Args:
        signal: 输入信号
        
    Returns:
        normalized: 归一化后的信号
        norm_factor: 归一化因子
    """
    norm_factor = np.max(np.abs(signal))
    if norm_factor == 0:
        norm_factor = 1.0
    normalized = signal / norm_factor
    return normalized, norm_factor


def prepare_input_tensor(signal, device):
    """
    将numpy信号转换为模型输入tensor
    
    Args:
        signal: (L,) numpy数组
        device: 目标设备
        
    Returns:
        tensor: (1, 1, L) tensor
        norm_factor: 归一化因子
    """
    normalized, norm_factor = normalize_signal(signal)
    tensor = torch.from_numpy(normalized).float().unsqueeze(0).unsqueeze(0).to(device)
    return tensor, norm_factor


def create_time_axis(length, fs=None):
    """
    创建时间轴
    
    Args:
        length: 信号长度
        fs: 采样率，如果为None则使用配置中的值
        
    Returns:
        time: 时间轴数组（秒）
    """
    if fs is None:
        fs = DATA_CONFIG['fs']
    return np.arange(length) / fs


def save_figure(fig, filename, subdir=''):
    """
    保存图像
    
    Args:
        fig: matplotlib figure对象
        filename: 文件名（不含扩展名）
        subdir: 子目录名
    """
    save_dir = os.path.join(FIGURE_DIR, subdir) if subdir else FIGURE_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    if EXPORT_CONFIG['save_png']:
        png_path = os.path.join(save_dir, f'{filename}.png')
        fig.savefig(png_path, dpi=VIS_CONFIG['dpi'], bbox_inches='tight')
        print(f'保存图像: {png_path}')
    
    if EXPORT_CONFIG['save_pdf']:
        pdf_path = os.path.join(save_dir, f'{filename}.pdf')
        fig.savefig(pdf_path, bbox_inches='tight')
    
    if EXPORT_CONFIG['save_svg']:
        svg_path = os.path.join(save_dir, f'{filename}.svg')
        fig.savefig(svg_path, bbox_inches='tight')


def save_data(data_dict, filename, subdir=''):
    """
    保存中间数据
    
    Args:
        data_dict: 要保存的数据字典
        filename: 文件名（不含扩展名）
        subdir: 子目录名
    """
    if not EXPORT_CONFIG['save_data']:
        return
    
    save_dir = os.path.join(DATA_OUTPUT_DIR, subdir) if subdir else DATA_OUTPUT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f'{filename}.npz')
    np.savez(save_path, **data_dict)
    print(f'保存数据: {save_path}')


def plot_signal(ax, time, signal, label='Signal', color=None, alpha=1.0, linewidth=1.0):
    """
    在指定的axes上绘制信号
    
    Args:
        ax: matplotlib axes对象
        time: 时间轴
        signal: 信号数据
        label: 标签
        color: 颜色
        alpha: 透明度
        linewidth: 线宽
    """
    ax.plot(time, signal, label=label, color=color, alpha=alpha, linewidth=linewidth)
    ax.set_xlabel('时间 (秒)')
    ax.set_ylabel('幅值')
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_signals_comparison(signals_dict, time=None, title='信号对比', 
                            figsize=None, save_name=None):
    """
    绘制多个信号的对比图
    
    Args:
        signals_dict: 字典 {name: signal}
        time: 时间轴，如果为None则自动生成
        title: 图标题
        figsize: 图尺寸
        save_name: 保存的文件名
        
    Returns:
        fig, axes
    """
    n_signals = len(signals_dict)
    if figsize is None:
        figsize = (12, 3 * n_signals)
    
    fig, axes = plt.subplots(n_signals, 1, figsize=figsize)
    if n_signals == 1:
        axes = [axes]
    
    # 获取第一个信号的长度来创建时间轴
    first_signal = list(signals_dict.values())[0]
    if time is None:
        time = create_time_axis(len(first_signal))
    
    for ax, (name, signal) in zip(axes, signals_dict.items()):
        color = VIS_CONFIG['colors'].get(name.lower(), None)
        plot_signal(ax, time, signal, label=name, color=color)
        ax.set_title(name)
    
    fig.suptitle(title, fontsize=VIS_CONFIG['title_size'], y=0.995)
    plt.tight_layout()
    
    if save_name:
        save_figure(fig, save_name)
    
    return fig, axes


def compute_frequency_spectrum(signal, fs=None):
    """
    计算信号的频谱
    
    Args:
        signal: 输入信号
        fs: 采样率
        
    Returns:
        freqs: 频率轴
        spectrum: 功率谱密度
    """
    if fs is None:
        fs = DATA_CONFIG['fs']
    
    # 使用FFT计算频谱
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, 1/fs)
    
    # 只取正频率部分
    positive_idx = freqs >= 0
    freqs = freqs[positive_idx]
    spectrum = np.abs(fft_result[positive_idx]) ** 2
    
    return freqs, spectrum


def plot_spectrum(ax, signal, fs=None, label='Spectrum', color=None, freq_range=None):
    """
    在指定的axes上绘制频谱
    
    Args:
        ax: matplotlib axes对象
        signal: 信号数据
        fs: 采样率
        label: 标签
        color: 颜色
        freq_range: 频率范围 (min, max)
    """
    freqs, spectrum = compute_frequency_spectrum(signal, fs)
    
    if freq_range:
        mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        freqs = freqs[mask]
        spectrum = spectrum[mask]
    
    ax.plot(freqs, spectrum, label=label, color=color)
    ax.set_xlabel('频率 (Hz)')
    ax.set_ylabel('功率谱密度')
    ax.legend()
    ax.grid(True, alpha=0.3)


def print_signal_stats(signal, name='Signal'):
    """
    打印信号统计信息
    
    Args:
        signal: 信号数据
        name: 信号名称
    """
    print(f'\n{name} 统计信息:')
    print(f'  长度: {len(signal)}')
    print(f'  最小值: {np.min(signal):.4f}')
    print(f'  最大值: {np.max(signal):.4f}')
    print(f'  均值: {np.mean(signal):.4f}')
    print(f'  标准差: {np.std(signal):.4f}')


def tensor_to_numpy(tensor):
    """
    将PyTorch tensor转换为numpy数组
    
    Args:
        tensor: PyTorch tensor
        
    Returns:
        numpy数组
    """
    return tensor.detach().cpu().numpy()


def create_subplots_grid(n_plots, ncols=2, figsize_per_plot=(6, 4)):
    """
    创建子图网格
    
    Args:
        n_plots: 子图数量
        ncols: 列数
        figsize_per_plot: 每个子图的尺寸
        
    Returns:
        fig, axes
    """
    nrows = (n_plots + ncols - 1) // ncols
    figsize = (figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten() if n_plots > 1 else [axes]
    
    # 隐藏多余的子图
    for i in range(n_plots, len(axes)):
        axes[i].axis('off')
    
    return fig, axes


# ==================== 初始化 ====================
def initialize():
    """初始化可视化环境"""
    import sys
    import io
    
    # 设置stdout编码为UTF-8
    if sys.stdout.encoding != 'utf-8':
        try:
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        except:
            pass
    
    setup_plotting_style()
    try:
        print('✓ 可视化环境初始化完成')
        print(f'✓ 输出目录: {OUTPUT_DIR}')
    except UnicodeEncodeError:
        print('[OK] 可视化环境初始化完成')
        print(f'[OK] 输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    # 测试工具函数
    print('测试 utils.py')
    initialize()
    
    # 测试数据加载
    test_input, test_target = load_test_data()
    print(f'\n测试数据加载成功: {test_input.shape}')
    
    # 测试模型加载
    model, device = load_model()
    print(f'\n模型加载成功')
    
    # 测试单个样本获取
    sample_input, sample_target = get_sample(test_input, test_target, idx=0)
    print_signal_stats(sample_input, '样本输入')
    print_signal_stats(sample_target, '样本目标')
    
    print('\n✓ 所有测试通过')
