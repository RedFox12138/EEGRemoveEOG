"""
真实数据集配置文件 - 基于阈值的方法
"""
import os

# ========== 数据集路径 ==========
REAL_DATA_PATH = r'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat'
DATA_KEY = 'eog_dataset'

# ========== 数据参数 ==========
SAMPLING_RATE = 250.0
WINDOW_SIZE = 1500

# ========== 结果保存路径 ==========
RESULTS_DIR = 'results'
PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, 'Threshold_real_data_predictions.mat')

# ========== 总结果保存路径 ==========
FINAL_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'results'
)
FINAL_PREDICTION_PATH = os.path.join(FINAL_RESULTS_DIR, 'Threshold_real_data_predictions.mat')
