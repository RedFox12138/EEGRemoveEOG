"""
真实数据集配置文件 - EEGIFNet
"""
import os

# ========== 数据集路径 ==========
REAL_DATA_PATH = r'D:\Pycharm_Projects\EOG Remove\真实数据集\eog_dataset.mat'
DATA_KEY = 'eog_dataset'

# ========== 数据参数 ==========
SAMPLING_RATE = 250.0
WINDOW_SIZE = 1500

# ========== 模型路径 ==========
# 使用训练好的最佳模型
INET_MODEL_PATH = r'D:\Pycharm_Projects\EOG Remove\复现的方法\训练完的模型和数据\全模拟数据集\模型\有监督\EEGIFNet_INet_best.pkl'
MNET_MODEL_PATH = r'D:\Pycharm_Projects\EOG Remove\复现的方法\训练完的模型和数据\全模拟数据集\模型\有监督\EEGIFNet_MNet_best.pkl'

# ========== 结果保存路径 ==========
RESULTS_DIR = 'results'
PREDICTION_SAVE_PATH = os.path.join(RESULTS_DIR, 'EEGIFNet_real_data_predictions.mat')

# ========== 总结果保存路径 ==========
FINAL_RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),
    'results'
)
FINAL_PREDICTION_PATH = os.path.join(FINAL_RESULTS_DIR, 'EEGIFNet_real_data_predictions.mat')
