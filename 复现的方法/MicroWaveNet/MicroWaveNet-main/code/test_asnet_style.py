
import os
import sys
import time
import scipy.io
import torch
import torch.utils.data as Data
import numpy as np

# 添加父目录以导入数据配置
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from data_config import DATA_DIR, TEST_CONTAMINATED_PATH, TEST_PURE_PATH, DATA_KEY
from cbamdropout import EEGNetMorletWindowCBAMDropout


BATCH_SIZE = 50


class EEGDatasetASNetStyle(Data.Dataset):
    def __init__(self, noisy_signals, clean_signals, is_train=False):
        self.noisy_signals = noisy_signals
        self.clean_signals = clean_signals
        self.is_train = is_train

    def __len__(self):
        return len(self.noisy_signals)

    def __getitem__(self, idx):
        noisy = self.noisy_signals[idx]
        clean = self.clean_signals[idx]

        norm_factor = np.max(np.abs(noisy))
        if norm_factor == 0:
            norm_factor = 1.0

        noisy_normalized = noisy / norm_factor
        noisy_normalized = noisy_normalized[np.newaxis, :].astype(np.float32)
        clean = clean[np.newaxis, :].astype(np.float32)

        return noisy_normalized, clean, np.array([norm_factor], dtype=np.float32)


def load_test_data(data_dir):
    test_input = scipy.io.loadmat(TEST_CONTAMINATED_PATH)[DATA_KEY]
    test_output = scipy.io.loadmat(TEST_PURE_PATH)[DATA_KEY]

    test_set = EEGDatasetASNetStyle(test_input, test_output, is_train=False)
    test_loader = Data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    return test_loader, test_output


def test_model(model, device, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    sample_count = 0
    start = time.time()
    with torch.no_grad():
        for noisy, clean, norm in test_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            norm = norm.to(device)

            out = model(noisy)
            eeg_pred = (out[0] * norm.view(-1, 1, 1)).cpu().numpy()

            all_preds.append(eeg_pred)
            all_targets.append(clean.cpu().numpy())
            sample_count += eeg_pred.shape[0]

    total_time = time.time() - start
    time_per_sample = total_time / max(1, sample_count)

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 去掉通道维度,从 (N, 1, L) 变为 (N, L)
    all_preds = np.squeeze(all_preds, axis=1)
    all_targets = np.squeeze(all_targets, axis=1)

    return all_preds, all_targets, time_per_sample


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_loader, test_targets = load_test_data(DATA_DIR)

    model = EEGNetMorletWindowCBAMDropout(device=device)
    model.to(device)

    model_path = 'MicroWaveNet_best.pt'
    if not os.path.exists(model_path):
        print(f'找不到最佳模型文件 {model_path}, 请先运行训练脚本')
        return

    model.load_state_dict(torch.load(model_path, map_location=device))

    preds, targets, tps = test_model(model, device, test_loader)

    results_dir = r'D:\Pycharm_Projects\EOG Remove\复现的方法\results'
    os.makedirs(results_dir, exist_ok=True)
    save_path = os.path.join(results_dir, 'MicroWaveNet_predictions.mat')
    scipy.io.savemat(save_path, {'predictions': preds, 'targets': targets, 'time_per_sample': tps})

    print(f'预测保存到: {save_path}, 预测形状: {preds.shape}, 单样本时间: {tps*1000:.3f}ms')


if __name__ == '__main__':
    main()
