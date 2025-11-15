import numpy as np
import tqdm

def segmentation(in_signals, new_seg_duration, sfreq):
    n_epochs = in_signals.shape[0]
    out_data = []
    pos_segs = in_signals.shape[-1] // (new_seg_duration * sfreq)

    for idx_e_epoch in range(n_epochs):
        for idx_e_seg in range(pos_segs):
            start_index = int(idx_e_seg * sfreq)
            end_index = int(start_index + sfreq)

            e_segment = in_signals[idx_e_epoch, :, start_index:end_index]

            out_data.append(e_segment)

    return np.array(out_data)

def generate_contaminated_data(clean_all, noisy_all, third=None, snr_db_list=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]):
    clean_data = None
    contaminated_data = None
    for snr_db in tqdm(snr_db_list):
        snr = np.power(10, snr_db/10)
        clean_norms = np.linalg.norm(clean_all, axis=1)
        noisy_norms = np.linalg.norm(noisy_all, axis=1)
        lambdas = clean_norms/noisy_norms/snr
        clean_data = np.concatenate([clean_data, clean_all]) if clean_data is not None else clean_all
        if third is not None:   
            noisy_norms_1 = np.linalg.norm(noisy_all, axis=1)
            noisy_norms_2 = np.linalg.norm(third, axis=1)
            lambdas_1 = clean_norms/noisy_norms_1/snr
            lambdas_2 = clean_norms/noisy_norms_2/snr
            contaminated = clean_all + noisy_all*lambdas_1.reshape(-1, 1) + third*lambdas_2.reshape(-1, 1)
        else:   
            contaminated = clean_all + noisy_all*lambdas.reshape(-1, 1)
        contaminated *= 20.0 ** (-max(0.0, -snr_db) / 10.0)
        contaminated_data = np.concatenate([contaminated_data, contaminated]) if contaminated_data is not None else contaminated
    return clean_data, contaminated_data


def get_eegdenoising_data(type, snr_db_list=[-7, -6, -5, -4, -3, -2, -1, 0, 1, 2]):
    clean_all = np.load('EEGdenoising_EEG.npy')
    noisy_eog = np.load('EEGdenoising_EOG.npy')
    noisy_emg = np.load('EEGdenoising_EMG.npy')

    if type == 'EMG':
        data_length = max(len(clean_all), len(noisy_emg))
        reuse_num = len(noisy_emg) - len(clean_all)
        EEG_reuse = clean_all[0: reuse_num]
        clean_all_truncated = np.concatenate((EEG_reuse, clean_all), axis=0)
        noisy_emg_truncated = noisy_emg[:data_length]
        noisy_eog_truncated = noisy_eog[:data_length]
        clean, contaminated = generate_contaminated_data(clean_all_truncated, noisy_emg_truncated, snr_db_list=snr_db_list)
        
    elif type == 'EOG':
        data_length = min(len(clean_all), len(noisy_eog))
        clean_all_truncated = clean_all[:data_length]
        noisy_emg_truncated = noisy_emg[:data_length]
        noisy_eog_truncated = noisy_eog[:data_length]
        clean, contaminated = generate_contaminated_data(clean_all_truncated, noisy_eog_truncated, snr_db_list=snr_db_list)
    else:
        data_length = min(len(clean_all), len(noisy_emg), len(noisy_eog))
        clean_all_truncated = clean_all[:data_length]
        noisy_emg_truncated = noisy_emg[:data_length]
        noisy_eog_truncated = noisy_eog[:data_length]
        clean, contaminated = generate_contaminated_data(clean_all_truncated, noisy_emg_truncated, noisy_eog_truncated, snr_db_list=snr_db_list)

    clean_signal, noisy_signal = np.expand_dims(clean, axis=1), np.expand_dims(contaminated, axis=1)

    signal_min, signal_max = np.min(noisy_signal), np.max(noisy_signal) 
    noisy_signal = (noisy_signal - signal_min) / (signal_max - signal_min)
    clean_signal = (clean_signal - signal_min) / (signal_max - signal_min)
    noisy_signal = 2 * noisy_signal - 1
    clean_signal = 2 * clean_signal - 1

    return clean_signal, noisy_signal


def getdata(name, snr_db_list):
    if name == 'semi':
        clean_all = np.load('reference_Semi-simulated_EOG.npy')
        noisy_eog = np.load('signal_Semi-simulated_EOG.npy')
        clean_1s_seg = segmentation(in_signals=clean_all, new_seg_duration=1, sfreq=200)
        noisy_1s_seg = segmentation(in_signals=noisy_eog, new_seg_duration=1, sfreq=200)

        mean_cont, std_cont = np.mean(noisy_1s_seg, axis=-1, keepdims=True), np.std(noisy_1s_seg, axis=-1, keepdims=True)
        clean_1s_seg = (clean_1s_seg - mean_cont) / std_cont
        noisy_1s_seg = (noisy_1s_seg - mean_cont) / std_cont

    elif name == 'phy':
        clean_1s_seg = np.load('physiobank_clean.npy')
        noisy_1s_seg = np.load('physiobank_contaminated.npy')

        mean_cont, std_cont = np.mean(noisy_1s_seg, axis=-1, keepdims=True), np.std(noisy_1s_seg, axis=-1, keepdims=True)
        clean_1s_seg = (clean_1s_seg - mean_cont) / std_cont
        noisy_1s_seg = (noisy_1s_seg - mean_cont) / std_cont
    else:
        clean_1s_seg, noisy_1s_seg = get_eegdenoising_data(name, snr_db_list)

    # print(clean_1s_seg.shape, noisy_1s_seg.shape)
    
    return clean_1s_seg, noisy_1s_seg
 