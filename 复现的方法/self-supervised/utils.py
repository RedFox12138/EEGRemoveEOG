import torch
import math

def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_pearson_correlation(clean, outputs_cpu):  
    corr_running = 0
    for cl, output_cpu in zip(clean, outputs_cpu):
        for ch_cl, ch_o in zip(cl, output_cpu):
            corr_coef = torch.corrcoef(torch.stack([ch_cl, ch_o]))[0][1]
            corr_running += corr_coef
    corr_avg = corr_running/torch.prod(torch.tensor(clean.shape[:-1]))
    return corr_avg


def get_matric(clean, outputs_cpu):
    err = ((clean - outputs_cpu)**2).sum(tuple(range(1, len(clean.shape))))
    max_diff_norm = torch.prod(torch.tensor(clean.shape[1:])) * 1
    metric = (1 - err / max_diff_norm).mean() 
    return metric


def get_snr(clean, outputs_cpu):
    noise = clean - outputs_cpu
    signal_power = torch.linalg.vector_norm(clean, dim=len(clean.shape)-1)**2
    noise_power = torch.linalg.vector_norm(noise, dim=len(noise.shape)-1)**2
    snr_db = 10*torch.log10(signal_power/noise_power).mean()
    return snr_db

def trrmse_metric(y_true, y_predict):
    return torch.sqrt((y_predict - y_true) ** 2).mean() / torch.sqrt(y_true ** 2).mean()

def frrmse_metric(y_true, y_predict):
    psd_y_true = _get_psd(y_true)
    psd_y_predict = _get_psd(y_predict)
    return torch.sqrt((psd_y_predict - psd_y_true) ** 2).mean() / torch.sqrt(psd_y_true ** 2).mean()

def _get_psd(tensor):
    if len(tensor.shape) == 3:
        data = tensor.squeeze(1)  
    else:
        data = tensor
    fft_data = torch.fft.fft(data, dim=-1) 
    psd = torch.abs(fft_data) ** 2 
    psd = psd[..., :psd.shape[-1] // 2] 
    return psd

