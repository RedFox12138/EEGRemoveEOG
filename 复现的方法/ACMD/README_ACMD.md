# ACMD-based OA Removal (MATLAB)

This folder provides a MATLAB implementation of the OA removal framework inspired by the paper in this directory (ACMD PDF). The pipeline follows the paper's sections:

1) DFT-based baseline wander removal (cut < 1 Hz); 2) amplitude normalization; 3) ACMD-based detection and removal using the first mode and a peak-point count criterion.

Implementation notes
- ACMD is approximated by a ridge-guided complex demodulation: we extract a low-frequency spectral ridge using the STFT, demodulate to baseband, low-pass filter, and remodulate to obtain the first mode. This captures the OA-dominated component while remaining toolbox-free and fast.
- OA detection uses Eq. (10)-(11): count peaks in |x1(t)|, compare with a threshold ξ. We recommend setting ξ to the midpoint between means of clean vs contaminated examples (see example script).

Files
- `oa_remove_acmd.m`: end-to-end denoiser implementing the paper's Algorithm 2 decision rule and Eq. (12) reconstruction.
- `acmd_extract_first_mode.m`: approximates the first ACMD mode using spectrogram ridge + demodulation.
- `peak_count.m`: implements Ψ_p.
- `synthesize_eeg_eog.m`: synthetic EEG+EOG generator (clean, EOG blinks, baseline wander).
- `example_synth_acmd.m`: runnable demo on synthetic data (无需外部数据文件)。
- `example_run_acmd.m`: runnable demo using the provided semi-simulated data。

Quick start
- Prefer synthetic demo: open `example_synth_acmd.m` and run。它会自动合成信号、估计阈值 ξ、完成去噪并绘图。
- 或用数据集：打开 `example_run_acmd.m`，它会从 `生成半模拟数据/` 读取 .mat 文件运行同样流程。

API
- `[z, info] = oa_remove_acmd(x, fs, opts)`
  - `opts.baselineHz` (default 1), `fmaxOA` (12), `winSec` (1.0), `ridgeSmooth` (~0.25*fs), `bandwidthHz` (2), `threshold` (empty=only analyze), `returnAll` (true).
  - `info` contains baseline, normalized signal, first mode, IF/phase, Ψ_p, detection flag, and parameters.

Troubleshooting
- If the first mode looks too narrow or too wide, adjust `bandwidthHz` (1–3 Hz typical). If ridge drifts to harmonics, reduce `fmaxOA` to 8–10 Hz. If IF fluctuates, increase `ridgeSmooth`.
- If your data are at a different sampling rate than 200 Hz, parameters auto-scale, but you may want to tweak `winSec` and `ridgeSmooth`.
