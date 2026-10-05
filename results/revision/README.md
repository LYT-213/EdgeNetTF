# Major-revision reproducibility files

This directory contains the additional analyses introduced during the Scientific Reports major revision.

- `hyperparameter_summary.csv` and `hyperparameter_statistics.csv`: kernel-size and channel-width sensitivity analysis.
- `fft_window_summary.csv` and `fft_window_statistics.csv`: direct FFT versus Hann/Hamming tapering.
- `loso_subject_level.csv` and `loso_summary.csv`: leakage-free leave-one-subject-out evaluation on NinaPro DB2.
- `db3_subject_level.csv` and `db3_summary.csv`: external validation on NinaPro DB3.
- `end_to_end_latency_summary.csv`: complete CPU pipeline latency including preprocessing and model inference.

Unless otherwise stated, standard deviations reported in the revised manuscript and Supplementary Information use the sample definition (`ddof=1`).

The experiment scripts in `scripts/revision/` generate the full run-level outputs. The original direct-FFT subject-level results used as the reference for sensitivity statistics are stored in `results/ablation_subject_level.csv` as the final `EdgeNetTF` variant.
