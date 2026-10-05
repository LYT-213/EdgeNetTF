# EdgeNetTF

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23155437.svg)](https://doi.org/10.5281/zenodo.23155437)

Official reproducibility repository for **EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition**.

Yutao Li, Junghun Kim, and Sang-Il Choi — Daegu Catholic University, Republic of Korea.

## Overview

EdgeNetTF is a lightweight dual-branch 1D network for sEMG hand-gesture recognition. It learns a temporal representation and a direct-FFT spectral representation, concatenates the two 128-dimensional branch outputs, and uses a compact classifier. The final model contains approximately **0.1891 M trainable parameters**.

The primary evaluation uses **NinaPro DB2** (40 intact subjects, 49 non-rest gestures). The Scientific Reports major revision additionally includes leakage-free DB2 LOSO evaluation, NinaPro DB3 validation, kernel/channel sensitivity, FFT-window sensitivity, UMAP feature visualization, gesture-level statistics, and end-to-end CPU latency profiling.

## Repository structure

```text
EdgeNetTF/
├── src/edgenettf/                 # model, baselines, preprocessing, smoothing
├── scripts/
│   ├── statistics.py
│   ├── reproduce_summary_tables.py
│   ├── profile_table4.py
│   └── revision/
│       ├── run_loso_db2.py
│       ├── run_db3_external_validation.py
│       ├── run_hyperparameter_sensitivity.py
│       ├── run_fft_window_sensitivity.py
│       ├── statistics_revision.py
│       ├── generate_umap_s1.py
│       ├── generate_gesture_statistics.py
│       ├── generate_time_fft_figure.py
│       └── profile_end_to_end_latency.py
└── results/
    ├── ablation_subject_level.csv
    ├── baseline_subject_level.csv
    ├── table1_ablation_summary.csv
    ├── table2_baseline_summary.csv
    ├── table4_computational_efficiency.csv
    ├── wilcoxon_ablation.csv
    ├── wilcoxon_baselines_holm.csv
    └── revision/                  # major-revision summaries/statistics/results
```

## Data and primary DB2 protocol

Raw NinaPro data are **not redistributed**. Obtain DB2/DB3 from the official NinaPro resource or another authorized source and edit `BASE_PATH` in the relevant script.

Primary DB2 settings:

- 40 subjects; 49 active gestures; rest excluded
- 2 kHz sampling; 600 samples = 300 ms
- train repetitions: 1, 3, 4, 6; test repetitions: 2, 5
- train stride: 60; test stride: 600
- split before windowing; windows do not cross gesture/repetition boundaries
- temporal and frequency inputs standardized independently using training-only statistics

Log compression:

```python
sign(x) * log(1 + 2048 * abs(x)) / log(2049)
```

Original spectral representation (no explicit taper; rectangular equivalent):

```python
log1p(abs(fft(window))[:window_size // 2] + 1e-8)
```

Inputs are `12 x 600` (temporal) and `12 x 300` (frequency).

## Training configuration

AdamW, learning rate 0.001, weight decay 1e-3, batch size 128, 60 epochs, label smoothing 0.05, dropout 0.3, Mixup probability 0.3 / alpha 0.2, CosineAnnealingWarmRestarts (`T_0=20`, `T_mult=2`), and gradient clipping 1.0.

Causal5 uses the current prediction and up to four preceding predictions within the same dataset-defined gesture segment; no future prediction is used.

## Installation

```bash
pip install -r requirements.txt
```

Run commands below from the repository root. The revision scripts are invoked as modules so the local `src` package is resolvable.

## Major-revision analyses

Hyperparameter sensitivity:

```bash
python -m scripts.revision.run_hyperparameter_sensitivity
```

FFT-window sensitivity (Hann/Hamming; direct FFT is the original reference):

```bash
python -m scripts.revision.run_fft_window_sensitivity
```

Leakage-free DB2 LOSO (held-out subject contributes neither training windows nor normalization statistics; stride 600 for both train/test):

```bash
python -m scripts.revision.run_loso_db2 --start 1 --end 40
```

NinaPro DB3 external validation (11 trans-radial amputee participants; repetitions 1/3/4/6 train and 2/5 test; 38–49 available active classes):

```bash
python -m scripts.revision.run_db3_external_validation
```

Feature visualization and supporting analyses:

```bash
python -m scripts.revision.generate_umap_s1
python -m scripts.revision.generate_gesture_statistics
python -m scripts.revision.generate_time_fft_figure
python -m scripts.revision.profile_end_to_end_latency
```

UMAP uses independently projected temporal, frequency-domain, and fused held-out S1 features (`n_neighbors=30`, `min_dist=0.1`, Euclidean distance, `random_state=42`).

Revision paired statistics after generating subject-level sensitivity CSVs:

```bash
python -m scripts.revision.statistics_revision
```

Primary ablation/baseline statistics and summary reproduction:

```bash
python scripts/statistics.py
python scripts/reproduce_summary_tables.py
```

Paired two-sided Wilcoxon tests are used at the subject level with Holm correction within each specified comparison family. The revised manuscript and Supplementary Information report **sample standard deviations (`ddof=1`)**.

## Computational profiling

Model-only Table 4 profiling:

```bash
python scripts/profile_table4.py
```

The model-only benchmark uses batch size 1, inputs `(1,12,600)` and `(1,12,300)` for EdgeNetTF, 100 warm-ups, 1,000 timed iterations, and FLOPs = `2 x MACs`.

The revision additionally measures the complete CPU pipeline (log compression + FFT + normalization + tensor conversion + model inference): **2.032 ± 0.141 ms** per 300-ms window, median **2.013 ms**, P95 **2.255 ms** in the reported Kaggle CPU environment.

## Main reported results

Primary within-subject DB2:

- Causal5 accuracy: **87.01 ± 4.30%**
- Causal5 macro-F1: **87.41 ± 3.94%**
- parameters: **0.1891 M**; MACs: **35.00 M**; FLOPs: **70.01 M**

Complementary DB2 LOSO:

- Raw accuracy / macro-F1: **14.93 ± 5.19% / 14.11 ± 5.37%**
- Causal5 accuracy / macro-F1: **16.82 ± 6.24% / 15.66 ± 6.34%**

NinaPro DB3:

- Raw accuracy: **57.16 ± 18.15%**
- Causal5 accuracy / macro-F1: **64.35 ± 20.18% / 63.65 ± 20.15%**

See `results/revision/README.md` for the revision result files.

## Citation

Archived software release (v1.0.0): **https://doi.org/10.5281/zenodo.23155437**

> Y. Li, J. Kim, and S.-I. Choi, "EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition."

## License

No open-source license has been assigned yet. The code is publicly available for research transparency and reproducibility; a formal license can be added separately by the authors.
