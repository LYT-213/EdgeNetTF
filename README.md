# EdgeNetTF

Official reproducibility repository for:

**EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition**

Yutao Li, Junghun Kim, and Sang-Il Choi  
Daegu Catholic University, Republic of Korea

## Overview

EdgeNetTF is a lightweight dual-branch one-dimensional neural network for surface electromyography (sEMG) hand gesture recognition. It combines a temporal representation with a fast Fourier transform (FFT)-based spectral representation and fuses two 128-dimensional branch outputs by direct concatenation.

The primary evaluation uses **NinaPro DB2** with 40 intact subjects and 49 non-rest gestures. Repetitions **1, 3, 4, and 6** are used for training and repetitions **2 and 5** for testing. The split is performed before sliding-window segmentation. The final model contains approximately **0.1891 M trainable parameters**.

The Scientific Reports major revision additionally includes leakage-free DB2 leave-one-subject-out (LOSO) evaluation, external validation on NinaPro DB3, kernel/channel sensitivity analysis, FFT-window sensitivity analysis, UMAP feature visualization, gesture-level sample statistics, and end-to-end CPU latency profiling.

## Repository structure

```text
EdgeNetTF/
├── README.md
├── CITATION.cff
├── requirements.txt
├── src/edgenettf/
│   ├── models.py
│   ├── baselines.py
│   ├── preprocessing.py
│   └── smoothing.py
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
    └── revision/
        ├── hyperparameter_summary.csv
        ├── hyperparameter_statistics.csv
        ├── fft_window_summary.csv
        ├── fft_window_statistics.csv
        ├── loso_subject_level.csv
        ├── loso_summary.csv
        ├── db3_subject_level.csv
        ├── db3_summary.csv
        └── end_to_end_latency_summary.csv
```

## Data

Raw NinaPro data are **not redistributed**. Obtain NinaPro DB2 and DB3 from the official NinaPro resource or another authorized source and set the dataset path at the top of the relevant experiment script.

## Primary DB2 protocol

- subjects: 40
- active gesture classes: 49 (rest excluded)
- sampling rate: 2 kHz
- window length: 600 samples (300 ms)
- train repetitions: 1, 3, 4, 6
- test repetitions: 2, 5
- train stride: 60 samples (30 ms)
- test stride: 600 samples (300 ms)
- windows never cross gesture or repetition boundaries
- temporal and frequency inputs are standardized independently using training-window statistics only

The logarithmic amplitude compression is:

```python
sign(x) * log(1 + 2048 * abs(x)) / log(2049)
```

The original spectral representation is a direct FFT without explicit tapering (rectangular-window equivalent):

```python
log1p(abs(fft(window))[:window_size // 2] + 1e-8)
```

For a 600-sample window, the temporal input is `12 x 600` and the frequency input is `12 x 300`.

## Training configuration

- optimizer: AdamW
- initial learning rate: 0.001
- weight decay: 1e-3
- batch size: 128
- epochs: 60
- label smoothing: 0.05
- dropout: 0.3
- Mixup probability: 0.3
- Mixup alpha: 0.2
- scheduler: CosineAnnealingWarmRestarts (`T_0=20`, `T_mult=2`)
- gradient clipping: 1.0

Causal5 majority voting uses the current prediction and up to four preceding predictions inside each dataset-defined gesture segment; future predictions are never used.

## Controlled models and ablations

`src/edgenettf/models.py` contains the final concatenation-based EdgeNetTF plus the Temporal-only, FFT-only, and gated-fusion ablations. `src/edgenettf/baselines.py` contains the one-dimensional controlled baselines:

- CNN1D
- MobileNetV2_1D
- SqueezeNet1D
- ShuffleNetV2_1D
- ResNet1D
- DenseNet1D
- InceptionTime1D

## Major-revision analyses

### Hyperparameter sensitivity

```bash
python scripts/revision/run_hyperparameter_sensitivity.py
```

Tests smaller/larger convolution kernels and narrower/wider channel configurations while retaining the primary DB2 protocol. Summary values and Holm-adjusted paired statistics are in `results/revision/`.

### FFT-window sensitivity

```bash
python scripts/revision/run_fft_window_sensitivity.py
```

Runs Hann and Hamming tapering under the same protocol. The original direct FFT results are the reference condition. Explicit tapering did not improve the reported recognition performance.

### Leakage-free LOSO evaluation

```bash
python scripts/revision/run_loso_db2.py --start 1 --end 40
```

Each fold completely holds out one DB2 participant; the remaining 39 subjects are used for training. The held-out participant contributes neither training windows nor normalization statistics. For computational feasibility, both training and testing use non-overlapping 600-sample windows (stride 600).

### NinaPro DB3 validation

```bash
python scripts/revision/run_db3_external_validation.py
```

Evaluates 11 trans-radial amputee participants using repetitions 1/3/4/6 for training and 2/5 for testing. The number of available active gesture classes varies from 38 to 49 across participants.

### Feature visualization and supporting analyses

```bash
python scripts/revision/generate_umap_s1.py
python scripts/revision/generate_gesture_statistics.py
python scripts/revision/generate_time_fft_figure.py
python scripts/revision/profile_end_to_end_latency.py
```

The UMAP visualization uses independently projected temporal, frequency-domain, and fused held-out S1 features with `n_neighbors=30`, `min_dist=0.1`, Euclidean distance, and `random_state=42`.

## Statistical analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Primary ablation/baseline statistics:

```bash
python scripts/statistics.py
```

Revision sensitivity statistics after generating the subject-level sensitivity CSVs:

```bash
python scripts/revision/statistics_revision.py
```

Paired two-sided Wilcoxon signed-rank tests are performed at the subject level and Holm correction is applied within each specified comparison family. The revised manuscript and Supplementary Information report **sample standard deviations (`ddof=1`)**.

To recompute the main summary tables from subject-level CSVs:

```bash
python scripts/reproduce_summary_tables.py
```

## Computational profiling

Model-only Table 4 profiling:

```bash
python scripts/profile_table4.py
```

The profiling protocol uses batch size 1, temporal input `(1, 12, 600)`, frequency input `(1, 12, 300)`, 100 warm-up iterations, 1,000 timed iterations, and FLOPs approximated as `2 x MACs`.

The revised manuscript additionally reports end-to-end CPU latency including logarithmic compression, FFT computation, normalization, tensor conversion, and model inference. The measured pipeline latency was **2.032 ± 0.141 ms** per 300-ms window (median 2.013 ms; P95 2.255 ms) in the reported Kaggle CPU environment.

## Main reported results

Primary within-subject DB2 protocol:

- Causal5 accuracy: **87.01 ± 4.30%**
- Causal5 macro-F1: **87.41 ± 3.94%**
- parameters: **0.1891 M**
- MACs: **35.00 M**
- FLOPs: **70.01 M**

Complementary LOSO DB2 evaluation:

- Raw accuracy: **14.93 ± 5.19%**
- Raw macro-F1: **14.11 ± 5.37%**
- Causal5 accuracy: **16.82 ± 6.24%**
- Causal5 macro-F1: **15.66 ± 6.34%**

NinaPro DB3 external validation:

- Raw accuracy: **57.16 ± 18.15%**
- Causal5 accuracy: **64.35 ± 20.18%**
- Causal5 macro-F1: **63.65 ± 20.15%**

## Citation

Publication details and the archived repository DOI will be added when available. Until then, please cite the manuscript title:

> Y. Li, J. Kim, and S.-I. Choi, "EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition."

## License

No open-source license has been assigned yet. The code is publicly available for research transparency and reproducibility; a formal license can be added separately by the authors.
