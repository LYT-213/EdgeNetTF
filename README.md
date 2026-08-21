# EdgeNetTF

Official reproducibility repository for:

**EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition**

Yutao Li, Junghun Kim, and Sang-Il Choi  
Daegu Catholic University, Republic of Korea

## Overview

EdgeNetTF is a lightweight dual-branch one-dimensional neural network for surface electromyography (sEMG) hand gesture recognition. It combines a temporal representation with a fast Fourier transform (FFT)-based spectral representation and fuses the two 128-dimensional branch outputs by direct concatenation.

The manuscript evaluates EdgeNetTF on **NinaPro DB2** using:

- 40 intact subjects
- 49 non-rest gestures
- training repetitions: **1, 3, 4, 6**
- test repetitions: **2, 5**
- 600-sample windows
- training stride: 60 samples
- test stride: 600 samples
- subject-specific standardization using training windows only
- Causal5 majority voting for the controlled within-study evaluation

The final concatenation-based EdgeNetTF contains approximately **0.1891 M trainable parameters**.

## Repository structure

```text
EdgeNetTF/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── edgenettf/
│       ├── __init__.py
│       ├── models.py
│       ├── baselines.py
│       ├── preprocessing.py
│       └── smoothing.py
├── scripts/
│   ├── statistics.py
│   ├── reproduce_summary_tables.py
│   └── profile_table4.py
└── results/
    ├── ablation_subject_level.csv
    ├── baseline_subject_level.csv
    ├── table1_ablation_summary.csv
    ├── table2_baseline_summary.csv
    ├── table4_computational_efficiency.csv
    ├── table4_environment.txt
    ├── wilcoxon_ablation.csv
    └── wilcoxon_baselines_holm.csv
```

## Data

The raw NinaPro DB2 data are **not redistributed in this repository**. Please obtain DB2 from the official NinaPro resource or another authorized source and keep the raw data outside the repository.

## Preprocessing

The experiments use logarithmic amplitude compression:

```python
sign(x) * log(1 + 2048 * abs(x)) / log(2049)
```

Windows are generated only inside continuous, single-label gesture segments so that no window crosses a gesture boundary.

For each temporal window, the spectral branch uses the first half of the FFT magnitude spectrum:

```python
log1p(abs(fft(window))[:window_size // 2] + 1e-8)
```

Temporal and frequency-domain inputs are standardized independently using statistics computed from the corresponding subject's **training windows only**.

## EdgeNetTF architecture

The temporal branch uses two Conv1D blocks with 64 and 128 channels, batch normalization, GELU activation, max pooling, and adaptive average pooling.

The frequency branch uses Conv1D layers with 64 and 128 channels, batch normalization, GELU activation, and adaptive average pooling.

The two 128-dimensional feature vectors are concatenated into a 256-dimensional representation and passed to the classifier:

```text
256 -> 256 -> 49 classes
```

The final manuscript model is the **direct-concatenation EdgeNetTF** implemented in `src/edgenettf/models.py`. `EdgeNetTFGated` in the same file is the adaptive gated-fusion ablation.

## Controlled baseline models

`src/edgenettf/baselines.py` contains the one-dimensional implementations used for the controlled comparisons:

- CNN1D
- MobileNetV2_1D
- SqueezeNet1D
- ShuffleNetV2_1D
- ResNet1D
- DenseNet1D
- InceptionTime1D

The parameter counts of these implementations match the models reported in the manuscript tables.

## Training configuration

The final experiments used:

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

Individual original experiment runs used their experiment-specific random seeds.

## Causal5 smoothing

Causal5 uses the current prediction and up to four immediately preceding predictions. It does not use future predictions.

In the reported experiments, smoothing is applied independently inside dataset-defined gesture segments. It should not be interpreted as a complete continuous-stream gesture-transition detector.

## Reproducing the statistical analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/statistics.py
```

The script performs two-sided paired Wilcoxon signed-rank tests using subject-level Causal5 macro-F1 values. Comparisons between EdgeNetTF and the seven controlled baseline models are adjusted using the Holm method.

To recompute the manuscript summary means and standard deviations:

```bash
python scripts/reproduce_summary_tables.py
```

The manuscript summary standard deviations use `ddof=0`, matching the original experiment summary convention.

## Computational profiling

Run:

```bash
python scripts/profile_table4.py
```

The Table 4 profiling protocol uses:

- batch size: 1
- temporal input: `(1, 12, 600)`
- frequency input for EdgeNetTF: `(1, 12, 300)`
- 100 warm-up iterations
- 1,000 timed iterations
- one CPU thread for CPU profiling
- CUDA events for GPU profiling
- forward-pass latency only
- data loading, window generation, preprocessing, and FFT excluded
- FLOPs reported as `2 x MACs`

The final reported rerun was performed in Kaggle with Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, and an NVIDIA Tesla T4 GPU. The reported values are stored in `results/table4_computational_efficiency.csv`.

## Main reported results

For the final concatenation-based EdgeNetTF across 40 subjects:

- Causal5 accuracy: approximately **87.02%**
- Causal5 macro-F1: approximately **87.41%**
- parameters: **0.1891 M**
- MACs: **35.00 M**
- FLOPs: **70.01 M**

Subject-level Causal5 macro-F1 values are included for all ablation variants and all seven controlled baseline models so that the reported summary values and paired statistical tests can be independently recomputed.

## Citation

Publication details will be added after publication. For now, please cite the manuscript title:

> Y. Li, J. Kim, and S.-I. Choi, "EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition."

## License

No open-source license has been assigned yet. The code is publicly available for research transparency and reproducibility; a formal license can be added separately by the authors.
