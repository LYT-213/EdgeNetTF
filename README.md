# EdgeNetTF

Official reproducibility repository for:

**EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition**

Yutao Li, Junghun Kim, and Sang-Il Choi  
Daegu Catholic University, Republic of Korea

## Overview

EdgeNetTF is a lightweight dual-branch one-dimensional neural network for surface electromyography (sEMG) hand gesture recognition. The model combines a temporal representation of the sEMG window with a fast Fourier transform (FFT)-based spectral representation and fuses the two 128-dimensional branch outputs by direct concatenation.

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

## Important model-version note

The final model reported as **EdgeNetTF** in the manuscript uses **direct feature concatenation**.

An older Kaggle notebook named:

`notebooks/original_kaggle/kaggle-edgenettf-db2-full.ipynb`

implements the **adaptive gated-fusion ablation** (EdgeNetTF-Gated), not the final concatenation model. It is retained unchanged as an original experiment record.

The final concatenation implementation is available in:

- `src/edgenettf/models.py` (`EdgeNetTF`)
- `notebooks/original_kaggle/kaggle-db2-ablation-py.ipynb` (`concat_fusion`)

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
│       ├── preprocessing.py
│       └── smoothing.py
├── scripts/
│   ├── statistics.py
│   ├── reproduce_summary_tables.py
│   └── profile_table4.py
├── results/
│   ├── ablation_subject_level.csv
│   ├── baseline_subject_level.csv
│   ├── table1_ablation_summary.csv
│   ├── table2_baseline_summary.csv
│   ├── table4_computational_efficiency.csv
│   ├── table4_environment.txt
│   ├── wilcoxon_ablation.csv
│   └── wilcoxon_baselines_holm.csv
└── notebooks/
    ├── README.md
    └── original_kaggle/
```

## Data

The raw NinaPro DB2 data are **not redistributed in this repository**.

Please obtain DB2 from the official NinaPro resource or an authorized mirror and keep the raw data outside the repository.

The original Kaggle notebooks use a Kaggle dataset path. If your dataset is stored elsewhere, update the `BASE_PATH` variable before running the notebooks.

## Preprocessing

The experimental notebooks use the following amplitude compression:

```python
sign(x) * log(1 + 2048 * abs(x)) / log(2049)
```

Windows are generated only inside continuous, single-label gesture segments, preventing windows from crossing gesture boundaries.

For each temporal window, the spectral branch uses the first half of the FFT magnitude spectrum:

```python
log1p(abs(fft(window))[:window_size // 2] + 1e-8)
```

Temporal and frequency-domain inputs are standardized independently using statistics computed from the corresponding subject's **training windows only**.

## EdgeNetTF architecture

The temporal branch contains two Conv1D blocks with 64 and 128 channels, GELU activation, batch normalization, max pooling, and adaptive average pooling.

The frequency branch contains Conv1D layers with 64 and 128 channels, GELU activation, batch normalization, and adaptive average pooling.

The two 128-dimensional feature vectors are concatenated into a 256-dimensional representation, followed by a lightweight classifier:

```text
256 -> 256 -> 49 classes
```

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

The original experiment notebooks are preserved because individual baseline runs used their original experiment-specific random seeds.

## Causal5 smoothing

Causal5 uses the current prediction and up to four immediately preceding predictions. It does not use future predictions.

In the reported experiments, smoothing is applied independently inside dataset-defined gesture segments. This should not be interpreted as a complete continuous-stream gesture-transition detector.

## Reproducing the statistical analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python scripts/statistics.py
```

The script performs two-sided paired Wilcoxon signed-rank tests on subject-level Causal5 macro-F1 values. For comparisons between EdgeNetTF and the seven controlled baseline models, p values are adjusted using the Holm method.

To recompute the summary tables:

```bash
python scripts/reproduce_summary_tables.py
```

## Computational profiling

The final unified Table 4 profiling script is:

```bash
python scripts/profile_table4.py
```

The manuscript profiling protocol uses:

- batch size: 1
- temporal input: `(1, 12, 600)`
- frequency input for EdgeNetTF: `(1, 12, 300)`
- 100 warm-up iterations
- 1,000 timed iterations
- one CPU thread for CPU profiling
- CUDA events for GPU profiling
- forward-pass latency only
- FFT preprocessing, data loading, and window generation excluded
- FLOPs reported as `2 x MACs`

The final profiling rerun was performed in a Kaggle environment with Python 3.12.13, PyTorch 2.10.0+cu128, CUDA 12.8, and an NVIDIA Tesla T4 GPU. Exact reported values are stored in `results/table4_computational_efficiency.csv`.

## Main reported results

For the final concatenation-based EdgeNetTF across 40 subjects:

- Causal5 accuracy: approximately **87.02%**
- Causal5 macro-F1: approximately **87.41%**
- parameters: **0.1891 M**
- MACs: **35.00 M**
- FLOPs: **70.01 M**

The subject-level result files are included so that the reported means, standard deviations, and statistical comparisons can be independently recomputed.

## Original notebooks

The `notebooks/original_kaggle/` directory contains the available original Kaggle experiment notebooks. These files are retained primarily for traceability. Some notebook filenames reflect earlier experiment naming and some figure/statistics notebooks predate the final manuscript formatting.

Use the cleaned source files and scripts at the repository root for the manuscript-facing model definition, statistical analysis, and final Table 4 profiling protocol.

## Citation

Publication details will be added after the manuscript is published.

For now, please cite the manuscript title:

> Y. Li, J. Kim, and S.-I. Choi, "EdgeNetTF for Lightweight Time Frequency Representation Learning in Multiclass sEMG Hand Gesture Recognition."

## License

No open-source license has been assigned yet. The source code is publicly available for research transparency and reproducibility. A formal license can be added separately by the authors.
