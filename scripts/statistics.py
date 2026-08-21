#!/usr/bin/env python3
"""Reproduce the paired Wilcoxon analyses reported in the manuscript."""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted_sorted[rank] = min(running, 1.0)

    adjusted = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def paired_results(reference, comparison, label):
    stat, p = wilcoxon(reference, comparison, alternative="two-sided")
    return {
        "Comparison": label,
        "Wilcoxon_W": float(stat),
        "Raw_p": float(p),
        "Mean_difference_percentage_points": float(
            (np.mean(reference) - np.mean(comparison)) * 100.0
        ),
    }


def main():
    baselines = pd.read_csv(RESULTS / "baseline_subject_level.csv")
    ablations = pd.read_csv(RESULTS / "ablation_subject_level.csv")

    edge = (
        ablations[ablations["Variant"] == "EdgeNetTF"]
        .set_index("Subject")["Causal5_F1_Macro"]
        .sort_index()
    )

    baseline_order = [
        "MobileNetV2_1D",
        "SqueezeNet1D",
        "ResNet1D",
        "ShuffleNetV2_1D",
        "InceptionTime1D",
        "DenseNet1D",
        "CNN1D",
    ]

    rows = []
    for model in baseline_order:
        other = (
            baselines[baselines["Model"] == model]
            .set_index("Subject")["Causal5_F1_Macro"]
            .reindex(edge.index)
        )
        rows.append(
            paired_results(
                edge.to_numpy(),
                other.to_numpy(),
                f"EdgeNetTF vs {model}",
            )
        )

    adjusted = holm_adjust([row["Raw_p"] for row in rows])
    for row, p_adj in zip(rows, adjusted):
        row["Holm_adjusted_p"] = float(p_adj)

    baseline_stats = pd.DataFrame(rows)
    baseline_stats.to_csv(
        RESULTS / "wilcoxon_baselines_holm.csv",
        index=False,
    )

    rows = []
    for variant in ["Temporal-only", "FFT-only", "EdgeNetTF-Gated"]:
        other = (
            ablations[ablations["Variant"] == variant]
            .set_index("Subject")["Causal5_F1_Macro"]
            .reindex(edge.index)
        )
        rows.append(
            paired_results(
                edge.to_numpy(),
                other.to_numpy(),
                f"EdgeNetTF vs {variant}",
            )
        )

    ablation_stats = pd.DataFrame(rows)
    ablation_stats.to_csv(
        RESULTS / "wilcoxon_ablation.csv",
        index=False,
    )

    print("\nBaseline comparisons (Holm adjusted)")
    print(baseline_stats.to_string(index=False))
    print("\nAblation comparisons")
    print(ablation_stats.to_string(index=False))


if __name__ == "__main__":
    main()
