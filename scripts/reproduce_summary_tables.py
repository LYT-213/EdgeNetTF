#!/usr/bin/env python3
"""Recompute manuscript summary values from the subject-level CSV files.

The manuscript summary tables use population standard deviation (ddof=0),
matching the original experiment summary code.
"""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def summarize(values):
    values = np.asarray(values, dtype=float)
    return values.mean() * 100.0, values.std(ddof=0) * 100.0


def main():
    ablation = pd.read_csv(RESULTS / "ablation_subject_level.csv")
    baselines = pd.read_csv(RESULTS / "baseline_subject_level.csv")

    ablation_order = [
        "Temporal-only",
        "FFT-only",
        "EdgeNetTF-Gated",
        "EdgeNetTF",
    ]

    table1_rows = []
    for variant in ablation_order:
        mean_pct, sd_pct = summarize(
            ablation.loc[
                ablation["Variant"] == variant,
                "Causal5_F1_Macro"
            ]
        )
        table1_rows.append({
            "Variant": variant,
            "Causal5_Macro_F1_mean_pct": mean_pct,
            "Causal5_Macro_F1_sd_pct": sd_pct,
        })

    table1 = pd.DataFrame(table1_rows)
    table1.to_csv(
        RESULTS / "table1_ablation_summary.csv",
        index=False,
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

    table2_rows = []
    for model in baseline_order:
        subset = baselines[baselines["Model"] == model]
        mean_pct, sd_pct = summarize(subset["Causal5_F1_Macro"])
        table2_rows.append({
            "Model": model,
            "Params_M": subset["Params_M"].iloc[0],
            "Causal5_Macro_F1_mean_pct": mean_pct,
            "Causal5_Macro_F1_sd_pct": sd_pct,
        })

    edge = ablation[ablation["Variant"] == "EdgeNetTF"]
    mean_pct, sd_pct = summarize(edge["Causal5_F1_Macro"])
    table2_rows.append({
        "Model": "EdgeNetTF",
        "Params_M": edge["Params_M"].iloc[0],
        "Causal5_Macro_F1_mean_pct": mean_pct,
        "Causal5_Macro_F1_sd_pct": sd_pct,
    })

    table2 = pd.DataFrame(table2_rows)
    table2.to_csv(
        RESULTS / "table2_baseline_summary.csv",
        index=False,
    )

    print("Table 1")
    print(table1.to_string(index=False))
    print("\nTable 2")
    print(table2.to_string(index=False))


if __name__ == "__main__":
    main()
