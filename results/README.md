# Result files

- `ablation_subject_level.csv`: subject-level metrics for Temporal-only, FFT-only, EdgeNetTF-Gated, and final concatenation-based EdgeNetTF.
- `baseline_subject_level.csv`: subject-level metrics for the seven controlled neural-network baselines.
- `table1_ablation_summary.csv`: mean and standard deviation used for the ablation summary.
- `table2_baseline_summary.csv`: controlled baseline summary.
- `wilcoxon_ablation.csv`: paired two-sided Wilcoxon tests for ablation comparisons.
- `wilcoxon_baselines_holm.csv`: paired two-sided Wilcoxon tests against the seven baselines with Holm-adjusted p values.
- `table4_computational_efficiency.csv`: final unified computational profiling rerun.
- `table4_environment.txt`: profiling environment and timing protocol.

All subject-level metrics are stored as fractions (for example, 0.8741 corresponds to 87.41%) unless a column explicitly contains percentage points.

Manuscript summary standard deviations are recomputed with `ddof=0`, matching the original experiment summary convention.
