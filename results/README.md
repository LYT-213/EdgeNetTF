# Result files

- `ablation_subject_level.csv`: subject-level metrics for Temporal-only, FFT-only, EdgeNetTF-Gated, and final concatenation-based EdgeNetTF.
- `baseline_subject_level.csv`: subject-level metrics for the seven controlled neural-network baselines.
- `table1_ablation_summary.csv`: ablation summary using sample standard deviation.
- `table2_baseline_summary.csv`: controlled baseline summary using sample standard deviation.
- `wilcoxon_ablation.csv`: paired two-sided Wilcoxon tests for ablation comparisons.
- `wilcoxon_baselines_holm.csv`: paired two-sided Wilcoxon tests against the seven baselines with Holm-adjusted p values.
- `table4_computational_efficiency.csv`: unified model-only computational profiling rerun.
- `table4_environment.txt`: profiling environment and timing protocol.
- `revision/`: additional major-revision results for hyperparameter sensitivity, FFT-window sensitivity, DB2 LOSO evaluation, DB3 validation, and end-to-end latency.

All subject-level metrics are stored as fractions (for example, 0.8741 corresponds to 87.41%) unless a column explicitly contains percentage points.

The revised manuscript and Supplementary Information report **sample standard deviations (`ddof=1`)** across subjects.
