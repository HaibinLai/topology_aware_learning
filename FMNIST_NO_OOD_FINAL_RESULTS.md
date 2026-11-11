# FMNIST / no-OOD — final results

Updated: 2026-11-11

## Status

- All 30 planned runs completed; queue recorded 30 starts, 30 completions, 0 failures.
- Each run used 33 clients, 40 rounds, 5 local epochs per round, batch size 16, Adam (`lr=1e-4`, `weight_decay=0`), seed 1, and `sample_alpha=1000`.
- Graphs: BA(33, m=1) and BA(33, m=3); `label_alpha` in {1, 10, 1000}; aggregators: `random`, `random_betCent`, `random_degCent`, `betCent`, `degCent`.
- No OOD data and no backdoor were used. Evaluation was performed after final aggregation.
- Reported numbers are means across the 33 clients’ final evaluation records at `round_idx=39`. Clean AUC is macro AUC; weighted AUC matches to 4 decimals because the test set is class-balanced.

## Results

| Graph | label_alpha | Aggregation | Clean Acc | Clean AUC (macro) | Clean AUC (weighted) |
|---|---:|---|---:|---:|---:|
| BA m=1 | 1 | betCent | 0.8192 | 0.9770 | 0.9770 |
| BA m=1 | 1 | degCent | 0.8239 | 0.9780 | 0.9780 |
| BA m=1 | 1 | random | 0.8313 | 0.9798 | 0.9798 |
| BA m=1 | 1 | random_betCent | 0.8311 | 0.9795 | 0.9795 |
| BA m=1 | 1 | random_degCent | 0.8273 | 0.9796 | 0.9796 |
| BA m=1 | 10 | betCent | 0.8507 | 0.9822 | 0.9822 |
| BA m=1 | 10 | degCent | 0.8513 | 0.9821 | 0.9821 |
| BA m=1 | 10 | random | 0.8501 | 0.9828 | 0.9828 |
| BA m=1 | 10 | random_betCent | 0.8529 | 0.9832 | 0.9832 |
| BA m=1 | 10 | random_degCent | 0.8515 | 0.9829 | 0.9829 |
| BA m=1 | 1000 | betCent | 0.8513 | 0.9824 | 0.9824 |
| BA m=1 | 1000 | degCent | 0.8478 | 0.9818 | 0.9818 |
| BA m=1 | 1000 | random | 0.8510 | 0.9825 | 0.9825 |
| BA m=1 | 1000 | random_betCent | 0.8537 | 0.9833 | 0.9833 |
| BA m=1 | 1000 | random_degCent | 0.8521 | 0.9826 | 0.9826 |
| BA m=3 | 1 | betCent | 0.8589 | 0.9841 | 0.9841 |
| BA m=3 | 1 | degCent | 0.8564 | 0.9842 | 0.9842 |
| BA m=3 | 1 | random | 0.8609 | 0.9855 | 0.9855 |
| BA m=3 | 1 | random_betCent | 0.8624 | 0.9852 | 0.9852 |
| BA m=3 | 1 | random_degCent | 0.8614 | 0.9852 | 0.9852 |
| BA m=3 | 10 | betCent | 0.8725 | 0.9862 | 0.9862 |
| BA m=3 | 10 | degCent | 0.8685 | 0.9858 | 0.9858 |
| BA m=3 | 10 | random | 0.8754 | 0.9868 | 0.9868 |
| BA m=3 | 10 | random_betCent | 0.8741 | 0.9867 | 0.9867 |
| BA m=3 | 10 | random_degCent | 0.8719 | 0.9864 | 0.9864 |
| BA m=3 | 1000 | betCent | 0.8698 | 0.9858 | 0.9858 |
| BA m=3 | 1000 | degCent | 0.8714 | 0.9864 | 0.9864 |
| BA m=3 | 1000 | random | 0.8758 | 0.9867 | 0.9867 |
| BA m=3 | 1000 | random_betCent | 0.8748 | 0.9866 | 0.9866 |
| BA m=3 | 1000 | random_degCent | 0.8753 | 0.9867 | 0.9867 |

## Aggregate summary

All summaries below are computed from the 30 selected final run CSVs. First, each run is averaged over its 33 clients at `round_idx=39` and final evaluation; the tables then average those 30 configuration means equally. Thus these are descriptive averages across this seed-1 experiment matrix, not averages across multiple seeds.

### Mean by aggregation method

Each method averages its six configurations (2 graphs × 3 `label_alpha` values).

| Aggregation | Mean Clean Acc | Mean Clean AUC (macro) | Mean Clean AUC (weighted) |
|---|---:|---:|---:|
| random | 0.8574 | 0.9840 | 0.9840 |
| random_betCent | 0.8582 | 0.9841 | 0.9841 |
| random_degCent | 0.8566 | 0.9839 | 0.9839 |
| betCent | 0.8538 | 0.9830 | 0.9830 |
| degCent | 0.8532 | 0.9830 | 0.9830 |

### Distribution across the 30 configuration means

| Metric | Mean | Median | Min | Max |
|---|---:|---:|---:|---:|
| Clean Acc | 0.8558 | 0.8550 | 0.8192 | 0.8758 |
| Clean AUC (macro) | 0.9836 | 0.9837 | 0.9770 | 0.9868 |
| Clean AUC (weighted) | 0.9836 | 0.9837 | 0.9770 | 0.9868 |

### Mean by graph and label alpha

| Group | Mean Clean Acc | Mean Clean AUC (macro) |
|---|---:|---:|
| BA m=1 (15 configs) | 0.8430 | 0.9813 |
| BA m=3 (15 configs) | 0.8686 | 0.9859 |
| `label_alpha=1` (10 configs) | 0.8433 | 0.9818 |
| `label_alpha=10` (10 configs) | 0.8619 | 0.9845 |
| `label_alpha=1000` (10 configs) | 0.8623 | 0.9845 |

## Quick read

- Best Clean Acc: 0.8758 — BA m=3, `label_alpha=1000`, `random`.
- Best Clean AUC (macro): 0.9868 — BA m=3, `label_alpha=10`, `random`.
- Across these conditions, all methods reached Clean AUC ≥ 0.9770. These are IID/no-OOD results; they do not measure robustness to poisoned or shifted data.

## Files

- Consolidated per-client final metrics (990 rows across 30 configurations): `FMNIST_NO_OOD_PER_CLIENT_FINAL.csv`.
- Original per-run records: each run’s `client_stats.csv` under `results/<run_id>/<args_hash>/`.
- Launch/finish queue: `queue_gpucache_resume.log`.
- Performance optimization record: `optimization_records/2026-11-11_runtime_optimization.md` and its PNG chart.
