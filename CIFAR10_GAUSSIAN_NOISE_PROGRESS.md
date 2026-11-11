# CIFAR-10 Gaussian-noise results — progress report

Updated: 2026-11-11 21:05 UTC

## Status

- This is a **partial** report: 12 of 20 configurations have completed final evaluation and have valid records for all 33 clients.
- Five configurations are still running; three configurations that failed on GPU 2 are being retried on GPUs 3, 4, and 6. GPU 1 and GPU 2 are excluded.
- Values below are means across the 33 clients at final `round_idx=39`. Clean AUC is macro AUC. Noisy/OOD metrics are evaluated on the corrupted CIFAR-10 test set; these are not backdoor attack metrics.

## Experiment settings

- Dataset/model: CIFAR-10 / VGG; BA graphs with 33 clients and attachment parameter `m=1` or `m=3`.
- Training: 40 rounds, 5 local epochs per round, batch size 16, Adam (`lr=1e-4`, `weight_decay=0`), full participation, seed 1, `label_alpha=sample_alpha=1000`.
- Corruption: additive Gaussian noise, `std=0.1` on `[0,1]` pixels, clipped then normalized. Ten percent of each configured source client's local training samples are corrupted; labels are preserved. The full test set receives the same deterministic corruption for OOD evaluation.
- One-source condition: client 14. Four-source condition: clients 14, 15, 24, 31. Node IDs are 0-based.
- `random`, `random+betCent`, and `random+degCent` refresh their random component each round.

## Completed results

| BA graph | Noisy source clients | Aggregation | Clean Acc | Clean AUC (macro) | OOD Acc | OOD AUC (macro) |
|---|---|---|---:|---:|---:|---:|
| m=1 | 14 | betCent | 0.5944 | 0.9087 | 0.4183 | 0.8284 |
| m=1 | 14 | degCent | 0.5859 | 0.9009 | 0.4214 | 0.8262 |
| m=1 | 14 | random | 0.5692 | 0.9028 | 0.4031 | 0.8266 |
| m=1 | 14 | random+degCent (dynamic) | 0.5833 | 0.9097 | 0.3823 | 0.8230 |
| m=1 | 14, 15, 24, 31 | degCent | 0.5908 | 0.9036 | 0.4554 | 0.8443 |
| m=1 | 14, 15, 24, 31 | random | 0.5715 | 0.9041 | 0.4018 | 0.8262 |
| m=1 | 14, 15, 24, 31 | random+betCent (dynamic) | 0.6020 | 0.9141 | 0.4087 | 0.8279 |
| m=1 | 14, 15, 24, 31 | random+degCent (dynamic) | 0.5783 | 0.9064 | 0.3977 | 0.8268 |
| m=3 | 14 | betCent | 0.7546 | 0.9645 | 0.4609 | 0.8652 |
| m=3 | 14 | random | 0.7727 | 0.9698 | 0.3626 | 0.8288 |
| m=3 | 14 | random+betCent (dynamic) | 0.7688 | 0.9683 | 0.3726 | 0.8397 |
| m=3 | 14 | random+degCent (dynamic) | 0.7706 | 0.9689 | 0.3801 | 0.8371 |


## Summary across completed configurations

The following statistics are across the 12 completed configuration scores in the table above. Each configuration contributes one value: the mean of its 33 clients' final evaluation records. These are not statistics over individual client records, and they will change as more configurations finish.

| Metric | Mean | Median | Minimum | Maximum |
|---|---:|---:|---:|---:|
| Clean Acc | 0.6452 | 0.5926 | 0.5692 | 0.7727 |
| Clean AUC (macro) | 0.9268 | 0.9092 | 0.9009 | 0.9698 |
| OOD Acc | 0.4054 | 0.4025 | 0.3626 | 0.4609 |
| OOD AUC (macro) | 0.8333 | 0.8281 | 0.8230 | 0.8652 |

## Current limits

These are single-seed results with fixed source-node placements. Compare aggregation methods within the same graph and source-node condition. Do not treat the partial table as the final 20-configuration matrix. The remaining configurations and GPU-2 retries will be added after valid final evaluation records are available.

Raw per-client results and logs are stored under `/ssd/haibin/logs/topology_aware_learning_cifar10_gaussian_noise_20261111/` on the experiment host.
