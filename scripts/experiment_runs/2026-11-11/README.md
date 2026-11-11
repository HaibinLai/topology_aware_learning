# Experiment run scripts from 2026-11-11

This directory archives the scripts used for the reported FMNIST/no-OOD,
CIFAR-10 Gaussian-noise, and BA(33, m=1) high-degree-node backdoor runs.
They are preserved with the original server paths and GPU assignments so the
executed parameters can be audited. Before reusing them on another machine,
update `ROOT`, `PY`, `BASE`, and `DATA`.

## FMNIST / no-OOD

The scripts show the execution sequence used while reducing scheduler and data
transfer overhead:

1. `fmnist_no_ood/run_matrix.sh`: initial serial 30-configuration matrix.
2. `fmnist_no_ood/run_matrix_8gpu.sh`: one queue worker per GPU.
3. `fmnist_no_ood/run_matrix_finalonly_8gpu.sh`: evaluate only after the final
   aggregation rather than after every local epoch.
4. `fmnist_no_ood/run_matrix_finalonly_2xgpu.sh`: allow two experiments per
   GPU and retry failures on another device.
5. `fmnist_no_ood/run_matrix_finalonly_7gpu.sh`: final in-process/GPU-cache
   runner, excluding GPU 1. This is the runner that produced the 30 results in
   `FMNIST_NO_OOD_FINAL_RESULTS.md`.
6. `fmnist_no_ood/monitor_status.py`: inspect queue counts, active processes,
   last observed rounds, stale logs, and GPU utilization.

## CIFAR-10 Gaussian noise

- `cifar10_gaussian_noise/run_cifar10_noise_20.sh`: main 20-configuration
  matrix, covering BA m=1/m=3, one/four noisy clients, and five aggregators.
- `cifar10_gaussian_noise/monitor_30min.sh`: periodic progress and GPU monitor.
- `cifar10_gaussian_noise/retry_failed_gpu2.sh`: original deferred retry plan.
- `cifar10_gaussian_noise/retry_failed_now.sh`: immediate retries moved away
  from GPU 2 after CUDA failures.
- `cifar10_gaussian_noise/run_moved_pending_gpu7.sh`: one pending m=3 degree
  run moved from GPU 0 to GPU 7 during queue rebalancing.

## High-degree-node backdoor placement

`ba33_backdoor_top4/run_ba33_backdoor_top4_m1.py` schedules the six aggregation
methods for BA(33, m=1), with BadNets data placed on clients 0, 2, 5, and 8.
Those are the four highest-degree nodes in the saved m=1 graph.

## Output and state files

The launchers create manifests, per-GPU queue files, failure TSVs, stdout and
stderr logs, result directories, and Parsl run directories beneath each
script's `BASE` path. Retry and migration scripts expect those state files to
exist. Generated logs, checkpoints, and datasets are intentionally excluded
from Git.

The final result reports and consolidated client CSV are stored at the
repository root. The report-generation commands were initially run as inline
analysis commands, so they were not standalone launch scripts in the original
run.
