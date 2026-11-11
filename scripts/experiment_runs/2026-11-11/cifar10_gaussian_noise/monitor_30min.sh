#!/usr/bin/env bash
set -uo pipefail
BASE=/ssd/haibin/logs/topology_aware_learning_cifar10_gaussian_noise_20261111
OUT="$BASE/monitor_30min.log"
while true; do
  {
    echo "=== $(date -Is) ==="
    starts=$(rg -c ' START ' "$BASE/queue.log" || true)
    dones=$(rg -c ' DONE ' "$BASE/queue.log" || true)
    fails=$(rg -c ' FAIL ' "$BASE/queue.log" || true)
    echo "starts=${starts:-0} done=${dones:-0} fail=${fails:-0}"
    nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
    echo "progress_logs=$(rg -l 'TRAIN_EPOCH_DONE|FINAL_EVAL_DONE' "$BASE/stdout" -g '*.stdout.log' | wc -l)"
    if rg -q 'ALL_DONE|^.* FAILED=' "$BASE/queue.log"; then
      echo "monitor_exit=queue_finished"
      exit 0
    fi
  } >> "$OUT" 2>&1
  sleep 1800
done
