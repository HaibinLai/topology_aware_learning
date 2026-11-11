#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_cifar10_gaussian_noise_20261111
DATA=/ssd/haibin/datasets
GPUS=(3 4 6)
cd "$ROOT"
: > "$BASE/retry_queue.log"
: > "$BASE/retry_failed.tsv"
n=0
pids=()
while IFS=$'\t' read -r old_gpu graph placement agg; do
  [[ -n "$graph" ]] || continue
  gpu=${GPUS[$((n % ${#GPUS[@]}))]}
  if [[ "$graph" == m1 ]]; then topo=src/create_topo/topology/topo_ba33.txt; else topo=src/create_topo/topology/topo_ba33_m3.txt; fi
  if [[ "$placement" == one ]]; then nodes=(14); else nodes=(14 15 24 31); fi
  nodes_csv=$(IFS=-; echo "${nodes[*]}")
  run_id="cifar10_ba33_${graph}_gaussian_std01_mix10_nodes${nodes_csv}_${agg}_s1_finalonly"
  echo "$(date -Is) RETRY_START gpu=$gpu old_gpu=$old_gpu $run_id" >> "$BASE/retry_queue.log"
  (
    if CUDA_VISIBLE_DEVICES="$gpu" TOPO_FL_LOCAL_THREADS=8 TOPO_FL_FINAL_EVAL_ONLY=1 TOPO_FL_INPROCESS=1 TOPO_FL_PIN_MEMORY=0 TOPO_FL_VALIDATE_TARGET_TRANSFER=1 \
      TOPO_FL_PARSL_RUN_DIR="$BASE/parsl/retry_parsl_runinfo_${run_id}" \
      TOPO_FL_PARSL_LOG="$BASE/parsl/retry_parsl_${run_id}.log" \
      "$PY" -m src.experiments.decentralized_main \
        --rounds 40 --checkpoint_every 5 --epochs 5 --batch_size 16 \
        --lr 0.0001 --optimizer adam --weight_decay 0 \
        --participation 1.0 --sample_alpha 1000 --label_alpha 1000 \
        --aggregation_strategy "$agg" --dataset cifar10_vgg \
        --topology_file "$topo" --noise_node_idxs "${nodes[@]}" \
        --noise_std 0.1 --noise_mix_ratio 0.1 \
        --out_dir "$BASE/results" --data_dir "$DATA" \
        --run_id "$run_id" --seed 1 --parsl_executor local \
        >"$BASE/stdout/retry_${run_id}.stdout.log" 2>"$BASE/stdout/retry_${run_id}.stderr.log"; then
      echo "$(date -Is) RETRY_DONE gpu=$gpu $run_id" >> "$BASE/retry_queue.log"
    else
      code=$?
      echo "$(date -Is) RETRY_FAIL gpu=$gpu exit=$code $run_id" >> "$BASE/retry_queue.log"
      printf '%s\t%s\t%s\t%s\n' "$gpu" "$graph" "$placement" "$agg" >> "$BASE/retry_failed.tsv"
    fi
  ) & pids+=("$!")
  n=$((n+1))
done < "$BASE/failed.tsv"
for pid in "${pids[@]}"; do wait "$pid" || true; done
echo "$(date -Is) RETRY_ALL_DONE" >> "$BASE/retry_queue.log"
