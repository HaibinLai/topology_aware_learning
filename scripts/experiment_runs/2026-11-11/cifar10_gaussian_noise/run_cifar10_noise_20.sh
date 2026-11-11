#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_cifar10_gaussian_noise_20261111
DATA=/ssd/haibin/datasets
GPUS=(0 3 4 5 6 7)
mkdir -p "$BASE/results" "$BASE/stdout" "$BASE/parsl" "$BASE/queues" /ssd/haibin/tmp
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
: > "$BASE/manifest.tsv"
idx=0
for graph in m1 m3; do
  for placement in one four; do
    for agg in random random_betCent random_degCent betCent degCent; do
      gpu=${GPUS[$((idx % ${#GPUS[@]}))]}
      printf '%s\t%s\t%s\t%s\n' "$gpu" "$graph" "$placement" "$agg" >> "$BASE/manifest.tsv"
      idx=$((idx+1))
    done
  done
done
for gpu in "${GPUS[@]}"; do awk -F '\t' -v g="$gpu" '$1==g {print $2 "\t" $3 "\t" $4}' "$BASE/manifest.tsv" > "$BASE/queues/gpu_${gpu}.tsv"; done
: > "$BASE/queue.log"
: > "$BASE/failed.tsv"
run_one() {
  gpu=$1 graph=$2 placement=$3 agg=$4
  if [[ "$graph" == m1 ]]; then topo=src/create_topo/topology/topo_ba33.txt; else topo=src/create_topo/topology/topo_ba33_m3.txt; fi
  if [[ "$placement" == one ]]; then nodes=(14); else nodes=(14 15 24 31); fi
  nodes_csv=$(IFS=-; echo "${nodes[*]}")
  run_id="cifar10_ba33_${graph}_gaussian_std01_mix10_nodes${nodes_csv}_${agg}_s1_finalonly"
  echo "$(date -Is) START gpu=$gpu $run_id" >> "$BASE/queue.log"
  if CUDA_VISIBLE_DEVICES="$gpu" TOPO_FL_LOCAL_THREADS=8 TOPO_FL_FINAL_EVAL_ONLY=1 TOPO_FL_INPROCESS=1 TOPO_FL_PIN_MEMORY=0 TOPO_FL_VALIDATE_TARGET_TRANSFER=1 \
    TOPO_FL_PARSL_RUN_DIR="$BASE/parsl/parsl_runinfo_${run_id}" \
    TOPO_FL_PARSL_LOG="$BASE/parsl/parsl_${run_id}.log" \
    "$PY" -m src.experiments.decentralized_main \
      --rounds 40 --checkpoint_every 5 --epochs 5 --batch_size 16 \
      --lr 0.0001 --optimizer adam --weight_decay 0 \
      --participation 1.0 --sample_alpha 1000 --label_alpha 1000 \
      --aggregation_strategy "$agg" \
      --dataset cifar10_vgg --topology_file "$topo" \
      --noise_node_idxs "${nodes[@]}" --noise_std 0.1 --noise_mix_ratio 0.1 \
      --out_dir "$BASE/results" --data_dir "$DATA" \
      --run_id "$run_id" --seed 1 --parsl_executor local \
      >"$BASE/stdout/${run_id}.stdout.log" 2>"$BASE/stdout/${run_id}.stderr.log"; then
    echo "$(date -Is) DONE gpu=$gpu $run_id" >> "$BASE/queue.log"
  else
    code=$?
    echo "$(date -Is) FAIL gpu=$gpu exit=$code $run_id" >> "$BASE/queue.log"
    printf '%s\t%s\t%s\t%s\n' "$gpu" "$graph" "$placement" "$agg" >> "$BASE/failed.tsv"
  fi
}
worker() {
  gpu=$1; batch=()
  while IFS=$'\t' read -r graph placement agg; do
    [[ -n "$graph" ]] || continue
    run_one "$gpu" "$graph" "$placement" "$agg" & batch+=("$!")
    if (( ${#batch[@]} == 2 )); then
      for pid in "${batch[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done < "$BASE/queues/gpu_${gpu}.tsv"
  for pid in "${batch[@]}"; do wait "$pid" || true; done
}
pids=()
for gpu in "${GPUS[@]}"; do worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || true; done
if [[ ! -s "$BASE/failed.tsv" ]]; then echo "$(date -Is) ALL_DONE" >> "$BASE/queue.log"; else echo "$(date -Is) FAILED=$(wc -l < "$BASE/failed.tsv")" >> "$BASE/queue.log"; fi
