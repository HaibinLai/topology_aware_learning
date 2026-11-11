#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_fmnist_no_noise_20261111
DATA=/ssd/haibin/datasets
mkdir -p "$BASE/results" "$BASE/stdout" "$BASE/parsl" "$BASE/queues" "$DATA" /ssd/haibin/tmp
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
# Split the 30 configurations round-robin across eight GPU-pinned workers.
: > "$BASE/manifest.tsv"
idx=0
for graph in m1 m3; do
  for alpha in 1 10 1000; do
    for agg in random random_betCent random_degCent betCent degCent; do
      printf '%s\t%s\t%s\t%s\n' "$((idx % 8))" "$graph" "$alpha" "$agg" >> "$BASE/manifest.tsv"
      idx=$((idx+1))
    done
  done
done
for gpu in {0..7}; do awk -F '\t' -v g="$gpu" '$1==g {print $2 "\t" $3 "\t" $4}' "$BASE/manifest.tsv" > "$BASE/queues/gpu${gpu}.tsv"; done
worker() {
  gpu=$1
  while IFS=$'\t' read -r graph alpha agg; do
    [[ -n "$graph" ]] || continue
    if [[ "$graph" == m1 ]]; then topo=src/create_topo/topology/topo_ba33.txt; else topo=src/create_topo/topology/topo_ba33_m3.txt; fi
    run_id="fmnist_ba33_${graph}_la${alpha}_${agg}_s1_finalonly_gpu8"
    echo "$(date -Is) START gpu=$gpu $run_id" >> "$BASE/queue.log"
    if CUDA_VISIBLE_DEVICES="$gpu" TOPO_FL_LOCAL_THREADS=8 TOPO_FL_FINAL_EVAL_ONLY=1 \
      TOPO_FL_PARSL_RUN_DIR="$BASE/parsl/parsl_runinfo_${run_id}" \
      TOPO_FL_PARSL_LOG="$BASE/parsl/parsl_${run_id}.log" \
      "$PY" -m src.experiments.decentralized_main \
        --rounds 40 --checkpoint_every 5 --epochs 5 --batch_size 16 \
        --lr 0.0001 --optimizer adam --weight_decay 0 \
        --participation 1.0 --sample_alpha 1000 --label_alpha "$alpha" \
        --dataset fmnist --topology_file "$topo" \
        --out_dir "$BASE/results" --data_dir "$DATA" \
        --run_id "$run_id" --seed 1 --parsl_executor local \
        >"$BASE/stdout/${run_id}.stdout.log" 2>"$BASE/stdout/${run_id}.stderr.log"; then
      echo "$(date -Is) DONE gpu=$gpu $run_id" >> "$BASE/queue.log"
    else
      code=$?
      echo "$(date -Is) FAIL gpu=$gpu exit=$code $run_id" >> "$BASE/queue.log"
    fi
  done < "$BASE/queues/gpu${gpu}.tsv"
}
pids=()
for gpu in {0..7}; do worker "$gpu" >"$BASE/stdout/gpu${gpu}_worker.log" 2>&1 & pids+=("$!"); done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if (( status == 0 )); then echo "$(date -Is) ALL_DONE" >> "$BASE/queue.log"; fi
exit "$status"
