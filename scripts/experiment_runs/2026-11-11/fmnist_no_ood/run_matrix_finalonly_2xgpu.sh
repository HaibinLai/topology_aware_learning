#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_fmnist_no_noise_20261111
DATA=/ssd/haibin/datasets
mkdir -p "$BASE/results" "$BASE/stdout" "$BASE/parsl" "$BASE/queues" "$DATA" /ssd/haibin/tmp
cd "$ROOT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
: > "$BASE/manifest_2xgpu.tsv"
idx=0
for graph in m1 m3; do
  for alpha in 1 10 1000; do
    for agg in random random_betCent random_degCent betCent degCent; do
      # Offset assignment so the configurations that errored on GPU 1 are
      # initially placed on different devices.
      printf '%s\t%s\t%s\t%s\n' "$(((idx + 3) % 8))" "$graph" "$alpha" "$agg" >> "$BASE/manifest_2xgpu.tsv"
      idx=$((idx+1))
    done
  done
done
for gpu in {0..7}; do awk -F '\t' -v g="$gpu" '$1==g {print $2 "\t" $3 "\t" $4}' "$BASE/manifest_2xgpu.tsv" > "$BASE/queues/2xgpu_${gpu}.tsv"; done
run_one() {
  gpu=$1 graph=$2 alpha=$3 agg=$4 suffix=$5
  if [[ "$graph" == m1 ]]; then topo=src/create_topo/topology/topo_ba33.txt; else topo=src/create_topo/topology/topo_ba33_m3.txt; fi
  run_id="fmnist_ba33_${graph}_la${alpha}_${agg}_s1_finalonly_${suffix}"
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
    printf '%s\t%s\t%s\t%s\n' "$gpu" "$graph" "$alpha" "$agg" >> "$BASE/failed_2xgpu.tsv"
  fi
}
worker() {
  gpu=$1
  batch=()
  while IFS=$'\t' read -r graph alpha agg; do
    [[ -n "$graph" ]] || continue
    run_one "$gpu" "$graph" "$alpha" "$agg" "2xgpu" >/dev/null 2>&1 &
    batch+=("$!")
    if (( ${#batch[@]} == 2 )); then
      for pid in "${batch[@]}"; do wait "$pid" || true; done
      batch=()
    fi
  done < "$BASE/queues/2xgpu_${gpu}.tsv"
  for pid in "${batch[@]}"; do wait "$pid" || true; done
}
: > "$BASE/failed_2xgpu.tsv"
pids=()
for gpu in {0..7}; do worker "$gpu" & pids+=("$!"); done
for pid in "${pids[@]}"; do wait "$pid" || true; done

# Retry failed configurations once on a different GPU, in pairs where possible.
if [[ -s "$BASE/failed_2xgpu.tsv" ]]; then
  cp "$BASE/failed_2xgpu.tsv" "$BASE/retry_2xgpu.tsv"
  : > "$BASE/failed_2xgpu.tsv"
  retry_pids=()
  n=0
  while IFS=$'\t' read -r old_gpu graph alpha agg; do
    [[ -n "$graph" ]] || continue
    gpu=$(((old_gpu + n + 1) % 8))
    run_one "$gpu" "$graph" "$alpha" "$agg" "retry2xgpu" >"$BASE/stdout/retry_${n}_2xgpu.log" 2>&1 &
    retry_pids+=("$!")
    n=$((n+1))
  done < "$BASE/retry_2xgpu.tsv"
  for pid in "${retry_pids[@]}"; do wait "$pid" || true; done
fi
if [[ ! -s "$BASE/failed_2xgpu.tsv" ]]; then
  echo "$(date -Is) ALL_DONE" >> "$BASE/queue.log"
else
  echo "$(date -Is) RETRIES_REMAIN=$(wc -l < "$BASE/failed_2xgpu.tsv")" >> "$BASE/queue.log"
fi
