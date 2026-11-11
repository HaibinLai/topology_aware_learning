#!/usr/bin/env bash
set -euo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_fmnist_no_noise_20261111
DATA=/ssd/haibin/datasets
mkdir -p "$BASE/results" "$BASE/stdout" "$BASE/parsl" "$DATA" /ssd/haibin/tmp
cd "$ROOT"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for graph in m1 m3; do
  if [[ "$graph" == m1 ]]; then topo=src/create_topo/topology/topo_ba33.txt; else topo=src/create_topo/topology/topo_ba33_m3.txt; fi
  for alpha in 1 10 1000; do
    for agg in random random_betCent random_degCent betCent degCent; do
      run_id="fmnist_ba33_${graph}_la${alpha}_${agg}_s1"
      echo "$(date -Is) START $run_id" | tee -a "$BASE/queue.log"
      TOPO_FL_PARSL_RUN_DIR="$BASE/parsl/parsl_runinfo_${run_id}" \
      TOPO_FL_PARSL_LOG="$BASE/parsl/parsl_${run_id}.log" \
      "$PY" -m src.experiments.decentralized_main \
        --rounds 40 --checkpoint_every 5 --epochs 5 --batch_size 16 \
        --lr 0.0001 --optimizer adam --weight_decay 0 \
        --participation 1.0 --sample_alpha 1000 --label_alpha "$alpha" \
        --dataset fmnist --topology_file "$topo" \
        --out_dir "$BASE/results" --data_dir "$DATA" \
        --run_id "$run_id" --seed 1 --parsl_executor local \
        >"$BASE/stdout/${run_id}.stdout.log" 2>"$BASE/stdout/${run_id}.stderr.log"
      echo "$(date -Is) DONE $run_id" | tee -a "$BASE/queue.log"
    done
  done
done
echo "$(date -Is) ALL_DONE" | tee -a "$BASE/queue.log"
