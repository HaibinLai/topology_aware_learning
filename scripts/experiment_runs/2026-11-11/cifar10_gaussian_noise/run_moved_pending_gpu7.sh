#!/usr/bin/env bash
set -uo pipefail
ROOT=/home/haibin/topo_fl/topology_aware_learning
PY=/ssd/haibin/miniconda3/envs/topo/bin/python
BASE=/ssd/haibin/logs/topology_aware_learning_cifar10_gaussian_noise_20261111
DATA=/ssd/haibin/datasets
cd "$ROOT"
run_id=cifar10_ba33_m3_gaussian_std01_mix10_nodes14_degCent_s1_finalonly
echo "$(date -Is) START gpu=7 moved_from_gpu=0 $run_id" >> "$BASE/queue.log"
if CUDA_VISIBLE_DEVICES=7 TOPO_FL_LOCAL_THREADS=8 TOPO_FL_FINAL_EVAL_ONLY=1 TOPO_FL_INPROCESS=1 TOPO_FL_PIN_MEMORY=0 TOPO_FL_VALIDATE_TARGET_TRANSFER=1 \
  TOPO_FL_PARSL_RUN_DIR="$BASE/parsl/parsl_runinfo_${run_id}" \
  TOPO_FL_PARSL_LOG="$BASE/parsl/parsl_${run_id}.log" \
  "$PY" -m src.experiments.decentralized_main \
    --rounds 40 --checkpoint_every 5 --epochs 5 --batch_size 16 \
    --lr 0.0001 --optimizer adam --weight_decay 0 \
    --participation 1.0 --sample_alpha 1000 --label_alpha 1000 \
    --aggregation_strategy degCent --dataset cifar10_vgg \
    --topology_file src/create_topo/topology/topo_ba33_m3.txt \
    --noise_node_idxs 14 --noise_std 0.1 --noise_mix_ratio 0.1 \
    --out_dir "$BASE/results" --data_dir "$DATA" \
    --run_id "$run_id" --seed 1 --parsl_executor local \
    >"$BASE/stdout/${run_id}.stdout.log" 2>"$BASE/stdout/${run_id}.stderr.log"; then
  echo "$(date -Is) DONE gpu=7 $run_id" >> "$BASE/queue.log"
else
  code=$?
  echo "$(date -Is) FAIL gpu=7 exit=$code $run_id" >> "$BASE/queue.log"
  printf '7\tm3\tone\tdegCent\n' >> "$BASE/failed.tsv"
fi
