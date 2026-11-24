#!/usr/bin/env bash

set -uo pipefail

repo_dir="/home/haibin/topo_fl/topology_aware_learning"
python_bin="/ssd/haibin/miniconda3/envs/topo/bin/python"
output_root="${OUTPUT_ROOT:-/ssd/haibin/logs/topology_aware_learning_fmnist_additional_ba_m3_20261124}"
results_dir="${output_root}/results"
stdout_dir="${output_root}/stdout"
status_log="${output_root}/queue.log"

mkdir -p "${results_dir}" "${stdout_dir}" "${output_root}/parsl"
cd "${repo_dir}" || exit 1

# Two independent experiment processes per usable GPU. GPUs 1 and 2 are
# intentionally excluded because they have been unreliable on this host.
slot_gpus=(0 0 3 3 4 4 5 5 6 6 7 7)
tasks=()

# Six no-OOD additions: three label alphas x MH/unweighted.
for alpha in 1 10 1000; do
    for method in mhCent unweighted; do
        tasks+=("noood|${alpha}|${method}")
    done
done

# Twenty-one four-node impulse-noise additions.
for alpha in 1 10 1000; do
    for method in mhCent unweighted random betCent degCent random_betCent random_degCent; do
        tasks+=("noise4|${alpha}|${method}")
    done
done

printf '%s CONFIG branch=%s commit=%s tasks=%s slots=%s gpus=0,3,4,5,6,7 topology=BA_m3 rounds=40 epochs=5 batch=16 optimizer=adam lr=1e-4 seed=1 sample_alpha=1000 noise_type=impulse noise_nodes=23,27,28,30 noise_proportion=0.1 noise_severity=5\n' \
    "$(date --iso-8601=seconds)" "$(git branch --show-current)" "$(git rev-parse HEAD)" \
    "${#tasks[@]}" "${#slot_gpus[@]}" >> "${status_log}"

run_task() {
    local slot="$1"
    local gpu="$2"
    local task_index="$3"
    local spec="${tasks[$task_index]}"
    local condition alpha method run_id stdout_log started exit_code elapsed
    IFS='|' read -r condition alpha method <<< "${spec}"

    if [[ "${condition}" == "noood" ]]; then
        run_id="fmnist_ba33_m3_la${alpha}_${method}_noood_s1"
    else
        run_id="fmnist_ba33_m3_la${alpha}_${method}_noise4n23-27-28-30_sev5_s1"
    fi
    stdout_log="${stdout_dir}/${run_id}.log"
    started="$(date +%s)"

    printf '%s START slot=%s gpu=%s task=%s/27 condition=%s alpha=%s method=%s run_id=%s\n' \
        "$(date --iso-8601=seconds)" "${slot}" "${gpu}" "$((task_index + 1))" \
        "${condition}" "${alpha}" "${method}" "${run_id}" >> "${status_log}"

    command=(
        "${python_bin}" -m src.experiments.decentralized_main
        --rounds 40
        --checkpoint_every 5
        --epochs 5
        --batch_size 16
        --dataset fmnist
        --topology_file src/create_topo/topology/topo_ba33_m3.txt
        --aggregation_strategy "${method}"
        --label_alpha "${alpha}"
        --sample_alpha 1000
        --seed 1
        --lr 0.0001
        --optimizer adam
        --weight_decay 0
        --prox_coeff 0
        --participation 1
        --out_dir "${results_dir}"
        --data_dir /ssd/haibin/datasets
        --parsl_executor local
        --run_id "${run_id}"
    )
    if [[ "${condition}" == "noise4" ]]; then
        command+=(
            --ood_type noise
            --ood_node_idxs 23 27 28 30
            --ood_proportion 0.1
            --ood_severity 5
        )
    fi

    env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        TOPO_FL_INPROCESS=1 \
        TOPO_FL_FINAL_EVAL_ONLY=1 \
        TOPO_FL_LOCAL_THREADS=8 \
        TOPO_FL_PARSL_RUN_DIR="${output_root}/parsl/${run_id}" \
        PYTHONUNBUFFERED=1 \
        "${command[@]}" > "${stdout_log}" 2>&1
    exit_code=$?
    elapsed="$(( $(date +%s) - started ))"

    printf '%s DONE slot=%s gpu=%s task=%s/27 condition=%s alpha=%s method=%s run_id=%s exit=%s elapsed_sec=%s\n' \
        "$(date --iso-8601=seconds)" "${slot}" "${gpu}" "$((task_index + 1))" \
        "${condition}" "${alpha}" "${method}" "${run_id}" "${exit_code}" \
        "${elapsed}" >> "${status_log}"
    return "${exit_code}"
}

worker() {
    local slot="$1"
    local gpu="${slot_gpus[$slot]}"
    local task_index failures=0
    for ((task_index=slot; task_index<${#tasks[@]}; task_index+=${#slot_gpus[@]})); do
        if ! run_task "${slot}" "${gpu}" "${task_index}"; then
            failures=$((failures + 1))
        fi
    done
    return "${failures}"
}

pids=()
for slot in "${!slot_gpus[@]}"; do
    worker "${slot}" &
    pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
        failures=$((failures + 1))
    fi
done

printf '%s ALL_DONE worker_failures=%s\n' "$(date --iso-8601=seconds)" "${failures}" \
    >> "${status_log}"
exit "${failures}"
