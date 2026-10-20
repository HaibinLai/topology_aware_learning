# Experiment Commands

All commands below are run from the repository root after activating the environment:

```bash
cd ~/topo_fl/topology_aware_learning
conda activate topo
```
The topology used in the initial CIFAR-10 experiments was:

```text
src/create_topo/topology/topo_8.txt
```

## Initial 5-round baseline

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u -m src.experiments.decentralized_main \
  --dataset cifar10 \
  --topology_file src/create_topo/topology/topo_8.txt \
  --parsl_executor local \
  --aggregation_strategy unweighted
```

## Full runs: batch size 256, learning rate 1e-4

```bash
CUDA_VISIBLE_DEVICES=3 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy unweighted --run_id gpu3_unweighted_b256_r40_e5_lr1e4_v3 --batch_size 256 --rounds 40 --epochs 5 --lr 1e-4

CUDA_VISIBLE_DEVICES=4 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy random --run_id gpu4_random_b256_r40_e5_lr1e4_v3 --batch_size 256 --rounds 40 --epochs 5 --lr 1e-4

CUDA_VISIBLE_DEVICES=5 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy betCent --run_id gpu5_betCent_b256_r40_e5_lr1e4_v3 --batch_size 256 --rounds 40 --epochs 5 --lr 1e-4

CUDA_VISIBLE_DEVICES=6 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy degCent --run_id gpu6_degCent_b256_r40_e5_lr1e4_v3 --batch_size 256 --rounds 40 --epochs 5 --lr 1e-4
```

## Full runs: batch size 16, learning rate 1e-3

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy unweighted --run_id gpu0_unweighted_b16_r40_e5_lr1e3_v4 --batch_size 16 --rounds 40 --epochs 5 --lr 1e-3

CUDA_VISIBLE_DEVICES=1 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy random --run_id gpu1_random_b16_r40_e5_lr1e3_v4 --batch_size 16 --rounds 40 --epochs 5 --lr 1e-3

CUDA_VISIBLE_DEVICES=2 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy betCent --run_id gpu2_betCent_b16_r40_e5_lr1e3_v4 --batch_size 16 --rounds 40 --epochs 5 --lr 1e-3

CUDA_VISIBLE_DEVICES=7 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy degCent --run_id gpu7_degCent_b16_r40_e5_lr1e3_v4 --batch_size 16 --rounds 40 --epochs 5 --lr 1e-3
```

## Combined centrality experiments

The implemented combination is an equal-weight sum:

```text
combined = 0.5 * random + 0.5 * centrality
```

```bash
CUDA_VISIBLE_DEVICES=2 python -u -m src.experiments.decentralized_main --dataset cifar10 --topology_file src/create_topo/topology/topo_8.txt --parsl_executor local --aggregation_strategy random_betCent --run_id gpu2_random_betCent_b16_r40_e5_lr1e3_v1 --batch_size 16 --rounds 40 --epochs 5 --lr 1e-3
```

This run stopped with a CUDA illegal-memory-access error on GPU 2. `random_degCent` has not yet been run.

## Backdoor multi-node template

The newer multi-client backdoor interface is:

```bash
CUDA_VISIBLE_DEVICES=0 python -u -m src.experiments.decentralized_main \
  --dataset cifar10 \
  --topology_file src/create_topo/topology/topo_8.txt \
  --parsl_executor local \
  --aggregation_strategy unweighted \
  --backdoor \
  --backdoor_node_idxs 1 7 12 20 \
  --run_id bd_nodes_1_7_12_20_seed0
```
