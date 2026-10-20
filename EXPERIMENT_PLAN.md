# OOD Knowledge Propagation Experiments

## Purpose

Study how OOD or backdoor knowledge propagates across decentralized clients under BA communication graphs, and compare topology-aware aggregation methods.

## Testbed

- **Topology:** Barabási–Albert (BA) graphs.
- **Topology sizes currently in scope:** 33 nodes only. BA-64 is deferred.
- **Dataset:** CIFAR-10.
- **Participation:** `P = 1`; all clients participate in every aggregation round.
- **OOD placement:** randomly select 4 nodes at once as OOD/backdoor source nodes for each run.
- **Repetitions:** 3 random seeds per configuration.

Each run should use a fresh run ID so that checkpoints from different configurations cannot be reused.

## Confirmed topology files

The currently confirmed graph files are:

| graph name | file | node count | edges | graph type/status |
|---|---|---:|---:|---|
| BA-33 | `src/create_topo/topology/topo_ba33.txt` | 33 | 32 | connected, symmetric, `P=1` confirmed |
| SB-low-connect-33 | `src/create_topo/topology/topo_sb_low_connect.txt` | 33 | 78 | connected, symmetric, `P=1` confirmed; exact generator type should remain documented |

BA-64 is intentionally not part of the current sweep.

## OOD conditions

The planned OOD conditions are:

1. **BD / backdoor:** source nodes receive backdoored examples. Evaluate both normal accuracy and backdoor attack success rate (ASR).
2. **Blur:** source nodes receive blurred images. This is intended as a harder-to-learn OOD distribution. Evaluate OOD accuracy and propagation speed.

Unless a separate experiment matrix is specified, BD and Blur should be treated as separate condition sets.

## Aggregation methods

Six aggregation methods are compared:

1. `unweighted` / uniform neighbor averaging
2. `betweenness` centrality weighting
3. `degree` centrality weighting
4. `random` weighting
5. `random + betweenness`
6. `random + degree`

The random-centrality variants should document whether randomness is applied once per run or regenerated every round. The seed must be recorded.

## Experiment matrix

There are 12 test groups before choosing an aggregation method. The base matrix is:

```text
topology size × OOD condition × placement setting
```

The current planned dimensions are:

- 2 BA topology sizes: 33 and 64 nodes;
- 2 OOD conditions: BD and Blur;
- 3 random four-node placement settings/runs per condition.

This gives:

```text
2 topology sizes × 2 OOD conditions × 3 placement groups = 12 groups
```

Each of the 12 groups is run with all 6 aggregation methods:

```text
12 groups × 6 methods = 72 configurations
```

With 3 random seeds per configuration:

```text
72 configurations × 3 seeds = 216 runs
```

If the three placement groups are not intended as a separate experimental dimension, revise this matrix before launching the full sweep. In that case, the number of groups should be recalculated explicitly.

## Required metadata

Every result should record:

- topology type and node count;
- topology file or graph-generation seed;
- OOD condition (`BD` or `Blur`);
- the four selected OOD node IDs;
- aggregation method;
- random seed;
- round and local epoch settings;
- batch size and learning rate;
- participation rate;
- global accuracy;
- OOD accuracy;
- backdoor ASR for BD experiments;
- checkpoint and output directory.

## Recommended run settings

The paper-style baseline is:

```text
aggregation rounds: 40
local epochs: 5
participation: 1.0
random seeds: 3
```

Use a unique `--run_id` for every configuration and seed.

## Current code readiness

The repository currently supports the following pieces directly:

- BA topologies can be supplied as adjacency-matrix text files through `--topology_file`.
- CIFAR-10 is supported through `--dataset cifar10`.
- `P=1` is `--participation 1.0` (the default).
- `unweighted`, `random`, `betCent`, and `degCent` are available aggregation strategies.
- Single-node backdoor injection is available with `--backdoor` and `--backdoor_node_idx`.

The following pieces require code additions before the full matrix is valid:

- **Four simultaneous random backdoor nodes:** supported with `--backdoor_node_idxs 1 7 12 20`; the list-valued argument injects backdoor data into each listed client. The legacy single-node `--backdoor_node_idx` remains supported.
- **Blur condition:** no blur data transform or blur-specific evaluation is currently implemented. Add a deterministic blur transform and record a separate OOD test set.
- **Random + betweenness** and **random + degree:** these are not currently exposed as aggregation strategies. They need explicit combined weighting rules and CLI choices.
- **Three placement groups:** define exactly how the three groups are generated and recorded. A seed alone is not enough; save the four selected node IDs.
- The current CLI defaults `label_alpha` and `sample_alpha` to `100`; use `1000` explicitly for the paper-style IID component.

## Commands for currently supported runs

After generating the BA topology files, a currently supported single-node backdoor run can be launched as:

```bash
CUDA_VISIBLE_DEVICES=0 \
python -u -m src.experiments.decentralized_main \
  --dataset cifar10 \
  --topology_file src/create_topo/topology/ba_33.txt \
  --parsl_executor local \
  --aggregation_strategy unweighted \
  --participation 1.0 \
  --label_alpha 1000 \
  --sample_alpha 1000 \
  --backdoor \
  --backdoor_node_idx 0 \
  --rounds 40 \
  --epochs 5 \
  --run_id ba33_bd_unweighted_seed0
```

Change the topology file, aggregation strategy, node index, seed, and run ID for each supported configuration. Do not launch the 12 × 6 sweep until the multi-node BD, Blur, and combined aggregation implementations are added and tested.
