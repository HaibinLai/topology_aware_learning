# CIFAR-10 topology-aware learning experiment report

Generated: 2026-10-17T21:50:43.161130+00:00

## Common configuration

- Dataset/model: CIFAR-10 with VGG16 (`cifar10_vgg`)
- Topologies: BA-33 or SB-low-connect-33; all clients participate (`participation=1.0`)
- 40 rounds, 5 local epochs, batch size 16
- Adam, learning rate 1e-4, weight decay 0, seed 1
- Backdoor source nodes: 14, 15, 24, 31
- `label_alpha=1000`, `sample_alpha=1000`; checkpoint every 3 rounds
- AUC: multiclass one-vs-rest ROC-AUC
- All logs/results: `/ssd/haibin/logs/topology_aware_learning_20261016/`

## Latest recorded metrics

| Run | Topology | Aggregation | Status | Round | Checkpoint | Clean acc | Clean AUC macro | Clean AUC weighted | Backdoor acc | Backdoor AUC macro |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| ba33_bd_betCent_s1_v2_auc | BA-33 | betCent | completed/ended | 39 | 39 | 0.6705151200294495 | 0.9376048834019205 | 0.9376048834019204 | 0.5615079402923584 | nan |
| ba33_bd_random_betCent_s1_v2_auc | BA-33 | random+betCent | completed/ended | 39 | 39 | 0.6762877702713013 | 0.9396751577503428 | 0.9396751577503429 | 0.2579365372657776 | nan |
| ba33_bd_random_s1_v2_auc | BA-33 | random | completed/ended | 39 | 39 | 0.5566163659095764 | 0.8897486419753087 | 0.8897486419753087 | 0.1150793731212616 | nan |
| ba33_unweighted | BA-33 | unweighted | completed/ended | 39 | 39 | 0.4730239808559418 | 0.8619326337448558 | 0.861932633744856 | 0.0704365149140358 | nan |
| ba33_bd_degCent_s1_v2_auc | BA-33 | degCent | running | 19 | 17 | 0.6437610983848572 | 0.9283942729766803 | 0.9283942729766804 | 0.2757936716079712 | nan |
| sb_low_connect_bd_betCent_s1_auc | SB-low-connect-33 | betCent | running | 19 | 17 | 0.5867007374763489 | 0.9096205967078189 | 0.9096205967078189 | 0.0873015895485878 | nan |

| ba33_bd_random_degCent_s1_v2_auc | BA-33 | random+degCent | running | 0 | - | - | - | - | - | - |
| sb_low_connect_bd_unweighted_s1_auc | SB-low-connect-33 | unweighted | running | 0 | - | - | - | - | - | - |

## Caveats

- Values are the latest client-level test values printed in each run log, not necessarily a final global aggregate.
- Backdoor AUC may be `NaN` when the backdoor test set contains one class only; ROC-AUC requires at least two classes.
- Clean accuracy/AUC higher is better. Backdoor metrics measure propagation strength for this study.

## Checkpoint root

`/ssd/haibin/logs/topology_aware_learning_20261016/results/`
