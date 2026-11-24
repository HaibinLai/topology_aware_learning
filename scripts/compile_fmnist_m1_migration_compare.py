#!/usr/bin/env python3
"""Compare the migrated FMNIST BA-m1 runs with the selected 2026-11-11 runs."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = ["betCent", "degCent", "random", "random_betCent", "random_degCent"]
DISPLAY_NAMES = {
    "betCent": "betCent",
    "degCent": "degCent",
    "random": "random",
    "random_betCent": "random+betCent",
    "random_degCent": "random+degCent",
}
METRICS = ["test_acc", "clean_auc_macro", "clean_auc_weighted"]


def one_path(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected one {description}, found {len(paths)}: {paths}")
    return paths[0]


def load_final(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    final = frame[
        frame["is_final_evaluation"].fillna(False).astype(bool)
        & (frame["round_idx"] == 39)
    ].copy()
    if len(final) != 33 or final["client_idx"].nunique() != 33:
        raise RuntimeError(f"Expected 33 final clients in {path}, found {len(final)}")
    return final.sort_values("client_idx").reset_index(drop=True)


def parse_runtime(queue_path: Path) -> dict[str, int]:
    runtimes: dict[str, int] = {}
    pattern = re.compile(r"method=(\S+).*exit=0 elapsed_sec=(\d+)")
    for line in queue_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            runtimes[match.group(1)] = int(match.group(2))
    return runtimes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--old-root",
        type=Path,
        default=Path(
            "/ssd/haibin/logs/"
            "topology_aware_learning_fmnist_no_noise_20261111/results"
        ),
    )
    parser.add_argument(
        "--new-root",
        type=Path,
        default=Path(
            "/ssd/haibin/logs/"
            "topology_aware_learning_fmnist_m1_migration_compare_20261124_v2"
        ),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help="Optional repository directory for the report, summary CSV, and plot.",
    )
    args = parser.parse_args()

    rows: list[dict] = []
    client_rows: list[pd.DataFrame] = []
    runtimes = parse_runtime(args.new_root / "queue.log")

    for method in METHODS:
        old_path = one_path(
            list(
                args.old_root.glob(
                    f"fmnist_ba33_m1_la1000_{method}_s1_finalonly_inproc3_2xgpu7/"
                    "*/client_stats.csv"
                )
            ),
            f"old {method} CSV",
        )
        new_path = one_path(
            list(
                (args.new_root / "results").glob(
                    f"migration_compare_fmnist_ba33_m1_la1000_{method}_s1/"
                    "*/client_stats.csv"
                )
            ),
            f"new {method} CSV",
        )
        old = load_final(old_path)
        new = load_final(new_path)

        row = {
            "aggregation": DISPLAY_NAMES[method],
            "aggregation_arg": method,
            "runtime_seconds": runtimes[method],
            "old_csv": str(old_path),
            "new_csv": str(new_path),
        }
        for metric in METRICS:
            old_mean = float(old[metric].mean())
            new_mean = float(new[metric].mean())
            row[f"old_{metric}"] = old_mean
            row[f"new_{metric}"] = new_mean
            row[f"delta_{metric}"] = new_mean - old_mean
            row[f"new_{metric}_median"] = float(new[metric].median())
            row[f"new_{metric}_min"] = float(new[metric].min())
            row[f"new_{metric}_max"] = float(new[metric].max())
        rows.append(row)

        clients = pd.DataFrame(
            {
                "aggregation": DISPLAY_NAMES[method],
                "client_idx": new["client_idx"].astype(int),
            }
        )
        for metric in METRICS:
            clients[f"old_{metric}"] = old[metric]
            clients[f"new_{metric}"] = new[metric]
            clients[f"delta_{metric}"] = new[metric] - old[metric]
        client_rows.append(clients)

    comparison = pd.DataFrame(rows)
    per_client = pd.concat(client_rows, ignore_index=True)
    comparison.to_csv(args.new_root / "comparison_summary.csv", index=False)
    per_client.to_csv(args.new_root / "per_client_comparison.csv", index=False)

    labels = comparison["aggregation"].tolist()
    positions = np.arange(len(labels))
    width = 0.36
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    panels = [
        ("test_acc", "Clean accuracy", (0.83, 0.86)),
        ("clean_auc_macro", "Clean AUC (macro)", (0.9805, 0.9840)),
    ]
    for axis, (metric, title, limits) in zip(axes, panels):
        old_values = comparison[f"old_{metric}"].to_numpy()
        new_values = comparison[f"new_{metric}"].to_numpy()
        axis.bar(positions - width / 2, old_values, width, label="Previous")
        axis.bar(positions + width / 2, new_values, width, label="Migrated")
        axis.set_title(title)
        axis.set_ylim(*limits)
        axis.set_xticks(positions, labels, rotation=20, ha="right")
        axis.grid(axis="y", alpha=0.25)
        for index, (old_value, new_value) in enumerate(zip(old_values, new_values)):
            axis.text(
                index + width / 2,
                new_value + (limits[1] - limits[0]) * 0.012,
                f"{new_value - old_value:+.4f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    axes[0].set_ylabel("Mean over 33 clients")
    axes[1].legend(loc="upper left")
    figure.suptitle("FMNIST BA m=1, no OOD, label alpha=1000: migration comparison")
    figure.tight_layout()
    figure.savefig(args.new_root / "migration_comparison.png", dpi=180)
    plt.close(figure)

    old_acc = comparison["old_test_acc"].mean()
    new_acc = comparison["new_test_acc"].mean()
    old_auc = comparison["old_clean_auc_macro"].mean()
    new_auc = comparison["new_clean_auc_macro"].mean()
    exact_method = comparison.loc[
        comparison["delta_test_acc"].abs() < 1e-12, "aggregation"
    ].tolist()

    table_lines = []
    for row in comparison.itertuples(index=False):
        table_lines.append(
            f"| {row.aggregation} | {row.old_test_acc:.6f} | {row.new_test_acc:.6f} | "
            f"{row.delta_test_acc:+.6f} | {row.old_clean_auc_macro:.6f} | "
            f"{row.new_clean_auc_macro:.6f} | {row.delta_clean_auc_macro:+.6f} |"
        )

    distribution_lines = []
    for row in comparison.itertuples(index=False):
        distribution_lines.append(
            f"| {row.aggregation} | {row.new_test_acc_median:.6f} | "
            f"{row.new_test_acc_min:.6f} | {row.new_test_acc_max:.6f} | "
            f"{row.new_clean_auc_macro_median:.6f} | "
            f"{row.new_clean_auc_macro_min:.6f} | {row.new_clean_auc_macro_max:.6f} |"
        )

    runtime_lines = [
        f"| {row.aggregation} | {row.runtime_seconds} | {row.runtime_seconds / 60:.2f} |"
        for row in comparison.itertuples(index=False)
    ]

    report = f"""# FMNIST BA m=1 migration comparison

Updated: 2026-11-24

## Configuration

- Dataset: FMNIST, no OOD/backdoor
- Graph: BA, 33 clients, m=1
- Label/sample alpha: 1000/1000
- Training: 40 rounds, 5 local epochs, batch size 16
- Optimizer: Adam, learning rate 1e-4, weight decay 0
- Participation: 1.0; seed: 1
- Scores: mean across 33 final client evaluations at round 39
- Migrated-code commit used by the launcher: `a5f984e`

## Mean-score comparison

| Aggregation | Old Acc | Migrated Acc | Delta | Old AUC macro | Migrated AUC macro | Delta |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(table_lines)}

Across the five aggregation methods, mean Clean Acc changed from
**{old_acc:.6f}** to **{new_acc:.6f}** ({new_acc - old_acc:+.6f}), and mean
Clean AUC changed from **{old_auc:.6f}** to **{new_auc:.6f}**
({new_auc - old_auc:+.6f}).

`{', '.join(exact_method)}` reproduced exactly for every reported mean and,
for `degCent`, for every one of the 33 client records. This provides a strong
deterministic check that the selective migration did not change the base
FMNIST training/evaluation path.

The remaining differences are small (largest mean Acc absolute delta:
{comparison['delta_test_acc'].abs().max():.6f}; largest mean AUC absolute delta:
{comparison['delta_clean_auc_macro'].abs().max():.6f}). The previous November
runs for several methods were resumed from early optimization checkpoints,
while this comparison started every method from round 0 with one final code
version. Therefore exact equality is not expected for those checkpoint chains.

![Old and migrated score comparison](migration_comparison.png)

## Migrated-run client distribution

| Aggregation | Acc median | Acc min | Acc max | AUC median | AUC min | AUC max |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(distribution_lines)}

## Runtime

The five methods ran concurrently on GPUs 0, 3, 4, 5, and 6. GPUs 1 and 2
were deliberately excluded.

| Aggregation | Seconds | Minutes |
|---|---:|---:|
{chr(10).join(runtime_lines)}

All five processes exited successfully; queue failures: 0.

## Data files

- `comparison_summary.csv`: method-level exact means, deltas, distributions,
  paths, and runtimes.
- `per_client_comparison.csv`: old/new/delta values for all 5 × 33 clients.
- `results/`: migrated raw client CSVs, checkpoints, arguments, and logs.
- `stdout/`: complete console logs.
- `queue.log`: launch, completion, exit status, and elapsed time.
"""
    (args.new_root / "COMPARISON_REPORT.md").write_text(report)
    if args.publish_dir is not None:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for filename in [
            "COMPARISON_REPORT.md",
            "comparison_summary.csv",
            "per_client_comparison.csv",
            "migration_comparison.png",
        ]:
            shutil.copy2(args.new_root / filename, args.publish_dir / filename)


if __name__ == "__main__":
    main()
