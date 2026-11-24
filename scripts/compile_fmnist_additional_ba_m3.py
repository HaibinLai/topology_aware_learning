#!/usr/bin/env python3
"""Validate and summarize the 27 FMNIST BA-m3 additional experiments."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ALPHAS = [1, 10, 1000]
NO_OOD_METHODS = ["mhCent", "unweighted"]
NOISE_METHODS = [
    "mhCent",
    "unweighted",
    "random",
    "betCent",
    "degCent",
    "random_betCent",
    "random_degCent",
]
DISPLAY = {
    "mhCent": "Metropolis-Hastings",
    "unweighted": "Unweighted",
    "random": "Random",
    "betCent": "BetCent",
    "degCent": "DegCent",
    "random_betCent": "BetCent+Random",
    "random_degCent": "DegCent+Random",
}
FINAL_METRICS = [
    "test_acc",
    "clean_auc_macro",
    "clean_auc_weighted",
    "ood_acc",
    "ood_auc_macro",
    "ood_auc_weighted",
]


def parse_runtime(queue_path: Path) -> dict[str, int]:
    runtimes: dict[str, int] = {}
    pattern = re.compile(r"run_id=(\S+).*exit=0 elapsed_sec=(\d+)")
    for line in queue_path.read_text().splitlines():
        match = pattern.search(line)
        if match:
            runtimes[match.group(1)] = int(match.group(2))
    return runtimes


def expected_runs() -> set[tuple[str, int, str]]:
    expected = {
        ("no-OOD", alpha, method)
        for alpha in ALPHAS
        for method in NO_OOD_METHODS
    }
    expected |= {
        ("noise-4-node", alpha, method)
        for alpha in ALPHAS
        for method in NOISE_METHODS
    }
    return expected


def identify(args: dict) -> tuple[str, int, str]:
    if args.get("ood_type") not in (None, "noise"):
        raise RuntimeError(f"Unexpected OOD type: {args.get('ood_type')}")
    condition = "noise-4-node" if args.get("ood_type") == "noise" else "no-OOD"
    return condition, int(args["label_alpha"]), args["aggregation_strategy"]


def validate_common_args(args: dict, run_id: str) -> None:
    expected = {
        "dataset": "fmnist",
        "topology_path": "topo_ba33_m3.txt",
        "batch_size": 16,
        "epochs": 5,
        "lr": 0.0001,
        "sample_alpha": 1000.0,
        "seed": 1,
        "optimizer": "adam",
        "final_eval_only": True,
        "inprocess_execution": True,
    }
    mismatches = {
        key: {"expected": value, "actual": args.get(key)}
        for key, value in expected.items()
        if args.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{run_id}: argument mismatch: {mismatches}")


def load_runs(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    runtimes = parse_runtime(root / "queue.log")
    summary_rows: list[dict] = []
    client_frames: list[pd.DataFrame] = []
    observed: set[tuple[str, int, str]] = set()

    for csv_path in sorted((root / "results").rglob("client_stats.csv")):
        args_path = csv_path.parent / "args.txt"
        args = json.loads(args_path.read_text())
        key = identify(args)
        if key in observed:
            raise RuntimeError(f"Duplicate completed configuration: {key}")
        observed.add(key)
        condition, alpha, method = key
        run_id = args["run_id"]
        validate_common_args(args, run_id)

        frame = pd.read_csv(csv_path)
        final = frame[
            frame["is_final_evaluation"].fillna(False).astype(bool)
            & (frame["round_idx"] == 39)
        ].copy()
        client_ids = set(final["client_idx"].astype(int))
        if len(final) != 33 or client_ids != set(range(33)):
            raise RuntimeError(
                f"{run_id}: expected final clients 0..32, found "
                f"{len(final)} rows and IDs {sorted(client_ids)}"
            )
        if condition == "noise-4-node":
            required = set(FINAL_METRICS)
            nodes = args.get("ood_node_idxs")
            if nodes != [23, 27, 28, 30]:
                raise RuntimeError(f"{run_id}: unexpected OOD nodes {nodes}")
            if args.get("ood_severity") != 5 or args.get("ood_proportion") != 0.1:
                raise RuntimeError(f"{run_id}: unexpected OOD settings")
        else:
            required = set(FINAL_METRICS[:3])
        missing = required - set(final.columns)
        if missing:
            raise RuntimeError(f"{run_id}: missing metrics {sorted(missing)}")
        if final[list(required)].isna().any().any():
            raise RuntimeError(f"{run_id}: null final metrics")
        if run_id not in runtimes:
            raise RuntimeError(f"{run_id}: missing successful runtime record")

        row = {
            "condition": condition,
            "label_alpha": alpha,
            "aggregation": DISPLAY[method],
            "aggregation_arg": method,
            "runtime_seconds": runtimes[run_id],
            "run_id": run_id,
            "client_csv": str(csv_path),
        }
        for metric in FINAL_METRICS:
            if metric in final.columns:
                row[metric] = float(final[metric].mean())
                row[f"{metric}_median"] = float(final[metric].median())
                row[f"{metric}_min"] = float(final[metric].min())
                row[f"{metric}_max"] = float(final[metric].max())
        summary_rows.append(row)

        clients = final[["client_idx", *[m for m in FINAL_METRICS if m in final]]].copy()
        clients.insert(0, "aggregation", DISPLAY[method])
        clients.insert(0, "label_alpha", alpha)
        clients.insert(0, "condition", condition)
        client_frames.append(clients)

    missing_runs = expected_runs() - observed
    extra_runs = observed - expected_runs()
    if missing_runs or extra_runs:
        raise RuntimeError(
            f"Matrix mismatch: missing={sorted(missing_runs)}, extra={sorted(extra_runs)}"
        )
    return pd.DataFrame(summary_rows), pd.concat(client_frames, ignore_index=True)


def table(frame: pd.DataFrame, noise: bool) -> str:
    lines = []
    ordered_methods = NOISE_METHODS if noise else NO_OOD_METHODS
    for alpha in ALPHAS:
        for method in ordered_methods:
            row = frame[
                (frame["label_alpha"] == alpha)
                & (frame["aggregation_arg"] == method)
            ].iloc[0]
            if noise:
                lines.append(
                    f"| {alpha} | {row.aggregation} | {row.test_acc:.6f} | "
                    f"{row.clean_auc_macro:.6f} | {row.clean_auc_weighted:.6f} | "
                    f"{row.ood_acc:.6f} | {row.ood_auc_macro:.6f} | "
                    f"{row.ood_auc_weighted:.6f} |"
                )
            else:
                lines.append(
                    f"| {alpha} | {row.aggregation} | {row.test_acc:.6f} | "
                    f"{row.clean_auc_macro:.6f} | {row.clean_auc_weighted:.6f} |"
                )
    return "\n".join(lines)


def make_plot(summary: pd.DataFrame, output: Path) -> None:
    no_ood = summary[summary["condition"] == "no-OOD"]
    noise = summary[summary["condition"] == "noise-4-node"]
    figure, axes = plt.subplots(2, 2, figsize=(16, 10))

    x = np.arange(len(ALPHAS))
    width = 0.35
    for index, method in enumerate(NO_OOD_METHODS):
        subset = no_ood[no_ood["aggregation_arg"] == method].set_index("label_alpha")
        values_acc = [subset.loc[a, "test_acc"] for a in ALPHAS]
        values_auc = [subset.loc[a, "clean_auc_macro"] for a in ALPHAS]
        axes[0, 0].bar(x + (index - 0.5) * width, values_acc, width, label=DISPLAY[method])
        axes[0, 1].bar(x + (index - 0.5) * width, values_auc, width, label=DISPLAY[method])
    for axis, title in zip(axes[0], ["No-OOD Clean Acc", "No-OOD Clean AUC"]):
        axis.set_title(title)
        axis.set_xticks(x, [str(a) for a in ALPHAS])
        axis.set_xlabel("Label alpha")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    method_x = np.arange(len(NOISE_METHODS))
    width = 0.25
    for index, alpha in enumerate(ALPHAS):
        subset = noise[noise["label_alpha"] == alpha].set_index("aggregation_arg")
        clean = [subset.loc[m, "clean_auc_macro"] for m in NOISE_METHODS]
        ood = [subset.loc[m, "ood_auc_macro"] for m in NOISE_METHODS]
        offset = (index - 1) * width
        axes[1, 0].bar(method_x + offset, clean, width, label=f"alpha={alpha}")
        axes[1, 1].bar(method_x + offset, ood, width, label=f"alpha={alpha}")
    for axis, title in zip(axes[1], ["Noise: Clean AUC", "Noise: OOD AUC"]):
        axis.set_title(title)
        axis.set_xticks(method_x, [DISPLAY[m] for m in NOISE_METHODS], rotation=25, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.legend()

    figure.suptitle("FMNIST BA m=3 additional experiments")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(
            "/ssd/haibin/logs/"
            "topology_aware_learning_fmnist_additional_ba_m3_20261124"
        ),
    )
    parser.add_argument(
        "--publish-dir",
        type=Path,
        default=None,
        help="Optional repository directory for report, CSVs, and comparison plot.",
    )
    args = parser.parse_args()
    summary, clients = load_runs(args.root)
    summary = summary.sort_values(["condition", "label_alpha", "aggregation_arg"])
    summary.to_csv(args.root / "final_summary.csv", index=False)
    clients.to_csv(args.root / "per_client_results.csv", index=False)
    make_plot(summary, args.root / "final_results.png")

    no_ood = summary[summary["condition"] == "no-OOD"]
    noise = summary[summary["condition"] == "noise-4-node"]
    best_clean = noise.loc[noise["clean_auc_macro"].idxmax()]
    best_ood = noise.loc[noise["ood_auc_macro"].idxmax()]
    runtime = summary["runtime_seconds"]
    report = f"""# FMNIST BA m=3 additional experiments

Updated: 2026-11-24

## Configuration

- 33 clients; BA m=3; 40 rounds; 5 local epochs; batch size 16
- Adam, learning rate 1e-4; seed 1; sample alpha 1000
- Label alpha: 1, 10, 1000
- Noise: impulse noise, severity 5, proportion 0.1
- Noise nodes: 23, 27, 28, 30
- Values are means over 33 client records at final round 39
- Raw output root: `{args.root}`

## No-OOD additions

| Label alpha | Aggregation | Clean Acc | Clean AUC macro | Clean AUC weighted |
|---:|---|---:|---:|---:|
{table(no_ood, noise=False)}

## Four-node noise experiments

| Label alpha | Aggregation | Clean Acc | Clean AUC macro | Clean AUC weighted | OOD Acc | OOD AUC macro | OOD AUC weighted |
|---:|---|---:|---:|---:|---:|---:|---:|
{table(noise, noise=True)}

## Highlights

- Best noise-condition Clean AUC: **{best_clean.clean_auc_macro:.6f}** —
  alpha={int(best_clean.label_alpha)}, {best_clean.aggregation}.
- Best OOD AUC: **{best_ood.ood_auc_macro:.6f}** —
  alpha={int(best_ood.label_alpha)}, {best_ood.aggregation}.
- Runtime per run: mean {runtime.mean() / 60:.2f} min, range
  {runtime.min() / 60:.2f}–{runtime.max() / 60:.2f} min.
- All 27 configurations passed the 33-client, round-39, non-null metric audit.

![Final experiment comparison](final_results.png)

## Artifacts

- `final_summary.csv`: exact configuration-level metrics and distributions.
- `per_client_results.csv`: all 27 × 33 final client records.
- `results/`: raw CSVs, arguments, logs, and checkpoints.
- `queue.log`: start/completion status and runtimes.
"""
    report_path = args.root / "FINAL_RESULTS.md"
    report_path.write_text(report)

    if args.publish_dir is not None:
        args.publish_dir.mkdir(parents=True, exist_ok=True)
        for filename in (
            "FINAL_RESULTS.md",
            "final_summary.csv",
            "per_client_results.csv",
            "final_results.png",
        ):
            shutil.copy2(args.root / filename, args.publish_dir / filename)


if __name__ == "__main__":
    main()
