from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot K-shot comparisons from registry evaluation records.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--records", nargs="*", type=Path, default=[])
    parser.add_argument("--records-dir", type=Path, default=None)
    parser.add_argument("--comparison-run-id", required=True)
    parser.add_argument("--model-keys", nargs="*", default=None)
    return parser.parse_args()

# Example usage:
# python scripts\plot_kshot_registry_comparison.py --comparison-run-id 20260530_raw_csi_domain_cross_vs_baselines --records-dir experiments\registry\evaluations --model-keys 20260530_raw_csi_domain_cross_proto raw_csi_mixed_proto doppler_featuremap_proto doppler_softmax_baseline doppler_pooled_head_proto

def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.experiments.artifacts import save_figure
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, SOURCE_DOMAINS
    from wifi_doppler.experiments.registry import load_record, make_record, record_dir, record_id, save_record

    records = _load_records(project_root, args.records, args.records_dir, load_record)
    records = [record for record in records if record["record_type"] == "evaluation.kshot"]
    if args.model_keys:
        records = [record for record in records if record["model"].get("model_id") in set(args.model_keys)]
    if not records:
        raise ValueError("No evaluation.kshot records found.")

    by_protocol = _group_records(records)
    comparison_id = record_id("comparison.figure", args.comparison_run_id)
    placeholder = make_record(
        record_type="comparison.figure",
        explicit_record_id=comparison_id,
        source={"mode": "computed", "legacy_paths": []},
        protocol={"name": args.comparison_run_id},
    )
    run_dir = record_dir(project_root, placeholder)
    run_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    if "mixed_source" in by_protocol:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_protocol(ax, by_protocol["mixed_source"], title="Mixed-source K-shot", ylabel="query accuracy")
        ax.legend()
        fig.tight_layout()
        plot_paths["mixed_source"] = save_figure(fig, run_dir, "mixed_source_kshot_comparison.png")

    for target_protocol in ("same_domain_PI-4a",):
        if target_protocol in by_protocol:
            fig, ax = plt.subplots(figsize=(8, 4.5))
            _plot_protocol(ax, by_protocol[target_protocol], title="Target same-domain K-shot: PI-4a", ylabel="query accuracy")
            ax.legend()
            fig.tight_layout()
            plot_paths["target_pi4"] = save_figure(fig, run_dir, "target_pi4_kshot_comparison.png")

    if all(f"same_domain_{domain}" in by_protocol for domain in SOURCE_DOMAINS):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        _plot_source_average(ax, by_protocol, SOURCE_DOMAINS)
        ax.legend()
        fig.tight_layout()
        plot_paths["source_same_domain_average"] = save_figure(fig, run_dir, "source_same_domain_average_kshot_comparison.png")

    if all(f"same_domain_{domain}" in by_protocol for domain in ALL_DOMAINS):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        for ax, domain in zip(axes.flat, ALL_DOMAINS, strict=True):
            _plot_protocol(ax, by_protocol[f"same_domain_{domain}"], title=domain, ylabel="accuracy")
        for ax in axes.flat:
            ax.set_xlabel("K")
            ax.set_ylabel("accuracy")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("Per-domain same-domain K-shot", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        plot_paths["per_domain"] = save_figure(fig, run_dir, "per_domain_kshot_comparison.png")

    record = make_record(
        record_type="comparison.figure",
        explicit_record_id=comparison_id,
        source={"mode": "computed", "legacy_paths": []},
        model={"model_ids": sorted({record["model"].get("model_id") for record in records})},
        dataset={"protocols": sorted(by_protocol)},
        protocol={"name": args.comparison_run_id},
        metrics={"input_record_ids": [record["record_id"] for record in records]},
        artifacts={
            "plots": plot_paths,
            "run_dir": run_dir,
            "input_records": [str(_record_source_path(project_root, record)) for record in records],
        },
    )
    record_path = save_record(project_root, record)
    print("run directory:", run_dir)
    for name, path in plot_paths.items():
        print(f"{name}: {path}")
    print("comparison record:", record_path)


def _load_records(project_root: Path, explicit: list[Path], records_dir: Path | None, load_record):
    paths = list(explicit)
    if records_dir is None:
        records_dir = project_root / "experiments" / "registry" / "evaluations"
    if records_dir.exists():
        paths.extend(sorted(records_dir.rglob("*.json")))
    unique_paths = list(dict.fromkeys(path.resolve() for path in paths))
    return [load_record(path) for path in unique_paths]


def _group_records(records: list[dict]) -> dict[str, dict[str, dict]]:
    grouped = defaultdict(dict)
    for record in records:
        protocol_name = record["protocol"].get("name")
        model_id = record["model"].get("model_id")
        if not protocol_name or not model_id:
            raise ValueError(f"Record missing protocol/model id: {record['record_id']}")
        if model_id in grouped[protocol_name]:
            raise ValueError(f"Duplicate record for protocol={protocol_name}, model={model_id}.")
        grouped[protocol_name][model_id] = record
    return dict(grouped)


def _record_source_path(project_root: Path, record: dict) -> Path:
    from wifi_doppler.experiments.registry import record_path

    return record_path(project_root, record).resolve()


def _plot_protocol(ax, records_by_model: dict[str, dict], *, title: str, ylabel: str) -> None:
    for model_id, record in sorted(records_by_model.items()):
        k_values, means, stds = _series(record)
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=3, label=record["model"].get("label", model_id))
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xticks(_series(next(iter(records_by_model.values())))[0])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("K enrollment windows per person")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both")


def _plot_source_average(ax, by_protocol: dict[str, dict[str, dict]], source_domains: tuple[str, ...]) -> None:
    model_ids = sorted(set.intersection(*(set(by_protocol[f"same_domain_{domain}"]) for domain in source_domains)))
    for model_id in model_ids:
        k_values = _series(by_protocol[f"same_domain_{source_domains[0]}"][model_id])[0]
        means = []
        stds = []
        for k_idx, _ in enumerate(k_values):
            domain_values = [
                _series(by_protocol[f"same_domain_{domain}"][model_id])[1][k_idx]
                for domain in source_domains
            ]
            means.append(float(np.mean(domain_values)))
            stds.append(float(np.std(domain_values)))
        label = by_protocol[f"same_domain_{source_domains[0]}"][model_id]["model"].get("label", model_id)
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=3, label=label)
    ax.set_title("Source same-domain K-shot average")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("K enrollment windows per person")
    ax.set_ylabel("source same-domain query accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both")


def _series(record: dict) -> tuple[list[int], list[float], list[float]]:
    results = record["metrics"]["results"]
    k_values = [int(k) for k in record["metrics"]["k_values"]]
    means = [float(results[str(k)]["mean"]) for k in k_values]
    stds = [float(results[str(k)]["std"]) for k in k_values]
    return k_values, means, stds


if __name__ == "__main__":
    main()
