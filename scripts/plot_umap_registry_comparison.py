from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAP projection records from the registry.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--records", nargs="+", type=Path, required=True)
    parser.add_argument("--comparison-run-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.experiments.artifacts import save_figure
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, DEFAULT_PERSONS
    from wifi_doppler.experiments.registry import load_record, make_record, record_dir, record_id, save_record

    records = [load_record(path) for path in args.records]
    for record in records:
        if record["record_type"] != "projection.umap":
            raise ValueError(f"Expected projection.umap, got {record['record_type']} in {record['record_id']}")

    comparison_id = record_id("comparison.figure", args.comparison_run_id)
    placeholder = make_record(
        record_type="comparison.figure",
        explicit_record_id=comparison_id,
        source={"mode": "computed", "legacy_paths": []},
        protocol={"name": args.comparison_run_id},
    )
    run_dir = record_dir(project_root, placeholder)
    run_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(records), 2, figsize=(13, 4.5 * len(records)), squeeze=False)
    person_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(DEFAULT_PERSONS)))
    domain_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(ALL_DOMAINS)))
    person_to_color = dict(zip(DEFAULT_PERSONS, person_colors, strict=True))
    domain_to_color = dict(zip(ALL_DOMAINS, domain_colors, strict=True))

    for row, record in enumerate(records):
        data = np.load(record["artifacts"]["coordinates"], allow_pickle=False)
        coords = data["umap"]
        persons = data["persons"].astype(str)
        domains = data["domains"].astype(str)
        label = record["model"].get("label", record["model"].get("model_id", record["record_id"]))

        ax = axes[row, 0]
        for person in DEFAULT_PERSONS:
            mask = persons == person
            ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.55, color=person_to_color[person], label=person, linewidths=0)
        ax.set_title(f"{label} colored by person")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row, 1]
        for domain in ALL_DOMAINS:
            mask = domains == domain
            ax.scatter(coords[mask, 0], coords[mask, 1], s=7, alpha=0.55, color=domain_to_color[domain], label=domain, linewidths=0)
        ax.set_title(f"{label} colored by domain")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].legend(loc="best", fontsize=7, markerscale=2, ncol=2)
    axes[0, 1].legend(loc="best", fontsize=8, markerscale=2)
    fig.suptitle("UMAP of PI embeddings", y=0.995)
    fig.tight_layout()
    plot_path = save_figure(fig, run_dir, "embedding_umap_person_domain.png", dpi=180)

    comparison = make_record(
        record_type="comparison.figure",
        explicit_record_id=comparison_id,
        source={"mode": "computed", "legacy_paths": []},
        model={"model_ids": [record["model"].get("model_id") for record in records]},
        dataset={"projection_record_ids": [record["record_id"] for record in records]},
        protocol={"name": args.comparison_run_id},
        metrics={},
        artifacts={"plot": plot_path, "input_records": [str(path.resolve()) for path in args.records]},
    )
    record_path = save_record(project_root, comparison)
    print("plot:", plot_path)
    print("comparison record:", record_path)


if __name__ == "__main__":
    main()
