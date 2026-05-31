from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create/update one run folder and compute its PI few-shot artifacts.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--label", default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--training-history", type=Path, default=None)
    parser.add_argument("--training-curves", type=Path, default=None)
    parser.add_argument("--training-objective", default=None)
    parser.add_argument("--episode-style", default=None)
    parser.add_argument("--notes", default=None)
    parser.add_argument("--protocols", nargs="+", default=["mixed_source", "same_domain_PI-1a", "same_domain_PI-2a", "same_domain_PI-3a", "same_domain_PI-4a"])
    parser.add_argument("--baseline-run-ids", nargs="*", default=[])
    parser.add_argument("--comparison-name", default="kshot_vs_baselines")
    parser.add_argument("--umap-comparison-name", default="umap_vs_baselines")
    parser.add_argument("--umap-baseline-run-ids", nargs="*", default=[])
    parser.add_argument("--compute-umap", action="store_true")
    parser.add_argument("--plot-kshot", action="store_true")
    parser.add_argument("--plot-umap", action="store_true")
    parser.add_argument("--skip-kshot", action="store_true")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 25, 50, 100])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=340)
    parser.add_argument("--window-stride", type=int, default=30)
    parser.add_argument("--split-guard", type=int, default=31)
    parser.add_argument("--embedding-fusion", default="mean")
    parser.add_argument("--metric", default="cosine")
    parser.add_argument("--projection-name", default="umap_all_domains_balanced")
    parser.add_argument("--sample-per-person-domain", type=int, default=75)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.experiments.model_builders import ModelSpec, model_spec_from_key
    from wifi_doppler.experiments.protocols import DEFAULT_PERSONS
    from wifi_doppler.experiments.runs import copy_training_artifacts, ensure_run, load_run, run_checkpoint_path

    base_spec = model_spec_from_key(project_root, args.model_key, checkpoint_path=args.checkpoint)
    run_path = ensure_run(
        project_root,
        model_run_id=args.model_run_id,
        label=args.label,
        model_key=args.model_key,
        representation=base_spec.representation,
        builder=base_spec.builder,
        checkpoint_path=args.checkpoint,
        training_objective=args.training_objective,
        episode_style=args.episode_style,
        notes=args.notes,
    )
    if args.training_history or args.training_curves:
        copied = copy_training_artifacts(
            project_root,
            args.model_run_id,
            history_path=args.training_history,
            curves_path=args.training_curves,
        )
        print("training artifacts:", copied)
    spec = ModelSpec(
        key=args.model_run_id,
        label=load_run(project_root, args.model_run_id).get("label", args.model_run_id),
        representation=base_spec.representation,
        checkpoint_path=run_checkpoint_path(project_root, args.model_run_id),
        builder=base_spec.builder,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.skip_kshot:
        for protocol_name in args.protocols:
            path = evaluate_kshot(
                project_root=project_root,
                spec=spec,
                protocol_name=protocol_name,
                device=device,
                k_values=tuple(args.k_values),
                n_trials=args.n_trials,
                seed=args.seed,
                batch_size=args.batch_size,
                window_size=args.window_size,
                window_stride=args.window_stride,
                split_guard=args.split_guard,
                embedding_fusion=args.embedding_fusion,
                metric=args.metric,
            )
            print("evaluation:", path)

    if args.compute_umap:
        path = extract_umap(
            project_root=project_root,
            spec=spec,
            device=device,
            projection_name=args.projection_name,
            sample_per_group=args.sample_per_person_domain,
            batch_size=args.batch_size,
            seed=args.seed,
            pca_components=args.pca_components,
            umap_neighbors=args.umap_neighbors,
            umap_min_dist=args.umap_min_dist,
            embedding_fusion=args.embedding_fusion,
        )
        print("projection:", path)

    if args.plot_kshot:
        run_ids = [args.model_run_id, *args.baseline_run_ids]
        path = plot_kshot_comparison(project_root, args.model_run_id, args.comparison_name, run_ids)
        print("kshot comparison:", path)

    if args.plot_umap:
        run_ids = [args.model_run_id, *args.umap_baseline_run_ids]
        path = plot_umap_comparison(project_root, args.model_run_id, args.umap_comparison_name, run_ids, args.projection_name)
        print("umap comparison:", path)

    print("run directory:", run_path)


def evaluate_kshot(
    *,
    project_root: Path,
    spec,
    protocol_name: str,
    device: torch.device,
    k_values: tuple[int, ...],
    n_trials: int,
    seed: int,
    batch_size: int,
    window_size: int,
    window_stride: int,
    split_guard: int,
    embedding_fusion: str,
    metric: str,
) -> Path:
    from wifi_doppler.evaluation.fewshot import evaluate_kshot
    from wifi_doppler.experiments.artifacts import save_json
    from wifi_doppler.experiments.model_builders import load_model_from_spec
    from wifi_doppler.experiments.protocols import DEFAULT_PERSONS, build_kshot_datasets, parse_kshot_protocol
    from wifi_doppler.experiments.runs import evaluation_path, utc_now

    protocol = parse_kshot_protocol(protocol_name)
    model, _ = load_model_from_spec(
        spec,
        device=device,
        num_classes=len(DEFAULT_PERSONS),
        window_size=window_size,
        embedding_fusion=embedding_fusion,
    )
    enrollment_dataset, query_dataset, dataset = build_kshot_datasets(
        project_root=project_root,
        representation=spec.representation,
        protocol=protocol,
        persons=DEFAULT_PERSONS,
        window_size=window_size,
        window_stride=window_stride,
        split_guard=split_guard,
    )
    results = evaluate_kshot(
        model,
        enrollment_dataset,
        query_dataset,
        k_values,
        device,
        n_trials=n_trials,
        seed=seed,
        batch_size=batch_size,
        embedding_fusion=embedding_fusion,
        metric=metric,
    )
    enrollment_dataset.clear_cache()
    query_dataset.clear_cache()
    record = {
        "record_type": "evaluation.kshot",
        "created_at": utc_now(),
        "model_run_id": spec.key,
        "protocol": protocol.to_dict(),
        "dataset": dataset,
        "parameters": {
            "k_values": list(k_values),
            "n_trials": n_trials,
            "seed": seed,
            "batch_size": batch_size,
            "embedding_fusion": embedding_fusion,
            "metric": metric,
        },
        "results": {
            str(k): {
                "mean": float(value["mean"]),
                "std": float(value["std"]),
                "trials": [float(v) for v in value["trials"]],
            }
            for k, value in results.items()
        },
    }
    return save_json(evaluation_path(project_root, spec.key, protocol.name), record)


def extract_umap(
    *,
    project_root: Path,
    spec,
    device: torch.device,
    projection_name: str,
    sample_per_group: int,
    batch_size: int,
    seed: int,
    pca_components: int,
    umap_neighbors: int,
    umap_min_dist: float,
    embedding_fusion: str,
) -> Path:
    from sklearn.decomposition import PCA
    from umap import UMAP

    from wifi_doppler.experiments.artifacts import save_json
    from wifi_doppler.experiments.model_builders import load_model_from_spec
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, DEFAULT_PERSONS
    from wifi_doppler.experiments.runs import projection_coordinates_path, projection_record_path, utc_now
    from wifi_doppler.representation.embeddings import extract_embeddings

    rng = np.random.default_rng(seed)
    model, _ = load_model_from_spec(spec, device=device, num_classes=len(DEFAULT_PERSONS), embedding_fusion=embedding_fusion)
    dataset, data_root = build_projection_dataset(project_root, spec.representation, domains=ALL_DOMAINS, persons=DEFAULT_PERSONS)
    indices, domains, persons = balanced_sample_indices(dataset, rng=rng, sample_per_group=sample_per_group, domains=ALL_DOMAINS, persons=DEFAULT_PERSONS)
    embeddings, _ = extract_embeddings(model, dataset, device, batch_size=batch_size, indices=indices, embedding_fusion=embedding_fusion)
    dataset.clear_cache()

    x = embeddings.numpy().astype(np.float32, copy=False)
    if x.shape[1] > pca_components:
        n_components = min(pca_components, x.shape[0] - 1, x.shape[1])
        x_for_umap = PCA(n_components=n_components, random_state=seed).fit_transform(x)
    else:
        n_components = x.shape[1]
        x_for_umap = x
    coords = UMAP(
        n_components=2,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        metric="euclidean",
        random_state=seed,
    ).fit_transform(x_for_umap)

    npz_path = projection_coordinates_path(project_root, spec.key, projection_name)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        indices=indices,
        domains=domains.astype(str),
        persons=persons.astype(str),
        embeddings=x,
        input_to_umap=x_for_umap.astype(np.float32, copy=False),
        umap=coords.astype(np.float32),
    )
    record = {
        "record_type": "projection.umap",
        "created_at": utc_now(),
        "model_run_id": spec.key,
        "projection_name": projection_name,
        "dataset": {
            "representation": spec.representation,
            "data_root": str(data_root.resolve()),
            "domains": list(ALL_DOMAINS),
            "persons": list(DEFAULT_PERSONS),
            "split": [0.0, 0.8],
            "sampled_windows": int(indices.size),
        },
        "parameters": {
            "sample_per_person_domain": sample_per_group,
            "pca_components": int(n_components),
            "umap_neighbors": umap_neighbors,
            "umap_min_dist": umap_min_dist,
            "seed": seed,
        },
        "artifacts": {"coordinates": str(npz_path.resolve())},
    }
    return save_json(projection_record_path(project_root, spec.key, projection_name), record)


def build_projection_dataset(project_root: Path, representation: str, *, domains: tuple[str, ...], persons: tuple[str, ...]):
    from wifi_doppler.data.doppler_dataset import DopplerWindowDataset
    from wifi_doppler.data.raw_csi_dataset import RawCsiWindowDataset

    if representation == "raw_csi":
        data_root = project_root / "data" / "raw_csi_traces_pi"
        return RawCsiWindowDataset(
            data_root,
            scenarios=list(domains),
            split=(0.0, 0.8),
            labels=persons,
            flatten_channels=True,
            cache_traces=True,
        ), data_root
    if representation == "doppler":
        data_root = project_root / "data" / "doppler_traces_pi"
        return DopplerWindowDataset(
            data_root,
            scenarios=list(domains),
            split=(0.0, 0.8),
            labels=persons,
        ), data_root
    raise ValueError(f"Unknown representation: {representation}")


def balanced_sample_indices(dataset, *, rng: np.random.Generator, sample_per_group: int, domains: tuple[str, ...], persons: tuple[str, ...]):
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for idx, window in enumerate(dataset.window_indexes):
        trace = dataset.traces[window.recording_idx]
        groups[(trace.scenario, trace.ground_truth)].append(idx)

    selected_indices = []
    selected_domains = []
    selected_persons = []
    for domain in domains:
        for person in persons:
            candidates = np.asarray(groups[(domain, person)], dtype=np.int64)
            if candidates.size == 0:
                raise ValueError(f"No windows found for domain={domain}, person={person}.")
            sample_size = min(sample_per_group, candidates.size)
            sampled = rng.choice(candidates, size=sample_size, replace=False)
            selected_indices.extend(sampled.tolist())
            selected_domains.extend([domain] * sample_size)
            selected_persons.extend([person] * sample_size)

    order = rng.permutation(len(selected_indices))
    return (
        np.asarray(selected_indices, dtype=np.int64)[order],
        np.asarray(selected_domains, dtype=object)[order],
        np.asarray(selected_persons, dtype=object)[order],
    )


def plot_kshot_comparison(project_root: Path, owner_run_id: str, comparison_name: str, run_ids: list[str]) -> Path:
    from wifi_doppler.experiments.artifacts import save_figure, save_json
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, SOURCE_DOMAINS
    from wifi_doppler.experiments.runs import comparison_dir, evaluation_path, load_json, load_run, utc_now

    records = []
    for run_id in run_ids:
        for path in sorted((project_root / "experiments" / "runs" / run_id / "evaluations").glob("*.json")):
            records.append(load_json(path))
    by_protocol: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        protocol = record["protocol"]["name"]
        run_id = record["model_run_id"]
        by_protocol[protocol][run_id] = record

    out_dir = comparison_dir(project_root, owner_run_id, comparison_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = {}
    if "mixed_source" in by_protocol:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_protocol(ax, project_root, by_protocol["mixed_source"], title="Mixed-source K-shot", ylabel="query accuracy")
        ax.legend()
        fig.tight_layout()
        plots["mixed_source"] = str(save_figure(fig, out_dir, "mixed_source_kshot_comparison.png").resolve())

    if "same_domain_PI-4a" in by_protocol:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_protocol(ax, project_root, by_protocol["same_domain_PI-4a"], title="Target same-domain K-shot: PI-4a", ylabel="query accuracy")
        ax.legend()
        fig.tight_layout()
        plots["target_pi4"] = str(save_figure(fig, out_dir, "target_pi4_kshot_comparison.png").resolve())

    if all(f"same_domain_{domain}" in by_protocol for domain in SOURCE_DOMAINS):
        fig, ax = plt.subplots(figsize=(8, 4.5))
        plot_source_average(ax, project_root, by_protocol, SOURCE_DOMAINS)
        ax.legend()
        fig.tight_layout()
        plots["source_same_domain_average"] = str(save_figure(fig, out_dir, "source_same_domain_average_kshot_comparison.png").resolve())

    if all(f"same_domain_{domain}" in by_protocol for domain in ALL_DOMAINS):
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
        for ax, domain in zip(axes.flat, ALL_DOMAINS, strict=True):
            plot_protocol(ax, project_root, by_protocol[f"same_domain_{domain}"], title=domain, ylabel="accuracy")
        for ax in axes.flat:
            ax.set_xlabel("K")
            ax.set_ylabel("accuracy")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2)
        fig.suptitle("Per-domain same-domain K-shot", y=0.98)
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        plots["per_domain"] = str(save_figure(fig, out_dir, "per_domain_kshot_comparison.png").resolve())

    record = {
        "record_type": "comparison.kshot",
        "created_at": utc_now(),
        "owner_run_id": owner_run_id,
        "run_ids": run_ids,
        "input_records": [
            str(evaluation_path(project_root, record["model_run_id"], record["protocol"]["name"]).resolve())
            for record in records
        ],
        "plots": plots,
    }
    return save_json(out_dir / "record.json", record)


def plot_protocol(ax, project_root: Path, records_by_run: dict[str, dict[str, Any]], *, title: str, ylabel: str) -> None:
    for run_id, record in sorted(records_by_run.items()):
        k_values, means, stds = series(record)
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=3, label=load_run_label(project_root, run_id))
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_xticks(series(next(iter(records_by_run.values())))[0])
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("K enrollment windows per person")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.grid(True, which="both")


def plot_source_average(ax, project_root: Path, by_protocol: dict[str, dict[str, dict[str, Any]]], source_domains: tuple[str, ...]) -> None:
    run_ids = sorted(set.intersection(*(set(by_protocol[f"same_domain_{domain}"]) for domain in source_domains)))
    for run_id in run_ids:
        k_values = series(by_protocol[f"same_domain_{source_domains[0]}"][run_id])[0]
        means = []
        stds = []
        for k_idx, _ in enumerate(k_values):
            domain_values = [
                series(by_protocol[f"same_domain_{domain}"][run_id])[1][k_idx]
                for domain in source_domains
            ]
            means.append(float(np.mean(domain_values)))
            stds.append(float(np.std(domain_values)))
        ax.errorbar(k_values, means, yerr=stds, marker="o", capsize=3, label=load_run_label(project_root, run_id))
    ax.set_title("Source same-domain K-shot average")
    ax.set_xscale("log")
    ax.set_xticks(k_values)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("K enrollment windows per person")
    ax.set_ylabel("source same-domain query accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, which="both")


def plot_umap_comparison(project_root: Path, owner_run_id: str, comparison_name: str, run_ids: list[str], projection_name: str) -> Path:
    from wifi_doppler.experiments.artifacts import save_figure, save_json
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, DEFAULT_PERSONS
    from wifi_doppler.experiments.runs import comparison_dir, load_json, projection_record_path, utc_now

    records = [load_json(projection_record_path(project_root, run_id, projection_name)) for run_id in run_ids]
    out_dir = comparison_dir(project_root, owner_run_id, comparison_name)
    out_dir.mkdir(parents=True, exist_ok=True)
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
        label = load_run_label(project_root, record["model_run_id"])

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
    plot_path = save_figure(fig, out_dir, "embedding_umap_person_domain.png", dpi=180)
    record = {
        "record_type": "comparison.umap",
        "created_at": utc_now(),
        "owner_run_id": owner_run_id,
        "run_ids": run_ids,
        "input_records": [str(projection_record_path(project_root, run_id, projection_name).resolve()) for run_id in run_ids],
        "plots": {"umap": str(plot_path.resolve())},
    }
    return save_json(out_dir / "record.json", record)


def series(record: dict[str, Any]) -> tuple[list[int], list[float], list[float]]:
    k_values = [int(k) for k in record["parameters"]["k_values"]]
    means = [float(record["results"][str(k)]["mean"]) for k in k_values]
    stds = [float(record["results"][str(k)]["std"]) for k in k_values]
    return k_values, means, stds


def load_run_label(project_root: Path, run_id: str) -> str:
    from wifi_doppler.experiments.runs import load_run

    metadata = load_run(project_root, run_id)
    return str(metadata.get("label") or run_id)


if __name__ == "__main__":
    main()
