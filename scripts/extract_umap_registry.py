from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract one model's UMAP projection as a registry record.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-run-id", required=True)
    parser.add_argument("--model-label", default=None)
    parser.add_argument("--projection-name", default="umap_all_domains_balanced")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--sample-per-person-domain", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--embedding-fusion", default="mean")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from sklearn.decomposition import PCA
    from umap import UMAP

    from wifi_doppler.experiments.model_builders import ModelSpec, load_model_from_spec, model_spec_from_key
    from wifi_doppler.experiments.protocols import ALL_DOMAINS, DEFAULT_PERSONS
    from wifi_doppler.experiments.registry import make_record, record_dir, record_id, save_record
    from wifi_doppler.representation.embeddings import extract_embeddings

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    base_spec = model_spec_from_key(project_root, args.model_key, checkpoint_path=args.checkpoint)
    spec = ModelSpec(
        key=args.model_run_id,
        label=args.model_label or args.model_run_id,
        representation=base_spec.representation,
        checkpoint_path=base_spec.checkpoint_path,
        builder=base_spec.builder,
    )
    model, _ = load_model_from_spec(spec, device=device, num_classes=len(DEFAULT_PERSONS), embedding_fusion=args.embedding_fusion)

    dataset, data_root = _build_projection_dataset(
        project_root,
        representation=spec.representation,
        domains=ALL_DOMAINS,
        persons=DEFAULT_PERSONS,
    )
    indices, domains, persons = _balanced_sample_indices(
        dataset,
        rng=rng,
        sample_per_group=args.sample_per_person_domain,
        domains=ALL_DOMAINS,
        persons=DEFAULT_PERSONS,
    )
    embeddings, _ = extract_embeddings(
        model,
        dataset,
        device,
        batch_size=args.batch_size,
        indices=indices,
        embedding_fusion=args.embedding_fusion,
    )
    x = embeddings.numpy().astype(np.float32, copy=False)
    if x.shape[1] > args.pca_components:
        n_components = min(args.pca_components, x.shape[0] - 1, x.shape[1])
        x_for_umap = PCA(n_components=n_components, random_state=args.seed).fit_transform(x)
    else:
        n_components = x.shape[1]
        x_for_umap = x

    coords = UMAP(
        n_components=2,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric="euclidean",
        random_state=args.seed,
    ).fit_transform(x_for_umap)

    rid = record_id("projection.umap", f"{spec.key}_{args.projection_name}")
    placeholder = make_record(
        record_type="projection.umap",
        explicit_record_id=rid,
        source={"mode": "computed", "legacy_paths": []},
        model=spec.to_record_model(),
        protocol={"name": args.projection_name},
    )
    output_dir = record_dir(project_root, placeholder)
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "coordinates.npz"
    np.savez_compressed(
        npz_path,
        indices=indices,
        domains=domains.astype(str),
        persons=persons.astype(str),
        embeddings=x,
        input_to_umap=x_for_umap.astype(np.float32, copy=False),
        umap=coords.astype(np.float32),
    )
    record = make_record(
        record_type="projection.umap",
        explicit_record_id=rid,
        source={"mode": "computed", "legacy_paths": []},
        model=spec.to_record_model(),
        dataset={
            "representation": spec.representation,
            "data_root": str(data_root.resolve()),
            "domains": list(ALL_DOMAINS),
            "persons": list(DEFAULT_PERSONS),
            "split": [0.0, 0.8],
            "sampled_windows": int(indices.size),
        },
        protocol={"name": args.projection_name},
        metrics={
            "sample_per_person_domain": args.sample_per_person_domain,
            "pca_components": int(n_components),
            "umap_neighbors": args.umap_neighbors,
            "umap_min_dist": args.umap_min_dist,
            "seed": args.seed,
        },
        artifacts={"coordinates": str(npz_path.resolve())},
    )
    record_path = save_record(project_root, record)
    dataset.clear_cache()
    print("projection record:", record_path)
    print("coordinates:", npz_path)


def _build_projection_dataset(
    project_root: Path,
    *,
    representation: str,
    domains: tuple[str, ...],
    persons: tuple[str, ...],
):
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


def _balanced_sample_indices(
    dataset,
    *,
    rng: np.random.Generator,
    sample_per_group: int,
    domains: tuple[str, ...],
    persons: tuple[str, ...],
):
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


if __name__ == "__main__":
    main()
