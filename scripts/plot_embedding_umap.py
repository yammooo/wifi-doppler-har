from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot UMAP projections of PI embeddings.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--sample-per-person-domain", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--run-name", default="raw_featuremap_vs_pooled_head_umap")
    parser.add_argument("--raw-proto-checkpoint", type=Path, default=None)
    parser.add_argument("--pooled-proto-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from sklearn.decomposition import PCA
    from umap import UMAP

    from wifi_doppler.data.doppler_dataset import DopplerWindowDataset
    from wifi_doppler.experiments.artifacts import create_run_dir, save_figure, save_json
    from wifi_doppler.models.sharp import (
        MultiAntennaEncoder,
        MultiAntennaModel,
        SingleAntennaModel,
        build_sharp_single_antenna_encoder,
    )
    from wifi_doppler.representation.embeddings import extract_embeddings

    doppler_dir = project_root / "data" / "doppler_traces_pi"
    raw_proto_checkpoint = args.raw_proto_checkpoint or (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_multi_antenna_vs_softmax_baseline_20260527_164722"
        / "proto_model.pt"
    )
    pooled_proto_checkpoint = args.pooled_proto_checkpoint or (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_pooled_head_vs_softmax_baseline_20260528_220334"
        / "proto_model.pt"
    )
    raw_proto_checkpoint = raw_proto_checkpoint.resolve()
    pooled_proto_checkpoint = pooled_proto_checkpoint.resolve()

    persons = ["p03", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13"]
    domains = ["PI-1a", "PI-2a", "PI-3a", "PI-4a"]
    window_size = 340
    window_stride = 30
    split = (0.0, 0.8)
    embedding_fusion = "mean"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed)
    print("device:", device)

    dataset = DopplerWindowDataset(
        doppler_dir,
        scenarios=domains,
        split=split,
        window_size=window_size,
        window_stride=window_stride,
        labels=persons,
    )

    def balanced_sample_indices() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
                sample_size = min(args.sample_per_person_domain, candidates.size)
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

    selected_indices, selected_domains, selected_persons = balanced_sample_indices()
    print("sampled windows:", selected_indices.size)

    def load_checkpoint(path: Path) -> dict:
        if not path.exists():
            raise FileNotFoundError(path)
        return torch.load(path, map_location=device, weights_only=False)

    def load_state_dict_with_lazy_init(
        model: torch.nn.Module,
        checkpoint: dict,
        *,
        use_forward: bool = False,
    ) -> torch.nn.Module:
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError:
            dummy = torch.zeros(1, 4, window_size, 100, device=device)
            with torch.no_grad():
                if use_forward:
                    model(dummy)
                else:
                    model.forward_embedding(dummy, fusion=embedding_fusion)
            model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def build_raw_featuremap_model(path: Path) -> torch.nn.Module:
        checkpoint = load_checkpoint(path)
        model = MultiAntennaModel(SingleAntennaModel(num_classes=len(persons))).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint, use_forward=True)

    def build_pooled_model(path: Path) -> torch.nn.Module:
        checkpoint = load_checkpoint(path)
        config = checkpoint.get("config", {})
        pool_size = tuple(config.get("proto_pool_size", (10, 10)))
        hidden_dim = config.get("proto_hidden_dim")
        hidden_dim = int(hidden_dim) if hidden_dim is not None else None
        model = MultiAntennaEncoder(
            build_sharp_single_antenna_encoder(
                encoder_type=str(config.get("proto_encoder_type", "pooled")),
                embedding_dim=int(config.get("proto_embedding_dim", 128)),
                hidden_dim=hidden_dim,
                pool_size=(int(pool_size[0]), int(pool_size[1])),
                dropout=float(config.get("proto_head_dropout", 0.0)),
                normalize=True,
            )
        ).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint)

    models = {
        "raw_featuremap_proto": {
            "label": "raw feature-map proto",
            "checkpoint": raw_proto_checkpoint,
            "model": build_raw_featuremap_model(raw_proto_checkpoint),
        },
        "pooled_head_proto": {
            "label": "pooled-head proto",
            "checkpoint": pooled_proto_checkpoint,
            "model": build_pooled_model(pooled_proto_checkpoint),
        },
    }

    def reduce_embeddings(name: str, embeddings: torch.Tensor) -> dict[str, np.ndarray]:
        x = embeddings.numpy().astype(np.float32, copy=False)
        if x.shape[1] > args.pca_components:
            n_components = min(args.pca_components, x.shape[0] - 1, x.shape[1])
            print(f"{name}: PCA {x.shape[1]} -> {n_components}")
            x_for_umap = PCA(n_components=n_components, random_state=args.seed).fit_transform(x)
        else:
            x_for_umap = x

        print(f"{name}: UMAP {x_for_umap.shape[1]} -> 2")
        coords = UMAP(
            n_components=2,
            n_neighbors=args.umap_neighbors,
            min_dist=args.umap_min_dist,
            metric="euclidean",
            random_state=args.seed,
        ).fit_transform(x_for_umap)
        return {
            "umap": coords.astype(np.float32),
            "input_to_umap": x_for_umap.astype(np.float32, copy=False),
        }

    reduced = {}
    for model_key, model_info in models.items():
        print(f"extracting embeddings: {model_info['label']}")
        embeddings, labels = extract_embeddings(
            model_info["model"],
            dataset,
            device,
            batch_size=args.batch_size,
            indices=selected_indices,
            embedding_fusion=embedding_fusion,
        )
        reduced[model_key] = reduce_embeddings(model_key, embeddings)

    run_dir = create_run_dir(project_root, "embedding_umap", args.run_name)

    person_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(persons)))
    domain_colors = plt.get_cmap("tab10")(np.linspace(0, 1, len(domains)))
    person_to_color = dict(zip(persons, person_colors, strict=True))
    domain_to_color = dict(zip(domains, domain_colors, strict=True))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for row_idx, (model_key, model_info) in enumerate(models.items()):
        coords = reduced[model_key]["umap"]

        ax = axes[row_idx, 0]
        for person in persons:
            mask = selected_persons == person
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=7,
                alpha=0.55,
                color=person_to_color[person],
                label=person,
                linewidths=0,
            )
        ax.set_title(f"{model_info['label']} colored by person")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = axes[row_idx, 1]
        for domain in domains:
            mask = selected_domains == domain
            ax.scatter(
                coords[mask, 0],
                coords[mask, 1],
                s=7,
                alpha=0.55,
                color=domain_to_color[domain],
                label=domain,
                linewidths=0,
            )
        ax.set_title(f"{model_info['label']} colored by domain")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0, 0].legend(loc="best", fontsize=7, markerscale=2, ncol=2)
    axes[0, 1].legend(loc="best", fontsize=8, markerscale=2)
    fig.suptitle("UMAP of PI Doppler embeddings", y=0.995)
    fig.tight_layout()

    plot_path = save_figure(fig, run_dir, "embedding_umap_person_domain.png", dpi=180)
    npz_path = run_dir / "embedding_umap_coordinates.npz"
    np.savez_compressed(
        npz_path,
        selected_indices=selected_indices,
        domains=selected_domains.astype(str),
        persons=selected_persons.astype(str),
        raw_featuremap_proto_umap=reduced["raw_featuremap_proto"]["umap"],
        pooled_head_proto_umap=reduced["pooled_head_proto"]["umap"],
    )
    config = {
        "doppler_dir": doppler_dir,
        "domains": domains,
        "persons": persons,
        "split": split,
        "window_size": window_size,
        "window_stride": window_stride,
        "sample_per_person_domain": args.sample_per_person_domain,
        "sampled_windows": int(selected_indices.size),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "pca_components": args.pca_components,
        "umap_neighbors": args.umap_neighbors,
        "umap_min_dist": args.umap_min_dist,
        "embedding_fusion": embedding_fusion,
        "checkpoints": {
            key: model_info["checkpoint"]
            for key, model_info in models.items()
        },
        "plot": plot_path,
        "coordinates": npz_path,
    }
    config_path = save_json(run_dir / "embedding_umap_config.json", config)

    dataset.clear_cache()
    print("run directory:", run_dir)
    print("plot:", plot_path)
    print("coordinates:", npz_path)
    print("config:", config_path)


if __name__ == "__main__":
    main()
