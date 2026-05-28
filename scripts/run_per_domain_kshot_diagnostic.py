from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run per-domain same-domain K-shot PI diagnostics.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.data.dataset import DopplerWindowDataset
    from wifi_doppler.evaluation.fewshot import evaluate_kshot
    from wifi_doppler.experiments.artifacts import create_run_dir, save_figure, save_json
    from wifi_doppler.models.sharp import (
        MultiAntennaEncoder,
        MultiAntennaModel,
        SharpSingleAntennaEncoder,
        SingleAntennaModel,
    )

    run_group = "few_shot_per_domain_evaluation"
    run_name = "per_domain_featuremap_vs_projection"
    doppler_dir = project_root / "data" / "doppler_traces_pi"

    softmax_checkpoint_path = (
        project_root
        / "experiments"
        / "pi_classification"
        / "pi_all_persons_123_train_4_test_sharp_model_20260525_165437"
        / "model.pt"
    )
    old_proto_checkpoint_path = (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_multi_antenna_vs_softmax_baseline_20260527_164722"
        / "proto_model.pt"
    )
    new_proto_checkpoint_path = (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_multi_antenna_vs_softmax_baseline_20260528_184419"
        / "proto_model_best.pt"
    )

    persons = ["p03", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13"]
    domains = ["PI-1a", "PI-2a", "PI-3a", "PI-4a"]
    k_values = [1, 3, 5, 10, 25, 50, 100]
    enrollment_split = (0.0, 0.6)
    query_split = (0.6, 0.8)
    window_size = 340
    window_stride = 30
    embedding_fusion = "mean"
    metric = "cosine"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

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

    def build_legacy_featuremap_model(path: Path) -> torch.nn.Module:
        checkpoint = load_checkpoint(path)
        model = MultiAntennaModel(SingleAntennaModel(num_classes=len(persons))).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint, use_forward=True)

    def build_new_proto_encoder(path: Path) -> torch.nn.Module:
        checkpoint = load_checkpoint(path)
        config = checkpoint.get("config", {})
        model = MultiAntennaEncoder(
            SharpSingleAntennaEncoder(
                embedding_dim=int(config.get("proto_embedding_dim", 128)),
                hidden_dim=int(config.get("proto_hidden_dim", 256)),
                normalize=True,
            )
        ).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint)

    models = {
        "softmax_featuremap": {
            "label": "softmax feature maps",
            "model": build_legacy_featuremap_model(softmax_checkpoint_path),
        },
        "old_proto_featuremap": {
            "label": "old proto feature maps",
            "model": build_legacy_featuremap_model(old_proto_checkpoint_path),
        },
        "new_proto_encoder": {
            "label": "new proto 128-D encoder",
            "model": build_new_proto_encoder(new_proto_checkpoint_path),
        },
    }

    run_dir = create_run_dir(project_root, run_group, run_name)
    print("run directory:", run_dir)

    results: dict[str, dict[str, dict[int, dict[str, float | list[float]]]]] = {}
    for domain in domains:
        print(f"\n=== {domain} same-domain K-shot ===")
        enrollment_dataset = DopplerWindowDataset(
            doppler_dir,
            scenarios=[domain],
            split=enrollment_split,
            window_size=window_size,
            window_stride=window_stride,
            labels=persons,
        )
        query_dataset = DopplerWindowDataset(
            doppler_dir,
            scenarios=[domain],
            split=query_split,
            window_size=window_size,
            window_stride=window_stride,
            labels=persons,
        )
        print("enrollment windows:", len(enrollment_dataset), "query windows:", len(query_dataset))

        results[domain] = {}
        for model_key, model_info in models.items():
            print(f"evaluating {model_info['label']}")
            domain_results = evaluate_kshot(
                model_info["model"],
                enrollment_dataset,
                query_dataset,
                k_values,
                device,
                n_trials=args.n_trials,
                seed=args.seed,
                batch_size=args.batch_size,
                embedding_fusion=embedding_fusion,
                metric=metric,
            )
            results[domain][model_key] = domain_results
            for k in k_values:
                mean = domain_results[k]["mean"]
                std = domain_results[k]["std"]
                print(f"  K={k:>3}: {mean:.4f} +/- {std:.4f}")

        enrollment_dataset.clear_cache()
        query_dataset.clear_cache()

    styles = {
        "softmax_featuremap": {"marker": "o"},
        "old_proto_featuremap": {"marker": "s"},
        "new_proto_encoder": {"marker": "^"},
    }

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, domain in zip(axes.flat, domains, strict=True):
        for model_key, model_info in models.items():
            means = [results[domain][model_key][k]["mean"] for k in k_values]
            stds = [results[domain][model_key][k]["std"] for k in k_values]
            ax.errorbar(
                k_values,
                means,
                yerr=stds,
                marker=styles[model_key]["marker"],
                capsize=3,
                label=model_info["label"],
            )
        ax.set_title(domain)
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.grid(True, which="both")
        ax.set_ylim(0, 1)

    for ax in axes[-1, :]:
        ax.set_xlabel("K enrollment windows per person")
    for ax in axes[:, 0]:
        ax.set_ylabel("same-domain query accuracy")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3)
    fig.suptitle("Per-domain same-domain K-shot PI diagnostic", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    plot_path = save_figure(fig, run_dir, "per_domain_kshot_comparison.png")
    config = {
        "doppler_dir": doppler_dir,
        "domains": domains,
        "persons": persons,
        "enrollment_split": enrollment_split,
        "query_split": query_split,
        "window_size": window_size,
        "window_stride": window_stride,
        "k_values": k_values,
        "n_trials": args.n_trials,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "embedding_fusion": embedding_fusion,
        "metric": metric,
        "softmax_checkpoint": softmax_checkpoint_path,
        "old_proto_featuremap_checkpoint": old_proto_checkpoint_path,
        "new_proto_encoder_checkpoint": new_proto_checkpoint_path,
    }
    results_path = save_json(
        run_dir / "per_domain_kshot_results.json",
        {
            "config": config,
            "results": results,
        },
    )

    print("\nplot:", plot_path)
    print("results:", results_path)


if __name__ == "__main__":
    main()
