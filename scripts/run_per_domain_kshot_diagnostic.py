from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run K-shot PI diagnostics across source/target protocols.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-name", default="model_kshot_protocol_comparison")
    parser.add_argument("--pooled-proto-checkpoint", type=Path, default=None)
    parser.add_argument("--raw-csi-proto-checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.data.doppler_dataset import DopplerWindowDataset
    from wifi_doppler.data.raw_csi_dataset import RawCsiWindowDataset
    from wifi_doppler.evaluation.fewshot import evaluate_kshot
    from wifi_doppler.experiments.artifacts import create_run_dir, save_figure, save_json
    from wifi_doppler.models.raw_csi import RawCsiTemporalEncoder
    from wifi_doppler.models.sharp import (
        MultiAntennaEncoder,
        MultiAntennaModel,
        SingleAntennaModel,
        build_sharp_single_antenna_encoder,
    )

    run_group = "few_shot_model_comparison"
    doppler_dir = project_root / "data" / "doppler_traces_pi"
    raw_csi_dir = project_root / "data" / "raw_csi_traces_pi"

    softmax_checkpoint_path = (
        project_root
        / "experiments"
        / "pi_classification"
        / "pi_all_persons_123_train_4_test_sharp_model_20260525_165437"
        / "model.pt"
    )
    old_proto_featuremap_checkpoint_path = (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_multi_antenna_vs_softmax_baseline_20260527_164722"
        / "proto_model.pt"
    )
    flatten_mlp_proto_checkpoint_path = (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_multi_antenna_vs_softmax_baseline_20260528_184419"
        / "proto_model.pt"
    )
    pooled_proto_checkpoint_path = args.pooled_proto_checkpoint or (
        project_root
        / "experiments"
        / "few_shot_proto_evaluation"
        / "proto_pooled_head_vs_softmax_baseline_20260528_220334"
        / "proto_model.pt"
    )
    pooled_proto_checkpoint_path = pooled_proto_checkpoint_path.resolve()
    raw_csi_proto_checkpoint_path = args.raw_csi_proto_checkpoint or (
        project_root
        / "experiments"
        / "few_shot_raw_csi_proto_evaluation"
        / "raw_csi_proto_vs_doppler_featuremap_proto_20260529_173630"
        / "proto_model_best.pt"
    )
    raw_csi_proto_checkpoint_path = raw_csi_proto_checkpoint_path.resolve()

    persons = ["p03", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13"]
    source_domains = ["PI-1a", "PI-2a", "PI-3a"]
    target_domains = ["PI-4a"]
    all_domains = source_domains + target_domains
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

    def build_legacy_featuremap_model(path: Path) -> tuple[torch.nn.Module, dict]:
        checkpoint = load_checkpoint(path)
        model = MultiAntennaModel(SingleAntennaModel(num_classes=len(persons))).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint, use_forward=True), checkpoint

    def build_metric_encoder(path: Path, *, default_encoder_type: str) -> tuple[torch.nn.Module, dict]:
        checkpoint = load_checkpoint(path)
        config = checkpoint.get("config", {})
        encoder_type = str(config.get("proto_encoder_type", default_encoder_type))
        pool_size = tuple(config.get("proto_pool_size", (10, 10)))
        hidden_dim = config.get("proto_hidden_dim")
        hidden_dim = int(hidden_dim) if hidden_dim is not None else None

        model = MultiAntennaEncoder(
            build_sharp_single_antenna_encoder(
                encoder_type=encoder_type,
                embedding_dim=int(config.get("proto_embedding_dim", 128)),
                hidden_dim=hidden_dim,
                pool_size=(int(pool_size[0]), int(pool_size[1])),
                dropout=float(config.get("proto_head_dropout", 0.0)),
                normalize=True,
            )
        ).to(device)
        return load_state_dict_with_lazy_init(model, checkpoint), checkpoint

    class RawCsiProtoModel(torch.nn.Module):
        def __init__(self, config: dict):
            super().__init__()
            self.encoder = RawCsiTemporalEncoder(
                in_channels=int(config.get("raw_in_channels", 4 * 242)),
                embedding_dim=int(config.get("proto_embedding_dim", 128)),
                channel_mixer_dim=int(config.get("raw_channel_mixer_dim", 128)),
                hidden_dim=int(config.get("raw_hidden_dim", 256)),
                normalize=True,
            )

        def forward_embedding(self, x, fusion=None):
            return self.encoder.forward_embedding(x)

        def forward(self, x):
            return self.forward_embedding(x)

    def build_raw_csi_proto_model(path: Path) -> tuple[torch.nn.Module, dict]:
        checkpoint = load_checkpoint(path)
        model = RawCsiProtoModel(checkpoint.get("config", {})).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    softmax_model, softmax_checkpoint = build_legacy_featuremap_model(softmax_checkpoint_path)
    old_proto_model, old_proto_checkpoint = build_legacy_featuremap_model(old_proto_featuremap_checkpoint_path)
    flatten_mlp_model, flatten_mlp_checkpoint = build_metric_encoder(
        flatten_mlp_proto_checkpoint_path,
        default_encoder_type="flatten_mlp",
    )
    pooled_model, pooled_checkpoint = build_metric_encoder(
        pooled_proto_checkpoint_path,
        default_encoder_type="pooled",
    )
    raw_csi_model, raw_csi_checkpoint = build_raw_csi_proto_model(raw_csi_proto_checkpoint_path)

    models = {
        "softmax_featuremap": {
            "label": "softmax feature maps",
            "model": softmax_model,
            "checkpoint": softmax_checkpoint_path,
        },
        "old_proto_featuremap": {
            "label": "old proto feature maps",
            "model": old_proto_model,
            "checkpoint": old_proto_featuremap_checkpoint_path,
        },
        "flatten_mlp_proto": {
            "label": "flatten-MLP proto 128-D",
            "model": flatten_mlp_model,
            "checkpoint": flatten_mlp_proto_checkpoint_path,
        },
        "pooled_proto": {
            "label": "pooled-head proto 128-D",
            "model": pooled_model,
            "checkpoint": pooled_proto_checkpoint_path,
            "representation": "doppler",
        },
        "raw_csi_proto": {
            "label": "raw CSI proto 128-D",
            "model": raw_csi_model,
            "checkpoint": raw_csi_proto_checkpoint_path,
            "representation": "raw_csi",
        },
    }
    for model_info in models.values():
        model_info.setdefault("representation", "doppler")

    run_dir = create_run_dir(project_root, run_group, args.run_name)
    print("run directory:", run_dir)

    def evaluate_protocol(
        protocol_name: str,
        *,
        enrollment_domains: Sequence[str],
        query_domains: Sequence[str],
    ) -> dict[str, dict[int, dict[str, float | list[float]]]]:
        print(f"\n=== {protocol_name} ===")
        print("enrollment domains:", list(enrollment_domains))
        print("query domains:", list(query_domains))

        protocol_results = {}
        for model_key, model_info in models.items():
            if model_info["representation"] == "raw_csi":
                enrollment_dataset = RawCsiWindowDataset(
                    raw_csi_dir,
                    scenarios=list(enrollment_domains),
                    split=enrollment_split,
                    window_size=window_size,
                    window_stride=window_stride,
                    labels=persons,
                    flatten_channels=True,
                    cache_traces=True,
                )
                query_dataset = RawCsiWindowDataset(
                    raw_csi_dir,
                    scenarios=list(query_domains),
                    split=query_split,
                    window_size=window_size,
                    window_stride=window_stride,
                    labels=persons,
                    flatten_channels=True,
                    cache_traces=True,
                )
            else:
                enrollment_dataset = DopplerWindowDataset(
                    doppler_dir,
                    scenarios=list(enrollment_domains),
                    split=enrollment_split,
                    window_size=window_size,
                    window_stride=window_stride,
                    labels=persons,
                )
                query_dataset = DopplerWindowDataset(
                    doppler_dir,
                    scenarios=list(query_domains),
                    split=query_split,
                    window_size=window_size,
                    window_stride=window_stride,
                    labels=persons,
                )
            print(f"evaluating {model_info['label']}")
            print("  enrollment windows:", len(enrollment_dataset), "query windows:", len(query_dataset))
            model_results = evaluate_kshot(
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
            protocol_results[model_key] = model_results
            for k in k_values:
                mean = model_results[k]["mean"]
                std = model_results[k]["std"]
                print(f"  K={k:>3}: {mean:.4f} +/- {std:.4f}")

            enrollment_dataset.clear_cache()
            query_dataset.clear_cache()
        return protocol_results

    results = {
        "mixed_source": evaluate_protocol(
            "mixed-source K-shot",
            enrollment_domains=source_domains,
            query_domains=source_domains,
        ),
        "per_domain": {},
    }
    for domain in all_domains:
        results["per_domain"][domain] = evaluate_protocol(
            f"{domain} same-domain K-shot",
            enrollment_domains=[domain],
            query_domains=[domain],
        )

    styles = {
        "softmax_featuremap": {"marker": "o"},
        "old_proto_featuremap": {"marker": "s"},
        "flatten_mlp_proto": {"marker": "^"},
        "pooled_proto": {"marker": "D"},
        "raw_csi_proto": {"marker": "P"},
    }

    def plot_protocol(ax, protocol_results, *, title: str, ylabel: str):
        for model_key, model_info in models.items():
            means = [protocol_results[model_key][k]["mean"] for k in k_values]
            stds = [protocol_results[model_key][k]["std"] for k in k_values]
            ax.errorbar(
                k_values,
                means,
                yerr=stds,
                marker=styles[model_key]["marker"],
                capsize=3,
                label=model_info["label"],
            )
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xticks(k_values)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax.set_xlabel("K enrollment windows per person")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.grid(True, which="both")

    mixed_fig, mixed_ax = plt.subplots(figsize=(8, 4.5))
    plot_protocol(
        mixed_ax,
        results["mixed_source"],
        title="Mixed-source K-shot: PI-1a/2a/3a pooled",
        ylabel="mixed-source query accuracy",
    )
    mixed_ax.legend()
    mixed_fig.tight_layout()
    mixed_plot_path = save_figure(mixed_fig, run_dir, "mixed_source_kshot_comparison.png")

    target_fig, target_ax = plt.subplots(figsize=(8, 4.5))
    plot_protocol(
        target_ax,
        results["per_domain"]["PI-4a"],
        title="Target same-domain K-shot: PI-4a",
        ylabel="PI-4a query accuracy",
    )
    target_ax.legend()
    target_fig.tight_layout()
    target_plot_path = save_figure(target_fig, run_dir, "target_pi4_kshot_comparison.png")

    per_domain_fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    for ax, domain in zip(axes.flat, all_domains, strict=True):
        plot_protocol(
            ax,
            results["per_domain"][domain],
            title=domain,
            ylabel="same-domain query accuracy",
        )
    for ax in axes.flat:
        ax.set_xlabel("K")
        ax.set_ylabel("accuracy")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    per_domain_fig.legend(handles, labels, loc="upper center", ncol=2)
    per_domain_fig.suptitle("Per-domain same-domain K-shot PI diagnostic", y=0.98)
    per_domain_fig.tight_layout(rect=(0, 0, 1, 0.9))
    per_domain_plot_path = save_figure(per_domain_fig, run_dir, "per_domain_kshot_comparison.png")

    config = {
        "doppler_dir": doppler_dir,
        "raw_csi_dir": raw_csi_dir,
        "source_domains": source_domains,
        "target_domains": target_domains,
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
        "checkpoints": {
            key: model_info["checkpoint"]
            for key, model_info in models.items()
        },
    }
    results_path = save_json(
        run_dir / "kshot_model_comparison_results.json",
        {
            "config": config,
            "results": results,
            "plots": {
                "mixed_source": mixed_plot_path,
                "target_pi4": target_plot_path,
                "per_domain": per_domain_plot_path,
            },
        },
    )

    print("\nplots:")
    print("  mixed source:", mixed_plot_path)
    print("  target PI-4a:", target_plot_path)
    print("  per domain:", per_domain_plot_path)
    print("results:", results_path)


if __name__ == "__main__":
    main()
