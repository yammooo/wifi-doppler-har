from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate one model/protocol and save a registry K-shot record.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-run-id", required=True, help="Readable model/run id used as the registry folder name.")
    parser.add_argument("--model-label", default=None, help="Human-readable model label for plots.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--protocol", required=True, help="mixed_source or same_domain_PI-4a, etc.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[1, 3, 5, 10, 25, 50, 100])
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--window-size", type=int, default=340)
    parser.add_argument("--window-stride", type=int, default=30)
    parser.add_argument("--split-guard", type=int, default=31)
    parser.add_argument("--embedding-fusion", default="mean")
    parser.add_argument("--metric", default="cosine")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.experiments.kshot_records import evaluate_kshot_record
    from wifi_doppler.experiments.model_builders import model_spec_from_key
    from wifi_doppler.experiments.protocols import parse_kshot_protocol
    from wifi_doppler.experiments.registry import save_record

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_spec = model_spec_from_key(project_root, args.model_key, checkpoint_path=args.checkpoint)
    from wifi_doppler.experiments.model_builders import ModelSpec

    model_spec = ModelSpec(
        key=args.model_run_id,
        label=args.model_label or args.model_run_id,
        representation=base_spec.representation,
        checkpoint_path=base_spec.checkpoint_path,
        builder=base_spec.builder,
    )
    protocol = parse_kshot_protocol(args.protocol)

    record = evaluate_kshot_record(
        project_root=project_root,
        model_spec=model_spec,
        protocol=protocol,
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
    path = save_record(project_root, record)
    print("saved evaluation record:", path)


if __name__ == "__main__":
    main()
