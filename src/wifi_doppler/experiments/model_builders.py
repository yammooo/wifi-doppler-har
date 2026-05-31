from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from wifi_doppler.experiments.runs import checkpoint_fingerprint
from wifi_doppler.models.raw_csi import RawCsiTemporalEncoder
from wifi_doppler.models.sharp import (
    MultiAntennaEncoder,
    MultiAntennaModel,
    SingleAntennaModel,
    build_sharp_single_antenna_encoder,
)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    representation: str
    checkpoint_path: Path
    builder: str

    def to_record_model(self) -> dict[str, Any]:
        return {
            "model_run_id": self.key,
            "model_id": self.key,
            "label": self.label,
            "representation": self.representation,
            "builder": self.builder,
            "checkpoint": checkpoint_fingerprint(self.checkpoint_path),
        }


class RawCsiProtoModel(torch.nn.Module):
    """Adapter that gives RawCsiTemporalEncoder the shared forward_embedding API."""

    def __init__(self, config: dict[str, Any]):
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


def default_model_specs(project_root: str | Path) -> dict[str, ModelSpec]:
    root = Path(project_root)
    return {
        "softmax_featuremap": ModelSpec(
            key="softmax_featuremap",
            label="Doppler softmax feature maps",
            representation="doppler",
            builder="legacy_sharp_classifier",
            checkpoint_path=root
            / "experiments"
            / "pi_classification"
            / "pi_all_persons_123_train_4_test_sharp_model_20260525_165437"
            / "model.pt",
        ),
        "old_proto_featuremap": ModelSpec(
            key="old_proto_featuremap",
            label="Doppler feature-map proto",
            representation="doppler",
            builder="legacy_sharp_classifier",
            checkpoint_path=root
            / "experiments"
            / "few_shot_proto_evaluation"
            / "proto_multi_antenna_vs_softmax_baseline_20260527_164722"
            / "proto_model.pt",
        ),
        "flatten_mlp_proto": ModelSpec(
            key="flatten_mlp_proto",
            label="Doppler flatten-MLP proto",
            representation="doppler",
            builder="sharp_metric_encoder",
            checkpoint_path=root
            / "experiments"
            / "few_shot_proto_evaluation"
            / "proto_multi_antenna_vs_softmax_baseline_20260528_184419"
            / "proto_model.pt",
        ),
        "pooled_proto": ModelSpec(
            key="pooled_proto",
            label="Doppler pooled-head proto",
            representation="doppler",
            builder="sharp_metric_encoder",
            checkpoint_path=root
            / "experiments"
            / "few_shot_proto_evaluation"
            / "proto_pooled_head_vs_softmax_baseline_20260528_220334"
            / "proto_model.pt",
        ),
        "raw_csi_proto": ModelSpec(
            key="raw_csi_proto",
            label="Raw CSI proto",
            representation="raw_csi",
            builder="raw_csi_proto",
            checkpoint_path=root
            / "experiments"
            / "few_shot_raw_csi_proto_evaluation"
            / "raw_csi_proto_vs_doppler_featuremap_proto_20260529_173630"
            / "proto_model_best.pt",
        ),
    }


def model_spec_from_key(
    project_root: str | Path,
    key: str,
    *,
    checkpoint_path: str | Path | None = None,
) -> ModelSpec:
    specs = default_model_specs(project_root)
    if key not in specs:
        raise ValueError(f"Unknown model key {key!r}. Available: {sorted(specs)}")
    spec = specs[key]
    if checkpoint_path is None:
        return spec
    return ModelSpec(
        key=spec.key,
        label=spec.label,
        representation=spec.representation,
        builder=spec.builder,
        checkpoint_path=Path(checkpoint_path),
    )


def load_model_from_spec(
    spec: ModelSpec,
    *,
    device: str | torch.device,
    num_classes: int = 10,
    window_size: int = 340,
    embedding_fusion: str = "mean",
) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = load_checkpoint(spec.checkpoint_path, device=device)
    if spec.builder == "legacy_sharp_classifier":
        model = MultiAntennaModel(SingleAntennaModel(num_classes=num_classes)).to(device)
        return _load_with_lazy_init(
            model,
            checkpoint,
            device=device,
            window_size=window_size,
            embedding_fusion=embedding_fusion,
            use_forward=True,
        ), checkpoint

    if spec.builder == "sharp_metric_encoder":
        config = checkpoint.get("config", {})
        model = MultiAntennaEncoder(_build_sharp_encoder_from_config(config)).to(device)
        return _load_with_lazy_init(
            model,
            checkpoint,
            device=device,
            window_size=window_size,
            embedding_fusion=embedding_fusion,
            use_forward=False,
        ), checkpoint

    if spec.builder == "raw_csi_proto":
        model = RawCsiProtoModel(checkpoint.get("config", {})).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model, checkpoint

    raise ValueError(f"Unknown model builder: {spec.builder!r}")


def load_checkpoint(path: str | Path, *, device: str | torch.device) -> dict[str, Any]:
    checkpoint_path = Path(path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    return torch.load(checkpoint_path, map_location=device, weights_only=False)


def _build_sharp_encoder_from_config(config: dict[str, Any]) -> torch.nn.Module:
    pool_size = tuple(config.get("proto_pool_size", (10, 10)))
    hidden_dim = config.get("proto_hidden_dim")
    hidden_dim = int(hidden_dim) if hidden_dim is not None else None
    return build_sharp_single_antenna_encoder(
        encoder_type=str(config.get("proto_encoder_type", "pooled")),
        embedding_dim=int(config.get("proto_embedding_dim", 128)),
        hidden_dim=hidden_dim,
        pool_size=(int(pool_size[0]), int(pool_size[1])),
        dropout=float(config.get("proto_head_dropout", 0.0)),
        normalize=True,
    )


def _load_with_lazy_init(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    *,
    device: str | torch.device,
    window_size: int,
    embedding_fusion: str,
    use_forward: bool,
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
