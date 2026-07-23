from __future__ import annotations

from typing import Any

import torch

from wifi_doppler.models.csi_to_doppler import CsiToDopplerUNet1D
from wifi_doppler.models.csi_to_doppler_2d import CsiToDopplerUNet2DDecoder
from wifi_doppler.models.csi_to_doppler_2d_head import CsiToDopplerUNet1DSpatialHead


_ARCHITECTURES: dict[str, tuple[type[torch.nn.Module], str, str]] = {
    "unet1d_legacy": (
        CsiToDopplerUNet1D,
        "csi_to_doppler_unet1d_v1",
        "wifi_doppler.models.csi_to_doppler.CsiToDopplerUNet1D",
    ),
    "unet2d_decoder": (
        CsiToDopplerUNet2DDecoder,
        "csi_to_doppler_unet2d_decoder_v1",
        "wifi_doppler.models.csi_to_doppler_2d.CsiToDopplerUNet2DDecoder",
    ),
    "unet1d_spatial_head": (
        CsiToDopplerUNet1DSpatialHead,
        "csi_to_doppler_unet1d_spatial_head_v1",
        "wifi_doppler.models.csi_to_doppler_2d_head.CsiToDopplerUNet1DSpatialHead",
    ),
}


def build_csi_to_doppler_model(config: dict[str, Any]) -> torch.nn.Module:
    options = dict(config)
    architecture = str(options.pop("architecture", "unet1d_legacy"))
    try:
        model_class = _ARCHITECTURES[architecture][0]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CSI-to-Doppler architecture {architecture!r}. "
            f"Available: {sorted(_ARCHITECTURES)}"
        ) from exc
    return model_class(**options)


def csi_to_doppler_model_metadata(config: dict[str, Any]) -> tuple[str, str]:
    architecture = str(config.get("architecture", "unet1d_legacy"))
    try:
        _, model_key, builder = _ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(
            f"Unknown CSI-to-Doppler architecture {architecture!r}. "
            f"Available: {sorted(_ARCHITECTURES)}"
        ) from exc
    return model_key, builder
