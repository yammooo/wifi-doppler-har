from __future__ import annotations

import torch

from wifi_doppler.models.csi_to_doppler_2d_head import CsiToDopplerUNet1DSpatialHead


class SharedAntennaCsiToDopplerUNet1DSpatialHead(torch.nn.Module):
    """Apply one single-antenna CSI-to-Doppler network to every antenna."""

    architecture_version = "v1"

    def __init__(
        self,
        num_antennas: int = 4,
        num_subcarriers: int = 30,
        input_parts: int = 2,
        output_doppler_bins: int = 100,
        output_time: int = 340,
        base_channels: int = 128,
        mid_channels: int = 192,
        bottleneck_channels: int = 256,
        head_channels: int = 8,
        head_coarse_bins: int = 25,
    ):
        super().__init__()
        if num_antennas < 1:
            raise ValueError("num_antennas must be >= 1.")
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_parts = input_parts
        self.output_doppler_bins = output_doppler_bins
        self.output_time = output_time
        self.single_antenna_model = CsiToDopplerUNet1DSpatialHead(
            num_antennas=1,
            num_subcarriers=num_subcarriers,
            input_parts=input_parts,
            output_doppler_bins=output_doppler_bins,
            output_time=output_time,
            base_channels=base_channels,
            mid_channels=mid_channels,
            bottleneck_channels=bottleneck_channels,
            head_channels=head_channels,
            head_coarse_bins=head_coarse_bins,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                "Expected input [batch, antenna, subcarrier, time, parts], "
                f"got {tuple(x.shape)}"
            )
        if (
            x.shape[1] < 1
            or x.shape[2] != self.num_subcarriers
            or x.shape[4] != self.input_parts
        ):
            raise ValueError(
                "Input shape does not match model configuration: "
                f"expected at least one antenna, subcarrier={self.num_subcarriers}, "
                f"parts={self.input_parts}; got {tuple(x.shape)}"
            )

        batch_size, num_input_antennas, num_subcarriers, time_steps, input_parts = x.shape
        x = x.reshape(
            batch_size * num_input_antennas,
            1,
            num_subcarriers,
            time_steps,
            input_parts,
        )
        x = self.single_antenna_model(x)
        return x.reshape(
            batch_size,
            num_input_antennas,
            self.output_time,
            self.output_doppler_bins,
        )
