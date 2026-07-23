from __future__ import annotations

import torch
import torch.nn.functional as F

from wifi_doppler.models.csi_to_doppler import CsiToDopplerUNet1D


class SpatialDopplerHead(torch.nn.Module):
    """Map temporal features to a locally coherent time-Doppler grid."""

    def __init__(
        self,
        in_channels: int,
        num_antennas: int,
        output_doppler_bins: int,
        *,
        head_channels: int,
        head_coarse_bins: int,
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.output_doppler_bins = output_doppler_bins
        self.head_channels = head_channels
        self.head_coarse_bins = head_coarse_bins

        self.grid_projection = torch.nn.Conv1d(
            in_channels,
            head_channels * head_coarse_bins,
            kernel_size=1,
        )
        self.spatial_refinement = torch.nn.Sequential(
            torch.nn.Conv2d(head_channels, head_channels, kernel_size=3, padding=1, bias=False),
            torch.nn.BatchNorm2d(head_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(head_channels, num_antennas, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, time_steps = x.shape
        x = self.grid_projection(x)
        x = x.reshape(
            batch_size,
            self.head_channels,
            self.head_coarse_bins,
            time_steps,
        ).permute(0, 1, 3, 2)
        x = F.interpolate(
            x,
            size=(time_steps, self.output_doppler_bins),
            mode="bilinear",
            align_corners=False,
        )
        x = self.spatial_refinement(x)
        return x.permute(0, 1, 3, 2).reshape(
            batch_size,
            self.num_antennas * self.output_doppler_bins,
            time_steps,
        )


class CsiToDopplerUNet1DSpatialHead(CsiToDopplerUNet1D):
    """Legacy temporal U-Net with only its output head made spatial."""

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
        if (
            not isinstance(head_channels, int)
            or isinstance(head_channels, bool)
            or head_channels < 1
        ):
            raise ValueError("head_channels must be an integer >= 1.")
        if (
            not isinstance(head_coarse_bins, int)
            or isinstance(head_coarse_bins, bool)
            or not 1 <= head_coarse_bins <= output_doppler_bins
        ):
            raise ValueError(
                "head_coarse_bins must be an integer between 1 and output_doppler_bins."
            )

        super().__init__(
            num_antennas=num_antennas,
            num_subcarriers=num_subcarriers,
            input_parts=input_parts,
            output_doppler_bins=output_doppler_bins,
            output_time=output_time,
            base_channels=base_channels,
            mid_channels=mid_channels,
            bottleneck_channels=bottleneck_channels,
        )
        self.output_head = SpatialDopplerHead(
            base_channels,
            num_antennas,
            output_doppler_bins,
            head_channels=head_channels,
            head_coarse_bins=head_coarse_bins,
        )
