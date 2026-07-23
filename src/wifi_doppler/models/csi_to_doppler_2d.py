from __future__ import annotations

import torch
import torch.nn.functional as F

from wifi_doppler.models.csi_to_doppler import ConvBnRelu1d, ResidualTemporalBlock


class ResidualSpatialBlock(torch.nn.Module):
    """Pre-activation residual block over time and Doppler."""

    def __init__(self, channels: int):
        super().__init__()
        self.bn1 = torch.nn.BatchNorm2d(channels)
        self.conv1 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = torch.nn.BatchNorm2d(channels)
        self.conv2 = torch.nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.bn1(x), inplace=True))
        h = self.conv2(F.relu(self.bn2(h), inplace=True))
        return x + h


class CsiToDopplerUNet2DDecoder(torch.nn.Module):
    """Temporal CSI encoder with a full time-Doppler 2D decoder."""

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
        decoder_channels: int = 16,
        decoder_coarse_bins: int = 25,
    ):
        super().__init__()
        if (
            not isinstance(decoder_channels, int)
            or isinstance(decoder_channels, bool)
            or decoder_channels < 1
        ):
            raise ValueError("decoder_channels must be an integer >= 1.")
        if (
            not isinstance(decoder_coarse_bins, int)
            or isinstance(decoder_coarse_bins, bool)
            or not 1 <= decoder_coarse_bins <= output_doppler_bins
        ):
            raise ValueError(
                "decoder_coarse_bins must be an integer between 1 and output_doppler_bins."
            )

        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_parts = input_parts
        self.output_doppler_bins = output_doppler_bins
        self.output_time = output_time
        self.decoder_channels = decoder_channels
        self.decoder_coarse_bins = decoder_coarse_bins
        self.input_channels = num_antennas * num_subcarriers * input_parts

        self.input_projection = ConvBnRelu1d(self.input_channels, base_channels, kernel_size=1)
        self.temporal_stem = ConvBnRelu1d(base_channels, base_channels, kernel_size=7)
        self.encoder1 = ResidualTemporalBlock(base_channels, kernel_size=5)
        self.down1 = ConvBnRelu1d(
            base_channels,
            mid_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.encoder2 = ResidualTemporalBlock(mid_channels, kernel_size=5)
        self.down2 = ConvBnRelu1d(
            mid_channels,
            bottleneck_channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.bottleneck = torch.nn.Sequential(
            ResidualTemporalBlock(bottleneck_channels, kernel_size=5, dilation=1),
            ResidualTemporalBlock(bottleneck_channels, kernel_size=5, dilation=2),
        )

        self.grid_projection = torch.nn.Conv1d(
            bottleneck_channels,
            decoder_channels * decoder_coarse_bins,
            kernel_size=1,
        )
        self.skip2_projection = torch.nn.Conv1d(mid_channels, decoder_channels, kernel_size=1)
        self.decoder1 = ResidualSpatialBlock(decoder_channels)
        self.skip1_projection = torch.nn.Conv1d(base_channels, decoder_channels, kernel_size=1)
        self.decoder2 = ResidualSpatialBlock(decoder_channels)
        self.output_head = torch.nn.Sequential(
            ResidualSpatialBlock(decoder_channels),
            torch.nn.BatchNorm2d(decoder_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(decoder_channels, num_antennas, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._flatten_input(x)

        x0 = self.input_projection(x)
        x1 = self.encoder1(self.temporal_stem(x0))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x = self.grid_projection(x3)
        x = x.reshape(
            batch_size,
            self.decoder_channels,
            self.decoder_coarse_bins,
            x3.shape[-1],
        ).permute(0, 1, 3, 2)

        middle_bins = min(self.output_doppler_bins, 2 * self.decoder_coarse_bins)
        x = F.interpolate(
            x,
            size=(x2.shape[-1], middle_bins),
            mode="bilinear",
            align_corners=False,
        )
        x = self.decoder1(x + self.skip2_projection(x2).unsqueeze(-1))

        x = F.interpolate(
            x,
            size=(x1.shape[-1], self.output_doppler_bins),
            mode="bilinear",
            align_corners=False,
        )
        x = self.decoder2(x + self.skip1_projection(x1).unsqueeze(-1))

        x = F.interpolate(
            x,
            size=(self.output_time, self.output_doppler_bins),
            mode="bilinear",
            align_corners=False,
        )
        return self.output_head(x)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(
                f"Expected input [batch, antenna, subcarrier, time, parts], got {tuple(x.shape)}"
            )
        if (
            x.shape[1] != self.num_antennas
            or x.shape[2] != self.num_subcarriers
            or x.shape[4] != self.input_parts
        ):
            raise ValueError(
                "Input shape does not match model configuration: "
                f"expected antenna={self.num_antennas}, subcarrier={self.num_subcarriers}, "
                f"parts={self.input_parts}; got {tuple(x.shape)}"
            )
        return x.permute(0, 1, 2, 4, 3).reshape(
            x.shape[0],
            self.input_channels,
            x.shape[3],
        )
