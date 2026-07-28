from __future__ import annotations

import torch
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    return next(groups for groups in range(min(8, channels), 0, -1) if channels % groups == 0)


class ConvGnRelu2d(torch.nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: tuple[int, int],
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int],
    ):
        super().__init__(
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            torch.nn.GroupNorm(_group_count(out_channels), out_channels),
            torch.nn.ReLU(inplace=True),
        )


class ResidualGroupBlock2d(torch.nn.Module):
    def __init__(self, channels: int, *, temporal_dilation: int = 1):
        super().__init__()
        dilation = (1, temporal_dilation)
        padding = (1, temporal_dilation)
        self.norm1 = torch.nn.GroupNorm(_group_count(channels), channels)
        self.conv1 = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.norm2 = torch.nn.GroupNorm(_group_count(channels), channels)
        self.conv2 = torch.nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=padding,
            dilation=dilation,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.relu(self.norm1(x), inplace=True))
        h = self.conv2(F.relu(self.norm2(h), inplace=True))
        return x + h


class FrequencyPool(torch.nn.Module):
    """Average frequency features, then mix channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.projection = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.norm = torch.nn.GroupNorm(_group_count(channels), channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.norm(self.projection(x.mean(dim=2))), inplace=True)


class SharedAntennaCsiToDopplerUNet2D(torch.nn.Module):
    """Frequency-aware shared-antenna encoder with a full-resolution 2D decoder."""

    architecture_version = "v1"

    def __init__(
        self,
        num_antennas: int = 4,
        num_subcarriers: int = 242,
        input_parts: int = 2,
        output_doppler_bins: int = 100,
        output_time: int = 340,
        temporal_context: int = 31,
        base_channels: int = 64,
        mid_channels: int = 96,
        bottleneck_channels: int = 128,
        decoder_channels: int = 16,
    ):
        super().__init__()
        if num_antennas < 1:
            raise ValueError("num_antennas must be >= 1.")
        if num_subcarriers < 1:
            raise ValueError("num_subcarriers must be >= 1.")
        if temporal_context < 1 or temporal_context % 2 == 0:
            raise ValueError("temporal_context must be a positive odd integer.")
        for name, channels in (
            ("base_channels", base_channels),
            ("mid_channels", mid_channels),
            ("bottleneck_channels", bottleneck_channels),
            ("decoder_channels", decoder_channels),
        ):
            if not isinstance(channels, int) or isinstance(channels, bool) or channels < 1:
                raise ValueError(f"{name} must be an integer >= 1.")

        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_parts = input_parts
        self.output_doppler_bins = output_doppler_bins
        self.output_time = output_time
        self.temporal_context = temporal_context
        self.decoder_channels = decoder_channels

        self.input_stem = ConvGnRelu2d(
            input_parts,
            base_channels,
            kernel_size=(9, 7),
            stride=(4, 1),
            padding=(4, 3),
        )
        self.encoder1 = ResidualGroupBlock2d(base_channels)
        self.down1 = ConvGnRelu2d(
            base_channels,
            mid_channels,
            kernel_size=(3, 4),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.encoder2 = ResidualGroupBlock2d(mid_channels)
        self.down2 = ConvGnRelu2d(
            mid_channels,
            bottleneck_channels,
            kernel_size=(3, 4),
            stride=(2, 2),
            padding=(1, 1),
        )
        self.bottleneck = torch.nn.Sequential(
            ResidualGroupBlock2d(bottleneck_channels),
            ResidualGroupBlock2d(bottleneck_channels, temporal_dilation=2),
        )

        self.collapse1 = FrequencyPool(base_channels)
        self.collapse2 = FrequencyPool(mid_channels)
        self.collapse3 = FrequencyPool(bottleneck_channels)

        self.grid_projection = torch.nn.Conv1d(
            bottleneck_channels,
            decoder_channels * output_doppler_bins,
            kernel_size=1,
        )
        self.skip2_projection = torch.nn.Conv1d(mid_channels, decoder_channels, kernel_size=1)
        self.decoder1 = ResidualGroupBlock2d(decoder_channels)
        self.skip1_projection = torch.nn.Conv1d(base_channels, decoder_channels, kernel_size=1)
        self.decoder2 = ResidualGroupBlock2d(decoder_channels)
        self.output_head = torch.nn.Sequential(
            ResidualGroupBlock2d(decoder_channels),
            torch.nn.GroupNorm(_group_count(decoder_channels), decoder_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(decoder_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_antennas = self._validate_input(x)
        x = x.reshape(
            batch_size * num_antennas,
            self.num_subcarriers,
            x.shape[3],
            self.input_parts,
        ).permute(0, 3, 1, 2)

        x1 = self.encoder1(self.input_stem(x))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x = self.grid_projection(self.collapse3(x3))
        x = x.reshape(
            batch_size * num_antennas,
            self.decoder_channels,
            self.output_doppler_bins,
            x3.shape[-1],
        ).permute(0, 1, 3, 2)

        x = self._resize_time(x, x2.shape[-1])
        x = self.decoder1(x + self.skip2_projection(self.collapse2(x2)).unsqueeze(-1))
        x = self._resize_time(x, x1.shape[-1])
        x = self.decoder2(x + self.skip1_projection(self.collapse1(x1)).unsqueeze(-1))
        x = self._crop_valid_time(self.output_head(x)).squeeze(1)

        return x.reshape(
            batch_size,
            num_antennas,
            self.output_time,
            self.output_doppler_bins,
        )

    def _validate_input(self, x: torch.Tensor) -> tuple[int, int]:
        expected_time = self.output_time + self.temporal_context - 1
        if x.ndim != 5:
            raise ValueError(
                "Expected input [batch, antenna, subcarrier, time, parts], "
                f"got {tuple(x.shape)}"
            )
        if (
            x.shape[1] < 1
            or x.shape[2] != self.num_subcarriers
            or x.shape[3] != expected_time
            or x.shape[4] != self.input_parts
        ):
            raise ValueError(
                "Input shape does not match model configuration: "
                f"expected at least one antenna, subcarrier={self.num_subcarriers}, "
                f"time={expected_time}, parts={self.input_parts}; got {tuple(x.shape)}"
            )
        return x.shape[0], x.shape[1]

    def _crop_valid_time(self, x: torch.Tensor) -> torch.Tensor:
        margin = (self.temporal_context - 1) // 2
        return x[:, :, margin : margin + self.output_time, :]

    @staticmethod
    def _resize_time(x: torch.Tensor, output_time: int) -> torch.Tensor:
        batch_size, channels, time_steps, doppler_bins = x.shape
        x = x.permute(0, 1, 3, 2).reshape(
            batch_size,
            channels * doppler_bins,
            time_steps,
        )
        x = F.interpolate(x, size=output_time, mode="linear", align_corners=False)
        return x.reshape(batch_size, channels, doppler_bins, output_time).permute(0, 1, 3, 2)
