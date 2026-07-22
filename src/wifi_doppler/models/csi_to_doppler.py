from __future__ import annotations

import torch
import torch.nn.functional as F


class ConvBnRelu1d(torch.nn.Module):
    """Small Conv1d block for temporal CSI processing."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        *,
        stride: int = 1,
        padding: int | None = None,
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            torch.nn.BatchNorm1d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualTemporalBlock(torch.nn.Module):
    """Two-layer residual temporal block with optional dilation."""

    def __init__(self, channels: int, *, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = dilation * (kernel_size // 2)
        self.conv1 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(channels)
        self.conv2 = torch.nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm1d(channels)

    def forward(self, x):
        h = self.conv1(F.relu(self.bn1(x), inplace=True))
        h = self.conv2(F.relu(self.bn2(h), inplace=True))
        return x + h


class CsiToDopplerUNet1D(torch.nn.Module):
    """Temporal 1D U-Net student that predicts SHARP Doppler from raw CSI.

    Expected input shape is ``[batch, antenna, subcarrier, raw_time, 2]``.
    The model flattens antenna/subcarrier/real-imag into Conv1d channels and
    predicts ``[batch, antenna, doppler_time, doppler_bin]``.
    """

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
    ):
        super().__init__()
        self.num_antennas = num_antennas
        self.num_subcarriers = num_subcarriers
        self.input_parts = input_parts
        self.output_doppler_bins = output_doppler_bins
        self.output_time = output_time
        self.input_channels = num_antennas * num_subcarriers * input_parts
        self.output_channels = num_antennas * output_doppler_bins

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

        self.up1 = ConvBnRelu1d(bottleneck_channels, mid_channels, kernel_size=3)
        self.decoder1 = ResidualTemporalBlock(mid_channels, kernel_size=5)

        self.up2 = ConvBnRelu1d(mid_channels, base_channels, kernel_size=3)
        self.decoder2 = ResidualTemporalBlock(base_channels, kernel_size=5)

        self.output_head = torch.nn.Sequential(
            torch.nn.Conv1d(base_channels, 2 * base_channels, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv1d(2 * base_channels, self.output_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x = self._flatten_input(x)

        x0 = self.input_projection(x)
        x1 = self.encoder1(self.temporal_stem(x0))

        x2 = self.encoder2(self.down1(x1))
        x3 = self.bottleneck(self.down2(x2))

        x = F.interpolate(x3, size=x2.shape[-1], mode="linear", align_corners=False)
        x = self.decoder1(self.up1(x) + x2)

        x = F.interpolate(x, size=x1.shape[-1], mode="linear", align_corners=False)
        x = self.decoder2(self.up2(x) + x1)

        x = F.interpolate(x, size=self.output_time, mode="linear", align_corners=False)
        x = self.output_head(x)
        return x.reshape(batch_size, self.num_antennas, self.output_doppler_bins, self.output_time).transpose(2, 3)

    def _flatten_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"Expected input [batch, antenna, subcarrier, time, parts], got {tuple(x.shape)}")
        if x.shape[1] != self.num_antennas or x.shape[2] != self.num_subcarriers or x.shape[4] != self.input_parts:
            raise ValueError(
                "Input shape does not match model configuration: "
                f"expected antenna={self.num_antennas}, subcarrier={self.num_subcarriers}, parts={self.input_parts}; "
                f"got {tuple(x.shape)}"
            )
        return x.permute(0, 1, 2, 4, 3).reshape(x.shape[0], self.input_channels, x.shape[3])


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
