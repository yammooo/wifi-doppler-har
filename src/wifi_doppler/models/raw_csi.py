import torch
import torch.nn.functional as F


class ConvBnRelu1d(torch.nn.Module):
    """Conv1d block used by the raw CSI encoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
    ):
        super().__init__()
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

    def forward(self, x):
        return self.block(x)


class ResidualBlock1d(torch.nn.Module):
    """Small ResNet-style temporal block for CSI time series."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = torch.nn.BatchNorm1d(out_channels)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = torch.nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                torch.nn.BatchNorm1d(out_channels),
            )
        else:
            self.shortcut = torch.nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual, inplace=True)


class RawCsiTemporalEncoder(torch.nn.Module):
    """SimID-inspired raw CSI encoder for prototype-learning experiments.

    Expected input shape is ``[batch, channels, time]``, where ``channels`` is
    the flattened antenna/subcarrier stream dimension.
    """

    def __init__(
        self,
        in_channels: int,
        embedding_dim: int = 128,
        channel_mixer_dim: int = 128,
        hidden_dim: int = 256,
        normalize: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.normalize = normalize

        self.channel_mixer = ConvBnRelu1d(
            in_channels=in_channels,
            out_channels=channel_mixer_dim,
            kernel_size=1,
        )
        self.temporal_stem = ConvBnRelu1d(
            in_channels=channel_mixer_dim,
            out_channels=channel_mixer_dim,
            kernel_size=7,
            stride=2,
            padding=3,
        )
        self.blocks = torch.nn.Sequential(
            ResidualBlock1d(channel_mixer_dim, channel_mixer_dim, stride=1),
            ResidualBlock1d(channel_mixer_dim, channel_mixer_dim, stride=1),
            ResidualBlock1d(channel_mixer_dim, hidden_dim, stride=2),
        )
        self.pool = torch.nn.AdaptiveAvgPool1d(1)
        self.projection = torch.nn.Linear(hidden_dim, embedding_dim)

    def forward_feature_maps(self, x):
        x = self.channel_mixer(x)
        x = self.temporal_stem(x)
        return self.blocks(x)

    def forward_embedding(self, x):
        x = self.forward_feature_maps(x)
        x = self.pool(x).flatten(start_dim=1)
        x = self.projection(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

    def forward(self, x):
        return self.forward_embedding(x)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
