import torch


class XRF55ConvNet(torch.nn.Module):
    """Small-but-deeper 2D CNN for single XRF55 Doppler images.

    Input shape is expected to be [batch, 1, time, doppler_bin]. Adaptive
    pooling keeps the classifier independent of the exact crop/spectrogram size.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4
        c4 = base_channels * 8

        self.features = torch.nn.Sequential(
            self._conv_block(in_channels, c1),
            self._conv_block(c1, c1),
            torch.nn.MaxPool2d(kernel_size=2),
            self._conv_block(c1, c2),
            self._conv_block(c2, c2),
            torch.nn.MaxPool2d(kernel_size=2),
            self._conv_block(c2, c3),
            self._conv_block(c3, c3),
            torch.nn.MaxPool2d(kernel_size=2),
            self._conv_block(c3, c4),
            torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(c4, num_classes),
        )

    @staticmethod
    def _conv_block(in_channels: int, out_channels: int) -> torch.nn.Sequential:
        return torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
