import torch
import torch.nn.functional as F


class SharpBackbone(torch.nn.Module):
    """SHARP single-antenna convolutional feature extractor."""

    def __init__(self):
        super().__init__()
        self.maxpool_branch1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_branch2 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2)

        self.conv1_branch3 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv2_branch3 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, padding="same")
        self.conv3_branch3 = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=1)

        self.conv_final = torch.nn.Conv2d(in_channels=1 + 5 + 9, out_channels=3, kernel_size=1)

    def forward(self, x):
        branch1 = F.relu(self.maxpool_branch1(x))

        branch2 = F.relu(self.conv1_branch2(x))

        branch3 = F.relu(self.conv1_branch3(x))
        branch3 = F.relu(self.conv2_branch3(branch3))
        branch3 = F.relu(self.conv3_branch3(branch3))

        concatenated = torch.cat((branch1, branch2, branch3), dim=1)
        return F.relu(self.conv_final(concatenated))


class SharpSingleAntennaClassifier(torch.nn.Module):
    """SHARP single-antenna classifier."""

    def __init__(self, num_classes: int = 5, backbone: SharpBackbone | None = None):
        super().__init__()
        self.backbone = backbone if backbone is not None else SharpBackbone()
        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=0.2),
            torch.nn.LazyLinear(out_features=num_classes),
        )

    def forward_feature_maps(self, x):
        return self.backbone(x)

    def forward_embedding(self, x):
        return torch.flatten(self.forward_feature_maps(x), start_dim=1)

    def forward(self, x):
        return self.classifier(self.forward_feature_maps(x))


class SharpLegacySingleAntennaClassifier(torch.nn.Module):
    """Original SHARP classifier layout kept for loading existing checkpoints."""

    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.maxpool_branch1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_branch2 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2)

        self.conv1_branch3 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv2_branch3 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, padding="same")
        self.conv3_branch3 = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=1)

        self.conv_final = torch.nn.Conv2d(in_channels=1 + 5 + 9, out_channels=3, kernel_size=1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=0.2),
            torch.nn.LazyLinear(out_features=num_classes),
        )

    def forward_feature_maps(self, x):
        branch1 = F.relu(self.maxpool_branch1(x))

        branch2 = F.relu(self.conv1_branch2(x))

        branch3 = F.relu(self.conv1_branch3(x))
        branch3 = F.relu(self.conv2_branch3(branch3))
        branch3 = F.relu(self.conv3_branch3(branch3))

        concatenated = torch.cat((branch1, branch2, branch3), dim=1)
        return F.relu(self.conv_final(concatenated))

    def forward_embedding(self, x):
        return torch.flatten(self.forward_feature_maps(x), start_dim=1)

    def forward(self, x):
        return self.classifier(self.forward_feature_maps(x))


class SharpSingleAntennaEncoder(torch.nn.Module):
    """SHARP backbone plus a projection head for metric-learning embeddings."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        normalize: bool = True,
        backbone: SharpBackbone | None = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else SharpBackbone()
        self.normalize = normalize
        self.projection = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.LazyLinear(hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(hidden_dim, embedding_dim),
        )

    def forward_feature_maps(self, x):
        return self.backbone(x)

    def forward_embedding(self, x):
        embeddings = self.projection(self.forward_feature_maps(x))
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings

    def forward(self, x):
        return self.forward_embedding(x)


class SharpPooledSingleAntennaEncoder(torch.nn.Module):
    """SHARP backbone plus a compact pooled projection head.

    The flattened SHARP feature map has 25,500 values for the current
    340x100 Doppler windows. Pooling before projection keeps this encoder much
    smaller than the legacy flatten-MLP encoder while preserving coarse
    time/Doppler layout.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pool_size: tuple[int, int] = (10, 10),
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        normalize: bool = True,
        backbone: SharpBackbone | None = None,
    ):
        super().__init__()
        self.backbone = backbone if backbone is not None else SharpBackbone()
        self.normalize = normalize
        self.pool_size = pool_size

        layers: list[torch.nn.Module] = [
            torch.nn.AdaptiveAvgPool2d(pool_size),
            torch.nn.Flatten(start_dim=1),
        ]
        pooled_dim = 3 * pool_size[0] * pool_size[1]
        if hidden_dim is None:
            layers.append(torch.nn.Linear(pooled_dim, embedding_dim))
        else:
            layers.extend(
                [
                    torch.nn.Linear(pooled_dim, hidden_dim),
                    torch.nn.ReLU(),
                ]
            )
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            layers.append(torch.nn.Linear(hidden_dim, embedding_dim))
        self.projection = torch.nn.Sequential(*layers)

    def forward_feature_maps(self, x):
        return self.backbone(x)

    def forward_embedding(self, x):
        embeddings = self.projection(self.forward_feature_maps(x))
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=1)
        return embeddings

    def forward(self, x):
        return self.forward_embedding(x)


class MultiAntennaClassifier(torch.nn.Module):
    """Apply one shared single-antenna classifier to each antenna stream."""

    def __init__(self, single_antenna_model: torch.nn.Module):
        super().__init__()
        self.backbone = single_antenna_model

    def forward_antennas(self, x):
        batch_size, num_antennas, time_steps, doppler_bins = x.shape
        x = x.reshape(batch_size * num_antennas, 1, time_steps, doppler_bins)
        logits = self.backbone(x)
        return logits.reshape(batch_size, num_antennas, -1)

    def forward_antenna_embeddings(self, x):
        batch_size, num_antennas, time_steps, doppler_bins = x.shape
        x = x.reshape(batch_size * num_antennas, 1, time_steps, doppler_bins)
        embeddings = self.backbone.forward_embedding(x)
        return embeddings.reshape(batch_size, num_antennas, -1)

    def forward_embedding(self, x, fusion="mean"):
        embeddings = self.forward_antenna_embeddings(x)
        return fuse_antennas(embeddings, fusion)

    def forward(self, x, fusion="sum"):
        logits = self.forward_antennas(x)
        if fusion == "sum":
            return logits.sum(dim=1)
        if fusion == "mean":
            return logits.mean(dim=1)
        if fusion == "sharp":
            return self.sharp_decision_fusion(logits)
        raise ValueError(f"Unknown fusion method: {fusion}")

    @staticmethod
    def sharp_decision_fusion(logits: torch.Tensor) -> torch.Tensor:
        batch_size, num_antennas, num_classes = logits.shape
        antenna_predictions = logits.argmax(dim=-1)
        summed_logits = logits.sum(dim=1)

        fused_logits = summed_logits.clone()
        for batch_idx in range(batch_size):
            counts = torch.bincount(antenna_predictions[batch_idx], minlength=num_classes)
            majority_class = counts.argmax()
            if counts[majority_class] >= num_antennas - 1:
                fused_logits[batch_idx] = torch.full_like(fused_logits[batch_idx], float("-inf"))
                fused_logits[batch_idx, majority_class] = 0.0
        return fused_logits


class MultiAntennaEncoder(torch.nn.Module):
    """Apply one shared single-antenna encoder to each antenna stream."""

    def __init__(self, single_antenna_encoder: torch.nn.Module):
        super().__init__()
        self.backbone = single_antenna_encoder

    def forward_antenna_embeddings(self, x):
        batch_size, num_antennas, time_steps, doppler_bins = x.shape
        x = x.reshape(batch_size * num_antennas, 1, time_steps, doppler_bins)
        embeddings = self.backbone.forward_embedding(x)
        return embeddings.reshape(batch_size, num_antennas, -1)

    def forward_embedding(self, x, fusion="mean"):
        embeddings = self.forward_antenna_embeddings(x)
        fused = fuse_antennas(embeddings, fusion)
        if fusion in {"mean", "sum"}:
            fused = F.normalize(fused, dim=1)
        return fused

    def forward(self, x, fusion="mean"):
        return self.forward_embedding(x, fusion=fusion)


def fuse_antennas(values: torch.Tensor, fusion: str):
    if fusion == "mean":
        return values.mean(dim=1)
    if fusion == "sum":
        return values.sum(dim=1)
    if fusion == "concat":
        return values.flatten(start_dim=1)
    raise ValueError(f"Unknown fusion method: {fusion}")


def build_sharp_single_antenna_encoder(
    encoder_type: str = "pooled",
    *,
    embedding_dim: int = 128,
    hidden_dim: int | None = None,
    pool_size: tuple[int, int] = (10, 10),
    dropout: float = 0.0,
    normalize: bool = True,
    backbone: SharpBackbone | None = None,
) -> torch.nn.Module:
    """Build a SHARP single-antenna encoder for metric-learning experiments."""
    if encoder_type == "pooled":
        return SharpPooledSingleAntennaEncoder(
            embedding_dim=embedding_dim,
            pool_size=pool_size,
            hidden_dim=hidden_dim,
            dropout=dropout,
            normalize=normalize,
            backbone=backbone,
        )

    if encoder_type == "flatten_mlp":
        if hidden_dim is None:
            raise ValueError("hidden_dim is required for encoder_type='flatten_mlp'.")
        return SharpSingleAntennaEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            normalize=normalize,
            backbone=backbone,
        )

    raise ValueError(f"Unknown SHARP encoder type: {encoder_type!r}.")


SingleAntennaModel = SharpLegacySingleAntennaClassifier
MultiAntennaModel = MultiAntennaClassifier
