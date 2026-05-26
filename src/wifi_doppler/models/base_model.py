import torch

class SingleAntennaModel(torch.nn.Module):
    """Single antenna SHARP paper model"""

    def __init__(self, num_classes: int = 5):
        super().__init__()

        self.maxpool_branch1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv1_branch2 = torch.nn.Conv2d(in_channels=1, out_channels=5, kernel_size=2, stride=2)
        
        self.conv1_branch3 = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1)
        self.conv2_branch3 = torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=2, padding="same")
        self.conv3_branch3 = torch.nn.Conv2d(in_channels=6, out_channels=9, kernel_size=4, stride=2, padding=1)

        self.conv_final = torch.nn.Conv2d(in_channels=1+5+9, out_channels=3, kernel_size=1)

        self.classifier = torch.nn.Sequential(
            torch.nn.Flatten(start_dim=1),
            torch.nn.Dropout(p=0.2),
            torch.nn.LazyLinear(out_features=num_classes),
        )

    def forward_feature_maps(self, x):
        """Return convolutional feature maps before the classifier."""
        branch1 = self.maxpool_branch1(x)
        branch1 = torch.nn.ReLU()(branch1)

        branch2 = self.conv1_branch2(x)
        branch2 = torch.nn.ReLU()(branch2)

        branch3 = self.conv1_branch3(x)
        branch3 = torch.nn.ReLU()(branch3)
        branch3 = self.conv2_branch3(branch3)
        branch3 = torch.nn.ReLU()(branch3)
        branch3 = self.conv3_branch3(branch3)
        branch3 = torch.nn.ReLU()(branch3)

        # Concatenate the branches along the channel dimension
        concatenated = torch.cat((branch1, branch2, branch3), dim=1)

        out = self.conv_final(concatenated)
        out = torch.nn.ReLU()(out)

        return out

    def forward_embedding(self, x):
        """Return flattened pre-classifier features for embedding evaluation."""
        feature_maps = self.forward_feature_maps(x)
        return torch.flatten(feature_maps, start_dim=1)

    def forward(self, x):
        out = self.forward_feature_maps(x)
        out = self.classifier(out)

        return out


class MultiAntennaModel(torch.nn.Module):
    """Applies one shared single-antenna model to each antenna stream."""

    def __init__(self, backbone: torch.nn.Module):
        super().__init__()
        self.backbone = backbone

    # Returns per-antenna logits without fusion as (batch_size, num_antennas, num_classes)
    def forward_antennas(self, x):
        # x: [batch, antennas, time, doppler]
        batch_size, num_antennas, time_steps, doppler_bins = x.shape

        x = x.reshape(batch_size * num_antennas, 1, time_steps, doppler_bins)
        logits = self.backbone(x) # (batch_size * num_antennas, num_classes)

        return logits.reshape(batch_size, num_antennas, -1)

    def forward_antenna_embeddings(self, x):
        """Return per-antenna embeddings as (batch_size, num_antennas, embedding_dim)."""
        batch_size, num_antennas, time_steps, doppler_bins = x.shape

        x = x.reshape(batch_size * num_antennas, 1, time_steps, doppler_bins)
        embeddings = self.backbone.forward_embedding(x)

        return embeddings.reshape(batch_size, num_antennas, -1)

    def forward_embedding(self, x, fusion="mean"):
        """Return one fused embedding per sample."""
        embeddings = self.forward_antenna_embeddings(x)

        if fusion == "mean":
            return embeddings.mean(dim=1)

        if fusion == "sum":
            return embeddings.sum(dim=1)

        if fusion == "concat":
            return embeddings.flatten(start_dim=1)

        raise ValueError(f"Unknown embedding fusion method: {fusion}")

    # Returns fused logits as (batch_size, num_classes)
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
        """Fuse antenna scores using SHARP's majority-then-score rule."""
        batch_size, num_antennas, num_classes = logits.shape
        antenna_predictions = logits.argmax(dim=-1)
        summed_logits = logits.sum(dim=1)

        fused_logits = summed_logits.clone()
        for batch_idx in range(batch_size):
            counts = torch.bincount(
                antenna_predictions[batch_idx],
                minlength=num_classes,
            )
            majority_class = counts.argmax()
            if counts[majority_class] >= num_antennas - 1:
                fused_logits[batch_idx] = torch.full_like(fused_logits[batch_idx], float("-inf"))
                fused_logits[batch_idx, majority_class] = 0.0

        return fused_logits
