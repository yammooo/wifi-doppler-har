import torch
import torch.nn.functional as F


def compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average embeddings into one prototype per class.

    Returns prototypes ordered by sorted class label, plus the class label for
    each prototype.
    """
    if normalize:
        embeddings = F.normalize(embeddings, dim=1)

    prototypes = []
    prototype_labels = []
    for label in sorted(labels.unique().tolist()):
        class_embeddings = embeddings[labels == label]
        prototype = class_embeddings.mean(dim=0)
        if normalize:
            prototype = F.normalize(prototype, dim=0)
        prototypes.append(prototype)
        prototype_labels.append(label)

    return torch.stack(prototypes), torch.tensor(prototype_labels, dtype=labels.dtype, device=labels.device)


def prototype_logits(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """Return class logits induced by query-to-prototype similarity/distance."""
    if metric == "cosine":
        query_embeddings = F.normalize(query_embeddings, dim=1)
        prototypes = F.normalize(prototypes, dim=1)
        return query_embeddings @ prototypes.T

    if metric == "euclidean":
        return -torch.cdist(query_embeddings, prototypes)

    raise ValueError(f"Unknown prototype metric: {metric}")


def prototype_predictions(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """Classify query embeddings by nearest prototype."""
    logits = prototype_logits(query_embeddings, prototypes, metric=metric)
    nearest = logits.argmax(dim=1)
    return prototype_labels[nearest]


def prototype_accuracy(
    query_embeddings: torch.Tensor,
    query_true_labels: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
    metric: str = "cosine",
) -> float:
    """Return nearest-prototype accuracy for query embeddings."""
    predictions = prototype_predictions(
        query_embeddings,
        prototypes,
        prototype_labels,
        metric=metric,
    )
    return (predictions == query_true_labels).float().mean().item()


def map_labels_to_prototype_indices(
    labels: torch.Tensor,
    prototype_labels: torch.Tensor,
) -> torch.Tensor:
    """Map global class labels to local prototype column indices.

    Cross-entropy over prototype logits expects targets in ``0..num_prototypes-1``.
    This helper maps original dataset labels to the matching prototype positions.
    """
    mapping = {int(label): idx for idx, label in enumerate(prototype_labels.tolist())}
    mapped = [mapping[int(label)] for label in labels.tolist()]
    return torch.tensor(mapped, dtype=torch.long, device=labels.device)
