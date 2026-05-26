from collections.abc import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from wifi_doppler.data.dataset import DopplerWindowDataset
from wifi_doppler.evaluation.embeddings import extract_embeddings


def window_true_labels_from_recordings(dataset: DopplerWindowDataset) -> np.ndarray:
    """Return the integer label for each window without loading window arrays."""
    labels = []
    for window in dataset.window_indexes:
        trace = dataset.traces[window.recording_idx]
        labels.append(dataset.label_to_idx[trace.ground_truth])
    return np.asarray(labels, dtype=np.int64)


def sample_k_per_class(labels: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sample K enrollment indices for every class present in labels."""
    selected = []
    for label in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == label)
        if class_indices.size < k:
            raise ValueError(f"Class {label} has only {class_indices.size} samples, cannot sample k={k}.")
        selected.extend(rng.choice(class_indices, size=k, replace=False).tolist())
    return np.asarray(selected, dtype=np.int64)


def compute_prototypes(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    normalize: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Average enrollment embeddings into one prototype per class."""
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

    return torch.stack(prototypes), torch.tensor(prototype_labels, dtype=labels.dtype)


def prototype_predictions(
    query_embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    prototype_labels: torch.Tensor,
    metric: str = "cosine",
) -> torch.Tensor:
    """Classify query embeddings by nearest prototype."""
    if metric == "cosine":
        query_embeddings = F.normalize(query_embeddings, dim=1)
        prototypes = F.normalize(prototypes, dim=1)

        # query_embeddings @ prototypes.T is (num_queries, num_prototypes)
        nearest = (query_embeddings @ prototypes.T).argmax(dim=1)
    elif metric == "euclidean":
        nearest = torch.cdist(query_embeddings, prototypes).argmin(dim=1)
    else:
        raise ValueError(f"Unknown prototype metric: {metric}")

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


def evaluate_kshot(
    model: torch.nn.Module,
    enrollment_dataset: DopplerWindowDataset,
    query_dataset: DopplerWindowDataset,
    k_values: Iterable[int],
    device: str | torch.device,
    n_trials: int = 20,
    seed: int = 0,
    batch_size: int = 128,
    embedding_fusion: str = "mean",
    metric: str = "cosine",
) -> dict[int, dict[str, float | list[float]]]:
    """Evaluate K-shot prototype inference.

    Query embeddings are computed once. For each trial, enrollment indices are
    sampled first, and only the selected enrollment windows are embedded.
    """
    rng = np.random.default_rng(seed)

    # Compute true labels for all enrollment windows without loading windows into memory
    enrollment_true_labels = window_true_labels_from_recordings(enrollment_dataset)

    # Compute all query embeddings
    query_embeddings, query_true_labels = extract_embeddings(
        model,
        query_dataset,
        device,
        batch_size=batch_size,
        embedding_fusion=embedding_fusion,
    )

    results = {}
    for k in k_values:
        trial_accuracies = []
        for _ in range(n_trials):
            # Does not compute window embeddings for enrollment dataset since we expect K << total windows
            enrollment_indices = sample_k_per_class(enrollment_true_labels, k, rng)
            enrollment_embeddings, selected_true_labels = extract_embeddings(
                model,
                enrollment_dataset,
                device,
                batch_size=batch_size,
                indices=enrollment_indices,
                embedding_fusion=embedding_fusion,
            )
            prototypes, prototype_labels = compute_prototypes(enrollment_embeddings, selected_true_labels)
            accuracy = prototype_accuracy(
                query_embeddings,
                query_true_labels,
                prototypes,
                prototype_labels,
                metric=metric,
            )
            trial_accuracies.append(accuracy)

        results[k] = {
            "mean": float(np.mean(trial_accuracies)),
            "std": float(np.std(trial_accuracies)),
            "trials": trial_accuracies,
        }

    return results
