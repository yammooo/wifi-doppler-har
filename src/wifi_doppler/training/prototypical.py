from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from wifi_doppler.evaluation.fewshot import window_true_labels_from_recordings
from wifi_doppler.representation.prototypes import (
    compute_prototypes,
    map_labels_to_prototype_indices,
    prototype_logits,
)


@dataclass(frozen=True)
class Episode:
    """Support/query tensors and labels for one prototypical-learning episode."""

    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def group_indices_by_label(labels: np.ndarray) -> dict[int, np.ndarray]:
    """Group dataset window indices by integer class label."""
    return {
        int(label): np.flatnonzero(labels == label)
        for label in sorted(np.unique(labels))
    }


def window_domains_from_recordings(dataset) -> np.ndarray:
    """Return the scenario/domain for each window without loading arrays."""
    domains = []
    for window in dataset.window_indexes:
        trace = dataset.traces[window.recording_idx]
        domains.append(trace.scenario)
    return np.asarray(domains)


def group_indices_by_label_and_domain(
    labels: np.ndarray,
    domains: np.ndarray,
) -> dict[tuple[int, str], np.ndarray]:
    """Group dataset window indices by class label and scenario/domain."""
    if labels.shape[0] != domains.shape[0]:
        raise ValueError(
            f"labels and domains must have the same length, got "
            f"{labels.shape[0]} and {domains.shape[0]}."
        )

    groups = {}
    for label in sorted(np.unique(labels)):
        for domain in sorted(np.unique(domains)):
            indices = np.flatnonzero((labels == label) & (domains == domain))
            if indices.size:
                groups[(int(label), str(domain))] = indices
    return groups


def sample_episode_indices(
    labels: np.ndarray,
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample support and query dataset indices for one episode."""
    groups = group_indices_by_label(labels)
    eligible_labels = [
        label for label, indices in groups.items()
        if indices.size >= k_shot + q_query
    ]
    if len(eligible_labels) < n_way:
        raise ValueError(
            f"Only {len(eligible_labels)} classes have at least "
            f"k_shot + q_query = {k_shot + q_query} samples; n_way={n_way}."
        )

    episode_labels = rng.choice(eligible_labels, size=n_way, replace=False)
    support_indices = []
    query_indices = []
    for label in episode_labels:
        sampled = rng.choice(groups[int(label)], size=k_shot + q_query, replace=False)
        support_indices.extend(sampled[:k_shot].tolist())
        query_indices.extend(sampled[k_shot:].tolist())

    return np.asarray(support_indices, dtype=np.int64), np.asarray(query_indices, dtype=np.int64)


def sample_domain_cross_episode_indices(
    labels: np.ndarray,
    domains: np.ndarray,
    *,
    support_domain: str,
    query_domain: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an episode with support/query constrained to specific domains.

    This is useful for domain-cross prototypical training: a prototype is built
    from one source domain, while queries for the same identities come from a
    different source domain.
    """
    groups = group_indices_by_label_and_domain(labels, domains)
    same_domain = support_domain == query_domain

    eligible_labels = []
    for label in sorted(np.unique(labels)):
        support_indices = groups.get((int(label), support_domain), np.asarray([], dtype=np.int64))
        query_indices = groups.get((int(label), query_domain), np.asarray([], dtype=np.int64))
        if same_domain:
            if support_indices.size >= k_shot + q_query:
                eligible_labels.append(int(label))
        elif support_indices.size >= k_shot and query_indices.size >= q_query:
            eligible_labels.append(int(label))

    if len(eligible_labels) < n_way:
        raise ValueError(
            f"Only {len(eligible_labels)} classes have enough samples for "
            f"{support_domain!r}->{query_domain!r}; n_way={n_way}, "
            f"k_shot={k_shot}, q_query={q_query}."
        )

    episode_labels = rng.choice(eligible_labels, size=n_way, replace=False)
    support_episode_indices = []
    query_episode_indices = []
    for label in episode_labels:
        support_pool = groups[(int(label), support_domain)]
        if same_domain:
            sampled = rng.choice(support_pool, size=k_shot + q_query, replace=False)
            support_episode_indices.extend(sampled[:k_shot].tolist())
            query_episode_indices.extend(sampled[k_shot:].tolist())
        else:
            query_pool = groups[(int(label), query_domain)]
            support_episode_indices.extend(
                rng.choice(support_pool, size=k_shot, replace=False).tolist()
            )
            query_episode_indices.extend(
                rng.choice(query_pool, size=q_query, replace=False).tolist()
            )

    return (
        np.asarray(support_episode_indices, dtype=np.int64),
        np.asarray(query_episode_indices, dtype=np.int64),
    )


def load_episode(dataset, support_indices: np.ndarray, query_indices: np.ndarray) -> Episode:
    """Load window tensors for sampled support/query indices."""
    support_samples = [dataset[int(index)] for index in support_indices]
    query_samples = [dataset[int(index)] for index in query_indices]

    support_x = torch.stack([x for x, _ in support_samples])
    support_y = torch.stack([y for _, y in support_samples])
    query_x = torch.stack([x for x, _ in query_samples])
    query_y = torch.stack([y for _, y in query_samples])

    return Episode(
        support_x=support_x,
        support_y=support_y,
        query_x=query_x,
        query_y=query_y,
    )


def sample_cross_dataset_episode_indices(
    support_labels: np.ndarray,
    query_labels: np.ndarray,
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample support/query indices when support and query are separate datasets."""
    support_groups = group_indices_by_label(support_labels)
    query_groups = group_indices_by_label(query_labels)
    eligible_labels = [
        label for label in sorted(set(support_groups) & set(query_groups))
        if support_groups[label].size >= k_shot and query_groups[label].size >= q_query
    ]
    if len(eligible_labels) < n_way:
        raise ValueError(
            f"Only {len(eligible_labels)} classes have enough support/query samples; "
            f"n_way={n_way}, k_shot={k_shot}, q_query={q_query}."
        )

    episode_labels = rng.choice(eligible_labels, size=n_way, replace=False)
    support_indices = []
    query_indices = []
    for label in episode_labels:
        support_indices.extend(
            rng.choice(support_groups[int(label)], size=k_shot, replace=False).tolist()
        )
        query_indices.extend(
            rng.choice(query_groups[int(label)], size=q_query, replace=False).tolist()
        )

    return np.asarray(support_indices, dtype=np.int64), np.asarray(query_indices, dtype=np.int64)


def load_cross_dataset_episode(
    support_dataset,
    support_indices: np.ndarray,
    query_dataset,
    query_indices: np.ndarray,
) -> Episode:
    """Load one episode with support and query windows from different datasets."""
    support_samples = [support_dataset[int(index)] for index in support_indices]
    query_samples = [query_dataset[int(index)] for index in query_indices]

    return Episode(
        support_x=torch.stack([x for x, _ in support_samples]),
        support_y=torch.stack([y for _, y in support_samples]),
        query_x=torch.stack([x for x, _ in query_samples]),
        query_y=torch.stack([y for _, y in query_samples]),
    )


def sample_episode(
    dataset,
    *,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
) -> Episode:
    """Sample and load one prototypical-learning episode from a dataset."""
    labels = window_true_labels_from_recordings(dataset)
    support_indices, query_indices = sample_episode_indices(
        labels,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        rng=rng,
    )
    return load_episode(dataset, support_indices, query_indices)


def prototypical_loss(
    model: torch.nn.Module,
    episode: Episode,
    *,
    device: str | torch.device,
    embedding_fusion: str = "mean",
    metric: str = "cosine",
    temperature: float = 1.0,
    normalize_prototypes: bool = True,
) -> tuple[torch.Tensor, float]:
    """Compute prototypical loss and query accuracy for one episode."""
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    support_x = episode.support_x.to(device)
    support_y = episode.support_y.to(device)
    query_x = episode.query_x.to(device)
    query_y = episode.query_y.to(device)

    support_embeddings = model.forward_embedding(support_x, fusion=embedding_fusion)
    query_embeddings = model.forward_embedding(query_x, fusion=embedding_fusion)

    prototypes, prototype_labels = compute_prototypes(
        support_embeddings,
        support_y,
        normalize=normalize_prototypes,
    )
    logits = prototype_logits(query_embeddings, prototypes, metric=metric) / temperature
    targets = map_labels_to_prototype_indices(query_y, prototype_labels)

    loss = F.cross_entropy(logits, targets)
    accuracy = (logits.argmax(dim=1) == targets).float().mean().item()
    return loss, accuracy


def run_prototypical_steps(
    model: torch.nn.Module,
    dataset,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    n_episodes: int,
    n_way: int,
    k_shot: int,
    q_query: int,
    rng: np.random.Generator,
    embedding_fusion: str = "mean",
    metric: str = "cosine",
    temperature: float = 1.0,
) -> dict[str, float]:
    """Train for a fixed number of sampled prototypical episodes."""
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    for _ in range(n_episodes):
        episode = sample_episode(
            dataset,
            n_way=n_way,
            k_shot=k_shot,
            q_query=q_query,
            rng=rng,
        )

        optimizer.zero_grad()
        loss, accuracy = prototypical_loss(
            model,
            episode,
            device=device,
            embedding_fusion=embedding_fusion,
            metric=metric,
            temperature=temperature,
        )
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy

    return {
        "loss": total_loss / n_episodes,
        "acc": total_acc / n_episodes,
    }
