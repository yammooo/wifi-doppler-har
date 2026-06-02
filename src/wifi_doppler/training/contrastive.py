from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from wifi_doppler.training.prototypical import load_windows_by_recording


@dataclass(frozen=True)
class ContrastiveBatch:
    """Window tensors, labels, and optional domain ids for contrastive training."""

    x: torch.Tensor
    y: torch.Tensor
    domains: torch.Tensor | None = None


def build_positive_weights(
    labels: torch.Tensor,
    domains: torch.Tensor | None = None,
    *,
    same_domain_positive_weight: float = 0.25,
    cross_domain_positive_weight: float = 1.0,
) -> torch.Tensor:
    """Return pairwise positive weights for supervised contrastive learning.

    Same-label pairs are positives. If domains are given, same-domain positives
    and cross-domain positives can be weighted differently. Self-pairs are
    always ignored.
    """
    if same_domain_positive_weight < 0 or cross_domain_positive_weight < 0:
        raise ValueError("positive weights must be non-negative.")

    labels = labels.view(-1)
    same_label = labels[:, None] == labels[None, :]
    not_self = ~torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)

    if domains is None:
        return (same_label & not_self).float()

    domains = domains.view(-1).to(labels.device)
    if domains.numel() != labels.numel():
        raise ValueError(
            f"domains and labels must have the same length, got "
            f"{domains.numel()} and {labels.numel()}."
        )

    same_domain = domains[:, None] == domains[None, :]
    weights = torch.zeros(
        labels.numel(),
        labels.numel(),
        dtype=torch.float32,
        device=labels.device,
    )
    weights[same_label & same_domain & not_self] = same_domain_positive_weight
    weights[same_label & ~same_domain & not_self] = cross_domain_positive_weight
    return weights


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    domains: torch.Tensor | None = None,
    *,
    temperature: float = 0.1,
    same_domain_positive_weight: float = 0.25,
    cross_domain_positive_weight: float = 1.0,
) -> torch.Tensor:
    """Compute weighted supervised contrastive loss.

    The denominator contains every non-self sample in the batch. Positive terms
    are weighted by ``build_positive_weights``. Anchors with no positive weight
    are excluded from the final average.
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}.")

    labels = labels.view(-1).to(embeddings.device)
    if domains is not None:
        domains = domains.view(-1).to(embeddings.device)

    embeddings = F.normalize(embeddings, dim=1)
    logits = embeddings @ embeddings.T / temperature

    not_self = ~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    logits = logits.masked_fill(~not_self, float("-inf"))
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    log_prob = log_prob.masked_fill(~not_self, 0.0)

    positive_weights = build_positive_weights(
        labels,
        domains,
        same_domain_positive_weight=same_domain_positive_weight,
        cross_domain_positive_weight=cross_domain_positive_weight,
    )
    positive_sums = positive_weights.sum(dim=1)
    valid_anchors = positive_sums > 0
    if not torch.any(valid_anchors):
        raise ValueError("contrastive batch has no positive pairs.")

    weighted_log_prob = (positive_weights * log_prob).sum(dim=1) / positive_sums.clamp_min(1e-12)
    return -weighted_log_prob[valid_anchors].mean()


def sample_contrastive_indices(
    labels: np.ndarray,
    domains: np.ndarray | None,
    *,
    classes_per_batch: int,
    samples_per_class_per_domain: int,
    domains_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a balanced contrastive batch.

    When domains are provided, each selected class contributes samples from
    ``domains_per_class`` domains. This creates cross-domain positives and
    same-domain hard negatives inside the same batch.
    """
    if classes_per_batch <= 0:
        raise ValueError("classes_per_batch must be positive.")
    if samples_per_class_per_domain <= 0:
        raise ValueError("samples_per_class_per_domain must be positive.")
    if domains_per_class <= 0:
        raise ValueError("domains_per_class must be positive.")

    labels = np.asarray(labels)
    if domains is None:
        groups = {
            int(label): np.flatnonzero(labels == label)
            for label in sorted(np.unique(labels))
        }
        eligible_labels = [
            label for label, indices in groups.items()
            if indices.size >= samples_per_class_per_domain
        ]
        if len(eligible_labels) < classes_per_batch:
            raise ValueError(
                f"Only {len(eligible_labels)} classes have enough samples; "
                f"classes_per_batch={classes_per_batch}."
            )

        batch_labels = rng.choice(eligible_labels, size=classes_per_batch, replace=False)
        indices = []
        for label in batch_labels:
            indices.extend(
                rng.choice(
                    groups[int(label)],
                    size=samples_per_class_per_domain,
                    replace=False,
                ).tolist()
            )
        return np.asarray(indices, dtype=np.int64)

    domains = np.asarray(domains)
    if labels.shape[0] != domains.shape[0]:
        raise ValueError(
            f"labels and domains must have the same length, got "
            f"{labels.shape[0]} and {domains.shape[0]}."
        )

    groups = {}
    eligible_domains_by_label = {}
    for label in sorted(np.unique(labels)):
        label_domains = []
        for domain in sorted(np.unique(domains)):
            group_indices = np.flatnonzero((labels == label) & (domains == domain))
            if group_indices.size >= samples_per_class_per_domain:
                groups[(int(label), str(domain))] = group_indices
                label_domains.append(str(domain))
        if len(label_domains) >= domains_per_class:
            eligible_domains_by_label[int(label)] = label_domains

    eligible_labels = sorted(eligible_domains_by_label)
    if len(eligible_labels) < classes_per_batch:
        raise ValueError(
            f"Only {len(eligible_labels)} classes have enough domain-balanced "
            f"samples; classes_per_batch={classes_per_batch}."
        )

    batch_labels = rng.choice(eligible_labels, size=classes_per_batch, replace=False)
    indices = []
    for label in batch_labels:
        selected_domains = rng.choice(
            eligible_domains_by_label[int(label)],
            size=domains_per_class,
            replace=False,
        )
        for domain in selected_domains:
            indices.extend(
                rng.choice(
                    groups[(int(label), str(domain))],
                    size=samples_per_class_per_domain,
                    replace=False,
                ).tolist()
            )

    return np.asarray(indices, dtype=np.int64)


def load_contrastive_batch(
    dataset,
    indices: np.ndarray,
    *,
    include_domains: bool = True,
) -> ContrastiveBatch:
    """Load sampled windows for one contrastive batch."""
    x, y = load_windows_by_recording(dataset, indices)
    if not include_domains:
        return ContrastiveBatch(x=x, y=y)

    domain_names = []
    for index in indices:
        window = dataset.window_indexes[int(index)]
        domain_names.append(str(dataset.traces[window.recording_idx].scenario))

    domain_to_id = {domain: idx for idx, domain in enumerate(sorted(set(domain_names)))}
    domains = torch.tensor([domain_to_id[domain] for domain in domain_names], dtype=torch.long)
    return ContrastiveBatch(x=x, y=y, domains=domains)


def contrastive_step(
    model: torch.nn.Module,
    batch: ContrastiveBatch,
    optimizer: torch.optim.Optimizer,
    *,
    device: str | torch.device,
    temperature: float = 0.1,
    same_domain_positive_weight: float = 0.25,
    cross_domain_positive_weight: float = 1.0,
) -> dict[str, float]:
    """Run one supervised contrastive optimizer step."""
    model.train()

    x = batch.x.to(device)
    y = batch.y.to(device)
    domains = batch.domains.to(device) if batch.domains is not None else None

    optimizer.zero_grad()
    embeddings = model.forward_embedding(x)
    loss = supervised_contrastive_loss(
        embeddings,
        y,
        domains,
        temperature=temperature,
        same_domain_positive_weight=same_domain_positive_weight,
        cross_domain_positive_weight=cross_domain_positive_weight,
    )
    loss.backward()
    optimizer.step()

    return {
        "loss": loss.item(),
        "batch_retrieval_acc": batch_retrieval_accuracy(embeddings.detach(), y),
    }


def batch_retrieval_accuracy(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Return same-batch nearest-neighbor label accuracy.

    This is a training diagnostic for contrastive learning, not the final PI
    metric. The final metric remains K-shot prototype accuracy.
    """
    labels = labels.view(-1).to(embeddings.device)
    embeddings = F.normalize(embeddings, dim=1)
    similarities = embeddings @ embeddings.T
    not_self = ~torch.eye(similarities.shape[0], dtype=torch.bool, device=similarities.device)
    similarities = similarities.masked_fill(~not_self, float("-inf"))
    nearest = similarities.argmax(dim=1)
    return (labels[nearest] == labels).float().mean().item()
