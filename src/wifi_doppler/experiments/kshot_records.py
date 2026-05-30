from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from wifi_doppler.evaluation.fewshot import evaluate_kshot
from wifi_doppler.experiments.model_builders import ModelSpec, load_model_from_spec
from wifi_doppler.experiments.protocols import (
    DEFAULT_K_VALUES,
    DEFAULT_PERSONS,
    KShotProtocol,
    build_kshot_datasets,
)
from wifi_doppler.experiments.registry import make_record


def evaluate_kshot_record(
    *,
    project_root: str | Path,
    model_spec: ModelSpec,
    protocol: KShotProtocol,
    device: str | torch.device,
    k_values: tuple[int, ...] = DEFAULT_K_VALUES,
    n_trials: int = 20,
    seed: int = 0,
    batch_size: int = 128,
    window_size: int = 340,
    window_stride: int = 30,
    split_guard: int = 31,
    embedding_fusion: str = "mean",
    metric: str = "cosine",
) -> dict[str, Any]:
    persons = DEFAULT_PERSONS
    model, _ = load_model_from_spec(
        model_spec,
        device=device,
        num_classes=len(persons),
        window_size=window_size,
        embedding_fusion=embedding_fusion,
    )
    enrollment_dataset, query_dataset, dataset_metadata = build_kshot_datasets(
        project_root=project_root,
        representation=model_spec.representation,
        protocol=protocol,
        persons=persons,
        window_size=window_size,
        window_stride=window_stride,
        split_guard=split_guard,
    )
    results = evaluate_kshot(
        model,
        enrollment_dataset,
        query_dataset,
        k_values,
        device,
        n_trials=n_trials,
        seed=seed,
        batch_size=batch_size,
        embedding_fusion=embedding_fusion,
        metric=metric,
    )
    enrollment_dataset.clear_cache()
    query_dataset.clear_cache()

    metrics = {
        "k_values": list(k_values),
        "n_trials": n_trials,
        "seed": seed,
        "batch_size": batch_size,
        "embedding_fusion": embedding_fusion,
        "metric": metric,
        "results": {
            str(k): {
                "mean": float(value["mean"]),
                "std": float(value["std"]),
                "trials": [float(v) for v in value["trials"]],
            }
            for k, value in results.items()
        },
    }
    return make_record(
        record_type="evaluation.kshot",
        stem=f"{model_spec.key}_{protocol.name}",
        source={"mode": "computed", "legacy_paths": []},
        model=model_spec.to_record_model(),
        dataset=dataset_metadata,
        protocol=protocol.to_dict(),
        metrics=metrics,
        artifacts={},
    )
