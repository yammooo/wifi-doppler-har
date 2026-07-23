from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
import math
from pathlib import Path
import queue
import random
import threading
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillationBatch:
    inputs: torch.Tensor
    targets: torch.Tensor
    filenames: tuple[str, ...]


@dataclass
class EpochResult:
    metrics: dict[str, float]
    num_batches: int
    num_samples: int
    global_step: int
    examples: list[dict[str, Any]]


def distillation_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    name: str = "mse",
    options: dict[str, Any] | None = None,
) -> torch.Tensor:
    """Compute the configured student/teacher regression objective."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {tuple(predictions.shape)} != {tuple(targets.shape)}"
        )
    if name == "mse":
        return F.mse_loss(predictions, targets)
    if name == "motion_weighted_wasserstein":
        return motion_weighted_wasserstein_loss(predictions, targets, **(options or {}))
    raise ValueError(f"Unknown distillation loss: {name!r}")


def motion_weighted_wasserstein_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    floor: float = 10**-1.2,
    center_half_width: int = 1,
    motion_weight: float = 4.0,
    wasserstein_weight: float = 0.1,
    raw_mse_weight: float = 0.05,
    smooth_l1_beta: float = 0.1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compare active Doppler amplitude and its location along the last axis."""
    if predictions.ndim != 4:
        raise ValueError(
            "Expected predictions [batch, antenna, time, doppler_bin], "
            f"got {tuple(predictions.shape)}"
        )
    if center_half_width < 0 or center_half_width >= predictions.shape[-1] // 2:
        raise ValueError("center_half_width must select a proper subset of Doppler bins.")

    # Keep the distribution calculations stable under CUDA autocast.
    predictions = predictions.float()
    targets = targets.float()
    pred_active = (predictions - floor).clamp_min(0)
    target_active = (targets - floor).clamp_min(0)

    target_mass = target_active.sum(dim=-1)
    center = predictions.shape[-1] // 2
    center_mass = target_active[
        ..., center - center_half_width : center + center_half_width + 1
    ].sum(dim=-1)
    motion_ratio = (target_mass - center_mass).clamp_min(0) / target_mass.clamp_min(eps)
    frame_weight = 1.0 + motion_weight * motion_ratio

    active_frame_error = F.smooth_l1_loss(
        pred_active,
        target_active,
        reduction="none",
        beta=smooth_l1_beta,
    ).mean(dim=-1)
    active_map_loss = (active_frame_error * frame_weight).sum() / frame_weight.sum()

    num_bins = predictions.shape[-1]
    pred_distribution = (pred_active + eps) / (
        pred_active.sum(dim=-1, keepdim=True) + eps * num_bins
    )
    target_distribution = (target_active + eps) / (
        target_active.sum(dim=-1, keepdim=True) + eps * num_bins
    )
    frame_wasserstein = (
        pred_distribution.cumsum(dim=-1) - target_distribution.cumsum(dim=-1)
    ).abs().sum(dim=-1) / (num_bins - 1)
    wasserstein_loss = (frame_wasserstein * frame_weight).sum() / frame_weight.sum()

    return (
        active_map_loss
        + wasserstein_weight * wasserstein_loss
        + raw_mse_weight * F.mse_loss(predictions, targets)
    )


class DistillationMetricAccumulator:
    """Accumulate regression metrics without retaining full predictions."""

    def __init__(self, num_antennas: int):
        self.num_antennas = num_antennas
        self._stats: torch.Tensor | None = None
        self.element_count = 0
        self.per_antenna_count = np.zeros(num_antennas, dtype=np.int64)
        self.peak_count = 0

    def update(self, predictions: torch.Tensor, targets: torch.Tensor) -> None:
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Prediction and target shapes differ: {tuple(predictions.shape)} != {tuple(targets.shape)}"
            )
        if predictions.ndim != 4 or predictions.shape[1] != self.num_antennas:
            raise ValueError(
                "Expected predictions [batch, antenna, time, doppler_bin] with "
                f"{self.num_antennas} antennas, got {tuple(predictions.shape)}"
            )

        error = predictions.detach() - targets.detach()
        squared_error = error.square()
        per_antenna_sse = squared_error.sum(dim=(0, 2, 3))
        predicted_peaks = predictions.detach().argmax(dim=-1)
        target_peaks = targets.detach().argmax(dim=-1)
        batch_stats = torch.cat(
            (
                per_antenna_sse.sum().reshape(1),
                error.abs().sum().reshape(1),
                per_antenna_sse,
                (predicted_peaks - target_peaks).abs().sum().reshape(1),
                (predictions.detach() < 0).sum().reshape(1),
            )
        ).double()
        if self._stats is None:
            self._stats = torch.zeros_like(batch_stats)
        self._stats.add_(batch_stats)

        self.element_count += error.numel()
        per_antenna_elements = predictions.shape[0] * predictions.shape[2] * predictions.shape[3]
        self.per_antenna_count += per_antenna_elements
        self.peak_count += predicted_peaks.numel()

    def compute(self) -> dict[str, float]:
        if self.element_count == 0 or self._stats is None:
            raise ValueError("No samples were accumulated.")
        stats = self._stats.cpu().numpy()
        total_sse = float(stats[0])
        total_absolute_error = float(stats[1])
        per_antenna_sse = stats[2 : 2 + self.num_antennas]
        peak_absolute_error = float(stats[-2])
        negative_count = float(stats[-1])
        mse = total_sse / self.element_count
        metrics = {
            "mse": mse,
            "mae": total_absolute_error / self.element_count,
            "rmse": math.sqrt(mse),
            "peak_bin_mae": peak_absolute_error / self.peak_count,
            "negative_fraction": negative_count / self.element_count,
        }
        for antenna in range(self.num_antennas):
            metrics[f"mse_antenna_{antenna}"] = per_antenna_sse[antenna] / self.per_antenna_count[antenna]
        return metrics


def count_recording_batches(dataset, batch_size: int) -> int:
    """Return the number of per-recording batches emitted for one full pass."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    counts = np.zeros(len(dataset.traces), dtype=np.int64)
    num_views = len(dataset.subcarrier_views)
    for window in dataset.window_indexes:
        counts[window.recording_idx] += num_views
    return sum(math.ceil(int(count) / batch_size) for count in counts if count)


def iter_recording_batches(
    dataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[DistillationBatch]:
    """Yield all windows while loading each backing recording once.

    Batches never cross recording boundaries. This bounds host memory to one
    recording and avoids re-reading large MAT/pickle files for every window.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    by_recording: dict[int, list[tuple[int, int]]] = {}
    for base_idx, window in enumerate(dataset.window_indexes):
        items = by_recording.setdefault(window.recording_idx, [])
        items.extend((base_idx, view_idx) for view_idx in range(len(dataset.subcarrier_views)))

    rng = np.random.default_rng(seed)
    recording_order = np.asarray(sorted(by_recording), dtype=np.int64)
    if shuffle:
        rng.shuffle(recording_order)

    for recording_idx_value in recording_order:
        recording_idx = int(recording_idx_value)
        recording = dataset.traces[recording_idx]
        items = list(by_recording[recording_idx])
        if shuffle:
            rng.shuffle(items)

        raw = recording.load_raw()
        doppler = recording.load_doppler()
        try:
            for offset in range(0, len(items), batch_size):
                batch_items = items[offset : offset + batch_size]
                inputs: list[np.ndarray] = []
                targets: list[np.ndarray] = []
                filenames: list[str] = []

                for base_idx, view_idx in batch_items:
                    window = dataset.window_indexes[base_idx]
                    selected_subcarriers = dataset.subcarrier_views[view_idx]
                    raw_start, raw_end = dataset.raw_bounds_for_doppler_window(window.start, window.end)

                    x_complex = raw[:, selected_subcarriers, raw_start:raw_end]
                    x = np.stack((x_complex.real, x_complex.imag), axis=-1).astype(np.float32, copy=False)
                    y = doppler[:, window.start:window.end].astype(np.float32, copy=False)
                    if not np.isfinite(x).all() or not np.isfinite(y).all():
                        raise ValueError(
                            f"Non-finite CSI/Doppler values in {recording.filename_stem} "
                            f"at Doppler window [{window.start}, {window.end})"
                        )

                    inputs.append(x)
                    targets.append(y)
                    filenames.append(
                        f"{recording.filename_stem}_d{window.start}-{window.end}_view{view_idx}"
                    )

                yield DistillationBatch(
                    inputs=torch.from_numpy(np.ascontiguousarray(np.stack(inputs))),
                    targets=torch.from_numpy(np.ascontiguousarray(np.stack(targets))),
                    filenames=tuple(filenames),
                )
        finally:
            recording.clear_cache()


@dataclass(frozen=True)
class _ProducerFailure:
    error: BaseException


_PREFETCH_END = object()


def prefetch_batches(
    batches: Iterator[DistillationBatch],
    *,
    max_prefetch: int,
    pin_memory: bool,
) -> Iterator[DistillationBatch]:
    """Prepare a bounded number of host batches on a background thread."""
    if max_prefetch <= 0:
        yield from batches
        return

    pending: queue.Queue[DistillationBatch | _ProducerFailure | object] = queue.Queue(max_prefetch)
    stopped = threading.Event()

    def put(item: DistillationBatch | _ProducerFailure | object) -> bool:
        while not stopped.is_set():
            try:
                pending.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def produce() -> None:
        try:
            for batch in batches:
                if pin_memory:
                    batch = DistillationBatch(
                        inputs=batch.inputs.pin_memory(),
                        targets=batch.targets.pin_memory(),
                        filenames=batch.filenames,
                    )
                if not put(batch):
                    return
        except BaseException as exc:
            put(_ProducerFailure(exc))
        finally:
            put(_PREFETCH_END)

    producer = threading.Thread(target=produce, name="csi-doppler-prefetch", daemon=True)
    producer.start()
    try:
        while True:
            item = pending.get()
            if item is _PREFETCH_END:
                break
            if isinstance(item, _ProducerFailure):
                raise RuntimeError("Background batch preparation failed") from item.error
            if not isinstance(item, DistillationBatch):
                raise TypeError(f"Unexpected prefetch item: {type(item).__name__}")
            yield item
    finally:
        stopped.set()


def move_batches_to_device(
    batches: Iterator[DistillationBatch],
    *,
    device: torch.device,
    cuda_prefetch: bool,
) -> Iterator[DistillationBatch]:
    """Move batches to the device, optionally one batch ahead on a CUDA stream."""
    if device.type != "cuda" or not cuda_prefetch:
        for batch in batches:
            yield DistillationBatch(
                inputs=batch.inputs.to(device, non_blocking=device.type == "cuda"),
                targets=batch.targets.to(device, non_blocking=device.type == "cuda"),
                filenames=batch.filenames,
            )
        return

    iterator = iter(batches)
    transfer_stream = torch.cuda.Stream(device=device)

    def transfer(batch: DistillationBatch) -> DistillationBatch:
        with torch.cuda.stream(transfer_stream):
            return DistillationBatch(
                inputs=batch.inputs.to(device, non_blocking=True),
                targets=batch.targets.to(device, non_blocking=True),
                filenames=batch.filenames,
            )

    try:
        next_batch = transfer(next(iterator))
    except StopIteration:
        return

    while True:
        current_stream = torch.cuda.current_stream(device)
        current_stream.wait_stream(transfer_stream)
        current_batch = next_batch
        current_batch.inputs.record_stream(current_stream)
        current_batch.targets.record_stream(current_stream)
        try:
            next_batch = transfer(next(iterator))
        except StopIteration:
            yield current_batch
            break
        yield current_batch


def run_distillation_epoch(
    model: torch.nn.Module,
    batches: Iterator[DistillationBatch],
    *,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    amp_enabled: bool = False,
    loss_name: str = "mse",
    loss_options: dict[str, Any] | None = None,
    global_step: int = 0,
    max_examples: int = 0,
    batch_callback: Callable[[int, dict[str, float]], None] | None = None,
    batch_callback_every: int = 1,
) -> EpochResult:
    """Run one training or evaluation epoch over pre-batched recordings."""
    if batch_callback_every < 1:
        raise ValueError("batch_callback_every must be >= 1")
    training = optimizer is not None
    model.train(training)
    accumulator = DistillationMetricAccumulator(num_antennas=int(model.num_antennas))
    examples: list[dict[str, Any]] = []
    num_batches = 0
    num_samples = 0
    objective_sum = 0.0
    started_at = time.perf_counter()

    grad_context = torch.enable_grad if training else torch.inference_mode
    with grad_context():
        for batch in batches:
            inputs = batch.inputs
            targets = batch.targets
            if inputs.device != device or targets.device != device:
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(inputs)
                loss = distillation_loss(
                    predictions,
                    targets,
                    name=loss_name,
                    options=loss_options,
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite {loss_name} loss for batch beginning with {batch.filenames[0]}"
                )

            if training:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                global_step += 1

            accumulator.update(predictions, targets)
            if batch_callback is not None and global_step % batch_callback_every == 0:
                batch_callback(global_step, {"loss": float(loss.detach().item())})

            remaining_examples = max_examples - len(examples)
            if remaining_examples > 0:
                for sample_idx in range(min(remaining_examples, predictions.shape[0])):
                    examples.append(
                        {
                            "filename": batch.filenames[sample_idx],
                            "prediction": predictions[sample_idx].detach().float().cpu(),
                            "target": targets[sample_idx].detach().float().cpu(),
                        }
                    )

            num_batches += 1
            num_samples += inputs.shape[0]
            objective_sum += float(loss.detach().item()) * inputs.shape[0]

    elapsed = time.perf_counter() - started_at
    metrics = accumulator.compute()
    metrics["loss"] = objective_sum / num_samples
    metrics["samples_per_second"] = num_samples / elapsed if elapsed > 0 else 0.0
    metrics["elapsed_seconds"] = elapsed
    return EpochResult(
        metrics=metrics,
        num_batches=num_batches,
        num_samples=num_samples,
        global_step=global_step,
        examples=examples,
    )


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    epoch: int,
    global_step: int,
    best_metric: float,
    patience_counter: int,
    config: dict[str, Any],
    history: list[dict[str, Any]],
    wandb_run_id: str | None,
) -> Path:
    checkpoint = {
        "checkpoint_type": "csi_to_doppler_training",
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "global_step": global_step,
        "best_metric": best_metric,
        "patience_counter": patience_counter,
        "config": config,
        "history": history,
        "wandb_run_id": wandb_run_id,
        "rng_state": capture_rng_state(),
    }
    return _atomic_torch_save(checkpoint, path)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Any | None,
    map_location: str | torch.device,
) -> dict[str, Any]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if checkpoint.get("checkpoint_type") != "csi_to_doppler_training":
        raise ValueError(f"Not a CSI-to-Doppler training checkpoint: {path}")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    restore_rng_state(checkpoint["rng_state"])
    return checkpoint


def save_inference_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    epoch: int,
    metrics: dict[str, float],
    config: dict[str, Any],
) -> Path:
    checkpoint = {
        "checkpoint_type": "csi_to_doppler_inference",
        "architecture": type(model).__name__,
        "architecture_version": getattr(model, "architecture_version", None),
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "config": config,
    }
    return _atomic_torch_save(checkpoint, path)


def _atomic_torch_save(value: dict[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(target)
    return target
