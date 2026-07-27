from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
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
    return distillation_loss_components(
        predictions,
        targets,
        name=name,
        options=options,
    )["loss"]


def distillation_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    name: str = "mse",
    options: dict[str, Any] | None = None,
) -> dict[str, torch.Tensor]:
    """Compute the objective and its separately loggable components."""
    if predictions.shape != targets.shape:
        raise ValueError(
            f"Prediction and target shapes differ: {tuple(predictions.shape)} != {tuple(targets.shape)}"
        )
    if name == "mse":
        loss = F.mse_loss(predictions, targets)
        return {"loss": loss, "loss_full_map_mse": loss}
    if name == "motion_weighted_wasserstein":
        return {
            "loss": motion_weighted_wasserstein_loss(
                predictions,
                targets,
                **(options or {}),
            )
        }
    if name == "motion_aware_mse":
        return motion_aware_mse_loss_components(
            predictions,
            targets,
            **(options or {}),
        )
    raise ValueError(f"Unknown distillation loss: {name!r}")


def motion_aware_mse_loss_components(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    floor: float = 10**-1.2,
    center_half_width: int = 5,
    center_taper_sigma_bins: float | None = None,
    motion_mse_weight: float = 0.25,
    background_leakage_weight: float = 0.25,
    wasserstein_weight: float = 0.05,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Emphasize target-active off-center bins while suppressing false power."""
    if predictions.ndim != 4:
        raise ValueError(
            "Expected predictions [batch, antenna, time, doppler_bin], "
            f"got {tuple(predictions.shape)}"
        )
    if not 0 <= floor < 1:
        raise ValueError("floor must be in [0, 1).")
    if center_taper_sigma_bins is None:
        if center_half_width < 0 or center_half_width >= predictions.shape[-1] // 2:
            raise ValueError("center_half_width must select a proper subset of Doppler bins.")
    elif center_taper_sigma_bins <= 0:
        raise ValueError("center_taper_sigma_bins must be positive.")
    if min(motion_mse_weight, background_leakage_weight, wasserstein_weight) < 0:
        raise ValueError("Loss weights must be non-negative.")

    predictions = predictions.float()
    targets = targets.float()
    num_bins = predictions.shape[-1]
    center = num_bins // 2
    if center_taper_sigma_bins is None:
        motion_bin_weight = torch.ones(
            num_bins,
            dtype=predictions.dtype,
            device=predictions.device,
        )
        motion_bin_weight[center - center_half_width : center + center_half_width + 1] = 0
    else:
        distance = torch.arange(
            num_bins,
            dtype=predictions.dtype,
            device=predictions.device,
        ) - center
        motion_bin_weight = 1 - torch.exp(
            -0.5 * (distance / center_taper_sigma_bins).square()
        )

    squared_error = (predictions - targets).square()
    full_map_mse = squared_error.mean()

    target_activity = (
        ((targets - floor) / (1 - floor)).clamp(0, 1) * motion_bin_weight
    )
    target_activity_sum = target_activity.sum()
    motion_mse = (target_activity * squared_error).sum() / target_activity_sum.clamp_min(eps)

    background = (targets <= floor + 1e-6).to(predictions.dtype)
    background_leakage = (
        background * (predictions - floor).clamp_min(0)
    ).sum() / background.sum().clamp_min(1)

    pred_active = (predictions - floor).clamp_min(0) * motion_bin_weight
    target_active = (targets - floor).clamp_min(0) * motion_bin_weight
    motion_bin_weight_sum = motion_bin_weight.sum()
    pred_distribution = (pred_active + eps * motion_bin_weight) / (
        pred_active.sum(dim=-1, keepdim=True) + eps * motion_bin_weight_sum
    )
    target_distribution = (target_active + eps * motion_bin_weight) / (
        target_active.sum(dim=-1, keepdim=True) + eps * motion_bin_weight_sum
    )
    frame_wasserstein = (
        pred_distribution.cumsum(dim=-1) - target_distribution.cumsum(dim=-1)
    ).abs().sum(dim=-1) / (num_bins - 1)
    frame_motion_weight = target_activity.sum(dim=-1)
    motion_wasserstein = (
        frame_wasserstein * frame_motion_weight
    ).sum() / frame_motion_weight.sum().clamp_min(eps)

    weighted_motion_mse = motion_mse_weight * motion_mse
    weighted_background_leakage = background_leakage_weight * background_leakage
    weighted_motion_wasserstein = wasserstein_weight * motion_wasserstein
    loss = (
        full_map_mse
        + weighted_motion_mse
        + weighted_background_leakage
        + weighted_motion_wasserstein
    )
    return {
        "loss": loss,
        "loss_full_map_mse": full_map_mse,
        "loss_motion_mse": weighted_motion_mse,
        "loss_background_leakage": weighted_background_leakage,
        "loss_motion_wasserstein": weighted_motion_wasserstein,
        "motion_mse": motion_mse,
        "background_leakage": background_leakage,
        "motion_wasserstein": motion_wasserstein,
    }


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
    recordings_per_batch: int = 1,
    window_shuffle_chunk_size: int = 1,
    batch_preparation_workers: int = 1,
) -> Iterator[DistillationBatch]:
    """Yield all windows from bounded pools of backing recordings."""
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    if recordings_per_batch < 1 or recordings_per_batch > batch_size:
        raise ValueError("recordings_per_batch must be between 1 and batch_size.")
    if window_shuffle_chunk_size < 1:
        raise ValueError("window_shuffle_chunk_size must be >= 1.")
    if batch_preparation_workers < 1:
        raise ValueError("batch_preparation_workers must be >= 1.")

    by_recording: dict[int, list[tuple[int, int]]] = {}
    for base_idx, window in enumerate(dataset.window_indexes):
        items = by_recording.setdefault(window.recording_idx, [])
        items.extend((base_idx, view_idx) for view_idx in range(len(dataset.subcarrier_views)))
    subcarrier_selectors: list[slice | np.ndarray] = []
    for selected in dataset.subcarrier_views:
        start = int(selected[0])
        if np.array_equal(selected, np.arange(start, start + len(selected))):
            subcarrier_selectors.append(slice(start, start + len(selected)))
        else:
            subcarrier_selectors.append(selected)

    rng = np.random.default_rng(seed)
    recording_order = np.asarray(sorted(by_recording), dtype=np.int64)
    if shuffle:
        rng.shuffle(recording_order)

    for pool_offset in range(0, len(recording_order), recordings_per_batch):
        pool = [int(value) for value in recording_order[pool_offset : pool_offset + recordings_per_batch]]
        pool_items: dict[int, list[tuple[int, int]]] = {}
        for recording_idx in pool:
            items = list(by_recording[recording_idx])
            if shuffle:
                chunks = [
                    items[offset : offset + window_shuffle_chunk_size]
                    for offset in range(0, len(items), window_shuffle_chunk_size)
                ]
                rng.shuffle(chunks)
                items = [item for chunk in chunks for item in chunk]
            pool_items[recording_idx] = items

        interleaved: list[tuple[int, int, int]] = []
        positions = {recording_idx: 0 for recording_idx in pool}
        active = list(pool)
        while active:
            if shuffle:
                rng.shuffle(active)
            remaining = []
            for recording_idx in active:
                position = positions[recording_idx]
                items = pool_items[recording_idx]
                base_idx, view_idx = items[position]
                interleaved.append((recording_idx, base_idx, view_idx))
                positions[recording_idx] = position + 1
                if position + 1 < len(items):
                    remaining.append(recording_idx)
            active = remaining

        loaded: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        try:
            for recording_idx in pool:
                loaded[recording_idx] = (
                    dataset.traces[recording_idx].load_raw(),
                    dataset.traces[recording_idx].load_doppler(),
                )
            for offset in range(0, len(interleaved), batch_size):
                batch_items = sorted(
                    interleaved[offset : offset + batch_size],
                    key=lambda item: (
                        item[0],
                        dataset.window_indexes[item[1]].start,
                        item[2],
                    ),
                )
                inputs: np.ndarray | None = None
                targets: np.ndarray | None = None
                filenames: list[str] = []
                grouped_items: dict[
                    tuple[int, int],
                    list[tuple[int, int, int, int]],
                ] = {}
                for sample_idx, (recording_idx, base_idx, view_idx) in enumerate(batch_items):
                    recording = dataset.traces[recording_idx]
                    raw, doppler = loaded[recording_idx]
                    window = dataset.window_indexes[base_idx]
                    selected_subcarriers = dataset.subcarrier_views[view_idx]
                    raw_start, raw_end = dataset.raw_bounds_for_doppler_window(window.start, window.end)

                    expected_input_shape = (
                        doppler.shape[0],
                        len(selected_subcarriers),
                        raw_end - raw_start,
                        2,
                    )
                    expected_target_shape = (
                        doppler.shape[0],
                        window.end - window.start,
                        doppler.shape[-1],
                    )
                    raw_slice = slice(raw_start, raw_end).indices(raw.shape[-1])
                    doppler_slice = slice(window.start, window.end).indices(doppler.shape[1])
                    actual_input_shape = (
                        raw.shape[0],
                        len(selected_subcarriers),
                        len(range(*raw_slice)),
                        2,
                    )
                    actual_target_shape = (
                        doppler.shape[0],
                        len(range(*doppler_slice)),
                        doppler.shape[-1],
                    )
                    if (
                        actual_input_shape != expected_input_shape
                        or actual_target_shape != expected_target_shape
                    ):
                        raise ValueError(
                            f"Incompatible CSI/Doppler window for {recording.filename_stem}: "
                            f"input shape {actual_input_shape}, expected {expected_input_shape}; "
                            f"target shape {actual_target_shape}, expected {expected_target_shape}; "
                            f"raw backing shape {raw.shape}, Doppler backing shape {doppler.shape}; "
                            f"raw bounds [{raw_start}, {raw_end}), "
                            f"Doppler bounds [{window.start}, {window.end}). "
                            "The paired recording does not contain the configured aligned window."
                        )
                    if inputs is None:
                        inputs = np.empty(
                            (len(batch_items), *expected_input_shape),
                            dtype=np.float32,
                        )
                        targets = np.empty(
                            (len(batch_items), *expected_target_shape),
                            dtype=np.float32,
                        )

                    grouped_items.setdefault((recording_idx, view_idx), []).append(
                        (sample_idx, base_idx, raw_start, raw_end)
                    )
                    filenames.append(
                        f"{recording.filename_stem}_d{window.start}-{window.end}_view{view_idx}"
                    )

                assert inputs is not None and targets is not None

                def fill_group(
                    grouped_item: tuple[
                        tuple[int, int],
                        list[tuple[int, int, int, int]],
                    ],
                ) -> None:
                    (recording_idx, view_idx), items = grouped_item
                    raw, doppler = loaded[recording_idx]
                    selector = subcarrier_selectors[view_idx]
                    items.sort(key=lambda item: dataset.window_indexes[item[1]].start)
                    runs: list[list[tuple[int, int, int, int]]] = []
                    run_raw_end = -1
                    run_doppler_end = -1
                    for item in items:
                        window = dataset.window_indexes[item[1]]
                        if (
                            not runs
                            or item[2] > run_raw_end
                            or window.start > run_doppler_end
                        ):
                            runs.append([])
                            run_raw_end = item[3]
                            run_doppler_end = window.end
                        else:
                            run_raw_end = max(run_raw_end, item[3])
                            run_doppler_end = max(run_doppler_end, window.end)
                        runs[-1].append(item)

                    for run in runs:
                        raw_start = min(item[2] for item in run)
                        raw_end = max(item[3] for item in run)
                        doppler_start = min(dataset.window_indexes[item[1]].start for item in run)
                        doppler_end = max(dataset.window_indexes[item[1]].end for item in run)
                        raw_slab = np.ascontiguousarray(
                            raw[:, selector, raw_start:raw_end],
                            dtype=np.complex64,
                        )
                        doppler_slab = np.ascontiguousarray(
                            doppler[:, doppler_start:doppler_end],
                            dtype=np.float32,
                        )
                        for sample_idx, base_idx, sample_raw_start, sample_raw_end in run:
                            window = dataset.window_indexes[base_idx]
                            x_complex = raw_slab[
                                :,
                                :,
                                sample_raw_start - raw_start : sample_raw_end - raw_start,
                            ]
                            y = doppler_slab[
                                :,
                                window.start - doppler_start : window.end - doppler_start,
                            ]
                            inputs[sample_idx, ..., 0] = x_complex.real
                            inputs[sample_idx, ..., 1] = x_complex.imag
                            targets[sample_idx] = y

                groups = list(grouped_items.items())
                if batch_preparation_workers == 1 or len(groups) == 1:
                    for group in groups:
                        fill_group(group)
                else:
                    with ThreadPoolExecutor(
                        max_workers=min(batch_preparation_workers, len(groups)),
                    ) as executor:
                        list(executor.map(fill_group, groups))

                yield DistillationBatch(
                    inputs=torch.from_numpy(inputs),
                    targets=torch.from_numpy(targets),
                    filenames=tuple(filenames),
                )
        finally:
            for recording_idx in pool:
                dataset.traces[recording_idx].clear_cache()
            loaded.clear()


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
    loss_component_sums: dict[str, torch.Tensor] = {}
    started_at = time.perf_counter()
    data_wait_seconds = 0.0

    def measured_batches() -> Iterator[DistillationBatch]:
        nonlocal data_wait_seconds
        iterator = iter(batches)
        while True:
            wait_started_at = time.perf_counter()
            try:
                batch = next(iterator)
            except StopIteration:
                return
            data_wait_seconds += time.perf_counter() - wait_started_at
            yield batch

    grad_context = torch.enable_grad if training else torch.inference_mode
    with grad_context():
        for batch in measured_batches():
            inputs = batch.inputs
            targets = batch.targets
            if inputs.device != device or targets.device != device:
                inputs = inputs.to(device, non_blocking=device.type == "cuda")
                targets = targets.to(device, non_blocking=device.type == "cuda")

            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(inputs)
                loss_components = distillation_loss_components(
                    predictions,
                    targets,
                    name=loss_name,
                    options=loss_options,
                )
                loss = loss_components["loss"]
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
                batch_callback(
                    global_step,
                    {
                        name: float(value.detach().item())
                        for name, value in loss_components.items()
                    },
                )

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
            for name, value in loss_components.items():
                if name == "loss":
                    continue
                batch_component = value.detach() * inputs.shape[0]
                loss_component_sums[name] = (
                    loss_component_sums[name] + batch_component
                    if name in loss_component_sums
                    else batch_component
                )

    elapsed = time.perf_counter() - started_at
    metrics = accumulator.compute()
    metrics["loss"] = objective_sum / num_samples
    metrics.update(
        {
            name: float(component_sum.item()) / num_samples
            for name, component_sum in loss_component_sums.items()
        }
    )
    metrics["samples_per_second"] = num_samples / elapsed if elapsed > 0 else 0.0
    metrics["elapsed_seconds"] = elapsed
    metrics["data_wait_seconds"] = data_wait_seconds
    metrics["data_wait_fraction"] = data_wait_seconds / elapsed if elapsed > 0 else 0.0
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
