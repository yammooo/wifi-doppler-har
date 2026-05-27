from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def create_run_dir(
    project_root: Path,
    group: str,
    run_name: str,
    *,
    timestamp: str | None = None,
) -> Path:
    """Create a timestamped experiment run directory."""
    timestamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(project_root) / "experiments" / group / f"{run_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_json(path: Path, value: Any) -> Path:
    """Save JSON with support for common experiment objects."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding="utf-8")
    return path


def save_figure(fig, run_dir: Path, name: str, *, dpi: int = 150) -> Path:
    """Save a Matplotlib figure inside a run directory."""
    path = Path(run_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    return path


def save_checkpoint(
    model: torch.nn.Module,
    run_dir: Path,
    *,
    labels: list[str] | tuple[str, ...],
    config: dict[str, Any],
    metrics: dict[str, Any],
    history: dict[str, Any] | None = None,
    name: str = "model.pt",
) -> Path:
    """Save model weights plus experiment metadata."""
    path = Path(run_dir) / name
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "labels": list(labels),
        "config": config,
        "metrics": metrics,
    }
    if history is not None:
        checkpoint["history"] = history
    torch.save(checkpoint, path)
    return path


def make_confusion_matrix(y_true, y_pred, num_classes: int) -> np.ndarray:
    """Build a count confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_idx, pred_idx in zip(y_true, y_pred):
        cm[int(true_idx), int(pred_idx)] += 1
    return cm


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str] | tuple[str, ...],
    *,
    normalize: bool = True,
    title: str = "Confusion matrix",
):
    """Plot a confusion matrix and return (fig, ax)."""
    values = cm.astype(float)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(im, ax=ax, label="recall" if normalize else "count")

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            text = f"{values[row, col]:.2f}" if normalize else str(cm[row, col])
            color = "white" if values[row, col] > 0.5 else "black"
            ax.text(col, row, text, ha="center", va="center", color=color, fontsize=8)

    fig.tight_layout()
    return fig, ax


def plot_history_curves(
    history: dict[str, list[float]],
    *,
    loss_keys: list[str] | tuple[str, ...] = (),
    acc_keys: list[str] | tuple[str, ...] = (),
    title: str | None = None,
):
    """Plot loss and accuracy curves from a history dictionary."""
    ncols = int(bool(loss_keys)) + int(bool(acc_keys))
    if ncols == 0:
        raise ValueError("At least one of loss_keys or acc_keys must be provided.")

    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    axes = np.atleast_1d(axes)
    axis_idx = 0

    if loss_keys:
        ax = axes[axis_idx]
        for key in loss_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.grid(True)
        ax.legend()
        axis_idx += 1

    if acc_keys:
        ax = axes[axis_idx]
        for key in acc_keys:
            ax.plot(history[key], label=key)
        ax.set_xlabel("epoch")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


def plot_step_curves(
    history: dict[str, list[float]],
    *,
    step_key: str = "step",
    loss_keys: list[str] | tuple[str, ...] = (),
    acc_keys: list[str] | tuple[str, ...] = (),
    title: str | None = None,
):
    """Plot loss and accuracy curves against an explicit step axis."""
    ncols = int(bool(loss_keys)) + int(bool(acc_keys))
    if ncols == 0:
        raise ValueError("At least one of loss_keys or acc_keys must be provided.")

    steps = history[step_key]
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 4))
    axes = np.atleast_1d(axes)
    axis_idx = 0

    if loss_keys:
        ax = axes[axis_idx]
        for key in loss_keys:
            ax.plot(steps, history[key], marker="o", markersize=2, label=key)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.grid(True)
        ax.legend()
        axis_idx += 1

    if acc_keys:
        ax = axes[axis_idx]
        for key in acc_keys:
            ax.plot(steps, history[key], marker="o", markersize=2, label=key)
        ax.set_xlabel("training step")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0, 1)
        ax.grid(True)
        ax.legend()

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes
