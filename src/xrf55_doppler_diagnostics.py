"""Diagnostics for XRF55 CSI-F-like Doppler variants."""

from argparse import ArgumentParser, BooleanOptionalAction
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xrf55_csif_doppler import compute_csif_like_doppler_result_for_file, format_doppler_for_plot
from xrf55_dataset import scan_xrf55_raw_wifi
from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins


DEFAULT_ACTIONS = "23,33,34,35,36,39"
DEFAULT_MODES = "amplitude,conj_phase,conj_complex"


def doppler_quality_metrics(doppler: np.ndarray, center_bins: int = 2) -> dict[str, float]:
    """Measure how much energy is stuck in the zero-Doppler center ridge."""
    center = doppler.shape[1] // 2
    start = max(0, center - center_bins)
    end = min(doppler.shape[1], center + center_bins + 1)
    center_mask = np.zeros(doppler.shape[1], dtype=bool)
    center_mask[start:end] = True

    total_energy = float(np.sum(doppler))
    center_energy = float(np.sum(doppler[:, center_mask]))
    off_center = doppler[:, ~center_mask]
    off_center_energy = float(np.sum(off_center))

    if total_energy <= 0:
        return {
            "center_energy_ratio": 0.0,
            "off_center_energy_ratio": 0.0,
            "non_center_peak": 0.0,
        }

    return {
        "center_energy_ratio": center_energy / total_energy,
        "off_center_energy_ratio": off_center_energy / total_energy,
        "non_center_peak": float(np.max(off_center)) if off_center.size else 0.0,
    }


def compute_doppler(path: Path, mode: str, args) -> tuple[np.ndarray, dict[str, float | str]]:
    result = compute_csif_like_doppler_result_for_file(
        path,
        mode=mode,
        rx_pair=args.rx_pair,
        auto_rx_pair=args.auto_rx_pair,
        sample_rate_hz=None if args.infer_sample_rate else args.sample_rate_hz,
        hampel_window=args.hampel_window,
        hampel_sigmas=args.hampel_sigmas,
        dwt_wavelet=args.dwt_wavelet,
        dwt_level=args.dwt_level,
        butter_low_hz=args.butter_low_hz,
        butter_high_hz=args.butter_high_hz,
        butter_order=args.butter_order,
        spectrogram_method=args.spectrogram_method,
        stft_nperseg=args.stft_nperseg,
        stft_noverlap=args.stft_noverlap,
        stft_nfft=args.stft_nfft,
        power_floor=args.power_floor,
        normalize=args.normalize,
    )
    doppler = mask_center_bins(result.doppler, args.mask_center_bins)
    metrics = doppler_quality_metrics(doppler, center_bins=args.metric_center_bins)
    metrics.update(
        {
            "sample_rate_hz": result.sample_rate_hz,
            "rx_pair": "" if result.rx_pair is None else f"{result.rx_pair[0]},{result.rx_pair[1]}",
        }
    )
    return doppler, metrics


def plot_trial_grid(args) -> None:
    actions = split_csv(args.actions)
    modes = split_csv(args.modes)
    root = Path(args.root)

    fig, axes = plt.subplots(
        len(modes),
        len(actions),
        figsize=(3.0 * len(actions), 2.2 * len(modes)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(modes), len(actions))

    print("mode,action,center_energy_ratio,off_center_energy_ratio,non_center_peak,sample_rate_hz,rx_pair")
    last_image = None
    for row_idx, mode in enumerate(modes):
        for col_idx, action in enumerate(actions):
            path = find_trial(root, args.scene, args.receiver, args.subject, action, args.repetition)
            doppler, metrics = compute_doppler(path, mode, args)
            ax = axes[row_idx, col_idx]
            display, vmin, vmax, colorbar_label = format_doppler_for_plot(
                doppler,
                plot_scale=args.plot_scale,
                db_min=args.db_min,
                db_max=args.db_max,
            )
            last_image = ax.imshow(
                display.T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{mode}\n{action} {ACTION_NAMES.get(action, '')}")
            if col_idx == 0:
                ax.set_ylabel("Doppler bin")
            if row_idx == len(modes) - 1:
                ax.set_xlabel("time")
            print(
                f"{mode},{action},"
                f"{metrics['center_energy_ratio']:.6f},"
                f"{metrics['off_center_energy_ratio']:.6f},"
                f"{metrics['non_center_peak']:.6f},"
                f"{metrics['sample_rate_hz']:.3f},"
                f"{metrics['rx_pair']}"
            )

    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.02, pad=0.01, label=colorbar_label)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


def plot_class_averages(args) -> None:
    if args.class_average_output is None:
        return

    actions = split_csv(args.actions)
    modes = split_csv(args.modes)
    recordings = [
        recording
        for recording in scan_xrf55_raw_wifi(args.root)
        if recording.scene == args.scene
        and recording.receiver == args.receiver
        and recording.action in set(actions)
    ]

    grouped: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    for mode in modes:
        per_action_count = defaultdict(int)
        for recording in recordings:
            if per_action_count[recording.action] >= args.class_average_limit:
                continue
            doppler, _ = compute_doppler(recording.path, mode, args)
            grouped[(mode, recording.action)].append(doppler)
            per_action_count[recording.action] += 1

    fig, axes = plt.subplots(
        len(modes),
        len(actions),
        figsize=(3.0 * len(actions), 2.2 * len(modes)),
        sharex=False,
        sharey=False,
        constrained_layout=True,
    )
    axes = np.asarray(axes).reshape(len(modes), len(actions))
    last_image = None

    for row_idx, mode in enumerate(modes):
        for col_idx, action in enumerate(actions):
            samples = grouped[(mode, action)]
            ax = axes[row_idx, col_idx]
            if not samples:
                ax.set_axis_off()
                continue
            average = np.mean(np.stack(samples, axis=0), axis=0)
            display, vmin, vmax, colorbar_label = format_doppler_for_plot(
                average,
                plot_scale=args.plot_scale,
                db_min=args.db_min,
                db_max=args.db_max,
            )
            last_image = ax.imshow(
                display.T,
                aspect="auto",
                origin="lower",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_title(f"{mode}\n{action} avg n={len(samples)}")
            if col_idx == 0:
                ax.set_ylabel("Doppler bin")
            if row_idx == len(modes) - 1:
                ax.set_xlabel("time")

    if last_image is not None:
        fig.colorbar(last_image, ax=axes, fraction=0.02, pad=0.01, label=colorbar_label)
    output = Path(args.class_average_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


def run_training_check(args) -> None:
    if args.train_cache_root is None:
        return

    import torch
    from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
    from torch.utils.data import DataLoader

    from models.base_model import SingleAntennaModel
    from xrf55_dataset import XRF55DopplerDataset

    train_repetitions = {f"{idx:02d}" for idx in range(args.train_rep_start, args.train_rep_end + 1)}
    val_repetitions = {f"{idx:02d}" for idx in range(args.val_rep_start, args.val_rep_end + 1)}
    crop_size = None if args.train_crop_size <= 0 else args.train_crop_size

    train_dataset = XRF55DopplerDataset(
        cache_root=args.train_cache_root,
        label_mode="action",
        scenes={args.scene},
        receivers={args.receiver},
        actions=set(split_csv(args.actions)),
        repetitions=train_repetitions,
        crop_size=crop_size,
        crops_per_recording=args.train_crops_per_recording,
        crop_strategy="none" if crop_size is None else "random_center_jitter",
        crop_jitter=args.train_crop_jitter,
        input_scale=args.input_scale,
        db_min=args.db_min,
        db_max=args.db_max,
        subtract_time_mean=not args.no_subtract_time_mean,
    )
    val_dataset = XRF55DopplerDataset(
        cache_root=args.train_cache_root,
        label_mode="action",
        scenes={args.scene},
        receivers={args.receiver},
        actions=set(split_csv(args.actions)),
        repetitions=val_repetitions,
        crop_size=crop_size,
        crops_per_recording=1,
        crop_strategy="none" if crop_size is None else "center",
        input_scale=args.input_scale,
        db_min=args.db_min,
        db_max=args.db_max,
        subtract_time_mean=not args.no_subtract_time_mean,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleAntennaModel(num_classes=len(train_dataset.label_to_idx)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        val_acc, _, _ = evaluate(model, val_loader, device)
        print(f"epoch {epoch:03d} val_acc={val_acc:.4f}")

    val_acc, y_true, y_pred = evaluate(model, val_loader, device)
    print(f"final_val_acc={val_acc:.4f}")

    labels = [val_dataset.idx_to_label[idx] for idx in range(len(val_dataset.idx_to_label))]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
    display = ConfusionMatrixDisplay(matrix, display_labels=labels)
    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    display.plot(ax=ax, cmap="Blues", colorbar=False)
    output = Path(args.confusion_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


def evaluate(model, loader, device):
    import torch

    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.to(device))
            pred = logits.argmax(dim=1).cpu()
            correct += int((pred == y).sum().item())
            total += int(y.numel())
            y_true.extend(y.tolist())
            y_pred.extend(pred.tolist())
    return correct / max(total, 1), y_true, y_pred


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_rx_pair(value: str) -> tuple[int, int] | None:
    if value.lower() in {"auto", "none"}:
        return None
    first, second = value.split(",", maxsplit=1)
    return int(first), int(second)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/XRF55_rawdata/WiFi")
    parser.add_argument("--scene", default="Scene_1")
    parser.add_argument("--receiver", default="lb")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--repetition", default="01")
    parser.add_argument("--actions", default=DEFAULT_ACTIONS)
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--rx-pair", type=parse_rx_pair, default=None)
    parser.add_argument("--auto-rx-pair", action=BooleanOptionalAction, default=True)
    parser.add_argument("--infer-sample-rate", action=BooleanOptionalAction, default=True)
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--dwt-wavelet", default="db4")
    parser.add_argument("--dwt-level", type=int, default=None)
    parser.add_argument("--butter-low-hz", type=float, default=0.2)
    parser.add_argument("--butter-high-hz", type=float, default=20.0)
    parser.add_argument("--butter-order", type=int, default=4)
    parser.add_argument("--spectrogram-method", choices=["sharp_fft", "stft"], default="stft")
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=120)
    parser.add_argument("--stft-nfft", type=int, default=256)
    parser.add_argument("--power-floor", type=float, default=1e-6)
    parser.add_argument("--normalize", choices=["frame", "global", "none"], default="frame")
    parser.add_argument("--mask-center-bins", type=int, default=0)
    parser.add_argument("--metric-center-bins", type=int, default=2)
    parser.add_argument("--plot-scale", choices=["linear", "db"], default="linear")
    parser.add_argument("--db-min", type=float, default=-30.0)
    parser.add_argument("--db-max", type=float, default=0.0)
    parser.add_argument("--output", default="outputs/xrf55_doppler_variant_grid.png")
    parser.add_argument("--class-average-output", default=None)
    parser.add_argument("--class-average-limit", type=int, default=5)

    parser.add_argument("--train-cache-root", default=None)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-rep-start", type=int, default=1)
    parser.add_argument("--train-rep-end", type=int, default=15)
    parser.add_argument("--val-rep-start", type=int, default=16)
    parser.add_argument("--val-rep-end", type=int, default=20)
    parser.add_argument("--train-crop-size", type=int, default=0)
    parser.add_argument("--train-crops-per-recording", type=int, default=1)
    parser.add_argument("--train-crop-jitter", type=int, default=0)
    parser.add_argument("--input-scale", choices=["linear", "db"], default="linear")
    parser.add_argument("--no-subtract-time-mean", action="store_true")
    parser.add_argument("--confusion-output", default="outputs/xrf55_doppler_confusion.png")
    args = parser.parse_args()

    plot_trial_grid(args)
    plot_class_averages(args)
    run_training_check(args)


if __name__ == "__main__":
    main()
