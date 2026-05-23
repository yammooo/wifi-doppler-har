"""Diagnostics for SHARP-like XRF55 Doppler traces across receivers and streams."""

from argparse import ArgumentParser
from pathlib import Path
import csv
import pickle

import matplotlib.pyplot as plt
import numpy as np

from xrf55_plot_doppler_grid import ACTION_NAMES
from xrf55_sharp_doppler import compute_sharp_like_doppler_for_file


def find_trial_path(root: Path, scene: str, receiver: str, subject: str, action: str, repetition: str) -> Path:
    path = root / scene / receiver / subject / f"{subject}_{action}_{repetition}.dat"
    if path.exists():
        return path
    mat_path = path.with_suffix(".mat")
    if mat_path.exists():
        return mat_path
    raise FileNotFoundError(path)


def cache_path(cache_dir: Path, scene: str, receiver: str, subject: str, action: str,
               repetition: str, stream: int, max_packets: int) -> Path:
    name = f"{scene}_{receiver}_s{subject}_a{action}_r{repetition}_stream{stream}_n{max_packets}.pkl"
    return cache_dir / name


def load_or_compute_doppler(
    raw_path: Path,
    cache_file: Path,
    stream: int,
    max_packets: int,
) -> np.ndarray:
    if cache_file.exists():
        with cache_file.open("rb") as fp:
            return pickle.load(fp)

    print(f"computing {cache_file.name}")
    doppler = compute_sharp_like_doppler_for_file(raw_path, stream_idx=stream, max_packets=max_packets)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with cache_file.open("wb") as fp:
        pickle.dump(doppler, fp)
    return doppler


def doppler_metrics(doppler: np.ndarray, center_bins: int) -> dict[str, float]:
    center = doppler.shape[1] // 2
    start = max(0, center - center_bins)
    end = min(doppler.shape[1], center + center_bins + 1)
    total_energy = float(np.sum(doppler))
    center_energy = float(np.sum(doppler[:, start:end]))
    off_center = doppler.copy()
    off_center[:, start:end] = np.min(doppler)
    return {
        "center_energy_ratio": center_energy / total_energy if total_energy else 0.0,
        "off_center_mean": float(np.mean(off_center)),
        "off_center_max": float(np.max(off_center)),
        "off_center_std": float(np.std(off_center)),
    }


def mask_center(doppler: np.ndarray, center_bins: int) -> np.ndarray:
    if center_bins <= 0:
        return doppler
    center = doppler.shape[1] // 2
    start = max(0, center - center_bins)
    end = min(doppler.shape[1], center + center_bins + 1)
    masked = doppler.copy()
    masked[:, start:end] = np.min(masked)
    return masked


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/XRF55_rawdata/WiFi")
    parser.add_argument("--scene", default="Scene_1")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--repetition", default="01")
    parser.add_argument("--receivers", default="lb,lf,rb")
    parser.add_argument("--streams", default="0,1,2")
    parser.add_argument("--actions", default="33,39")
    parser.add_argument("--max-packets", type=int, default=300)
    parser.add_argument("--mask-center-bins", type=int, default=2)
    parser.add_argument("--cache-dir", default="outputs/xrf55_sharp_like_cache")
    parser.add_argument("--plot-output", default="outputs/xrf55_sharp_like_receiver_stream_grid.png")
    parser.add_argument("--metrics-output", default="outputs/xrf55_sharp_like_receiver_stream_metrics.csv")
    args = parser.parse_args()

    root = Path(args.root)
    cache_dir = Path(args.cache_dir)
    receivers = [item.strip() for item in args.receivers.split(",") if item.strip()]
    streams = [int(item.strip()) for item in args.streams.split(",") if item.strip()]
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]

    entries = []
    for receiver in receivers:
        for stream in streams:
            for action in actions:
                raw_path = find_trial_path(root, args.scene, receiver, args.subject, action, args.repetition)
                cache_file = cache_path(
                    cache_dir,
                    args.scene,
                    receiver,
                    args.subject,
                    action,
                    args.repetition,
                    stream,
                    args.max_packets,
                )
                doppler = load_or_compute_doppler(raw_path, cache_file, stream, args.max_packets)
                metrics = doppler_metrics(doppler, args.mask_center_bins)
                entries.append({
                    "receiver": receiver,
                    "stream": stream,
                    "action": action,
                    "label": f"{action} {ACTION_NAMES.get(action, '')}".strip(),
                    "doppler": doppler,
                    **metrics,
                })

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", newline="") as fp:
        fieldnames = [
            "receiver",
            "stream",
            "action",
            "label",
            "center_energy_ratio",
            "off_center_mean",
            "off_center_max",
            "off_center_std",
        ]
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for entry in entries:
            writer.writerow({key: entry[key] for key in fieldnames})

    nrows = len(receivers) * len(streams)
    ncols = len(actions)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 1.9 * nrows), squeeze=False)

    for row_idx, (receiver, stream) in enumerate((receiver, stream) for receiver in receivers for stream in streams):
        row_entries = [
            entry for entry in entries
            if entry["receiver"] == receiver and entry["stream"] == stream
        ]
        for col_idx, action in enumerate(actions):
            ax = axes[row_idx][col_idx]
            entry = next(item for item in row_entries if item["action"] == action)
            doppler = mask_center(entry["doppler"], args.mask_center_bins)
            im = ax.imshow(doppler.T, aspect="auto", origin="lower", cmap="viridis")
            if row_idx == 0:
                ax.set_title(entry["label"])
            if col_idx == 0:
                ax.set_ylabel(f"{receiver} s{stream}\nDoppler bin")
            if row_idx == nrows - 1:
                ax.set_xlabel("time window")
            ax.text(
                0.01,
                0.95,
                f"center {entry['center_energy_ratio']:.2f}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                color="white",
                fontsize=8,
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 2},
            )

    fig.suptitle(
        f"XRF55 SHARP-like Doppler diagnostics: {args.scene}/subject {args.subject}/rep {args.repetition}",
        y=0.995,
    )
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    fig.tight_layout(rect=(0, 0, 0.985, 0.985))

    plot_path = Path(args.plot_output)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=150)
    print(plot_path)
    print(metrics_path)


if __name__ == "__main__":
    main()
