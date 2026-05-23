"""Simple non-SHARP Doppler diagnostics for XRF55 raw Wi-Fi CSI."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xrf55_csi import read_xrf55_wifi_file, records_to_csi_array
from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins, temporal_fft_profile


def simple_doppler_for_file(path: str | Path, stream_idx: int, mode: str) -> np.ndarray:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)[:, :, stream_idx, 0]

    if mode == "amplitude":
        matrix = np.abs(csi)
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    elif mode == "phase_diff":
        phase = np.unwrap(np.angle(csi), axis=0)
        matrix = np.diff(phase, axis=0, prepend=phase[:1])
        matrix = matrix - np.mean(matrix, axis=1, keepdims=True)
    elif mode == "complex":
        amp_mean = np.mean(np.abs(csi), axis=1, keepdims=True)
        amp_mean[amp_mean == 0] = 1
        matrix = csi / amp_mean
        matrix = matrix - np.mean(matrix, axis=0, keepdims=True)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return temporal_fft_profile(matrix)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/XRF55_rawdata/WiFi")
    parser.add_argument("--scene", default="Scene_1")
    parser.add_argument("--receiver", default="lb")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--repetition", default="01")
    parser.add_argument("--stream", type=int, default=2)
    parser.add_argument("--actions", default="23,33,34,35,36,39")
    parser.add_argument("--mode", choices=["amplitude", "phase_diff", "complex"], default="amplitude")
    parser.add_argument("--mask-center-bins", type=int, default=2)
    parser.add_argument("--output", default="outputs/xrf55_simple_doppler_grid.png")
    args = parser.parse_args()

    root = Path(args.root)
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    dopplers = []
    labels = []

    for action in actions:
        path = find_trial(root, args.scene, args.receiver, args.subject, action, args.repetition)
        print(f"processing {path}")
        doppler = simple_doppler_for_file(path, stream_idx=args.stream, mode=args.mode)
        doppler = mask_center_bins(doppler, args.mask_center_bins)
        dopplers.append(doppler)
        labels.append(f"{action} {ACTION_NAMES.get(action, '')}".strip())

    fig, axes = plt.subplots(len(dopplers), 1, figsize=(9, 2.2 * len(dopplers)), sharex=True, sharey=True)
    if len(dopplers) == 1:
        axes = [axes]

    for ax, doppler, label in zip(axes, dopplers, labels):
        im = ax.imshow(doppler.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(label)
        ax.set_ylabel("Doppler bin")

    axes[-1].set_xlabel("time window")
    fig.suptitle(
        f"XRF55 simple {args.mode} Doppler: {args.scene}/{args.receiver}/subject {args.subject}/rep {args.repetition}/stream {args.stream}",
        y=0.995,
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)
    fig.tight_layout(rect=(0, 0, 0.98, 0.98))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
