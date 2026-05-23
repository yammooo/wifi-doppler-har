"""Plot a small grid of SHARP-like XRF55 Doppler traces for visual inspection."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins
from xrf55_sharp_doppler import compute_sharp_like_doppler_for_file


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/XRF55_rawdata/WiFi")
    parser.add_argument("--scene", default="Scene_1")
    parser.add_argument("--receiver", default="lb")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--repetition", default="01")
    parser.add_argument("--stream", type=int, default=0)
    parser.add_argument("--max-packets", type=int, default=180)
    parser.add_argument("--actions", default="23,33,34,35,36,39")
    parser.add_argument("--output", default="outputs/xrf55_sharp_like_action_grid.png")
    parser.add_argument("--mask-center-bins", type=int, default=0)
    parser.add_argument("--per-panel-scale", action="store_true")
    args = parser.parse_args()

    root = Path(args.root)
    actions = [action.strip() for action in args.actions.split(",") if action.strip()]

    dopplers = []
    labels = []
    for action in actions:
        path = find_trial(root, args.scene, args.receiver, args.subject, action, args.repetition)
        print(f"processing {path}")
        doppler = compute_sharp_like_doppler_for_file(path, stream_idx=args.stream, max_packets=args.max_packets)
        doppler = mask_center_bins(doppler, args.mask_center_bins)
        dopplers.append(doppler)
        labels.append(f"{action} {ACTION_NAMES.get(action, '')}".strip())

    shared_vmax = max(float(np.max(doppler)) for doppler in dopplers)
    shared_vmin = min(float(np.min(doppler)) for doppler in dopplers)

    fig, axes = plt.subplots(len(dopplers), 1, figsize=(9, 2.2 * len(dopplers)), sharex=False, sharey=True)
    if len(dopplers) == 1:
        axes = [axes]

    for ax, doppler, label in zip(axes, dopplers, labels):
        if args.per_panel_scale:
            vmin = float(np.min(doppler))
            vmax = float(np.max(doppler))
        else:
            vmin = shared_vmin
            vmax = shared_vmax
        im = ax.imshow(doppler.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_ylabel("Doppler bin")

    axes[-1].set_xlabel("time window")
    fig.suptitle(
        f"XRF55 SHARP-like Doppler: {args.scene}/{args.receiver}/subject {args.subject}/rep {args.repetition}/stream {args.stream}",
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
