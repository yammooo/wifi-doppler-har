"""Generate the report figures from saved experiment histories and evaluation arrays."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


BLUE = "#2f6690"
ORANGE = "#d97706"
GREEN = "#287271"
RED = "#b23a48"
GRAY = "#4b5563"
LIGHT = "#f3f4f6"
FLOOR = 10**-1.2


def parse_args() -> argparse.Namespace:
    root = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--final-history",
        type=Path,
        default=root / "data/az9k6ori_epoch_history.csv",
    )
    parser.add_argument(
        "--overfit-30-history",
        type=Path,
        default=root / "data/cwj0myug_overfit_history.csv",
    )
    parser.add_argument(
        "--overfit-242-history",
        type=Path,
        default=root / "data/tm8nyt19_overfit_history.csv",
    )
    parser.add_argument(
        "--qualitative",
        type=Path,
        default=root / "data/s1a_c_16920_legacy_comparison.npz",
    )
    parser.add_argument("--output-dir", type=Path, default=root)
    return parser.parse_args()


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "pdf.fonttype": 42,
        }
    )


def box(ax, x, y, width, height, text, color=GRAY, fontsize=6.5):
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.003,rounding_size=0.008",
        facecolor=LIGHT,
        edgecolor=color,
        linewidth=1.2,
    )
    ax.add_patch(patch)
    ax.text(
        x + width / 2,
        y + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def arrow(ax, start, end, color=GRAY, connectionstyle="arc3"):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": color,
            "lw": 1.1,
            "shrinkA": 0.5,
            "shrinkB": 0.5,
            "connectionstyle": connectionstyle,
        },
    )


def make_pipeline(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.00))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.015, 0.38, 0.13, 0.24, "Raw complex CSI\n$4\\times242\\times370$", BLUE)
    paths = [
        (0.72, "SHARP teacher\nsparse paths + phase\nsanitization", BLUE),
        (0.39, "Neural student\ndirect map regression", RED),
        (0.06, "Affine phase\ncorrection", GREEN),
    ]
    for y, label, color in paths:
        box(ax, 0.20, y, 0.22, 0.22, label, color)
        arrow(ax, (0.145, 0.50), (0.20, y + 0.11), color)

    box(ax, 0.48, 0.72, 0.17, 0.22, "SHARP Doppler\n(reference map)", BLUE)
    box(ax, 0.48, 0.39, 0.17, 0.22, "Student Doppler", RED)
    box(ax, 0.46, 0.06, 0.14, 0.22, "Fixed Hann\nSTFT", GREEN)
    box(ax, 0.67, 0.06, 0.15, 0.22, "Affine-STFT\nDoppler", GREEN)
    arrow(ax, (0.42, 0.83), (0.48, 0.83), BLUE)
    arrow(ax, (0.42, 0.50), (0.48, 0.50), RED)
    arrow(ax, (0.42, 0.17), (0.46, 0.17), GREEN)
    arrow(ax, (0.60, 0.17), (0.67, 0.17), GREEN)

    box(ax, 0.85, 0.38, 0.135, 0.24, "Frozen SHARP\nHAR classifier", ORANGE)
    arrow(ax, (0.65, 0.83), (0.85, 0.57), BLUE)
    arrow(ax, (0.65, 0.50), (0.85, 0.50), RED)
    arrow(ax, (0.82, 0.17), (0.85, 0.43), GREEN)
    ax.text(0.65, 0.665, "pixel fidelity", ha="center", va="center", color=GRAY, fontsize=6.5)
    ax.annotate(
        "",
        xy=(0.565, 0.615),
        xytext=(0.565, 0.715),
        arrowprops={"arrowstyle": "<->", "color": GRAY, "lw": 0.9, "shrinkA": 1, "shrinkB": 1},
    )
    ax.text(0.9175, 0.675, "task fidelity", ha="center", va="center", color=GRAY, fontsize=6.5)
    fig.savefig(output_dir / "pipeline.pdf")
    plt.close(fig)


def make_architecture(output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.15, 2.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    stages = [
        (0.010, 0.080, "Input\n$[B,A,242,$\n$370,2]$", BLUE),
        (0.115, 0.120, "Shared antennas\n$[BA,2,242,$\n$370]$", BLUE),
        (
            0.260,
            0.175,
            "2-D encoder\n$64\\!\\times\\!61\\!\\times\\!370$\n"
            "$96\\!\\times\\!31\\!\\times\\!185$\n"
            "$128\\!\\times\\!16\\!\\times\\!92$",
            GREEN,
        ),
        (0.460, 0.135, "Frequency mean\n+ $1\\!\\times\\!1$ conv\n$[BA,128,92]$", GREEN),
        (0.620, 0.125, "Direct grid\n$[BA,16,$\n$92,100]$", RED),
        (0.770, 0.135, "2-D decoder\n+ two skips\n$92\\!\\to\\!185\\!\\to\\!370$", RED),
        (0.930, 0.060, "Output\n$[B,A,$\n$340,100]$", ORANGE),
    ]
    centers = []
    for x, width, label, color in stages:
        height = 0.46
        y = 0.34
        box(ax, x, y, width, height, label, color, 5.7)
        centers.append((x, y, width, height))
    for left, right in zip(centers, centers[1:]):
        arrow(
            ax,
            (left[0] + left[2], left[1] + left[3] / 2),
            (right[0], right[1] + right[3] / 2),
        )

    ax.text(
        0.4275,
        0.19,
        "GroupNorm residual blocks preserve\njoint frequency-time structure",
        ha="center",
        color=GREEN,
        fontsize=7,
    )
    ax.text(
        0.7625,
        0.19,
        "Direct 100-bin prediction;\nno spectral upsampling",
        ha="center",
        color=RED,
        fontsize=7,
    )
    ax.text(
        0.88,
        0.88,
        "exact valid crop: $15+340+15$",
        ha="center",
        color=ORANGE,
        fontsize=7,
    )
    fig.savefig(output_dir / "architecture.pdf")
    plt.close(fig)


def read_overfit(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    return data[data["eval/mse"].notna()].sort_values("global_step")


def make_diagnostics(
    final_history: Path,
    overfit_30_history: Path,
    overfit_242_history: Path,
    output_dir: Path,
) -> None:
    overfit_30 = read_overfit(overfit_30_history)
    overfit_242 = read_overfit(overfit_242_history)
    final = pd.read_csv(final_history)
    final = final[final["epoch"].notna()].sort_values("epoch")

    fig, axes = plt.subplots(1, 2, figsize=(7.15, 2.25))
    ax = axes[0]
    ax.plot(
        overfit_30["global_step"],
        overfit_30["eval/mse"],
        color=BLUE,
        label="30 subcarriers",
    )
    ax.plot(
        overfit_242["global_step"],
        overfit_242["eval/mse"],
        color=ORANGE,
        label="242 subcarriers",
    )
    ax.set_yscale("log")
    ax.set_xlabel("optimization step")
    ax.set_ylabel("fixed-set MSE")
    ax.set_title("(a) Memorizing 16 motion-rich windows")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.98,
        0.06,
        "$3.9\\times10^{-5}$ / $4.2\\times10^{-5}$",
        transform=ax.transAxes,
        ha="right",
        color=GRAY,
    )

    ax = axes[1]
    ax.plot(final["epoch"], final["train/loss"], color=BLUE, label="train")
    ax.plot(
        final["epoch"],
        final["source_val/loss"],
        color=ORANGE,
        label="all-scenario validation",
    )
    ax.plot(
        final["epoch"],
        final["target_val/loss"],
        color=GREEN,
        label="S1a/b/c validation",
    )
    ax.axvline(45, color=RED, linestyle="--", linewidth=1, label="best checkpoint")
    ax.set_xlabel("epoch")
    ax.set_ylabel("motion-aware loss")
    ax.set_title("(b) Precomputed-SHARP full-data experiment")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2, loc="upper right")
    fig.tight_layout(w_pad=1.4)
    fig.savefig(output_dir / "diagnostics.pdf", bbox_inches="tight")
    plt.close(fig)


def make_qualitative(path: Path, output_dir: Path) -> None:
    values = np.load(path)
    methods = (("target", "Precomputed SHARP"), ("student", "Neural student"), ("affine", "Affine + fixed STFT"))
    antennas = (2, 3)
    fig, axes = plt.subplots(2, 3, figsize=(7.15, 3.25), sharex=True, sharey=True)
    image = None
    for row, antenna in enumerate(antennas):
        for column, (key, title) in enumerate(methods):
            ax = axes[row, column]
            image = ax.imshow(
                values[key][antenna],
                origin="lower",
                aspect="auto",
                vmin=FLOOR,
                vmax=1.0,
                cmap="viridis",
                interpolation="nearest",
            )
            if row == 0:
                ax.set_title(title)
            if column == 0:
                ax.set_ylabel(f"time (antenna {antenna})")
            if row == len(antennas) - 1:
                ax.set_xlabel("Doppler bin")
            ax.set_xticks((0, 25, 50, 75, 99))
    fig.subplots_adjust(left=0.07, right=0.91, bottom=0.13, top=0.91, wspace=0.13, hspace=0.12)
    colorbar = fig.add_axes((0.93, 0.16, 0.012, 0.71))
    fig.colorbar(image, cax=colorbar, label="normalized power")
    fig.savefig(output_dir / "qualitative.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_style()
    make_pipeline(args.output_dir)
    make_architecture(args.output_dir)
    make_diagnostics(
        args.final_history,
        args.overfit_30_history,
        args.overfit_242_history,
        args.output_dir,
    )
    make_qualitative(args.qualitative, args.output_dir)


if __name__ == "__main__":
    main()
