"""CSI-F-inspired Doppler extraction for XRF55 Intel 5300 CSI.

This is not a full reproduction of CSI-F. It keeps the parts that transfer
directly to XRF55 raw Wi-Fi files: amplitude or conjugate-Rx features, Hampel
filtering, PCA, and a temporal FFT spectrogram.
"""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrf55_csi import read_xrf55_wifi_file, records_to_csi_array
from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins, temporal_fft_profile


def hampel_filter_matrix(matrix: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Replace temporal outliers per feature column with the local median."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    matrix = matrix.astype(np.float64, copy=False)
    half_window = window_size // 2
    scale = 1.4826

    padded = np.pad(matrix, ((half_window, half_window), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)
    medians = np.median(windows, axis=-1)
    mad = scale * np.median(np.abs(windows - np.expand_dims(medians, axis=-1)), axis=-1)
    outliers = (mad > 0) & (np.abs(matrix - medians) > n_sigmas * mad)

    filtered = matrix.copy()
    filtered[outliers] = medians[outliers]
    return filtered


def build_feature_matrix(
    csi: np.ndarray,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] = (0, 1),
) -> np.ndarray:
    """Build CSI-F-style temporal features from CSI shaped [packet, subcarrier, rx, tx]."""
    csi = csi[:, :, :, 0]

    if mode == "amplitude":
        if stream_idx is None:
            features = np.abs(csi).reshape(csi.shape[0], -1)
        else:
            features = np.abs(csi[:, :, stream_idx])
    elif mode == "conj_phase":
        first, second = rx_pair
        conjugate_product = csi[:, :, first] * np.conj(csi[:, :, second])
        features = np.unwrap(np.angle(conjugate_product), axis=0)
    elif mode == "conj_amplitude":
        first, second = rx_pair
        conjugate_product = csi[:, :, first] * np.conj(csi[:, :, second])
        features = np.abs(conjugate_product)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.nan_to_num(features)


def first_principal_component(features: np.ndarray, standardize: bool = True) -> np.ndarray:
    if standardize:
        features = StandardScaler().fit_transform(features)
    component = PCA(n_components=1).fit_transform(features)
    return component


def maybe_butterworth(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    if low_hz is None and high_hz is None:
        return signal

    if low_hz is not None and high_hz is not None:
        btype = "bandpass"
        cutoff = [low_hz, high_hz]
    elif low_hz is not None:
        btype = "highpass"
        cutoff = low_hz
    else:
        btype = "lowpass"
        cutoff = high_hz

    b, a = butter(order, cutoff, btype=btype, fs=sample_rate_hz)
    return filtfilt(b, a, signal, axis=0)


def compute_csif_like_doppler_for_file(
    path: str | Path,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] = (0, 1),
    sample_rate_hz: float = 200.0,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    butter_low_hz: float | None = None,
    butter_high_hz: float | None = None,
) -> np.ndarray:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)
    features = build_feature_matrix(csi, mode=mode, stream_idx=stream_idx, rx_pair=rx_pair)
    features = hampel_filter_matrix(features, window_size=hampel_window, n_sigmas=hampel_sigmas)
    component = first_principal_component(features)
    component = maybe_butterworth(component, sample_rate_hz, butter_low_hz, butter_high_hz)
    component = component - np.mean(component, axis=0, keepdims=True)
    return temporal_fft_profile(component)


def parse_rx_pair(value: str) -> tuple[int, int]:
    first, second = value.split(",", maxsplit=1)
    return int(first), int(second)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--root", default="data/XRF55_rawdata/WiFi")
    parser.add_argument("--scene", default="Scene_1")
    parser.add_argument("--receiver", default="lb")
    parser.add_argument("--subject", default="01")
    parser.add_argument("--repetition", default="01")
    parser.add_argument("--actions", default="23,33,34,35,36,39")
    parser.add_argument("--mode", choices=["amplitude", "conj_phase", "conj_amplitude"], default="amplitude")
    parser.add_argument("--stream", type=int, default=None)
    parser.add_argument("--rx-pair", default="0,1")
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--butter-low-hz", type=float, default=None)
    parser.add_argument("--butter-high-hz", type=float, default=None)
    parser.add_argument("--mask-center-bins", type=int, default=0)
    parser.add_argument("--output", default="outputs/xrf55_csif_like_doppler_grid.png")
    args = parser.parse_args()

    root = Path(args.root)
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    rx_pair = parse_rx_pair(args.rx_pair)
    dopplers = []
    labels = []

    for action in actions:
        path = find_trial(root, args.scene, args.receiver, args.subject, action, args.repetition)
        print(f"processing {path}")
        doppler = compute_csif_like_doppler_for_file(
            path,
            mode=args.mode,
            stream_idx=args.stream,
            rx_pair=rx_pair,
            sample_rate_hz=args.sample_rate_hz,
            hampel_window=args.hampel_window,
            hampel_sigmas=args.hampel_sigmas,
            butter_low_hz=args.butter_low_hz,
            butter_high_hz=args.butter_high_hz,
        )
        dopplers.append(mask_center_bins(doppler, args.mask_center_bins))
        labels.append(f"{action} {ACTION_NAMES.get(action, '')}".strip())

    fig, axes = plt.subplots(
        len(dopplers),
        1,
        figsize=(9, 2.2 * len(dopplers)),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )
    if len(dopplers) == 1:
        axes = [axes]

    for ax, doppler, label in zip(axes, dopplers, labels):
        im = ax.imshow(doppler.T, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(label)
        ax.set_ylabel("Doppler bin")

    stream_text = "all streams" if args.stream is None else f"stream {args.stream}"
    axes[-1].set_xlabel(
        f"time window - XRF55 CSI-F-like {args.mode}: "
        f"{args.scene}/{args.receiver}/subject {args.subject}/rep {args.repetition}/{stream_text}"
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
