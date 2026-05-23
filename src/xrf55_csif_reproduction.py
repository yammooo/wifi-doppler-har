"""CSI-F-style Doppler extraction for XRF55 raw Intel 5300 CSI.

This file follows the CSI-F preprocessing sequence as closely as the paper
allows without using unpublished implementation details:

1. Select two receive antennas with strong amplitude and stable CSI.
2. Remove common phase offset by conjugate multiplication.
3. Hampel-filter burst outliers.
4. Reduce subcarriers to the first PCA component.
5. Denoise the component with a discrete wavelet transform.
6. Smooth with a Butterworth filter.
7. Use STFT to create a Doppler spectrogram.

The paper does not specify several numeric choices, including wavelet family,
thresholds, and STFT window sizes. Those assumptions are explicit CLI options.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, stft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrf55_csi import read_xrf55_wifi_file, records_to_csi_array
from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins


@dataclass(frozen=True)
class CsifMetadata:
    rx_pair: tuple[int, int]
    feature_kind: str
    component_mean: float
    component_std: float


def select_rx_pair(csi: np.ndarray) -> tuple[int, int]:
    """Choose the two Rx antennas emphasized by CSI-F: high amplitude, low variance."""
    amplitudes = np.abs(csi[:, :, :, 0])
    mean_amplitude = amplitudes.mean(axis=(0, 1))
    variance = amplitudes.var(axis=(0, 1))

    amp_score = mean_amplitude / (mean_amplitude.max() + 1e-12)
    stable_score = 1.0 - variance / (variance.max() + 1e-12)
    score = amp_score + stable_score

    ranked = np.argsort(score)[::-1]
    return int(ranked[0]), int(ranked[1])


def conjugate_pair_features(
    csi: np.ndarray,
    rx_pair: tuple[int, int],
    feature_kind: str = "amplitude",
) -> np.ndarray:
    """Apply antenna-pair conjugate multiplication and expose real features."""
    csi = csi[:, :, :, 0]
    first, second = rx_pair
    paired = csi[:, :, first] * np.conj(csi[:, :, second])

    if feature_kind == "amplitude":
        features = np.abs(paired)
    elif feature_kind == "phase":
        features = np.unwrap(np.angle(paired), axis=0)
    elif feature_kind == "real_imag":
        features = np.concatenate([paired.real, paired.imag], axis=1)
    else:
        raise ValueError(f"Unknown feature kind: {feature_kind}")

    return np.nan_to_num(features)


def hampel_filter_matrix(matrix: np.ndarray, window_size: int = 7, n_sigmas: float = 3.0) -> np.ndarray:
    """Temporal Hampel filtering per feature column."""
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")

    matrix = matrix.astype(np.float64, copy=False)
    half_window = window_size // 2
    padded = np.pad(matrix, ((half_window, half_window), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=window_size, axis=0)

    medians = np.median(windows, axis=-1)
    mad = 1.4826 * np.median(np.abs(windows - np.expand_dims(medians, axis=-1)), axis=-1)
    outliers = (mad > 0) & (np.abs(matrix - medians) > n_sigmas * mad)

    filtered = matrix.copy()
    filtered[outliers] = medians[outliers]
    return filtered


def first_principal_component(features: np.ndarray) -> np.ndarray:
    """Select CSI-F's first principal component for time-frequency analysis."""
    scaled = StandardScaler().fit_transform(features)
    component = PCA(n_components=1).fit_transform(scaled)[:, 0]
    return component - component.mean()


def _haar_decompose(signal: np.ndarray, levels: int) -> tuple[np.ndarray, list[np.ndarray]]:
    details = []
    approx = signal.astype(np.float64, copy=True)
    for _ in range(levels):
        if approx.size % 2:
            approx = np.pad(approx, (0, 1), mode="edge")
        even = approx[0::2]
        odd = approx[1::2]
        details.append((even - odd) / np.sqrt(2.0))
        approx = (even + odd) / np.sqrt(2.0)
    return approx, details


def _haar_reconstruct(approx: np.ndarray, details: list[np.ndarray], output_length: int) -> np.ndarray:
    signal = approx
    for detail in reversed(details):
        rebuilt = np.empty(detail.size * 2, dtype=np.float64)
        rebuilt[0::2] = (signal[:detail.size] + detail) / np.sqrt(2.0)
        rebuilt[1::2] = (signal[:detail.size] - detail) / np.sqrt(2.0)
        signal = rebuilt
    return signal[:output_length]


def dwt_denoise_haar(signal: np.ndarray, levels: int = 3, threshold_scale: float = 1.0) -> np.ndarray:
    """Small dependency-free Haar DWT denoiser matching CSI-F's DWT stage."""
    original_length = signal.size
    approx, details = _haar_decompose(signal, levels)

    finest_detail = details[0]
    sigma = np.median(np.abs(finest_detail - np.median(finest_detail))) / 0.6745
    threshold = threshold_scale * sigma * np.sqrt(2.0 * np.log(max(original_length, 2)))

    denoised_details = [
        np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0.0)
        for detail in details
    ]
    return _haar_reconstruct(approx, denoised_details, original_length)


def butterworth_filter(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int,
) -> np.ndarray:
    """Apply the CSI-F smoothing stage with explicit cutoff assumptions."""
    nyquist = sample_rate_hz / 2.0
    if high_hz is not None and high_hz >= nyquist:
        high_hz = None

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
    return filtfilt(b, a, signal)


def stft_doppler_spectrogram(
    signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 120,
    nfft: int = 256,
    noise_floor: float = 1e-3,
) -> np.ndarray:
    """Create a normalized, fftshifted STFT Doppler spectrogram."""
    _, _, spectrum = stft(
        signal,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    power = np.abs(spectrum) ** 2
    power = np.fft.fftshift(power, axes=0).T
    row_max = power.max(axis=1, keepdims=True)
    row_max[row_max == 0] = 1.0
    power = power / row_max
    power[power < noise_floor] = noise_floor
    return power


def compute_csif_reproduction_for_file(
    path: str | Path,
    feature_kind: str = "amplitude",
    rx_pair: tuple[int, int] | None = None,
    sample_rate_hz: float = 200.0,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    dwt_levels: int = 3,
    dwt_threshold_scale: float = 1.0,
    butter_low_hz: float | None = 0.2,
    butter_high_hz: float | None = 30.0,
    butter_order: int = 4,
    stft_nperseg: int = 128,
    stft_noverlap: int = 120,
    stft_nfft: int = 256,
) -> tuple[np.ndarray, CsifMetadata]:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)

    selected_pair = select_rx_pair(csi) if rx_pair is None else rx_pair
    features = conjugate_pair_features(csi, selected_pair, feature_kind=feature_kind)
    features = hampel_filter_matrix(features, window_size=hampel_window, n_sigmas=hampel_sigmas)

    component = first_principal_component(features)
    component = dwt_denoise_haar(component, levels=dwt_levels, threshold_scale=dwt_threshold_scale)
    component = butterworth_filter(
        component,
        sample_rate_hz=sample_rate_hz,
        low_hz=butter_low_hz,
        high_hz=butter_high_hz,
        order=butter_order,
    )
    component = component - component.mean()

    doppler = stft_doppler_spectrogram(
        component,
        sample_rate_hz=sample_rate_hz,
        nperseg=stft_nperseg,
        noverlap=stft_noverlap,
        nfft=stft_nfft,
    )
    metadata = CsifMetadata(
        rx_pair=selected_pair,
        feature_kind=feature_kind,
        component_mean=float(component.mean()),
        component_std=float(component.std()),
    )
    return doppler, metadata


def parse_rx_pair(value: str) -> tuple[int, int] | None:
    if value == "auto":
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
    parser.add_argument("--actions", default="23,33,34,35,36,39")
    parser.add_argument("--feature-kind", choices=["amplitude", "phase", "real_imag"], default="amplitude")
    parser.add_argument("--rx-pair", default="auto")
    parser.add_argument("--sample-rate-hz", type=float, default=200.0)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--dwt-levels", type=int, default=3)
    parser.add_argument("--dwt-threshold-scale", type=float, default=1.0)
    parser.add_argument("--butter-low-hz", type=float, default=0.2)
    parser.add_argument("--butter-high-hz", type=float, default=30.0)
    parser.add_argument("--butter-order", type=int, default=4)
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=120)
    parser.add_argument("--stft-nfft", type=int, default=256)
    parser.add_argument("--mask-center-bins", type=int, default=2)
    parser.add_argument("--output", default="outputs/xrf55_csif_reproduction_grid.png")
    args = parser.parse_args()

    root = Path(args.root)
    actions = [item.strip() for item in args.actions.split(",") if item.strip()]
    rx_pair = parse_rx_pair(args.rx_pair)
    dopplers = []
    labels = []

    for action in actions:
        path = find_trial(root, args.scene, args.receiver, args.subject, action, args.repetition)
        print(f"processing {path}")
        doppler, metadata = compute_csif_reproduction_for_file(
            path,
            feature_kind=args.feature_kind,
            rx_pair=rx_pair,
            sample_rate_hz=args.sample_rate_hz,
            hampel_window=args.hampel_window,
            hampel_sigmas=args.hampel_sigmas,
            dwt_levels=args.dwt_levels,
            dwt_threshold_scale=args.dwt_threshold_scale,
            butter_low_hz=args.butter_low_hz,
            butter_high_hz=args.butter_high_hz,
            butter_order=args.butter_order,
            stft_nperseg=args.stft_nperseg,
            stft_noverlap=args.stft_noverlap,
            stft_nfft=args.stft_nfft,
        )
        dopplers.append(mask_center_bins(doppler, args.mask_center_bins))
        labels.append(f"{action} {ACTION_NAMES.get(action, '')} rx={metadata.rx_pair}".strip())

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

    axes[-1].set_xlabel(
        f"time window - CSI-F reproduction: {args.scene}/{args.receiver}/"
        f"subject {args.subject}/rep {args.repetition}/{args.feature_kind}"
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
