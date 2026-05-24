"""CSI-F-inspired Doppler extraction for XRF55 Intel 5300 CSI.

The production path stays compatible with the earlier amplitude baseline, but
also exposes the stronger CSI-F-like variants needed for diagnosis:
conjugate-Rx features, timestamp-based sampling rate inference, optional DWT
denoising, Butterworth filtering, and STFT spectrograms.
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, detrend, filtfilt, stft
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from xrf55_csi import CsiRecord, read_xrf55_wifi_file, records_to_csi_array
from xrf55_doppler import ACTION_NAMES, find_trial, mask_center_bins, temporal_fft_profile


DEFAULT_SAMPLE_RATE_HZ = 200.0
CONJUGATE_MODES = {"conj_phase", "conj_complex", "conj_amplitude"}


@dataclass(frozen=True)
class CsifDopplerResult:
    """Doppler output plus extraction metadata that should be cached."""

    doppler: np.ndarray
    sample_rate_hz: float
    rx_pair: tuple[int, int] | None
    mode: str
    spectrogram_method: str


def infer_sample_rate_hz(records: list[CsiRecord], default_hz: float = DEFAULT_SAMPLE_RATE_HZ) -> float:
    """Infer packet sampling rate from Intel 5300 microsecond timestamps."""
    if len(records) < 2:
        return default_hz

    timestamps = np.asarray([record.timestamp_low for record in records], dtype=np.int64)
    diffs = np.diff(timestamps)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return default_hz

    median_diff_us = float(np.median(diffs))
    if median_diff_us <= 0:
        return default_hz
    return 1_000_000.0 / median_diff_us


def select_rx_pair(csi: np.ndarray) -> tuple[int, int]:
    """Select the two strongest/stablest RX antennas using CSI-F-style scores."""
    rx_csi = csi[:, :, :, 0]
    amp = np.abs(rx_csi)
    mean_amp = amp.mean(axis=(0, 1))
    temporal_amp = amp.mean(axis=1)
    temporal_var = temporal_amp.var(axis=0)

    score = _zscore(mean_amp) - _zscore(temporal_var)
    ranked = sorted(range(rx_csi.shape[2]), key=lambda idx: (-score[idx], idx))
    if len(ranked) < 2:
        raise ValueError(f"Need at least two RX antennas for conjugate features, got {len(ranked)}")
    return ranked[0], ranked[1]


def _zscore(values: np.ndarray) -> np.ndarray:
    std = float(np.std(values))
    if std == 0:
        return np.zeros_like(values, dtype=np.float64)
    return (values - np.mean(values)) / std


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
    rx_pair: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build temporal features from CSI shaped [packet, subcarrier, rx, tx]."""
    csi = csi[:, :, :, 0]

    if mode == "amplitude":
        if stream_idx is None:
            features = np.abs(csi).reshape(csi.shape[0], -1)
        else:
            features = np.abs(csi[:, :, stream_idx])
    elif mode in CONJUGATE_MODES:
        if rx_pair is None:
            raise ValueError(f"rx_pair is required for mode={mode}")
        first, second = rx_pair
        conjugate_product = csi[:, :, first] * np.conj(csi[:, :, second])
        if mode == "conj_phase":
            features = np.unwrap(np.angle(conjugate_product), axis=0)
            features = detrend(features, axis=0, type="linear")
        elif mode == "conj_complex":
            features = np.concatenate(
                [np.real(conjugate_product), np.imag(conjugate_product)],
                axis=1,
            )
            features = detrend(features, axis=0, type="linear")
        else:
            features = np.abs(conjugate_product)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.nan_to_num(features)


def first_principal_component(features: np.ndarray, standardize: bool = True) -> np.ndarray:
    """Project feature columns to the dominant temporal component."""
    if standardize:
        features = StandardScaler().fit_transform(features)
    component = PCA(n_components=1).fit_transform(features)
    return component


def dwt_denoise_matrix(
    matrix: np.ndarray,
    wavelet: str = "db4",
    level: int | None = None,
) -> np.ndarray:
    """Apply soft-threshold DWT denoising independently per feature column."""
    try:
        import pywt
    except ImportError as exc:
        raise RuntimeError("PyWavelets is required for DWT denoising; install requirements.txt") from exc

    denoised = np.empty_like(matrix, dtype=np.float64)
    for col_idx in range(matrix.shape[1]):
        coeffs = pywt.wavedec(matrix[:, col_idx], wavelet=wavelet, level=level, mode="symmetric")
        detail = coeffs[-1]
        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745 if detail.size else 0.0
        threshold = sigma * np.sqrt(2.0 * np.log(matrix.shape[0])) if sigma > 0 else 0.0
        filtered_coeffs = [coeffs[0]] + [pywt.threshold(coeff, threshold, mode="soft") for coeff in coeffs[1:]]
        reconstructed = pywt.waverec(filtered_coeffs, wavelet=wavelet, mode="symmetric")
        denoised[:, col_idx] = reconstructed[: matrix.shape[0]]
    return denoised


def maybe_butterworth(
    signal: np.ndarray,
    sample_rate_hz: float,
    low_hz: float | None,
    high_hz: float | None,
    order: int = 4,
) -> np.ndarray:
    """Apply optional low/high/band-pass Butterworth filtering."""
    if low_hz is None and high_hz is None:
        return signal

    nyquist = sample_rate_hz / 2.0
    if low_hz is not None and low_hz <= 0:
        low_hz = None
    if high_hz is not None and high_hz >= nyquist:
        high_hz = None

    if low_hz is None and high_hz is None:
        return signal
    if low_hz is not None and high_hz is not None and low_hz >= high_hz:
        raise ValueError(f"Invalid Butterworth band: low_hz={low_hz}, high_hz={high_hz}")

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


def stft_profile(
    signal: np.ndarray,
    sample_rate_hz: float,
    nperseg: int = 128,
    noverlap: int = 120,
    nfft: int = 256,
    power_floor: float = 1e-6,
    normalize: str = "frame",
) -> np.ndarray:
    """Compute centered STFT power as [time, Doppler bin]."""
    vector = np.asarray(signal).reshape(-1)
    if nperseg > vector.shape[0]:
        nperseg = vector.shape[0]
    if noverlap >= nperseg:
        noverlap = max(0, nperseg - 1)

    _, _, spectrum = stft(
        vector,
        fs=sample_rate_hz,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        return_onesided=False,
        boundary=None,
        padded=False,
    )
    power = np.abs(np.fft.fftshift(spectrum, axes=0)) ** 2
    profile = power.T
    return normalize_power_profile(profile, power_floor=power_floor, normalize=normalize)


def normalize_power_profile(
    profile: np.ndarray,
    power_floor: float = 1e-6,
    normalize: str = "frame",
) -> np.ndarray:
    """Normalize power profile while preserving [time, Doppler bin] shape."""
    profile = np.nan_to_num(profile.astype(np.float64, copy=False))
    if normalize == "frame":
        denom = np.max(profile, axis=1, keepdims=True)
    elif normalize == "global":
        denom = np.asarray(np.max(profile), dtype=np.float64)
    elif normalize == "none":
        denom = np.asarray(1.0, dtype=np.float64)
    else:
        raise ValueError("normalize must be one of: frame, global, none")

    denom = np.where(denom == 0, 1.0, denom)
    profile = profile / denom
    profile[profile < power_floor] = power_floor
    return profile


def format_doppler_for_plot(
    doppler: np.ndarray,
    plot_scale: str = "linear",
    db_min: float = -30.0,
    db_max: float = 0.0,
) -> tuple[np.ndarray, float | None, float | None, str]:
    """Return display-only Doppler values and color limits."""
    if plot_scale == "linear":
        return doppler, None, None, "normalized power"
    if plot_scale == "db":
        db = 10.0 * np.log10(np.maximum(doppler, 1e-12))
        return np.clip(db, db_min, db_max), db_min, db_max, "normalized power [dB]"
    raise ValueError("plot_scale must be one of: linear, db")


def compute_csif_like_doppler_for_file(
    path: str | Path,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] | None = (0, 1),
    auto_rx_pair: bool = False,
    sample_rate_hz: float | None = DEFAULT_SAMPLE_RATE_HZ,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    dwt_wavelet: str | None = None,
    dwt_level: int | None = None,
    butter_low_hz: float | None = None,
    butter_high_hz: float | None = None,
    butter_order: int = 4,
    spectrogram_method: str = "sharp_fft",
    stft_nperseg: int = 128,
    stft_noverlap: int = 120,
    stft_nfft: int = 256,
    power_floor: float = 1e-6,
    normalize: str = "frame",
) -> np.ndarray:
    result = compute_csif_like_doppler_result_for_file(
        path,
        mode=mode,
        stream_idx=stream_idx,
        rx_pair=rx_pair,
        auto_rx_pair=auto_rx_pair,
        sample_rate_hz=sample_rate_hz,
        hampel_window=hampel_window,
        hampel_sigmas=hampel_sigmas,
        dwt_wavelet=dwt_wavelet,
        dwt_level=dwt_level,
        butter_low_hz=butter_low_hz,
        butter_high_hz=butter_high_hz,
        butter_order=butter_order,
        spectrogram_method=spectrogram_method,
        stft_nperseg=stft_nperseg,
        stft_noverlap=stft_noverlap,
        stft_nfft=stft_nfft,
        power_floor=power_floor,
        normalize=normalize,
    )
    return result.doppler


def compute_csif_like_doppler_result_for_file(
    path: str | Path,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] | None = (0, 1),
    auto_rx_pair: bool = False,
    sample_rate_hz: float | None = DEFAULT_SAMPLE_RATE_HZ,
    **kwargs,
) -> CsifDopplerResult:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)
    inferred_sample_rate_hz = infer_sample_rate_hz(records)
    return compute_csif_like_doppler_result(
        csi,
        mode=mode,
        stream_idx=stream_idx,
        rx_pair=rx_pair,
        auto_rx_pair=auto_rx_pair,
        sample_rate_hz=inferred_sample_rate_hz if sample_rate_hz is None else sample_rate_hz,
        **kwargs,
    )


def compute_csif_like_doppler(
    csi: np.ndarray,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] | None = (0, 1),
    auto_rx_pair: bool = False,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    **kwargs,
) -> np.ndarray:
    result = compute_csif_like_doppler_result(
        csi,
        mode=mode,
        stream_idx=stream_idx,
        rx_pair=rx_pair,
        auto_rx_pair=auto_rx_pair,
        sample_rate_hz=sample_rate_hz,
        **kwargs,
    )
    return result.doppler


def compute_csif_like_doppler_result(
    csi: np.ndarray,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] | None = (0, 1),
    auto_rx_pair: bool = False,
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ,
    hampel_window: int = 7,
    hampel_sigmas: float = 3.0,
    dwt_wavelet: str | None = None,
    dwt_level: int | None = None,
    butter_low_hz: float | None = None,
    butter_high_hz: float | None = None,
    butter_order: int = 4,
    spectrogram_method: str = "sharp_fft",
    stft_nperseg: int = 128,
    stft_noverlap: int = 120,
    stft_nfft: int = 256,
    power_floor: float = 1e-6,
    normalize: str = "frame",
) -> CsifDopplerResult:
    """Compute one CSI-F-like Doppler trace from [packet, subcarrier, rx, tx]."""
    selected_rx_pair = rx_pair
    if mode in CONJUGATE_MODES and (auto_rx_pair or selected_rx_pair is None):
        selected_rx_pair = select_rx_pair(csi)

    features = build_feature_matrix(csi, mode=mode, stream_idx=stream_idx, rx_pair=selected_rx_pair)
    features = hampel_filter_matrix(features, window_size=hampel_window, n_sigmas=hampel_sigmas)

    component = first_principal_component(features)
    if dwt_wavelet is not None:
        component = dwt_denoise_matrix(component, wavelet=dwt_wavelet, level=dwt_level)
    component = maybe_butterworth(component, sample_rate_hz, butter_low_hz, butter_high_hz, order=butter_order)
    component = component - np.mean(component, axis=0, keepdims=True)

    if spectrogram_method == "sharp_fft":
        doppler = temporal_fft_profile(component, noise_level=np.log10(power_floor))
    elif spectrogram_method == "stft":
        doppler = stft_profile(
            component,
            sample_rate_hz=sample_rate_hz,
            nperseg=stft_nperseg,
            noverlap=stft_noverlap,
            nfft=stft_nfft,
            power_floor=power_floor,
            normalize=normalize,
        )
    else:
        raise ValueError("spectrogram_method must be one of: sharp_fft, stft")

    return CsifDopplerResult(
        doppler=doppler,
        sample_rate_hz=float(sample_rate_hz),
        rx_pair=selected_rx_pair if mode in CONJUGATE_MODES else None,
        mode=mode,
        spectrogram_method=spectrogram_method,
    )


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
    parser.add_argument("--actions", default="23,33,34,35,36,39")
    parser.add_argument("--mode", choices=["amplitude", "conj_phase", "conj_complex", "conj_amplitude"], default="amplitude")
    parser.add_argument("--stream", type=int, default=None)
    parser.add_argument("--rx-pair", default="0,1")
    parser.add_argument("--auto-rx-pair", action="store_true")
    parser.add_argument("--infer-sample-rate", action="store_true")
    parser.add_argument("--sample-rate-hz", type=float, default=DEFAULT_SAMPLE_RATE_HZ)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigmas", type=float, default=3.0)
    parser.add_argument("--dwt-wavelet", default=None)
    parser.add_argument("--dwt-level", type=int, default=None)
    parser.add_argument("--butter-low-hz", type=float, default=None)
    parser.add_argument("--butter-high-hz", type=float, default=None)
    parser.add_argument("--butter-order", type=int, default=4)
    parser.add_argument("--spectrogram-method", choices=["sharp_fft", "stft"], default="sharp_fft")
    parser.add_argument("--stft-nperseg", type=int, default=128)
    parser.add_argument("--stft-noverlap", type=int, default=120)
    parser.add_argument("--stft-nfft", type=int, default=256)
    parser.add_argument("--power-floor", type=float, default=1e-6)
    parser.add_argument("--normalize", choices=["frame", "global", "none"], default="frame")
    parser.add_argument("--mask-center-bins", type=int, default=0)
    parser.add_argument("--plot-scale", choices=["linear", "db"], default="linear")
    parser.add_argument("--db-min", type=float, default=-30.0)
    parser.add_argument("--db-max", type=float, default=0.0)
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
        result = compute_csif_like_doppler_result_for_file(
            path,
            mode=args.mode,
            stream_idx=args.stream,
            rx_pair=rx_pair,
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
        dopplers.append(mask_center_bins(result.doppler, args.mask_center_bins))
        labels.append(
            f"{action} {ACTION_NAMES.get(action, '')} "
            f"fs={result.sample_rate_hz:.1f}Hz rx={result.rx_pair}"
        )

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
        display, vmin, vmax, colorbar_label = format_doppler_for_plot(
            doppler,
            plot_scale=args.plot_scale,
            db_min=args.db_min,
            db_max=args.db_max,
        )
        im = ax.imshow(display.T, aspect="auto", origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(label)
        ax.set_ylabel("Doppler bin")

    stream_text = "all streams" if args.stream is None else f"stream {args.stream}"
    axes[-1].set_xlabel(
        f"time window - XRF55 CSI-F-like {args.mode}/{args.spectrogram_method}: "
        f"{args.scene}/{args.receiver}/subject {args.subject}/rep {args.repetition}/{stream_text}"
    )
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.015, label=colorbar_label)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(output)


if __name__ == "__main__":
    main()
