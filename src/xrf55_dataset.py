"""Dataset indexing and cache helpers for XRF55 Wi-Fi data."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from recordings import XRF55DopplerRecording, XRF55RawRecording


XRF55_FILE_PATTERN = re.compile(
    r"^(?P<subject>\d{2})_(?P<action>\d{2})_(?P<repetition>\d{2})\.(?P<ext>dat|mat)$"
)

XRF55_DOPPLER_CACHE_PATTERN = re.compile(
    r"^(?P<scene>Scene_\d+)_(?P<receiver>[^_]+)_"
    r"(?P<subject>\d{2})_(?P<action>\d{2})_(?P<repetition>\d{2})\.npz$"
)


@dataclass(frozen=True)
class DopplerCropIndex:
    """Index of one crop within one cached Doppler recording."""

    recording_idx: int
    crop_idx: int


@dataclass(frozen=True)
class XRF55CacheConfig:
    """Parameters for one XRF55 Doppler cache variant."""

    cache_root: Path
    overwrite: bool
    mode: str
    stream_idx: int | None
    rx_pair: tuple[int, int] | None
    auto_rx_pair: bool
    infer_sample_rate: bool
    sample_rate_hz: float
    hampel_window: int
    hampel_sigmas: float
    dwt_wavelet: str | None
    dwt_level: int | None
    butter_low_hz: float | None
    butter_high_hz: float | None
    butter_order: int
    spectrogram_method: str
    stft_nperseg: int
    stft_noverlap: int
    stft_nfft: int
    power_floor: float
    normalize: str
    center_mask_bins: int


@dataclass(frozen=True)
class XRF55CacheTask:
    """One raw recording plus its expected cache output path."""

    raw_recording: XRF55RawRecording
    output_path: Path


def scan_xrf55_raw_wifi(root: str | Path = "data/XRF55_rawdata/WiFi") -> list[XRF55RawRecording]:
    """Scan XRF55 raw Wi-Fi files into typed recording objects."""
    root = Path(root)
    recordings: list[XRF55RawRecording] = []

    for path in sorted(root.glob("Scene_*/*/*/*")):
        if not path.is_file():
            continue

        match = XRF55_FILE_PATTERN.match(path.name)
        if not match:
            continue

        receiver = path.parent.parent.name
        scene = path.parent.parent.parent.name
        subject_dir = path.parent.name
        parts = match.groupdict()
        if subject_dir != parts["subject"]:
            raise ValueError(f"Subject directory and filename disagree: {path}")

        recordings.append(
            XRF55RawRecording(
                scene=scene,
                receiver=receiver,
                subject=parts["subject"],
                action=parts["action"],
                repetition=parts["repetition"],
                path=path,
            )
        )

    return recordings


def xrf55_doppler_cache_path(cache_root: str | Path, recording: XRF55RawRecording) -> Path:
    """Return the standard cache path for one XRF55 raw recording."""
    filename = (
        f"{recording.scene}_{recording.receiver}_"
        f"{recording.subject}_{recording.action}_{recording.repetition}.npz"
    )
    return Path(cache_root) / recording.scene / recording.receiver / recording.subject / filename


def cache_xrf55_csif_doppler(
    recordings: list[XRF55RawRecording],
    cache_root: str | Path = "data/XRF55_doppler/csif",
    overwrite: bool = False,
    mode: str = "amplitude",
    stream_idx: int | None = None,
    rx_pair: tuple[int, int] | None = (0, 1),
    auto_rx_pair: bool = False,
    infer_sample_rate: bool = False,
    sample_rate_hz: float = 200.0,
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
    center_mask_bins: int = 0,
    num_workers: int | None = 0,
    show_progress: bool = True,
) -> list[XRF55DopplerRecording]:
    """Precompute CSI-F-like Doppler caches for XRF55 raw recordings."""
    config = XRF55CacheConfig(
        cache_root=Path(cache_root),
        overwrite=overwrite,
        mode=mode,
        stream_idx=stream_idx,
        rx_pair=rx_pair,
        auto_rx_pair=auto_rx_pair,
        infer_sample_rate=infer_sample_rate,
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
        center_mask_bins=center_mask_bins,
    )

    tasks = [
        XRF55CacheTask(
            raw_recording=recording,
            output_path=xrf55_doppler_cache_path(config.cache_root, recording),
        )
        for recording in recordings
    ]
    cached = [_task_to_cached_doppler(task) for task in tasks]
    missing = [
        task
        for task in tasks
        if overwrite or not _cache_matches_config(task.output_path, config)
    ]

    if num_workers is None:
        num_workers = max(1, min((os.cpu_count() or 2) - 1, 8))

    if missing and num_workers == 1:
        num_workers = 0

    if missing and num_workers > 0:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [
                executor.submit(_write_xrf55_doppler_cache, task, config)
                for task in missing
            ]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="Caching XRF55 Doppler")
            for future in iterator:
                future.result()
    else:
        iterator = missing
        if show_progress and missing:
            iterator = tqdm(iterator, desc="Caching XRF55 Doppler")
        for task in iterator:
            _write_xrf55_doppler_cache(task, config)

    return cached


def _task_to_cached_doppler(
    task: XRF55CacheTask,
) -> XRF55DopplerRecording:
    recording = task.raw_recording
    return XRF55DopplerRecording(
        scene=recording.scene,
        receiver=recording.receiver,
        subject=recording.subject,
        action=recording.action,
        repetition=recording.repetition,
        path=task.output_path,
    )


def _write_xrf55_doppler_cache(task: XRF55CacheTask, config: XRF55CacheConfig) -> None:
    from xrf55_csif_doppler import compute_csif_like_doppler_result_for_file
    from xrf55_doppler import mask_center_bins

    recording = task.raw_recording
    result = compute_csif_like_doppler_result_for_file(
        recording.path,
        mode=config.mode,
        stream_idx=config.stream_idx,
        rx_pair=config.rx_pair,
        auto_rx_pair=config.auto_rx_pair,
        sample_rate_hz=None if config.infer_sample_rate else config.sample_rate_hz,
        hampel_window=config.hampel_window,
        hampel_sigmas=config.hampel_sigmas,
        dwt_wavelet=config.dwt_wavelet,
        dwt_level=config.dwt_level,
        butter_low_hz=config.butter_low_hz,
        butter_high_hz=config.butter_high_hz,
        butter_order=config.butter_order,
        spectrogram_method=config.spectrogram_method,
        stft_nperseg=config.stft_nperseg,
        stft_noverlap=config.stft_noverlap,
        stft_nfft=config.stft_nfft,
        power_floor=config.power_floor,
        normalize=config.normalize,
    )
    doppler = mask_center_bins(result.doppler, config.center_mask_bins)
    task.output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        task.output_path,
        doppler=doppler,
        scene=recording.scene,
        receiver=recording.receiver,
        subject=recording.subject,
        action=recording.action,
        repetition=recording.repetition,
        source_path=str(recording.path),
        extractor="xrf55_csif_doppler",
        mode=config.mode,
        stream_idx="" if config.stream_idx is None else str(config.stream_idx),
        rx_pair="" if result.rx_pair is None else f"{result.rx_pair[0]},{result.rx_pair[1]}",
        auto_rx_pair=str(config.auto_rx_pair),
        sample_rate_hz=str(result.sample_rate_hz),
        infer_sample_rate=str(config.infer_sample_rate),
        hampel_window=str(config.hampel_window),
        hampel_sigmas=str(config.hampel_sigmas),
        dwt_wavelet="" if config.dwt_wavelet is None else config.dwt_wavelet,
        dwt_level="" if config.dwt_level is None else str(config.dwt_level),
        butter_low_hz="" if config.butter_low_hz is None else str(config.butter_low_hz),
        butter_high_hz="" if config.butter_high_hz is None else str(config.butter_high_hz),
        butter_order=str(config.butter_order),
        spectrogram_method=config.spectrogram_method,
        stft_nperseg=str(config.stft_nperseg),
        stft_noverlap=str(config.stft_noverlap),
        stft_nfft=str(config.stft_nfft),
        power_floor=str(config.power_floor),
        normalize=config.normalize,
        center_mask_bins=str(config.center_mask_bins),
    )


def _cache_matches_config(path: Path, config: XRF55CacheConfig) -> bool:
    if not path.exists():
        return False

    expected = _cache_metadata(config)
    try:
        with np.load(path, allow_pickle=False) as data:
            if "doppler" not in data:
                return False
            for key, value in expected.items():
                if key not in data or str(data[key]) != value:
                    return False
    except Exception:
        return False

    return True


def _cache_metadata(config: XRF55CacheConfig) -> dict[str, str]:
    return {
        "extractor": "xrf55_csif_doppler",
        "mode": config.mode,
        "stream_idx": "" if config.stream_idx is None else str(config.stream_idx),
        "auto_rx_pair": str(config.auto_rx_pair),
        "infer_sample_rate": str(config.infer_sample_rate),
        "hampel_window": str(config.hampel_window),
        "hampel_sigmas": str(config.hampel_sigmas),
        "dwt_wavelet": "" if config.dwt_wavelet is None else config.dwt_wavelet,
        "dwt_level": "" if config.dwt_level is None else str(config.dwt_level),
        "butter_low_hz": "" if config.butter_low_hz is None else str(config.butter_low_hz),
        "butter_high_hz": "" if config.butter_high_hz is None else str(config.butter_high_hz),
        "butter_order": str(config.butter_order),
        "spectrogram_method": config.spectrogram_method,
        "stft_nperseg": str(config.stft_nperseg),
        "stft_noverlap": str(config.stft_noverlap),
        "stft_nfft": str(config.stft_nfft),
        "power_floor": str(config.power_floor),
        "normalize": config.normalize,
        "center_mask_bins": str(config.center_mask_bins),
    }


def scan_xrf55_doppler_cache(
    root: str | Path = "data/XRF55_doppler/csif",
) -> list[XRF55DopplerRecording]:
    """Scan cached XRF55 Doppler files into typed recording objects."""
    root = Path(root)
    recordings: list[XRF55DopplerRecording] = []

    for path in sorted(root.glob("Scene_*/*/*/*.npz")):
        match = XRF55_DOPPLER_CACHE_PATTERN.match(path.name)
        if not match:
            continue

        parts = match.groupdict()
        recordings.append(
            XRF55DopplerRecording(
                scene=parts["scene"],
                receiver=parts["receiver"],
                subject=parts["subject"],
                action=parts["action"],
                repetition=parts["repetition"],
                path=path,
            )
        )

    return recordings


class XRF55DopplerDataset(Dataset):
    """PyTorch dataset for cached XRF55 Doppler trials."""

    def __init__(
        self,
        cache_root: str | Path = "data/XRF55_doppler/csif",
        label_mode: str = "action",
        scenes: set[str] | None = None,
        receivers: set[str] | None = None,
        subjects: set[str] | None = None,
        actions: set[str] | None = None,
        repetitions: set[str] | None = None,
        crop_size: int | None = None,
        crops_per_recording: int = 1,
        crop_strategy: str = "none",
        crop_jitter: int = 0,
        input_scale: str = "linear",
        db_min: float = -30.0,
        db_max: float = 0.0,
        subtract_time_mean: bool = True,
    ):
        if label_mode not in {"action", "subject"}:
            raise ValueError("label_mode must be 'action' or 'subject'")
        if crop_strategy not in {"none", "center", "random_center_jitter", "even"}:
            raise ValueError("crop_strategy must be one of: none, center, random_center_jitter, even")
        if crops_per_recording < 1:
            raise ValueError("crops_per_recording must be >= 1")
        if crop_strategy != "none" and crop_size is None:
            raise ValueError("crop_size is required when crop_strategy is not 'none'")
        if input_scale not in {"linear", "db"}:
            raise ValueError("input_scale must be one of: linear, db")
        if db_min >= db_max:
            raise ValueError("db_min must be less than db_max")

        self.cache_root = Path(cache_root)
        self.label_mode = label_mode
        self.crop_size = crop_size
        self.crops_per_recording = crops_per_recording
        self.crop_strategy = crop_strategy
        self.crop_jitter = crop_jitter
        self.input_scale = input_scale
        self.db_min = db_min
        self.db_max = db_max
        self.subtract_time_mean = subtract_time_mean
        self.recordings = [
            recording
            for recording in scan_xrf55_doppler_cache(self.cache_root)
            if _matches_filter(recording.scene, scenes)
            and _matches_filter(recording.receiver, receivers)
            and _matches_filter(recording.subject, subjects)
            and _matches_filter(recording.action, actions)
            and _matches_filter(recording.repetition, repetitions)
        ]
        self.crop_indexes = [
            DopplerCropIndex(recording_idx=recording_idx, crop_idx=crop_idx)
            for recording_idx in range(len(self.recordings))
            for crop_idx in range(self.crops_per_recording)
        ]

        labels = sorted({self._recording_label(recording) for recording in self.recordings})
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.crop_indexes)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        crop_index = self.crop_indexes[idx]
        recording = self.recordings[crop_index.recording_idx]
        doppler = recording.load_doppler()
        doppler = self._crop_doppler(doppler, crop_index.crop_idx)
        doppler = self._scale_doppler(doppler)
        x = torch.from_numpy(doppler).float().unsqueeze(0)
        if self.subtract_time_mean:
            x = x - x.mean(dim=1, keepdim=True)
        y = torch.tensor(self.label_to_idx[self._recording_label(recording)], dtype=torch.long)
        return x, y

    def _recording_label(self, recording: XRF55DopplerRecording) -> str:
        return recording.action if self.label_mode == "action" else recording.subject

    def _crop_doppler(self, doppler: np.ndarray, crop_idx: int) -> np.ndarray:
        if self.crop_strategy == "none" or self.crop_size is None:
            return doppler

        if self.crop_size > doppler.shape[0]:
            raise ValueError(f"crop_size {self.crop_size} exceeds Doppler length {doppler.shape[0]}")

        start = self._crop_start(doppler.shape[0], crop_idx)
        end = start + self.crop_size
        return doppler[start:end]

    def _crop_start(self, trace_length: int, crop_idx: int) -> int:
        max_start = trace_length - self.crop_size
        center_start = max_start // 2

        if self.crop_strategy == "center":
            return center_start

        if self.crop_strategy == "random_center_jitter":
            low = max(0, center_start - self.crop_jitter)
            high = min(max_start, center_start + self.crop_jitter)
            return int(torch.randint(low, high + 1, size=(1,)).item())

        if self.crop_strategy == "even":
            if self.crops_per_recording == 1:
                return center_start
            return round(crop_idx * max_start / (self.crops_per_recording - 1))

        raise ValueError(f"Unknown crop strategy: {self.crop_strategy}")

    def _scale_doppler(self, doppler: np.ndarray) -> np.ndarray:
        if self.input_scale == "linear":
            return doppler

        db = 10.0 * np.log10(np.maximum(doppler, 1e-12))
        db = np.clip(db, self.db_min, self.db_max)
        return (db - self.db_min) / (self.db_max - self.db_min)


def _matches_filter(value: str, allowed: set[str] | None) -> bool:
    return allowed is None or value in allowed


if __name__ == "__main__":
    recordings = scan_xrf55_raw_wifi()
    print(f"Raw Wi-Fi recordings: {len(recordings)}")
    print("By scene/receiver:")
    for key, count in sorted(Counter((r.scene, r.receiver) for r in recordings).items()):
        print(f"  {key[0]} {key[1]}: {count}")

    first = recordings[0]
    print(f"First recording: {first.recording_id}")
    print(f"Path: {first.path}")
    print(f"CSI shape: {first.load_csi().shape}")
