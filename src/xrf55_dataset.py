"""Dataset indexing and cache helpers for XRF55 Wi-Fi data."""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset

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
    rx_pair: tuple[int, int] = (0, 1),
) -> list[XRF55DopplerRecording]:
    """Precompute CSI-F-like Doppler caches for XRF55 raw recordings."""
    cached: list[XRF55DopplerRecording] = []

    for recording in recordings:
        output_path = xrf55_doppler_cache_path(cache_root, recording)
        if overwrite or not output_path.exists():
            from xrf55_csif_doppler import compute_csif_like_doppler

            doppler = compute_csif_like_doppler(
                recording.load_csi(strict=False),
                mode=mode,
                stream_idx=stream_idx,
                rx_pair=rx_pair,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                output_path,
                doppler=doppler,
                scene=recording.scene,
                receiver=recording.receiver,
                subject=recording.subject,
                action=recording.action,
                repetition=recording.repetition,
                source_path=str(recording.path),
                extractor="xrf55_csif_doppler",
                mode=mode,
                stream_idx="" if stream_idx is None else str(stream_idx),
                rx_pair=f"{rx_pair[0]},{rx_pair[1]}",
            )

        cached.append(
            XRF55DopplerRecording(
                scene=recording.scene,
                receiver=recording.receiver,
                subject=recording.subject,
                action=recording.action,
                repetition=recording.repetition,
                path=output_path,
            )
        )

    return cached


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
    ):
        if label_mode not in {"action", "subject"}:
            raise ValueError("label_mode must be 'action' or 'subject'")
        if crop_strategy not in {"none", "center", "random_center_jitter", "even"}:
            raise ValueError("crop_strategy must be one of: none, center, random_center_jitter, even")
        if crops_per_recording < 1:
            raise ValueError("crops_per_recording must be >= 1")
        if crop_strategy != "none" and crop_size is None:
            raise ValueError("crop_size is required when crop_strategy is not 'none'")

        self.cache_root = Path(cache_root)
        self.label_mode = label_mode
        self.crop_size = crop_size
        self.crops_per_recording = crops_per_recording
        self.crop_strategy = crop_strategy
        self.crop_jitter = crop_jitter
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
        x = torch.from_numpy(doppler).float().unsqueeze(0)
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
