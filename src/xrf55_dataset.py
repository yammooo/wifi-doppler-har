"""Dataset indexing and cache helpers for XRF55 Wi-Fi data."""

from collections import Counter
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
    ):
        if label_mode not in {"action", "subject"}:
            raise ValueError("label_mode must be 'action' or 'subject'")

        self.cache_root = Path(cache_root)
        self.label_mode = label_mode
        self.recordings = [
            recording
            for recording in scan_xrf55_doppler_cache(self.cache_root)
            if _matches_filter(recording.scene, scenes)
            and _matches_filter(recording.receiver, receivers)
            and _matches_filter(recording.subject, subjects)
            and _matches_filter(recording.action, actions)
            and _matches_filter(recording.repetition, repetitions)
        ]

        labels = sorted({self._recording_label(recording) for recording in self.recordings})
        self.label_to_idx = {label: idx for idx, label in enumerate(labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self) -> int:
        return len(self.recordings)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        recording = self.recordings[idx]
        doppler = recording.load_doppler()
        x = torch.from_numpy(doppler).float().unsqueeze(0)
        y = torch.tensor(self.label_to_idx[self._recording_label(recording)], dtype=torch.long)
        return x, y

    def _recording_label(self, recording: XRF55DopplerRecording) -> str:
        return recording.action if self.label_mode == "action" else recording.subject


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
