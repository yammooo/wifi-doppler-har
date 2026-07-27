from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

from wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset


PREPARED_FORMAT_VERSION = 1


@dataclass
class PreparedCsiDopplerRecording:
    """One memory-mapped CSI/Doppler recording produced by the converter."""

    scenario: str
    label: str
    repetition: str
    ground_truth: str
    raw_path: Path
    doppler_path: Path
    raw_shape: tuple[int, ...]
    doppler_shape: tuple[int, ...]
    _raw: np.ndarray | None = field(default=None, init=False, repr=False)
    _doppler: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def length(self) -> int:
        return self.doppler_shape[1]

    @property
    def filename_stem(self) -> str:
        if self.repetition:
            return f"{self.scenario}_{self.label}{self.repetition}"
        return f"{self.scenario}_{self.label}"

    def load_raw(self) -> np.ndarray:
        if self._raw is None:
            self._raw = np.load(self.raw_path, mmap_mode="r", allow_pickle=False)
            if self._raw.shape != self.raw_shape or self._raw.dtype != np.complex64:
                raise ValueError(
                    f"Prepared raw CSI does not match its manifest: {self.raw_path} "
                    f"has shape={self._raw.shape}, dtype={self._raw.dtype}"
                )
        return self._raw

    def load_doppler(self) -> np.ndarray:
        if self._doppler is None:
            self._doppler = np.load(self.doppler_path, mmap_mode="r", allow_pickle=False)
            if self._doppler.shape != self.doppler_shape or self._doppler.dtype != np.float32:
                raise ValueError(
                    f"Prepared Doppler does not match its manifest: {self.doppler_path} "
                    f"has shape={self._doppler.shape}, dtype={self._doppler.dtype}"
                )
        return self._doppler

    def clear_cache(self) -> None:
        _close_memmap(self._raw)
        _close_memmap(self._doppler)
        self._raw = None
        self._doppler = None


class PreparedCsiToSharpDopplerDataset(CsiToSharpDopplerDataset):
    """Paired dataset backed by uncompressed, memory-mapped NumPy arrays."""

    def __init__(self, prepared_root: str | Path, **kwargs):
        self.prepared_root = Path(prepared_root)
        if kwargs.get("target_transform", "none") != "none":
            raise ValueError("Prepared Doppler targets support target_transform='none' only.")
        kwargs["target_transform"] = "none"
        super().__init__(
            raw_root=self.prepared_root,
            doppler_root=self.prepared_root,
            **kwargs,
        )

    def _parse_traces(self) -> list[PreparedCsiDopplerRecording]:
        manifest_path = self.prepared_root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Prepared dataset manifest not found: {manifest_path}. "
                "Run scripts/convert_csi_doppler_memmap.py first."
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format_version") != PREPARED_FORMAT_VERSION:
            raise ValueError(
                f"Unsupported prepared dataset format {manifest.get('format_version')!r}; "
                f"expected {PREPARED_FORMAT_VERSION}."
            )

        traces = []
        for item in manifest.get("recordings", []):
            if item["scenario"] not in self.scenarios:
                continue
            raw_path = self.prepared_root / item["raw_path"].replace("\\", "/")
            doppler_path = self.prepared_root / item["doppler_path"].replace("\\", "/")
            if not raw_path.is_file() or not doppler_path.is_file():
                raise FileNotFoundError(f"Prepared files are missing for {item['filename_stem']}")
            traces.append(
                PreparedCsiDopplerRecording(
                    scenario=item["scenario"],
                    label=item["label"],
                    repetition=item["repetition"],
                    ground_truth="surrogate",
                    raw_path=raw_path,
                    doppler_path=doppler_path,
                    raw_shape=tuple(item["raw_shape"]),
                    doppler_shape=tuple(item["doppler_shape"]),
                )
            )

        if not traces:
            raise ValueError(
                f"No prepared recordings found for scenarios {self.scenarios} in {manifest_path}"
            )
        return sorted(traces, key=lambda trace: (trace.scenario, trace.label, trace.repetition))

    def pairing_report(self) -> dict[str, Any]:
        report = super().pairing_report()
        report.update(
            {
                "storage": "memmap",
                "prepared_root": str(self.prepared_root.resolve()),
                "format_version": PREPARED_FORMAT_VERSION,
            }
        )
        return report


def _close_memmap(array: np.ndarray | None) -> None:
    mapping = getattr(array, "_mmap", None)
    if mapping is not None:
        mapping.close()
