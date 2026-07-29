from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
import os
import pickle

import numpy as np
import scipy.io as sio
import torch

from wifi_doppler.data.doppler_dataset import parse_trace_filename
from wifi_doppler.data.windowing import WindowedTraceDataset, WindowIndex


SHARP_DELETED_SUBCARRIERS = np.asarray(
    [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255],
    dtype=int,
)
NUM_SHARP_DATA_SUBCARRIERS = 256 - len(SHARP_DELETED_SUBCARRIERS)

def canonical_scenario(name: str) -> str:
        """Normalize legacy, lowercase, or unhyphenated scenario folder names."""
        clean = name.strip().upper().replace("_", "").replace("-", "")
        if clean.startswith("S"):
            return f"AR-{clean[1:]}"
        for prefix in ("AR", "PC", "PI"):
            if clean.startswith(prefix):
                rest = clean[len(prefix):]
                return f"{prefix}-{rest}" if rest else prefix
        return os.name.upper()

@dataclass
class CsiDopplerPairRecording:
    """One raw CSI recording paired with its four SHARP Doppler streams."""

    scenario: str
    label: str
    repetition: str
    ground_truth: str
    raw_path: Path
    doppler_stream_paths: tuple[Path, ...]
    raw_start_packet: int = 0
    n_streams: int = 4
    target_transform: str = "sharp_centered"
    cache_raw: bool = False
    cache_doppler: bool = False

    _raw: np.ndarray | None = field(default=None, init=False, repr=False)
    _doppler: np.ndarray | None = field(default=None, init=False, repr=False)


    
    @property
    def length(self) -> int:
        """Number of Doppler time frames."""
        return self.load_doppler().shape[1]

    @property
    def filename_stem(self) -> str:
        if self.repetition:
            return f"{self.scenario}_{self.label}{self.repetition}"
        return f"{self.scenario}_{self.label}"


    def load_raw(self) -> np.ndarray:
        """Load raw SHARP/Nexmon CSI as [antenna, subcarrier, packet_time]."""
        if self.cache_raw and self._raw is not None:
            return self._raw

        csi_buff = sio.loadmat(self.raw_path)["csi_buff"]
        raw = split_sharp_monitor_streams(
            csi_buff,
            n_streams=self.n_streams,
            start_packet=self.raw_start_packet,
        )
        if self.cache_raw:
            self._raw = raw
        return raw

    def load_doppler(self) -> np.ndarray:
        """Load paired SHARP Doppler streams as [antenna, time, doppler_bin]."""
        if self.cache_doppler and self._doppler is not None:
            return self._doppler

        streams = tuple(_load_doppler_stream(path, self.target_transform) for path in self.doppler_stream_paths)
        shapes = {stream.shape for stream in streams}
        if len(shapes) != 1:
            raise ValueError(f"Doppler stream shapes do not match for {self.filename_stem}: {sorted(shapes)}")

        doppler = np.stack(streams, axis=0).astype(np.float32, copy=False)
        if self.cache_doppler:
            self._doppler = doppler
        return doppler

    def clear_cache(self) -> None:
        self._raw = None
        self._doppler = None


class CsiToSharpDopplerDataset(WindowedTraceDataset):
    """Paired raw CSI windows and matching SHARP Doppler targets.

    Returns ``(x_raw, y_doppler, filename)`` where:
    * ``x_raw`` is ``[4, selected_subcarriers, raw_time, 2]`` real/imag CSI.
    * ``y_doppler`` is ``[4, doppler_time, 100]`` SHARP Doppler.
    * ``filename`` is provenance, not a class target.
    """

    traces: list[CsiDopplerPairRecording]
    window_indexes: list[WindowIndex]

    def __init__(
        self,
        raw_root: str | Path = "data/CSI-80Mhz",
        doppler_root: str | Path = "data/doppler_traces",
        scenarios: Sequence[str] = ("S1a", "S1b", "S1c"),
        split: tuple[float, float] = (0.0, 0.6),
        doppler_window_size: int = 340,
        window_stride: int = 30,
        split_guard: int = 31,
        doppler_start: int = 800,
        doppler_sample_length: int = 31,
        doppler_sliding: int = 1,
        selected_subcarriers: Sequence[int] | None = None,
        num_subcarriers: int | None = None,
        subcarrier_sampling: str = "fixed_uniform",
        num_subcarrier_views: int = 1,
        subcarrier_seed: int = 0,
        target_transform: str = "sharp_centered",
        cache_raw: bool = False,
        cache_doppler: bool = False,
    ):
        self.raw_root = Path(raw_root)
        self.doppler_root = Path(doppler_root)
        self.doppler_start = doppler_start
        self.doppler_sample_length = doppler_sample_length
        self.doppler_sliding = doppler_sliding
        self.target_transform = target_transform
        self.cache_raw = cache_raw
        self.cache_doppler = cache_doppler
        self.unmatched_raw_files: list[Path] = []
        self.unmatched_doppler_keys: list[tuple[str, str, str]] = []
        self.subcarrier_views = build_subcarrier_views(
            selected_subcarriers=selected_subcarriers,
            num_subcarriers=num_subcarriers,
            sampling=subcarrier_sampling,
            num_views=num_subcarrier_views,
            seed=subcarrier_seed,
        )

        super().__init__(
            scenarios=scenarios,
            split=split,
            window_size=doppler_window_size,
            window_stride=window_stride,
            split_guard=split_guard,
            labels=("surrogate",), # TODO fix
        )

    def __len__(self) -> int:
        return len(self.window_indexes) * len(self.subcarrier_views)

    def __getitem__(self, idx: int):
        base_idx = idx // len(self.subcarrier_views)
        view_idx = idx % len(self.subcarrier_views)

        window = self.window_indexes[base_idx]
        recording = self.traces[window.recording_idx]
        selected_subcarriers = self.subcarrier_views[view_idx]

        raw = recording.load_raw()
        doppler = recording.load_doppler()

        raw_start, raw_end = self.raw_bounds_for_doppler_window(window.start, window.end)
        x = raw[:, selected_subcarriers, raw_start:raw_end]
        y = doppler[:, window.start:window.end]

        x = np.stack((x.real, x.imag), axis=-1).astype(np.float32, copy=False)

        # TODO fix
        filename = f"{recording.filename_stem}_d{window.start}-{window.end}_view{view_idx}"

        return (
            torch.from_numpy(np.ascontiguousarray(x)).float(),
            torch.from_numpy(np.ascontiguousarray(y)).float(),
            filename,
        )

    def raw_bounds_for_doppler_window(self, doppler_start_frame: int, doppler_end_frame: int) -> tuple[int, int]:
        raw_start = self.doppler_start + doppler_start_frame * self.doppler_sliding
        raw_end = self.doppler_start + (doppler_end_frame - 1) * self.doppler_sliding + self.doppler_sample_length
        return raw_start, raw_end

    def pairing_report(self) -> dict[str, object]:
        return {
            "num_pairs": len(self.traces),
            "num_unmatched_raw_files": len(self.unmatched_raw_files),
            "num_unmatched_doppler_keys": len(self.unmatched_doppler_keys),
            "unmatched_raw_files": [str(path) for path in self.unmatched_raw_files],
            "unmatched_doppler_keys": [list(key) for key in self.unmatched_doppler_keys],
            "subcarrier_views": [view.tolist() for view in self.subcarrier_views],
        }

    def _parse_traces(self) -> list[CsiDopplerPairRecording]:
        doppler_groups = self._collect_doppler_groups()
        raw_paths = self._collect_raw_paths()

        traces: list[CsiDopplerPairRecording] = []
        matched_doppler_keys: set[tuple[str, str, str]] = set()

        for key, raw_path in sorted(raw_paths.items()):
            stream_paths = doppler_groups.get(key)
            if stream_paths is None:
                self.unmatched_raw_files.append(raw_path)
                continue
            matched_doppler_keys.add(key)
            scenario, label, repetition = key
            traces.append(
                CsiDopplerPairRecording(
                    scenario=scenario,
                    label=label,
                    repetition=repetition,
                    ground_truth="surrogate",
                    raw_path=raw_path,
                    doppler_stream_paths=tuple(stream_paths),
                    target_transform=self.target_transform,
                    cache_raw=self.cache_raw,
                    cache_doppler=self.cache_doppler,
                )
            )

        self.unmatched_doppler_keys = sorted(set(doppler_groups) - matched_doppler_keys)
        if not traces:
            raise ValueError(
                f"No paired CSI/Doppler recordings found for scenarios {self.scenarios} "
                f"under raw_root={self.raw_root} and doppler_root={self.doppler_root}"
            )
        return traces

    def _collect_doppler_groups(self) -> dict[tuple[str, str, str], list[Path]]:
        groups: dict[tuple[str, str, str], list[Path]] = {}
        available_dirs = {path.name: path for path in self.doppler_root.iterdir() if path.is_dir()}
        
        for scenario in self.scenarios:
            matched_dir_name = next(
                (name for name in available_dirs if canonical_scenario(name) == canonical_scenario(scenario)),
                scenario
            )
            scenario_dir = self.doppler_root / matched_dir_name
            if not scenario_dir.is_dir():
                raise FileNotFoundError(f"Missing Doppler scenario directory: {scenario_dir}")
            
            canonical_name = canonical_scenario(scenario)

            for entry in os.scandir(scenario_dir):
                if not entry.is_file():
                    continue
                info = parse_trace_filename(entry.name)
                if not info:
                    continue
                key = (canonical_name, str(info["label"]), str(info["repetition"]))
                antenna = int(info["antenna"])
                groups.setdefault(key, [None] * 4)
                groups[key][antenna] = Path(entry.path)

        complete: dict[tuple[str, str, str], list[Path]] = {}
        for key, stream_paths in groups.items():
            if any(path is None for path in stream_paths):
                raise ValueError(f"Missing Doppler stream for {key}: {stream_paths}")
            complete[key] = stream_paths
        return complete

    def _collect_raw_paths(self) -> dict[tuple[str, str, str], Path]:
        paths: dict[tuple[str, str, str], Path] = {}
        for scenario in self.scenarios:
            canonical_name = canonical_scenario(scenario)
            candidates = [self.raw_root / canonical_name, self.raw_root / scenario]
            raw_dir_candidate = self.raw_root / raw_scenario_dir(canonical_name)
            if raw_dir_candidate not in candidates:
                candidates.append(raw_dir_candidate)
                
            raw_dir = next((path for path in candidates if path.is_dir()), None)
            if raw_dir is None:
                expected = ", ".join(str(path) for path in candidates)
                raise FileNotFoundError(
                    f"Missing raw CSI scenario directory; expected one of: {expected}"
                )

            for entry in os.scandir(raw_dir):
                if not entry.is_file() or not entry.name.endswith(".mat"):
                    continue
                key = raw_file_key(canonical_name, Path(entry.name).stem)
                if key is not None:
                    paths[key] = Path(entry.path)
        return paths


def split_sharp_monitor_streams(csi_buff: np.ndarray, *, n_streams: int = 4, start_packet: int = 0) -> np.ndarray:
    """Apply SHARP's raw stream split and subcarrier cleanup without Doppler extraction."""
    if csi_buff.ndim != 2 or csi_buff.shape[1] != 256:
        raise ValueError(f"Expected csi_buff shape [packets, 256], got {csi_buff.shape}")

    csi_buff = np.fft.fftshift(csi_buff, axes=1)
    csi_buff = csi_buff[np.sum(csi_buff, axis=1) != 0]

    stream_length = int(np.floor(csi_buff.shape[0] / n_streams))
    streams: list[np.ndarray] = []

    for stream_idx in range(n_streams):
        stream = csi_buff[stream_idx : stream_length * n_streams + 1 : n_streams, :]
        stream = stream[start_packet:stream_length, :]
        stream[:, 64:] = -stream[:, 64:]
        stream = np.delete(stream, SHARP_DELETED_SUBCARRIERS, axis=1)

        mean_signal = np.mean(np.abs(stream), axis=1, keepdims=True)
        mean_signal[mean_signal == 0] = 1.0
        streams.append((stream / mean_signal).T.astype(np.complex64, copy=False))

    return np.stack(streams, axis=0)


def build_subcarrier_views(
    *,
    selected_subcarriers: Sequence[int] | None,
    num_subcarriers: int | None,
    sampling: str,
    num_views: int,
    seed: int,
) -> tuple[np.ndarray, ...]:
    if num_views < 1:
        raise ValueError("num_subcarrier_views must be >= 1")
    if selected_subcarriers is not None:
        if num_subcarriers is not None:
            raise ValueError("Use either selected_subcarriers or num_subcarriers, not both.")
        if num_views != 1:
            raise ValueError("selected_subcarriers defines one exact view; use num_subcarriers for multiple views.")
        return (_validate_subcarriers(np.asarray(selected_subcarriers, dtype=int)),)

    if num_subcarriers is None:
        if num_views != 1:
            raise ValueError("Multiple subcarrier views require num_subcarriers.")
        return (np.arange(NUM_SHARP_DATA_SUBCARRIERS, dtype=int),)
    if num_subcarriers < 1 or num_subcarriers > NUM_SHARP_DATA_SUBCARRIERS:
        raise ValueError(f"num_subcarriers must be in [1, {NUM_SHARP_DATA_SUBCARRIERS}]")

    if sampling == "fixed_uniform":
        if num_views != 1:
            raise ValueError("fixed_uniform supports one view. Use random_views for multiple views.")
        return (np.linspace(0, NUM_SHARP_DATA_SUBCARRIERS - 1, num_subcarriers, dtype=int),)
    if sampling == "fixed_random":
        if num_views != 1:
            raise ValueError("fixed_random supports one view. Use random_views for multiple views.")
        rng = np.random.default_rng(seed)
        return (np.sort(rng.choice(NUM_SHARP_DATA_SUBCARRIERS, size=num_subcarriers, replace=False)),)
    if sampling == "random_views":
        rng = np.random.default_rng(seed)
        return tuple(
            np.sort(rng.choice(NUM_SHARP_DATA_SUBCARRIERS, size=num_subcarriers, replace=False))
            for _ in range(num_views)
        )
    raise ValueError(f"Unknown subcarrier_sampling: {sampling}")


def raw_scenario_dir(doppler_scenario: str) -> str:
    canonical = canonical_scenario(doppler_scenario)
    if canonical.startswith("S"):
        return f"AR-{canonical[1:]}"
    return canonical


def raw_file_key(doppler_scenario: str, raw_stem: str) -> tuple[str, str, str] | None:
    canonical = canonical_scenario(doppler_scenario)
    if canonical.startswith("AR-"):
        num_part = canonical[3:]
        prefixes = (f"AR-{num_part}_", f"AR{num_part}_", f"S{num_part}_")
        prefix = next((value for value in prefixes if raw_stem.startswith(value)), None)
        if prefix is None:
            return None
        token = raw_stem[len(prefix) :]
        if not token:
            return None
        label = token[:1]
        repetition = token[1:].lstrip("_")
        return canonical, label, repetition

    prefix = canonical.replace("-", "")
    if raw_stem.startswith(prefix + "_"):
        return canonical, raw_stem[len(prefix) + 1 :], ""
    if raw_stem.startswith(canonical + "_"):
        return canonical, raw_stem[len(canonical) + 1 :], ""
    return None

def _load_doppler_stream(path: Path, target_transform: str = "none") -> np.ndarray:
    with path.open("rb") as fp:
        arr = pickle.load(fp)
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        raise ValueError(f"Expected 2D NumPy Doppler stream in {path}, got {type(arr)} {getattr(arr, 'shape', None)}")

    # This is Sharp implementation, but for distillation leaks validation/test statistics and makes target at one time now depending on the whole recording
    if target_transform == "sharp_centered":
        arr = arr - arr.mean(axis=0, keepdims=True)
    elif target_transform != "none":
        raise ValueError(f"Unknown target_transform: {target_transform}")
    return arr.astype(np.float32, copy=False)


def _validate_subcarriers(indices: np.ndarray) -> np.ndarray:
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("selected_subcarriers must be a non-empty 1D sequence")
    if np.any(indices < 0) or np.any(indices >= NUM_SHARP_DATA_SUBCARRIERS):
        raise ValueError(f"selected_subcarriers must be within [0, {NUM_SHARP_DATA_SUBCARRIERS - 1}]")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("selected_subcarriers contains duplicates")
    return np.sort(indices).astype(int, copy=False)
