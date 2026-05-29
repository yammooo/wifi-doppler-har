"""Recording containers and lazy trace loading helpers."""

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np

from wifi_doppler.data.windowing import WindowIndex


@dataclass
class TraceRecording:
    """Doppler recording metadata plus lazily loaded antenna streams."""

    scenario: str
    label: str
    repetition: str
    ground_truth: str
    stream_paths: tuple[Path, ...]

    _stream_data: list[np.ndarray | None] | None = field(default=None, init=False, repr=False)

    def _load_stream(self, antenna_idx: int) -> np.ndarray:
        """Load and cache one antenna stream."""
        if antenna_idx < 0 or antenna_idx >= len(self.stream_paths):
            raise IndexError(f"Invalid antenna index {antenna_idx}")

        # Initialize the stream data cache on first access
        if self._stream_data is None:
            self._stream_data = [None] * len(self.stream_paths)

        # Load the stream for the specified antenna index if not already loaded
        if self._stream_data[antenna_idx] is None:
            path = self.stream_paths[antenna_idx]
            if not path.is_file():
                raise FileNotFoundError(f"Missing stream file: {path}")

            # Load the Doppler trace from the file, expecting a pickled NumPy array.
            with path.open("rb") as f:
                arr = pickle.load(f)

            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected NumPy array in {path}, got {type(arr)}")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D Doppler trace in {path}, got shape {arr.shape}")

            # Normalize the trace as SHARP did
            arr = arr - arr.mean(axis=0, keepdims=True)

            self._stream_data[antenna_idx] = arr

        return self._stream_data[antenna_idx]

    def load_all_streams(self) -> tuple[np.ndarray, ...]:
        """Return all streams, requiring matching shapes."""
        streams = tuple(self._load_stream(i) for i in range(len(self.stream_paths)))
        shapes = [stream.shape for stream in streams]
        if len(set(shapes)) != 1:
            raise ValueError(f"Stream shapes do not match: {shapes}")
        return streams

    def clear_cache(self) -> None:
        """Release loaded stream arrays."""
        self._stream_data = None

    @property
    def length(self) -> int:
        """Number of time steps in the recording."""
        return self._load_stream(0).shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (streams, time, doppler bins)."""
        return (len(self.stream_paths), *self._load_stream(0).shape)

    @property
    def activity(self) -> str:
        """Backward-compatible alias for older HAR code."""
        return self.label


@dataclass
class RawCsiTraceRecording:
    """Raw CSI recording metadata plus lazy loading for one saved .npz trace."""

    scenario: str
    label: str
    repetition: str
    ground_truth: str
    trace_path: Path
    cache: bool = False

    _csi: np.ndarray | None = field(default=None, init=False, repr=False)

    def load(self) -> np.ndarray:
        """Load and cache one raw CSI trace as [antenna, subcarrier, time]."""
        if self.cache and self._csi is not None:
            return self._csi

        if not self.trace_path.is_file():
            raise FileNotFoundError(f"Missing raw CSI trace file: {self.trace_path}")

        with np.load(self.trace_path) as data:
            csi = data["csi"]

        if csi.ndim != 3:
            raise ValueError(f"Expected raw CSI trace [antenna, subcarrier, time], got {csi.shape}")
        if not np.isfinite(csi).all():
            raise ValueError(f"Raw CSI trace contains NaN or Inf values: {self.trace_path}")

        csi = csi.astype(np.float32, copy=False)
        if self.cache:
            self._csi = csi
        return csi

    def clear_cache(self) -> None:
        """Release loaded trace array."""
        self._csi = None

    @property
    def length(self) -> int:
        """Number of packet-time samples in the recording."""
        return self.load().shape[-1]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (antennas, subcarriers, time)."""
        return self.load().shape

    @property
    def activity(self) -> str:
        """Backward-compatible alias used by older helpers."""
        return self.label
