"""Recording containers and lazy stream loading helpers."""

from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np


@dataclass
class TraceRecording:
    """Doppler recording metadata plus lazily loaded antenna streams."""

    scenario: str
    activity: str
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

                # Normalize the trace as SHARP did
                arr = arr - arr.mean(axis=0, keepdims=True)

            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected NumPy array in {path}, got {type(arr)}")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D Doppler trace in {path}, got shape {arr.shape}")

            self._stream_data[antenna_idx] = arr

        return self._stream_data[antenna_idx]

    def load_all_streams(self) -> tuple[np.ndarray, ...]:
        """Return all streams, requiring matching shapes."""
        streams = tuple(self._load_stream(i) for i in range(len(self.stream_paths)))
        shapes = [stream.shape for stream in streams]
        if len(set(shapes)) != 1:
            raise ValueError(f"Stream shapes do not match: {shapes}")
        return streams

    @property
    def length(self) -> int:
        """Number of time steps in the recording."""
        return self._load_stream(0).shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        """Shape as (streams, time, doppler bins)."""
        return (len(self.stream_paths), *self._load_stream(0).shape)


@dataclass(frozen=True)
class WindowIndex:
    """Index of a window within a recording, used for dataset indexing and slicing."""

    recording_idx: int
    start: int
    end: int
