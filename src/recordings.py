from dataclasses import dataclass, field
from pathlib import Path
import pickle
import numpy as np

@dataclass
class TraceRecording:
    scenario: str
    activity: str
    repetition: str
    ground_truth: str
    stream_paths: tuple[Path, ...]

    _stream_data: list[np.ndarray | None] | None = field(default=None, init=False, repr=False)

    def _load_stream(self, antenna_idx: int) -> np.ndarray:
        if antenna_idx < 0 or antenna_idx >= len(self.stream_paths):
            raise IndexError(f"Invalid antenna index {antenna_idx}")

        if self._stream_data is None:
            self._stream_data = [None] * len(self.stream_paths)

        if self._stream_data[antenna_idx] is None:
            path = self.stream_paths[antenna_idx]
            if not path.is_file():
                raise FileNotFoundError(f"Missing stream file: {path}")

            with path.open("rb") as f:
                arr = pickle.load(f)

            if not isinstance(arr, np.ndarray):
                raise TypeError(f"Expected NumPy array in {path}, got {type(arr)}")
            if arr.ndim != 2:
                raise ValueError(f"Expected 2D Doppler trace in {path}, got shape {arr.shape}")

            self._stream_data[antenna_idx] = arr

        return self._stream_data[antenna_idx]

    def load_all_streams(self) -> tuple[np.ndarray, ...]:
        streams = tuple(self._load_stream(i) for i in range(len(self.stream_paths)))
        shapes = [stream.shape for stream in streams]
        if len(set(shapes)) != 1:
            raise ValueError(f"Stream shapes do not match: {shapes}")
        return streams

    @property
    def length(self) -> int:
        return self._load_stream(0).shape[0]

    @property
    def shape(self) -> tuple[int, int, int]:
        return (len(self.stream_paths), *self._load_stream(0).shape)


@dataclass(frozen=True)
class WindowIndex:
    recording_idx: int
    start: int
    end: int