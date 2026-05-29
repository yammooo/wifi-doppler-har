from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch.utils.data import Dataset


@dataclass(frozen=True)
class WindowIndex:
    """Index of a fixed-size window within one trace recording."""

    recording_idx: int
    start: int
    end: int


class WindowedTraceDataset(Dataset):
    """Shared label mapping and time-window indexing for trace datasets."""

    traces: list
    window_indexes: list[WindowIndex]

    def __init__(
        self,
        *,
        scenarios: Sequence[str],
        split: tuple[float, float],
        window_size: int,
        window_stride: int,
        split_guard: int,
        labels: Sequence[str],
    ):
        self.scenarios = list(scenarios)
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.split_guard = split_guard
        self.labels = tuple(labels)
        self.activities = self.labels
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.traces = self._parse_traces()
        self.window_indexes = self._create_windows()

    def __len__(self):
        return len(self.window_indexes)

    def clear_cache(self) -> None:
        """Release cached arrays held by trace recordings."""
        for trace in self.traces:
            trace.clear_cache()

    def _label_to_index(self, label: str) -> int:
        try:
            return self.label_to_idx[label]
        except KeyError as exc:
            raise ValueError(f"Unknown label {label!r}. Expected one of {self.labels}") from exc

    def _create_windows(self) -> list[WindowIndex]:
        windows: list[WindowIndex] = []

        for recording_idx, trace in enumerate(self.traces):
            trace_length = trace.length

            range_start = int(trace_length * self.split[0])
            range_end = int(trace_length * self.split[1])

            # Keep a gap between adjacent time splits to reduce leakage from
            # overlapping or near-overlapping windows.
            if self.split[0] > 0:
                range_start += self.split_guard

            last_start = range_end - self.window_size
            for start in range(range_start, last_start + 1, self.window_stride):
                windows.append(WindowIndex(recording_idx=recording_idx, start=start, end=start + self.window_size))

        return windows

    def _parse_traces(self) -> list:
        raise NotImplementedError
