from __future__ import annotations

from collections.abc import Sequence
import os
import re
from pathlib import Path

import numpy as np
import torch

from wifi_doppler.data.recordings import RawCsiTraceRecording
from wifi_doppler.data.windowing import WindowedTraceDataset


PI_RAW_CSI_PATTERN = re.compile(
    r"^(?P<scenario>PI(?P<scenario_id>\d+)(?P<campaign>[a-z]))_"
    r"(?P<label>p\d+)\.npz$"
)


def parse_raw_csi_filename(filename: str) -> dict[str, str] | None:
    """Parse raw PI CSI trace filenames produced by preprocess_raw_csi_pi.py."""
    match = PI_RAW_CSI_PATTERN.match(filename)
    if not match:
        return None
    info = match.groupdict()
    return {
        "scenario": info["scenario"],
        "label": info["label"],
        "repetition": "",
    }


class RawCsiWindowDataset(WindowedTraceDataset):
    """Dataset of fixed-length raw CSI windows.

    Saved traces are expected as ``.npz`` files containing ``csi`` with shape
    ``[antenna, subcarrier, time]``. By default each sample is returned as
    ``[antenna * subcarrier, time]`` to match Conv1d raw CSI encoders.
    """

    def __init__(
        self,
        raw_csi_traces_dir: str | Path = "data/raw_csi_traces_pi",
        scenarios: Sequence[str] = ("PI-1a", "PI-2a", "PI-3a", "PI-4a"),
        split: tuple[float, float] = (0, 0.6),
        window_size: int = 340,
        window_stride: int = 30,
        split_guard: int = 31,
        labels: Sequence[str] | None = None,
        flatten_channels: bool = True,
    ):
        self.raw_csi_traces_dir = Path(raw_csi_traces_dir)
        self.flatten_channels = flatten_channels
        super().__init__(
            scenarios=scenarios,
            split=split,
            window_size=window_size,
            window_stride=window_stride,
            split_guard=split_guard,
            labels=labels if labels is not None else self._discover_labels(scenarios),
        )

    def __getitem__(self, idx):
        """Return one raw CSI window and its label index."""
        window = self.window_indexes[idx]
        recording = self.traces[window.recording_idx]

        csi = recording.load()
        x = csi[:, :, window.start:window.end]
        if self.flatten_channels:
            x = x.reshape(x.shape[0] * x.shape[1], x.shape[2])

        x = torch.from_numpy(np.ascontiguousarray(x)).float()
        y = torch.tensor(self._label_to_index(recording.ground_truth), dtype=torch.long)
        return x, y

    def _discover_labels(self, scenarios: Sequence[str]) -> tuple[str, ...]:
        labels = set()
        if not self.raw_csi_traces_dir.is_dir():
            raise FileNotFoundError(f"Missing raw CSI traces directory: {self.raw_csi_traces_dir}")

        for scenario in scenarios:
            scenario_dir = self.raw_csi_traces_dir / scenario
            if not scenario_dir.is_dir():
                continue
            for entry in os.scandir(scenario_dir):
                if entry.is_file():
                    info = parse_raw_csi_filename(entry.name)
                    if info:
                        labels.add(info["label"])

        if not labels:
            raise ValueError(
                f"No raw CSI labels found under {self.raw_csi_traces_dir} for scenarios {list(scenarios)}"
            )
        return tuple(sorted(labels))

    def _parse_traces(self) -> list[RawCsiTraceRecording]:
        traces: list[RawCsiTraceRecording] = []

        dirs = [entry.name for entry in os.scandir(self.raw_csi_traces_dir) if entry.is_dir()]
        missing = set(self.scenarios) - set(dirs)
        if missing:
            raise ValueError(f"Some specified scenarios are not present in the directory: {missing}")

        for scenario_dir in self.scenarios:
            for entry in os.scandir(self.raw_csi_traces_dir / scenario_dir):
                if not entry.is_file():
                    continue

                trace_info = parse_raw_csi_filename(entry.name)
                if not trace_info:
                    continue

                label = trace_info["label"]
                if label not in self.label_to_idx:
                    continue

                traces.append(
                    RawCsiTraceRecording(
                        scenario=scenario_dir,
                        label=label,
                        repetition=trace_info["repetition"],
                        ground_truth=label,
                        trace_path=Path(entry.path),
                    )
                )

        return traces
