from collections.abc import Sequence
import os
import re
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

from recordings import TraceRecording, WindowIndex

TRACE_PATTERN = re.compile(
    r"^(?P<scenario>S(?P<scenario_id>\d+)(?P<campaign>[a-z]))_"
    r"(?P<activity>[A-Z])_?(?P<repetition>\d*)_"
    r"stream_(?P<antenna>\d+)\.txt$"
)

ACTIVITY_NAMES = {
    "W": "walking",
    "R": "running",
    "J": "jumping",
    "L": "sitting_still",
    "S": "standing_still",
    "C": "sit_down_stand_up",
    "G": "arm_exercises",
    "H": "arm_exercises",
    "E": "empty_room",
}

DEFAULT_ACTIVITIES = ("E", "L", "W", "R", "J")

# TODO: we now expect 4 antennas, but we should make this more flexible in the future
class DopplerWindowDataset(Dataset):
    """Dataset of fixed-length Doppler windows labeled by activity."""

    def __init__(self,
                 doppler_traces_dir='data/doppler_traces/',
                 scenarios: list[str] = ["S1a", "S1b", "S1c"],
                 split: tuple[float, float] = (0, 0.6),
                 window_size: int = 340,
                 window_stride: int = 30,
                 split_guard: int = 31,
                 activities: Sequence[str] = DEFAULT_ACTIVITIES):
        self.doppler_traces_dir = doppler_traces_dir
        self.scenarios = scenarios
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.split_guard = split_guard
        self.activities = tuple(activities)
        self.label_to_idx = {label: idx for idx, label in enumerate(self.activities)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        self.traces = self._parse_traces()
        self.window_indexes = self._create_windows()

    def __len__(self):
        """Number of available windows."""
        return len(self.window_indexes)

    def __getitem__(self, idx):
        """Return one window as [antenna, time, doppler_bin] and its label index."""
        window = self.window_indexes[idx]
        recording = self.traces[window.recording_idx]

        streams = recording.load_all_streams()
        x = np.stack(
            [stream[window.start:window.end] for stream in streams],
            axis=0,
        )  # [4, 340, 100]

        x = torch.from_numpy(x).float()
        y = torch.tensor(self._label_to_index(recording.ground_truth), dtype=torch.long)
        return x, y

    def _label_to_index(self, label: str) -> int:
        """Map an activity code to the class index used by PyTorch."""
        try:
            return self.label_to_idx[label]
        except KeyError as exc:
            raise ValueError(f"Unknown activity label {label!r}. Expected one of {self.activities}") from exc
    
    def _parse_traces(self) -> list[TraceRecording]:
        """Group per-antenna stream files into complete recordings."""
        traces = []

        # Fail early when a requested scenario directory is missing.
        dirs = [entry.name for entry in os.scandir(self.doppler_traces_dir) if entry.is_dir()]

        if list(set(self.scenarios) - set(dirs)):
            raise ValueError(f"Some specified scenarios are not present in the directory: {set(self.scenarios) - set(dirs)}")
        
        temp_stream_data = {}

        # Collect matching antenna files metadata by grouping them by scenario/activity/repetition.
        for scenario_dir in self.scenarios:
            for entry in os.scandir(os.path.join(self.doppler_traces_dir, scenario_dir)):
                if entry.is_file():
                    match = TRACE_PATTERN.match(entry.name)
                    if match:
                        trace_info = match.groupdict()
                        scenario = trace_info["scenario"]
                        activity = trace_info["activity"]
                        repetition = trace_info["repetition"]
                        antenna = int(trace_info["antenna"])
                        stream_path = Path(self.doppler_traces_dir) / scenario_dir / entry.name

                        if activity not in self.label_to_idx:
                            continue

                        key = (scenario, activity, repetition)
                        if key not in temp_stream_data:
                            temp_stream_data[key] = [None] * 4
                        temp_stream_data[key][antenna] = stream_path
        
        # Create list of TraceRecording
        for (scenario, activity, repetition), stream_paths in temp_stream_data.items():
            if None in stream_paths:
                raise ValueError(f"Missing stream files for scenario {scenario}, activity {activity}, repetition {repetition}")
            trace_recording = TraceRecording(
                scenario=scenario,
                activity=activity,
                repetition=repetition,
                ground_truth=activity,
                stream_paths=tuple(stream_paths)
            )
            traces.append(trace_recording)

        return traces
    
    def _create_windows(self) -> list[WindowIndex]:
        """Create a list of all valid windows across all recordings"""
        windows = []

        for recording_idx, trace in enumerate(self.traces):
            trace_length = trace.length

            range_start = int(trace_length * self.split[0])
            range_end = int(trace_length * self.split[1])

            # Guard to ensure no data leakage between splits
            if self.split[0] > 0:
                range_start += self.split_guard

            last_start = range_end - self.window_size

            for start in range(range_start, last_start + 1, self.window_stride):
                end = start + self.window_size
                windows.append(
                    WindowIndex(
                        recording_idx=recording_idx,
                        start=start,
                        end=end,
                    )
                )

        return windows


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from collections import Counter

    dataset = DopplerWindowDataset()
    print(f"Scenarios: {dataset.scenarios}")
    print(f"Activities: {dataset.activities}")
    print(f"Trace recordings: {len(dataset.traces)}")
    print(f"Total windows: {len(dataset)}")

    trace_label_counts = Counter(trace.ground_truth for trace in dataset.traces)
    window_label_counts = Counter(dataset.traces[window.recording_idx].ground_truth for window in dataset.window_indexes)
    print(f"Trace label counts: {dict(sorted(trace_label_counts.items()))}")
    print(f"Window label counts: {dict(sorted(window_label_counts.items()))}")
    
    x, y = dataset[0]
    print(f"Example window shape: {x.shape}, label index: {y.item()}, label: {dataset.idx_to_label[y.item()]}")

    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    print(f"Batch shape: {batch_x.shape}, labels shape: {batch_y.shape}")

    plot_dir = Path("outputs") / "dataset_smoke"
    plot_dir.mkdir(parents=True, exist_ok=True)

    for sample_idx in [0, len(dataset) // 2, len(dataset) - 1]:
        x, y = dataset[sample_idx]
        window = dataset.window_indexes[sample_idx]
        trace = dataset.traces[window.recording_idx]

        fig, axes = plt.subplots(1, 4, figsize=(14, 3), sharex=True, sharey=True)
        for antenna_idx, ax in enumerate(axes):
            ax.imshow(x[antenna_idx].T, aspect="auto", origin="lower", cmap="viridis")
            ax.set_title(f"stream {antenna_idx}")
            ax.set_xlabel("time")
        axes[0].set_ylabel("Doppler bin")
        fig.suptitle(
            f"{trace.scenario}_{trace.activity}{trace.repetition} "
            f"label={dataset.idx_to_label[y.item()]} window={window.start}:{window.end}"
        )
        fig.tight_layout()
        output_path = plot_dir / f"sample_{sample_idx}_{trace.scenario}_{trace.activity}{trace.repetition}.png"
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        print(f"Saved {output_path}")
