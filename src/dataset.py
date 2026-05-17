import os
import re
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import Dataset

from recording_structures import TraceRecording, WindowIndex

TRACE_PATTERN = re.compile(
    r"^(?P<scenario>S(?P<scenario_id>\d+)(?P<campaign>[a-z]))_(?P<activity>[A-Z])(?P<repetition>\d*)_stream_(?P<antenna>\d+)\.txt$"
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
    def __init__(self,
                 doppler_traces_dir='data/doppler_traces/',
                 scenarios=["S1a", "S1b", "S1c"],
                 split = (0, 0.6),
                 window_size=340,
                 window_stride=30,
                 split_guard=31,
                 activities=DEFAULT_ACTIVITIES):
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
        return len(self.window_indexes)

    def __getitem__(self, idx):
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
        try:
            return self.label_to_idx[label]
        except KeyError as exc:
            raise ValueError(f"Unknown activity label {label!r}. Expected one of {self.activities}") from exc
    
    # This function parses the traces (trace is [n. antennas x antenna trace]) from the specified directory and returns a list of trace data along with their ground truth labels.
    # input: 
    # Takes the traces dir,
    # output: returns a list of TraceRecording objects
    def _parse_traces(self) -> list[TraceRecording]:
        traces = []

        # Check if all specified scenarios are present in the directory
        dirs = [entry.name for entry in os.scandir(self.doppler_traces_dir) if entry.is_dir()]

        if list(set(self.scenarios) - set(dirs)):
            raise ValueError(f"Some specified scenarios are not present in the directory: {set(self.scenarios) - set(dirs)}")
        
        temp_stream_data = {}

        # Iterate over the specified scenarios and parse the traces
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
    
    # This function creates window metadata of the specified size and stride. It returns a list of WindowIndex.
    def _create_windows(self) -> list[WindowIndex]:
        windows = []

        for recording_idx, trace in enumerate(self.traces):
            trace_length = trace.length

            range_start = int(trace_length * self.split[0])
            range_end = int(trace_length * self.split[1])

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
    dataset = DopplerWindowDataset()
    print(f"Total windows: {len(dataset)}")
    
    x, y = dataset[0]
    print(f"Example window shape: {x.shape}, label index: {y.item()}, label: {dataset.idx_to_label[y.item()]}")
    