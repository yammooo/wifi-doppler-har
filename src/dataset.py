import os
import pickle
import pandas as pd
import re
from pathlib import Path
from torch.utils.data import Dataset

from recording_structures import TraceRecording, WindowIndex

TRACE_PATTERN = re.compile(
    r"^(?P<scenario>S(?P<scenario_id>\d+)(?P<campaign>[a-z]))_(?P<activity>[A-Z])(?P<repetition>\d*)_stream_(?P<antenna>\d+)\.txt$"
)

# TODO: we now expect 4 antennas, but we should make this more flexible in the future
class DopplerWindowDataset(Dataset):
    def __init__(self,
                 doppler_traces_dir='data/doppler_traces/',
                 scenarios=["S1a", "S1b", "S1c"],
                 split = "train",
                 window_size=340,
                 window_stride=30,
                 split_guard=31):
        self.doppler_traces_dir = doppler_traces_dir
        self.scenarios = scenarios
        self.split = split
        self.window_size = window_size
        self.window_stride = window_stride
        self.split_stride = split_guard

    def __len__(self):
        return len(self.window_indexes)

    def __getitem__(self, idx):
        return self.window_indexes[idx]["streams"]
    
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
    
    # This function takes a trace and creates window metadata of the specified size and stride. It returns a list of WindowIndex.
    def _create_windows(self, trace) -> list[WindowIndex]:
        
        # Load the trace data
        full_traces = self._parse_traces()
        
        # 

        return windows
