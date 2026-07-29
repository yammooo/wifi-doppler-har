from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import re

import numpy as np


TRACE_PATTERN = re.compile(r"^(S\d+[a-z])_(.+)_stream_0\.txt$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer legacy S* to canonical AR-* aliases from trace lengths."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--doppler-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    raw: dict[str, dict[str, int]] = {}
    for item in manifest["recordings"]:
        if item["scenario"].startswith("AR-"):
            raw.setdefault(item["scenario"], {})[item["label"]] = item["raw_shape"][-1]

    for scenario_dir in sorted(args.doppler_root.glob("S*")):
        if not scenario_dir.is_dir():
            continue
        targets = {}
        for path in scenario_dir.glob("*_stream_0.txt"):
            match = TRACE_PATTERN.match(path.name)
            if not match:
                continue
            with path.open("rb") as stream:
                targets[match.group(2)] = pickle.load(stream).shape[0]

        candidates = []
        for raw_scenario, lengths in raw.items():
            common = sorted(set(targets) & set(lengths))
            deltas = {label: lengths[label] - targets[label] for label in common}
            aligned = sum(delta in (1631, 1632) for delta in deltas.values())
            distance = (
                float(np.mean([abs(delta - 1631.5) for delta in deltas.values()]))
                if deltas
                else float("inf")
            )
            candidates.append((aligned, len(common), -distance, raw_scenario, deltas))

        candidates.sort(reverse=True)
        print(f"\n{scenario_dir.name}")
        for aligned, common, _, raw_scenario, deltas in candidates[:3]:
            print(
                f"  {raw_scenario}: aligned={aligned}/{common}, "
                f"deltas={deltas}"
            )


if __name__ == "__main__":
    main()
