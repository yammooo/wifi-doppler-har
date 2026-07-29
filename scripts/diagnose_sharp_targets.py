from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import pickle

import numpy as np


SCENARIO_PAIRS = (("AR-1a", "S1a"), ("AR-1b", "S1b"), ("AR-1c", "S1c"))
LABELS = frozenset("ELWRJ")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare recomputed SHARP traces with their legacy counterparts."
    )
    parser.add_argument("--recomputed-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-shift", type=int, default=5)
    return parser.parse_args()


def load_trace(path: Path) -> np.ndarray:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, np.ndarray) or value.ndim != 2:
        raise ValueError(f"Expected a 2D NumPy trace in {path}, got {type(value)}.")
    return value.astype(np.float64, copy=False)


def compare(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    difference = left - right
    left_flat = left.ravel()
    right_flat = right.ravel()
    left_std = left_flat.std()
    right_std = right_flat.std()
    correlation = (
        float(np.corrcoef(left_flat, right_flat)[0, 1])
        if left_std and right_std
        else float("nan")
    )
    return {
        "mse": float(np.mean(difference**2)),
        "mae": float(np.mean(np.abs(difference))),
        "correlation": correlation,
    }


def aligned_for_shift(
    legacy: np.ndarray,
    recomputed: np.ndarray,
    shift: int,
) -> tuple[np.ndarray, np.ndarray]:
    if shift >= 0:
        length = min(len(recomputed), len(legacy) - shift)
        return legacy[shift : shift + length], recomputed[:length]
    length = min(len(recomputed) + shift, len(legacy))
    return legacy[:length], recomputed[-shift : -shift + length]


def centered(trace: np.ndarray) -> np.ndarray:
    return trace - trace.mean(axis=0, keepdims=True)


def trace_summary(trace: np.ndarray) -> dict[str, float]:
    off_center = np.concatenate((trace[:, :45], trace[:, 56:]), axis=1)
    return {
        "mean": float(trace.mean()),
        "std": float(trace.std()),
        "floor_fraction": float(np.mean(trace <= 10**-1.2 + 1e-8)),
        "off_center_active_fraction": float(np.mean(off_center.max(axis=1) > 0.2)),
    }


def main() -> None:
    args = parse_args()
    if args.max_shift < 0:
        raise ValueError("--max-shift must be non-negative.")

    comparisons = []
    missing = []
    for recomputed_scenario, legacy_scenario in SCENARIO_PAIRS:
        recomputed_dir = args.recomputed_root / recomputed_scenario
        legacy_dir = args.legacy_root / legacy_scenario
        prefix = recomputed_scenario.replace("-", "")
        for recomputed_path in sorted(recomputed_dir.glob(f"{prefix}_*_stream_*.txt")):
            suffix = recomputed_path.name.removeprefix(f"{prefix}_")
            activity = suffix.split("_", 1)[0][:1]
            if activity not in LABELS:
                continue
            legacy_path = legacy_dir / f"{legacy_scenario}_{suffix}"
            if not legacy_path.is_file():
                missing.append(str(legacy_path))
                continue

            legacy = load_trace(legacy_path)
            recomputed = load_trace(recomputed_path)
            legacy_centered = centered(legacy)
            recomputed_centered = centered(recomputed)
            legacy_aligned, recomputed_aligned = aligned_for_shift(
                legacy, recomputed, 0
            )
            legacy_centered_aligned, recomputed_centered_aligned = aligned_for_shift(
                legacy_centered, recomputed_centered, 0
            )
            shifted = {}
            for shift in range(-args.max_shift, args.max_shift + 1):
                shifted_legacy, shifted_recomputed = aligned_for_shift(
                    legacy_centered, recomputed_centered, shift
                )
                shifted[str(shift)] = compare(shifted_legacy, shifted_recomputed)
            best_shift = min(shifted, key=lambda value: shifted[value]["mse"])

            comparisons.append(
                {
                    "recording": f"{recomputed_scenario}/{suffix.rsplit('_stream_', 1)[0]}",
                    "antenna": int(suffix.rsplit("_stream_", 1)[1].split(".", 1)[0]),
                    "legacy_shape": list(legacy.shape),
                    "recomputed_shape": list(recomputed.shape),
                    "identity": compare(legacy_aligned, recomputed_aligned),
                    "centered": compare(
                        legacy_centered_aligned, recomputed_centered_aligned
                    ),
                    "centered_bin_flip": compare(
                        legacy_centered_aligned,
                        recomputed_centered_aligned[:, ::-1],
                    ),
                    "best_centered_time_shift": int(best_shift),
                    "best_centered_time_shift_metrics": shifted[best_shift],
                    "legacy_summary": trace_summary(legacy),
                    "recomputed_summary": trace_summary(recomputed),
                }
            )

    if not comparisons:
        raise ValueError("No matching classifier-compatible traces were found.")

    scalar_fields = (
        ("identity_mse", "identity", "mse"),
        ("identity_correlation", "identity", "correlation"),
        ("centered_mse", "centered", "mse"),
        ("centered_correlation", "centered", "correlation"),
        ("centered_bin_flip_mse", "centered_bin_flip", "mse"),
    )
    aggregate = {
        name: float(np.mean([item[group][field] for item in comparisons]))
        for name, group, field in scalar_fields
    }
    aggregate["pairs"] = len(comparisons)
    aggregate["shape_deltas"] = dict(
        Counter(
            item["recomputed_shape"][0] - item["legacy_shape"][0]
            for item in comparisons
        )
    )
    aggregate["best_centered_time_shifts"] = dict(
        Counter(item["best_centered_time_shift"] for item in comparisons)
    )
    for source in ("legacy", "recomputed"):
        for field in ("mean", "std", "floor_fraction", "off_center_active_fraction"):
            aggregate[f"{source}_{field}"] = float(
                np.mean([item[f"{source}_summary"][field] for item in comparisons])
            )

    result = {
        "scenario_pairs": [list(pair) for pair in SCENARIO_PAIRS],
        "labels": sorted(LABELS),
        "aggregate": aggregate,
        "missing": missing,
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
