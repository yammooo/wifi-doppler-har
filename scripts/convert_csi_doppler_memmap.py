from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


DEFAULT_SCENARIOS = ("PI-1a", "PI-2a", "PI-3a", "PI-4a")


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired MAT/pickle CSI-Doppler recordings to memory-mapped NumPy arrays."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--raw-root", type=Path, default=Path("data/CSI-80Mhz"))
    parser.add_argument("--doppler-root", type=Path, default=Path("data/doppler_traces_pi"))
    parser.add_argument("--output-root", type=Path, default=Path("data/csi_doppler_memmap"))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=list(DEFAULT_SCENARIOS),
        help="Scenario names or family selectors AR, PC, PI, or all.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--append",
        action="store_true",
        help="Preserve recordings in an existing output manifest and add these scenarios.",
    )
    return parser.parse_args()


def resolve_path(project_root: Path, path: Path) -> Path:
    return (project_root / path).resolve() if not path.is_absolute() else path.resolve()


def resolve_scenarios(doppler_root: Path, selectors: list[str]) -> list[str]:
    available = sorted(path.name for path in doppler_root.iterdir() if path.is_dir())
    scenarios: list[str] = []
    for selector in selectors:
        if selector.lower() == "all":
            matches = available
        elif selector.upper() in {"AR", "PC", "PI"}:
            matches = [name for name in available if name.startswith(f"{selector.upper()}-")]
        else:
            matches = [selector]
        for scenario in matches:
            if scenario not in scenarios:
                scenarios.append(scenario)
    if not scenarios:
        raise ValueError(f"No scenarios matched {selectors} under {doppler_root}")
    return scenarios


def save_npy_atomic(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as output:
        np.save(output, array, allow_pickle=False)
    temporary.replace(path)


def save_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def convert_recording(recording, output_root: Path, *, overwrite: bool) -> dict[str, Any]:
    from wifi_doppler.data.prepared_csi_doppler_dataset import PREPARED_FORMAT_VERSION

    recording_dir = output_root / recording.scenario / recording.filename_stem
    raw_path = recording_dir / "raw.npy"
    doppler_path = recording_dir / "doppler.npy"
    metadata_path = recording_dir / "metadata.json"
    if not overwrite and raw_path.is_file() and doppler_path.is_file() and metadata_path.is_file():
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    started_at = time.perf_counter()
    raw = recording.load_raw().astype(np.complex64, copy=False)
    doppler = recording.load_doppler().astype(np.float32, copy=False)
    if raw.shape[0] != doppler.shape[0]:
        raise ValueError(f"Antenna mismatch for {recording.filename_stem}: {raw.shape} vs {doppler.shape}")

    save_npy_atomic(raw_path, raw)
    save_npy_atomic(doppler_path, doppler)
    metadata = {
        "format_version": PREPARED_FORMAT_VERSION,
        "filename_stem": recording.filename_stem,
        "scenario": recording.scenario,
        "label": recording.label,
        "repetition": recording.repetition,
        "raw_path": str(raw_path.relative_to(output_root)),
        "doppler_path": str(doppler_path.relative_to(output_root)),
        "raw_shape": list(raw.shape),
        "doppler_shape": list(doppler.shape),
        "raw_dtype": str(raw.dtype),
        "doppler_dtype": str(doppler.dtype),
        "source_raw_path": str(recording.raw_path.resolve()),
        "source_doppler_paths": [str(path.resolve()) for path in recording.doppler_stream_paths],
        "converted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    recording_dir.mkdir(parents=True, exist_ok=True)
    save_json_atomic(metadata_path, metadata)
    elapsed = time.perf_counter() - started_at
    size_gib = (raw.nbytes + doppler.nbytes) / 1024**3
    print(f"converted {recording.filename_stem}: {size_gib:.2f} GiB in {elapsed:.1f}s", flush=True)
    recording.clear_cache()
    return metadata


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset
    from wifi_doppler.data.prepared_csi_doppler_dataset import PREPARED_FORMAT_VERSION

    raw_root = resolve_path(project_root, args.raw_root)
    doppler_root = resolve_path(project_root, args.doppler_root)
    output_root = resolve_path(project_root, args.output_root)
    scenarios = resolve_scenarios(doppler_root, args.scenarios)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = output_root / "manifest.json"
    previous_manifest = None
    recordings_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    sources: list[dict[str, Any]] = []
    if args.append:
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Cannot append without an existing manifest: {manifest_path}")
        previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous_manifest.get("format_version") != PREPARED_FORMAT_VERSION:
            raise ValueError(
                f"Cannot append to prepared format {previous_manifest.get('format_version')!r}; "
                f"expected {PREPARED_FORMAT_VERSION}."
            )
        for item in previous_manifest.get("recordings", []):
            key = (item["scenario"], item["label"], item["repetition"])
            recordings_by_key[key] = item
        sources = list(previous_manifest.get("sources", []))
        if not sources:
            sources.append(
                {
                    "raw_root": previous_manifest.get("raw_root"),
                    "doppler_root": previous_manifest.get("doppler_root"),
                    "scenarios": sorted({item["scenario"] for item in recordings_by_key.values()}),
                }
            )

    for scenario in scenarios:
        dataset = CsiToSharpDopplerDataset(
            raw_root=raw_root,
            doppler_root=doppler_root,
            scenarios=(scenario,),
            split=(0.0, 1.0),
            doppler_window_size=1,
            window_stride=1_000_000_000,
            split_guard=0,
            target_transform="none",
            cache_raw=False,
            cache_doppler=True,
        )
        try:
            for recording in dataset.traces:
                item = convert_recording(recording, output_root, overwrite=args.overwrite)
                key = (item["scenario"], item["label"], item["repetition"])
                recordings_by_key[key] = item
        finally:
            dataset.clear_cache()

    source = {
        "raw_root": str(raw_root),
        "doppler_root": str(doppler_root),
        "scenarios": sorted(scenarios),
    }
    if source not in sources:
        sources.append(source)
    recordings = sorted(
        recordings_by_key.values(),
        key=lambda item: (item["scenario"], item["label"], item["repetition"]),
    )
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "format_version": PREPARED_FORMAT_VERSION,
        "created_at": previous_manifest.get("created_at", now) if previous_manifest else now,
        "updated_at": now,
        "raw_root": (
            previous_manifest.get("raw_root", str(raw_root))
            if previous_manifest
            else str(raw_root)
        ),
        "doppler_root": (
            previous_manifest.get("doppler_root", str(doppler_root))
            if previous_manifest
            else str(doppler_root)
        ),
        "sources": sources,
        "recordings": recordings,
    }
    save_json_atomic(manifest_path, manifest)
    total_gib = sum(
        math.prod(item["raw_shape"]) * np.dtype(item["raw_dtype"]).itemsize
        + math.prod(item["doppler_shape"]) * np.dtype(item["doppler_dtype"]).itemsize
        for item in recordings
    ) / 1024**3
    print(f"prepared {len(recordings)} recordings at {output_root} ({total_gib:.2f} GiB)")


if __name__ == "__main__":
    main()
