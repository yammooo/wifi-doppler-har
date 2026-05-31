"""Preprocess SHARP PI raw CSI traces for SimID-style raw CSI baselines.

This script intentionally sits between the SHARP and SimID pipelines:

* From SHARP/Nexmon we copy only the file-format parsing steps needed to turn
  the raw ``csi_buff`` matrix into antenna/subcarrier time series:
  FFT-shifting the OFDM bins, removing Nexmon control/non-data subcarriers,
  splitting interleaved monitor-antenna streams, and per-packet amplitude
  normalization.
* From SimID we copy the raw-CSI denoising idea: apply a second-order
  Butterworth low-pass filter independently to every CSI stream over time.

We explicitly do not run SHARP phase sanitization, H reconstruction, or Doppler
extraction here. The output remains a raw-amplitude CSI representation.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy import signal
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PI_SUBSETS = ("PI-1a", "PI-2a", "PI-3a", "PI-4a")
PI_FILE_PATTERN = re.compile(r"^PI(?P<domain>\d)a_p(?P<person>\d{2})$")

# Same OFDM-bin deletion used by the original SHARP preprocessing after
# fftshift. The 80 MHz Nexmon buffers expose 256 bins, but these 14 positions
# are non-data/control bins, leaving the 242 CFR data subchannels described in
# the SHARP/dataset papers.
SHARP_DELETED_SUBCARRIERS = np.asarray(
    [0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255],
    dtype=int,
)


def raw_name_to_subset(name: str) -> str:
    match = PI_FILE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unexpected PI trace name: {name}")
    return f"PI-{match.group('domain')}a"


def raw_name_to_label(name: str) -> str:
    match = PI_FILE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unexpected PI trace name: {name}")
    return f"p{match.group('person')}"


def iter_pi_files(input_root: Path, subsets: tuple[str, ...], include_empty: bool) -> list[Path]:
    files: list[Path] = []
    for subset in subsets:
        subset_dir = input_root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing PI subset directory: {subset_dir}")
        for path in sorted(subset_dir.glob("*.mat")):
            if not include_empty and path.stem.endswith("_p00"):
                continue
            files.append(path)
    return files


def split_sharp_monitor_streams(
    csi_buff: np.ndarray,
    *,
    nss: int,
    ncore: int,
    start_packet: int,
    end_packet: int | None,
) -> np.ndarray:
    """Convert SHARP/Nexmon ``csi_buff`` to ``[antenna, subcarrier, time]``.

    The PI raw files store monitor antenna streams interleaved on the row axis.
    This reproduces the stream splitting and OFDM-bin cleanup from SHARP's
    ``CSI_phase_sanitization_signal_preprocessing.py`` but stops before any
    phase sanitization or Doppler computation.
    """
    n_streams = nss * ncore
    if csi_buff.ndim != 2 or csi_buff.shape[1] != 256:
        raise ValueError(f"Expected csi_buff shape [packets, 256], got {csi_buff.shape}")

    csi_buff = np.fft.fftshift(csi_buff, axes=1)

    # SHARP drops all-zero packet rows before deinterleaving streams. In normal
    # PI files this is often a no-op, but keeping it preserves the original raw
    # parsing semantics.
    nonzero_rows = np.sum(csi_buff, axis=1) != 0
    csi_buff = csi_buff[nonzero_rows]

    stream_length = int(np.floor(csi_buff.shape[0] / n_streams))
    if end_packet is None or end_packet < 0:
        end_packet = stream_length
    if start_packet < 0 or end_packet <= start_packet:
        raise ValueError(f"Invalid packet crop start={start_packet}, end={end_packet}")
    if end_packet > stream_length:
        raise ValueError(
            f"Requested end_packet={end_packet}, but trace has only {stream_length} packets per stream"
        )

    streams = []
    for stream_idx in range(n_streams):
        stream = csi_buff[stream_idx : stream_length * n_streams + 1 : n_streams, :]
        stream = stream[start_packet:end_packet, :]

        # The original SHARP code flips the sign of bins >=64 before phase
        # processing. For amplitude-only raw CSI, abs(.) makes this sign change
        # irrelevant, so we do not apply it here.
        stream = np.delete(stream, SHARP_DELETED_SUBCARRIERS, axis=1)

        amplitude = np.abs(stream).astype(np.float32, copy=False)
        mean_amplitude = np.mean(amplitude, axis=1, keepdims=True)
        mean_amplitude[mean_amplitude == 0] = 1.0
        streams.append((amplitude / mean_amplitude).T)

    return np.stack(streams, axis=0).astype(np.float32, copy=False)


def butterworth_filter_csi(
    csi: np.ndarray,
    *,
    order: int,
    cutoff: float,
    show_progress: bool,
) -> np.ndarray:
    """Apply SimID-style low-pass filtering over packet time.

    SimID filters each real-valued CSI row independently. Our equivalent row is
    one antenna/subcarrier amplitude trace over time.
    """
    if csi.ndim != 3:
        raise ValueError(f"Expected csi shape [antenna, subcarrier, time], got {csi.shape}")

    b, a = signal.butter(order, cutoff, btype="low")
    padlen = 3 * max(len(a), len(b)) - 1
    if csi.shape[-1] <= padlen:
        raise ValueError(f"Trace is too short for filtfilt: time={csi.shape[-1]}, padlen={padlen}")

    filtered = np.empty_like(csi, dtype=np.float32)
    iterator = range(csi.shape[0])
    if show_progress:
        iterator = tqdm(iterator, desc="filter antennas", unit="ant", leave=False)

    for antenna_idx in iterator:
        filtered[antenna_idx] = signal.filtfilt(
            b,
            a,
            csi[antenna_idx],
            axis=-1,
            padlen=padlen,
        ).astype(np.float32, copy=False)

    return filtered


def save_trace(
    output_path: Path,
    *,
    csi: np.ndarray,
    label: str,
    scenario: str,
    source_file: Path,
    compressed: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "csi": csi,
        "label": np.asarray(label),
        "scenario": np.asarray(scenario),
        "source_file": np.asarray(str(source_file)),
    }
    if compressed:
        np.savez_compressed(output_path, **payload)
    else:
        np.savez(output_path, **payload)


def preprocess_trace(
    mat_path: Path,
    *,
    output_root: Path,
    nss: int,
    ncore: int,
    start_packet: int,
    end_packet: int | None,
    butter_order: int,
    butter_cutoff: float,
    compressed: bool,
    force: bool,
) -> Path:
    scenario = raw_name_to_subset(mat_path.stem)
    label = raw_name_to_label(mat_path.stem)
    output_path = output_root / scenario / f"{mat_path.stem}.npz"
    if output_path.exists() and not force:
        return output_path

    csi_buff = sio.loadmat(mat_path)["csi_buff"]
    csi = split_sharp_monitor_streams(
        csi_buff,
        nss=nss,
        ncore=ncore,
        start_packet=start_packet,
        end_packet=end_packet,
    )
    filtered = butterworth_filter_csi(
        csi,
        order=butter_order,
        cutoff=butter_cutoff,
        show_progress=True,
    )

    if not np.isfinite(filtered).all():
        raise ValueError(f"Filtered trace contains NaN or Inf values: {mat_path}")

    save_trace(
        output_path,
        csi=filtered,
        label=label,
        scenario=scenario,
        source_file=mat_path,
        compressed=compressed,
    )
    return output_path


def write_metadata(output_root: Path, args: argparse.Namespace, processed_paths: list[Path]) -> None:
    kept_subcarriers = [
        idx for idx in range(256) if idx not in set(SHARP_DELETED_SUBCARRIERS.tolist())
    ]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "input_root": str(args.input_root),
        "output_root": str(args.output_root),
        "subsets": args.subsets,
        "include_empty": args.include_empty,
        "nss": args.nss,
        "ncore": args.ncore,
        "num_streams": args.nss * args.ncore,
        "representation": "amplitude",
        "layout": "[antenna, subcarrier, time]",
        "source_format_steps_copied_from_sharp": [
            "fftshift over OFDM bins",
            "all-zero packet row removal",
            "interleaved monitor stream splitting",
            "Nexmon non-data/control subcarrier deletion",
            "per-packet mean-amplitude normalization",
        ],
        "simid_steps": [
            "second-order Butterworth low-pass filtering over time per CSI row",
        ],
        "intentionally_not_applied": [
            "SHARP phase sanitization",
            "SHARP H reconstruction",
            "SHARP Doppler FFT/profile extraction",
        ],
        "deleted_subcarriers_after_fftshift": SHARP_DELETED_SUBCARRIERS.tolist(),
        "kept_subcarriers_after_fftshift": kept_subcarriers,
        "num_kept_subcarriers": len(kept_subcarriers),
        "start_packet": args.start_packet,
        "end_packet": args.end_packet,
        "butterworth_order": args.butter_order,
        "butterworth_cutoff": args.butter_cutoff,
        "compressed": args.compressed,
        "processed_files": [str(path) for path in processed_paths],
    }
    with (output_root / "metadata.json").open("w", encoding="utf-8") as fp:
        json.dump(metadata, fp, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "CSI-80Mhz")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "raw_csi_traces_pi")
    parser.add_argument("--subsets", default=",".join(DEFAULT_PI_SUBSETS))
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--compressed", action="store_true")
    parser.add_argument("--nss", type=int, default=1)
    parser.add_argument("--ncore", type=int, default=4)
    parser.add_argument("--start-packet", type=int, default=0)
    parser.add_argument("--end-packet", type=int, default=-1)
    parser.add_argument("--butter-order", type=int, default=2)
    parser.add_argument("--butter-cutoff", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = tuple(item.strip() for item in args.subsets.split(",") if item.strip())
    args.output_root.mkdir(parents=True, exist_ok=True)

    mat_files = iter_pi_files(args.input_root, subsets, args.include_empty)
    if args.limit is not None:
        mat_files = mat_files[: args.limit]

    end_packet = None if args.end_packet < 0 else args.end_packet
    processed_paths: list[Path] = []
    for mat_path in tqdm(mat_files, desc="raw CSI traces", unit="trace"):
        processed_paths.append(
            preprocess_trace(
                mat_path,
                output_root=args.output_root,
                nss=args.nss,
                ncore=args.ncore,
                start_packet=args.start_packet,
                end_packet=end_packet,
                butter_order=args.butter_order,
                butter_cutoff=args.butter_cutoff,
                compressed=args.compressed,
                force=args.force,
            )
        )

    write_metadata(args.output_root, args, processed_paths)
    print(f"Processed {len(processed_paths)} traces into {args.output_root}")


if __name__ == "__main__":
    main()
