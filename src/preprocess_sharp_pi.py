"""Run SHARP preprocessing on the PI subset of the 80 MHz CSI dataset.

This script vendors the original SHARP phase preprocessing and H-estimation
scripts from external/sharp, then writes PI Doppler traces with stable PI
folder names such as PI-1a, PI-2a, PI-3a, and PI-4a.
"""

from __future__ import annotations

import argparse
import math as mt
import os
import pickle
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARP_DIR = PROJECT_ROOT / "external" / "sharp"
PHASE_DIR = SHARP_DIR / "phase_processing"
DEFAULT_PI_SUBSETS = ("PI-1a", "PI-2a", "PI-3a", "PI-4a")
PI_FILE_PATTERN = re.compile(r"^PI(?P<domain>\d)a_p(?P<person>\d{2})$")


def run_sharp_script(script: str, *args: object) -> None:
    command = [sys.executable, script, *(str(arg) for arg in args)]
    subprocess.run(command, cwd=SHARP_DIR, check=True)


def raw_name_to_subset(name: str) -> str:
    match = PI_FILE_PATTERN.match(name)
    if not match:
        raise ValueError(f"Unexpected PI trace name: {name}")
    return f"PI-{match.group('domain')}a"


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


def sharp_preprocess_and_estimate(
    mat_path: Path,
    *,
    preprocess_start: int,
    h_start: int,
    h_end: int,
    nss: int,
    ncore: int,
    force: bool,
) -> None:
    """Run the original SHARP signal preprocessing and H-estimation scripts."""
    stem = mat_path.stem
    input_dir = str(mat_path.parent.resolve()) + os.sep
    signal_path = PHASE_DIR / f"signal_{stem}.txt"
    tr_paths = [PHASE_DIR / f"Tr_vector_{stem}_stream_{stream}.txt" for stream in range(nss * ncore)]

    if force or not signal_path.exists():
        run_sharp_script(
            "CSI_phase_sanitization_signal_preprocessing.py",
            input_dir,
            0,
            stem,
            nss,
            ncore,
            preprocess_start,
        )

    if force or not all(path.exists() for path in tr_paths):
        run_sharp_script(
            "CSI_phase_sanitization_H_estimation.py",
            input_dir,
            0,
            stem,
            nss,
            ncore,
            h_start,
            h_end,
        )


def reconstruct_streams(
    mat_path: Path,
    processed_root: Path,
    *,
    reconstruct_start: int,
    reconstruct_end: int,
    nss: int,
    ncore: int,
    force: bool,
) -> list[Path]:
    """Reconstruct sanitized amplitude/phase matrices using SHARP's logic."""
    stem = mat_path.stem
    subset = raw_name_to_subset(stem)
    subset_dir = processed_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for stream in tqdm(range(nss * ncore), desc=f"{stem} reconstruct", unit="stream", leave=False):
        output_path = subset_dir / f"{stem}_stream_{stream}.mat"
        output_paths.append(output_path)
        if output_path.exists() and not force:
            continue

        tr_path = PHASE_DIR / f"Tr_vector_{stem}_stream_{stream}.txt"
        with tr_path.open("rb") as fp:
            h_est = pickle.load(fp)

        end_h = h_est.shape[1]
        h_est = h_est[:, reconstruct_start : end_h - reconstruct_end]
        f_frequency = 256
        csi_matrix_processed = np.zeros((h_est.shape[1], f_frequency, 2))

        csi_matrix_processed[:, 6:-5, 0] = np.abs(h_est[6:-5, :]).T

        phase_before = np.unwrap(np.angle(h_est[6:-5, :]), axis=0)
        ones_vector = np.ones((2, phase_before.shape[0]))
        ones_vector[1, :] = np.arange(0, phase_before.shape[0])

        for tidx in tqdm(
            range(1, phase_before.shape[1]),
                    desc=f"{output_path.stem} unwrap",
            unit="pkt",
            leave=False,
        ):
            stop = False
            idx_prec = -1
            while not stop:
                phase_err = phase_before[:, tidx] - phase_before[:, tidx - 1]
                diff_phase_err = np.diff(phase_err)
                idxs_invert_up = np.argwhere(diff_phase_err > 0.9 * mt.pi)[:, 0]
                idxs_invert_down = np.argwhere(diff_phase_err < -0.9 * mt.pi)[:, 0]
                if idxs_invert_up.shape[0] > 0:
                    idx_act = idxs_invert_up[0]
                    if idx_act == idx_prec:
                        stop = True
                    else:
                        phase_before[idx_act + 1 :, tidx] -= 2 * mt.pi
                        idx_prec = idx_act
                elif idxs_invert_down.shape[0] > 0:
                    idx_act = idxs_invert_down[0]
                    if idx_act == idx_prec:
                        stop = True
                    else:
                        phase_before[idx_act + 1 :, tidx] += 2 * mt.pi
                        idx_prec = idx_act
                else:
                    stop = True

        for tidx in tqdm(
            range(1, h_est.shape[1] - 1),
                    desc=f"{output_path.stem} detrend",
            unit="pkt",
            leave=False,
        ):
            val_prec = phase_before[:, tidx - 1 : tidx]
            val_act = phase_before[:, tidx : tidx + 1]
            error = val_act - val_prec
            correction = np.linalg.lstsq(ones_vector.T, error, rcond=None)[0]
            phase_before[:, tidx] = phase_before[:, tidx] - (np.dot(ones_vector.T, correction)).T

        csi_matrix_processed[:, 6:-5, 1] = phase_before.T
        sio.savemat(output_path, {"csi_matrix_processed": csi_matrix_processed[:, 6:-5, :]})

    return output_paths


def compute_doppler(
    processed_paths: list[Path],
    output_root: Path,
    *,
    doppler_start: int,
    doppler_end: int,
    sample_length: int,
    sliding: int,
    noise_level: float,
    force: bool,
) -> None:
    """Compute SHARP Doppler profiles from reconstructed amplitude/phase."""
    for processed_path in tqdm(processed_paths, desc="doppler streams", unit="stream", leave=False):
        subset = processed_path.parent.name
        output_dir = output_root / subset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{processed_path.stem}.txt"
        if output_path.exists() and not force:
            continue

        mdic = sio.loadmat(processed_path)
        csi_matrix_processed = mdic["csi_matrix_processed"]

        end_slice = -doppler_end if doppler_end else None
        csi_matrix_processed = csi_matrix_processed[doppler_start:end_slice, :, :]
        if csi_matrix_processed.shape[0] <= sample_length:
            raise ValueError(
                "Not enough sanitized CSI samples to compute Doppler for "
                f"{processed_path}: got {csi_matrix_processed.shape[0]} samples "
                f"after cropping start={doppler_start}, end={doppler_end}, "
                f"but sample_length={sample_length}. If this trace was created by "
                "a smoke test or partial run, rerun with --force or delete the "
                "corresponding files under external/sharp/phase_processing and "
                "the processed/output roots."
            )
        csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(
            csi_matrix_processed[:, :, 0],
            axis=1,
            keepdims=True,
        )
        csi_matrix_complete = csi_matrix_processed[:, :, 0] * np.exp(1j * csi_matrix_processed[:, :, 1])

        csi_d_profile_list = []
        for idx in tqdm(
            range(0, csi_matrix_complete.shape[0] - sample_length, sliding),
            desc=f"{processed_path.stem} doppler",
            unit="win",
            leave=False,
        ):
            csi_matrix_cut = np.nan_to_num(csi_matrix_complete[idx : idx + sample_length, :])
            csi_matrix_wind = csi_matrix_cut * np.expand_dims(hann(sample_length), axis=-1)
            csi_doppler_prof = fft(csi_matrix_wind, n=100, axis=0)
            csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)
            csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
            csi_d_profile_list.append(np.sum(csi_d_map, axis=1))

        csi_d_profile_array = np.asarray(csi_d_profile_list)
        csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
        csi_d_profile_array = csi_d_profile_array / csi_d_profile_array_max
        csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_level)] = mt.pow(10, noise_level)

        with output_path.open("wb") as fp:
            pickle.dump(csi_d_profile_array, fp)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "CSI-80Mhz")
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "sharp_processed_pi")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "doppler_traces_pi")
    parser.add_argument("--subsets", default=",".join(DEFAULT_PI_SUBSETS))
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--nss", type=int, default=1)
    parser.add_argument("--ncore", type=int, default=4)
    parser.add_argument("--preprocess-start", type=int, default=0)
    parser.add_argument("--h-start", type=int, default=0)
    parser.add_argument("--h-end", type=int, default=-1)
    parser.add_argument("--reconstruct-start", type=int, default=0)
    parser.add_argument("--reconstruct-end", type=int, default=0)
    parser.add_argument("--doppler-start", type=int, default=800)
    parser.add_argument("--doppler-end", type=int, default=800)
    parser.add_argument("--sample-length", type=int, default=31)
    parser.add_argument("--sliding", type=int, default=1)
    parser.add_argument("--noise-level", type=float, default=-1.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subsets = tuple(item.strip() for item in args.subsets.split(",") if item.strip())
    PHASE_DIR.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    mat_files = iter_pi_files(args.input_root, subsets, args.include_empty)
    if args.limit is not None:
        mat_files = mat_files[: args.limit]

    for mat_path in tqdm(mat_files, desc="PI traces", unit="trace"):
        tqdm.write(f"Processing {mat_path}")
        sharp_preprocess_and_estimate(
            mat_path,
            preprocess_start=args.preprocess_start,
            h_start=args.h_start,
            h_end=args.h_end,
            nss=args.nss,
            ncore=args.ncore,
            force=args.force,
        )
        processed_paths = reconstruct_streams(
            mat_path,
            args.processed_root,
            reconstruct_start=args.reconstruct_start,
            reconstruct_end=args.reconstruct_end,
            nss=args.nss,
            ncore=args.ncore,
            force=args.force,
        )
        compute_doppler(
            processed_paths,
            args.output_root,
            doppler_start=args.doppler_start,
            doppler_end=args.doppler_end,
            sample_length=args.sample_length,
            sliding=args.sliding,
            noise_level=args.noise_level,
            force=args.force,
        )


if __name__ == "__main__":
    main()
