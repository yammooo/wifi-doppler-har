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
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import scipy
import scipy.io as sio
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARP_DIR = PROJECT_ROOT / "external" / "sharp"
PHASE_DIR = SHARP_DIR / "phase_processing"
DEFAULT_PI_SUBSETS = ("PI-1a", "PI-2a", "PI-3a", "PI-4a")
PI_FILE_PATTERN = re.compile(r"^PI(?P<domain>\d)a_p(?P<person>\d{2})$")
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

sys.path.insert(0, str(SHARP_DIR))
from optimization_utility import build_T_matrix, lasso_regression_osqp_fast  # noqa: E402


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


def ensure_sharp_signal(
    mat_path: Path,
    *,
    preprocess_start: int,
    nss: int,
    ncore: int,
    force: bool,
) -> None:
    """Run the original SHARP signal preprocessing script."""
    stem = mat_path.stem
    input_dir = str(mat_path.parent.resolve()) + os.sep
    signal_path = PHASE_DIR / f"signal_{stem}.txt"

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


def _atomic_pickle_dump(path: Path, value: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fp:
        pickle.dump(value, fp)
    os.replace(tmp_path, path)


def _estimate_h_stream(task: tuple[str, int, int, int, bool, int]) -> tuple[str, int, int, bool]:
    """Estimate one SHARP stream using the same OSQP logic as the original script."""
    stem, stream, h_start, h_end_arg, force, checkpoint_every = task
    signal_path = PHASE_DIR / f"signal_{stem}.txt"
    tr_path = PHASE_DIR / f"Tr_vector_{stem}_stream_{stream}.txt"
    r_path = PHASE_DIR / f"r_vector_{stem}_stream_{stream}.txt"
    checkpoint_path = PHASE_DIR / f"checkpoint_H_{stem}_stream_{stream}.pkl"

    if tr_path.exists() and r_path.exists() and not force:
        return stem, stream, 0, True

    if force and checkpoint_path.exists():
        checkpoint_path.unlink()

    with signal_path.open("rb") as fp:
        signal_complete = pickle.load(fp)

    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255], dtype=int)
    subcarriers_space = 2
    delta_t = 1e-7
    delta_t_refined = 5e-9
    range_refined_up = 2.5e-7
    range_refined_down = 2e-7

    end_r = h_end_arg if h_end_arg != -1 else signal_complete.shape[1]
    signal_considered = signal_complete[:, h_start:end_r, stream]

    f_frequency = 256
    delta_f = 312.5e3
    frequency_vector_complete = np.zeros(f_frequency)
    f_frequency_2 = f_frequency // 2
    for row in range(f_frequency_2):
        freq_n = delta_f * (row - f_frequency / 2)
        frequency_vector_complete[row] = freq_n
        freq_p = delta_f * row
        frequency_vector_complete[row + f_frequency_2] = freq_p
    frequency_vector = np.delete(frequency_vector_complete, delete_idxs)

    t_min = -3e-7
    t_max = 5e-7

    t_matrix, time_matrix = build_T_matrix(frequency_vector, delta_t, t_min, t_max)
    r_length = int((t_max - t_min) / delta_t_refined)

    select_subcarriers = np.arange(0, frequency_vector.shape[0], subcarriers_space)

    row_t = int(t_matrix.shape[0] / subcarriers_space)
    col_t = t_matrix.shape[1]
    m = 2 * row_t
    n = 2 * col_t
    in_matrix = scipy.sparse.eye(n)
    im_matrix = scipy.sparse.eye(m)
    on_matrix = scipy.sparse.csc_matrix((n, n))
    onm_matrix = scipy.sparse.csc_matrix((n, m))
    p_matrix = scipy.sparse.block_diag([on_matrix, im_matrix, on_matrix], format="csc")
    q_vector = np.zeros(2 * n + m)
    a2_matrix = scipy.sparse.hstack([in_matrix, onm_matrix, -in_matrix])
    a3_matrix = scipy.sparse.hstack([in_matrix, onm_matrix, in_matrix])
    ones_n = np.ones(n)
    zeros_n = np.zeros(n)
    zeros_nm = np.zeros(n + m)

    n_steps = end_r - h_start
    next_step = 0
    if checkpoint_path.exists() and not force:
        with checkpoint_path.open("rb") as fp:
            checkpoint = pickle.load(fp)
        r_optim = checkpoint["r_optim"]
        tr_matrix = checkpoint["tr_matrix"]
        next_step = checkpoint["next_step"]
    else:
        r_optim = np.zeros((r_length, n_steps), dtype=complex)
        tr_matrix = np.zeros((frequency_vector_complete.shape[0], n_steps), dtype=complex)

    for time_step in range(next_step, n_steps):
        signal_time = signal_considered[:, time_step]
        complex_opt_r = lasso_regression_osqp_fast(
            signal_time,
            t_matrix,
            select_subcarriers,
            row_t,
            col_t,
            im_matrix,
            onm_matrix,
            p_matrix,
            q_vector,
            a2_matrix,
            a3_matrix,
            ones_n,
            zeros_n,
            zeros_nm,
        )

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_matrix[position_max_r]

        t_matrix_refined, time_matrix_refined = build_T_matrix(
            frequency_vector,
            delta_t_refined,
            max(time_max_r - range_refined_down, t_min),
            min(time_max_r + range_refined_up, t_max),
        )

        col_t_refined = t_matrix_refined.shape[1]
        n_refined = 2 * col_t_refined
        in_refined = scipy.sparse.eye(n_refined)
        on_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
        onm_refined = scipy.sparse.csc_matrix((n_refined, m))
        p_refined = scipy.sparse.block_diag([on_refined, im_matrix, on_refined], format="csc")
        q_refined = np.zeros(2 * n_refined + m)
        a2_refined = scipy.sparse.hstack([in_refined, onm_refined, -in_refined])
        a3_refined = scipy.sparse.hstack([in_refined, onm_refined, in_refined])
        ones_refined = np.ones(n_refined)
        zeros_refined = np.zeros(n_refined)
        zeros_nm_refined = np.zeros(n_refined + m)

        complex_opt_r_refined = lasso_regression_osqp_fast(
            signal_time,
            t_matrix_refined,
            select_subcarriers,
            row_t,
            col_t_refined,
            im_matrix,
            onm_refined,
            p_refined,
            q_refined,
            a2_refined,
            a3_refined,
            ones_refined,
            zeros_refined,
            zeros_nm_refined,
        )

        position_max_r_refined = np.argmax(abs(complex_opt_r_refined))

        t_matrix_refined, time_matrix_refined = build_T_matrix(
            frequency_vector_complete,
            delta_t_refined,
            max(time_max_r - range_refined_down, t_min),
            min(time_max_r + range_refined_up, t_max),
        )

        tr_values = np.multiply(t_matrix_refined, complex_opt_r_refined)
        trr_values = np.multiply(
            tr_values,
            np.conj(tr_values[:, position_max_r_refined : position_max_r_refined + 1]),
        )
        tr_matrix[:, time_step] = np.sum(trr_values, axis=1)

        start_r_opt = int((time_matrix_refined[0] - t_min) / delta_t_refined)
        end_r_opt = start_r_opt + complex_opt_r_refined.shape[0]
        r_optim[start_r_opt:end_r_opt, time_step] = complex_opt_r_refined

        if checkpoint_every > 0 and (time_step + 1) % checkpoint_every == 0:
            _atomic_pickle_dump(
                checkpoint_path,
                {
                    "r_optim": r_optim,
                    "tr_matrix": tr_matrix,
                    "next_step": time_step + 1,
                    "h_start": h_start,
                    "h_end": h_end_arg,
                },
            )

    _atomic_pickle_dump(r_path, r_optim)
    _atomic_pickle_dump(tr_path, tr_matrix)
    if checkpoint_path.exists():
        checkpoint_path.unlink()
    return stem, stream, n_steps - next_step, False


def estimate_h_streams(
    mat_files: list[Path],
    *,
    h_start: int,
    h_end: int,
    nss: int,
    ncore: int,
    force: bool,
    jobs: int,
    checkpoint_every: int,
) -> None:
    tasks = []
    task_totals = {}
    for mat_path in mat_files:
        stem = mat_path.stem
        with (PHASE_DIR / f"signal_{stem}.txt").open("rb") as fp:
            signal_complete = pickle.load(fp)
        end_r = h_end if h_end != -1 else signal_complete.shape[1]
        n_steps = end_r - h_start
        for stream in range(nss * ncore):
            tr_path = PHASE_DIR / f"Tr_vector_{stem}_stream_{stream}.txt"
            r_path = PHASE_DIR / f"r_vector_{stem}_stream_{stream}.txt"
            if not force and tr_path.exists() and r_path.exists():
                continue
            task = (stem, stream, h_start, h_end, force, checkpoint_every)
            tasks.append(task)
            task_totals[(stem, stream)] = n_steps

    if not tasks:
        return

    if jobs <= 1:
        for task in tqdm(
            tasks,
            desc="H-estimation streams",
            unit="stream",
            dynamic_ncols=True,
            bar_format=TQDM_BAR_FORMAT,
        ):
            _estimate_h_stream(task)
        return

    with ProcessPoolExecutor(max_workers=jobs) as executor:
        future_to_key = {
            executor.submit(_estimate_h_stream, task): (task[0], task[1])
            for task in tasks
        }
        pending = set(future_to_key)
        last_packet_progress = {}
        completed_packets = 0
        for key in task_totals:
            checkpoint_path = PHASE_DIR / f"checkpoint_H_{key[0]}_stream_{key[1]}.pkl"
            if checkpoint_path.exists() and not force:
                with checkpoint_path.open("rb") as fp:
                    checkpoint = pickle.load(fp)
                last_packet_progress[key] = checkpoint["next_step"]
                completed_packets += checkpoint["next_step"]
            else:
                last_packet_progress[key] = 0
        total_packets = sum(task_totals.values())
        with tqdm(
            total=len(pending),
            desc="H-estimation streams",
            unit="stream",
            dynamic_ncols=True,
            bar_format=TQDM_BAR_FORMAT,
        ) as stream_progress:
            with tqdm(
                total=total_packets,
                initial=completed_packets,
                desc="H-estimation total",
                unit="pkt",
                dynamic_ncols=True,
                bar_format=TQDM_BAR_FORMAT,
            ) as packet_progress:
                while pending:
                    done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)

                    for future, key in list(future_to_key.items()):
                        if key not in last_packet_progress:
                            continue
                        checkpoint_path = PHASE_DIR / f"checkpoint_H_{key[0]}_stream_{key[1]}.pkl"
                        if checkpoint_path.exists():
                            with checkpoint_path.open("rb") as fp:
                                checkpoint = pickle.load(fp)
                            current = checkpoint["next_step"]
                            delta = current - last_packet_progress[key]
                            if delta > 0:
                                packet_progress.update(delta)
                                last_packet_progress[key] = current

                    for future in done:
                        stem, stream, _, _ = future.result()
                        key = (stem, stream)
                        final_delta = task_totals[key] - last_packet_progress[key]
                        if final_delta > 0:
                            packet_progress.update(final_delta)
                        del last_packet_progress[key]
                        stream_progress.update(1)


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
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=500)
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

    for mat_path in tqdm(mat_files, desc="Signal preprocessing", unit="trace"):
        ensure_sharp_signal(
            mat_path,
            preprocess_start=args.preprocess_start,
            nss=args.nss,
            ncore=args.ncore,
            force=args.force,
        )

    estimate_h_streams(
        mat_files,
        h_start=args.h_start,
        h_end=args.h_end,
        nss=args.nss,
        ncore=args.ncore,
        force=args.force,
        jobs=args.jobs,
        checkpoint_every=args.checkpoint_every,
    )

    for mat_path in tqdm(mat_files, desc="PI traces", unit="trace"):
        tqdm.write(f"Post-processing {mat_path}")
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
