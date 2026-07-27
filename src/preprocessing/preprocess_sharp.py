"""Generate aligned SHARP Doppler traces from raw 80 MHz CSI recordings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math as mt
import os
import pickle
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import osqp
import scipy
import scipy.io as sio
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHARP_DIR = PROJECT_ROOT / "external" / "sharp"
PHASE_DIR = SHARP_DIR / "phase_processing"
DEFAULT_PI_SUBSETS = ("PI-1a", "PI-2a", "PI-3a", "PI-4a")
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"

sys.path.insert(0, str(SHARP_DIR))


def resolve_subsets(input_root: Path, selectors: tuple[str, ...]) -> tuple[str, ...]:
    available = sorted(path.name for path in input_root.iterdir() if path.is_dir())
    selected: list[str] = []
    for selector in selectors:
        if selector.lower() == "all":
            matches = available
        elif selector.upper() in {"AR", "PC", "PI"}:
            matches = [name for name in available if name.startswith(f"{selector.upper()}-")]
        else:
            matches = [selector]
        for name in matches:
            if name not in selected:
                selected.append(name)
    if not selected:
        raise ValueError(f"No subsets matched selectors {selectors} under {input_root}")
    return tuple(selected)


def iter_mat_files(input_root: Path, subsets: tuple[str, ...], include_empty: bool) -> list[Path]:
    files: list[Path] = []
    stems: set[str] = set()
    for subset in subsets:
        subset_dir = input_root / subset
        if not subset_dir.is_dir():
            raise FileNotFoundError(f"Missing CSI subset directory: {subset_dir}")
        for path in sorted(subset_dir.glob("*.mat")):
            if not include_empty and path.stem.endswith("_p00"):
                continue
            if path.stem in stems:
                raise ValueError(
                    f"Duplicate raw stem {path.stem!r}; SHARP intermediate filenames must be unique"
                )
            stems.add(path.stem)
            files.append(path)
    return files


def _atomic_pickle_dump(path: Path, value: object) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("wb") as fp:
        pickle.dump(value, fp)
    os.replace(tmp_path, path)


def ensure_sharp_signal(
    mat_path: Path,
    *,
    preprocess_start: int,
    nss: int,
    ncore: int,
    force: bool,
    phase_dir: Path = PHASE_DIR,
) -> None:
    """Normalize and clean one raw CSI recording using SHARP's logic."""
    stem = mat_path.stem
    signal_path = phase_dir / f"signal_{stem}.txt"
    if signal_path.exists() and not force:
        return

    csi_buff = sio.loadmat(mat_path)["csi_buff"]
    csi_buff = np.fft.fftshift(csi_buff, axes=1)
    csi_buff = np.delete(csi_buff, np.flatnonzero(np.sum(csi_buff, axis=1) == 0), axis=0)
    delete_idxs = np.asarray([0, 1, 2, 3, 4, 5, 127, 128, 129, 251, 252, 253, 254, 255])
    stream_count = nss * ncore
    end = int(np.floor(csi_buff.shape[0] / stream_count))
    signal_complete = np.zeros(
        (csi_buff.shape[1] - delete_idxs.shape[0], end - preprocess_start, stream_count),
        dtype=complex,
    )
    for stream in range(stream_count):
        signal_stream = csi_buff[stream : end * stream_count + 1 : stream_count][
            preprocess_start:end
        ]
        signal_stream[:, 64:] = -signal_stream[:, 64:]
        signal_stream = np.delete(signal_stream, delete_idxs, axis=1)
        signal_complete[:, :, stream] = (
            signal_stream / np.mean(np.abs(signal_stream), axis=1, keepdims=True)
        ).T
    _atomic_pickle_dump(signal_path, signal_complete)


def _ensure_sharp_signal_task(
    task: tuple[Path, int, int, int, bool, Path],
) -> None:
    mat_path, preprocess_start, nss, ncore, force, phase_dir = task
    ensure_sharp_signal(
        mat_path,
        preprocess_start=preprocess_start,
        nss=nss,
        ncore=ncore,
        force=force,
        phase_dir=phase_dir,
    )


def build_t_matrix(
    frequency_vector: np.ndarray,
    delta_t: float,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build SHARP's Fourier dictionary without Python scalar loops."""
    time_vector = t_min + delta_t * np.arange(int((t_max - t_min) / delta_t))
    return (
        np.exp(-2j * np.pi * frequency_vector[:, None] * time_vector[None, :]),
        time_vector,
    )


class _PreparedLasso:
    """An OSQP LASSO problem whose fixed matrix factorization is reused."""

    def __init__(self, t_matrix: np.ndarray, selected_subcarriers: np.ndarray) -> None:
        t_selected = t_matrix[selected_subcarriers]
        row_t, col_t = t_selected.shape
        m = 2 * row_t
        n = 2 * col_t

        t_real = np.zeros((m, n))
        t_real[:row_t, :col_t] = t_selected.real
        t_real[row_t:, col_t:] = t_selected.real
        t_real[row_t:, :col_t] = t_selected.imag
        t_real[:row_t, col_t:] = -t_selected.imag

        identity_m = scipy.sparse.eye(m)
        identity_n = scipy.sparse.eye(n)
        zeros_n = scipy.sparse.csc_matrix((n, n))
        zeros_nm = scipy.sparse.csc_matrix((n, m))
        p_matrix = scipy.sparse.block_diag(
            [zeros_n, identity_m, zeros_n],
            format="csc",
        )
        a_matrix = scipy.sparse.vstack(
            [
                scipy.sparse.hstack([t_real, -identity_m, zeros_nm.T]),
                scipy.sparse.hstack([identity_n, zeros_nm, -identity_n]),
                scipy.sparse.hstack([identity_n, zeros_nm, identity_n]),
            ],
            format="csc",
        )

        self._row_t = row_t
        self._n = n
        self._selected_subcarriers = selected_subcarriers
        self._lower = np.hstack([np.zeros(m), -np.inf * np.ones(n), np.zeros(n)])
        self._upper = np.hstack([np.zeros(m), np.zeros(n), np.inf * np.ones(n)])
        q_vector = np.hstack([np.zeros(n + m), 1e-1 * np.ones(n)])
        self._zero_x = np.zeros(q_vector.shape)
        self._zero_y = np.zeros(self._lower.shape)
        self._solver = osqp.OSQP()
        self._solver.setup(
            p_matrix,
            q_vector,
            a_matrix,
            self._lower,
            self._upper,
            warm_starting=True,
            verbose=False,
        )

    def solve(self, signal: np.ndarray) -> np.ndarray:
        signal_selected = signal[self._selected_subcarriers]
        h_real = np.hstack([signal_selected.real, signal_selected.imag])
        m = 2 * self._row_t
        self._lower[:m] = h_real
        self._upper[:m] = h_real
        self._solver.update(l=self._lower, u=self._upper)
        # Keep each result equivalent to SHARP's fresh-solver behavior while
        # retaining the cached symbolic/numeric matrix factorization.
        self._solver.warm_start(x=self._zero_x, y=self._zero_y)
        result = self._solver.solve()
        return result.x[: self._n // 2] + 1j * result.x[self._n // 2 : self._n]


def _estimate_h_stream(
    task: tuple[str, int, int, int, bool, int, Path],
) -> tuple[str, int, int, bool]:
    """Estimate one SHARP stream using the same OSQP logic as the original script."""
    stem, stream, h_start, h_end_arg, force, checkpoint_every, phase_dir = task
    signal_path = phase_dir / f"signal_{stem}.txt"
    tr_path = phase_dir / f"Tr_vector_{stem}_stream_{stream}.txt"
    checkpoint_path = phase_dir / f"checkpoint_H_{stem}_stream_{stream}.pkl"

    if tr_path.exists() and not force:
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

    t_matrix, time_matrix = build_t_matrix(frequency_vector, delta_t, t_min, t_max)
    select_subcarriers = np.arange(0, frequency_vector.shape[0], subcarriers_space)
    coarse_solver = _PreparedLasso(t_matrix, select_subcarriers)
    refined_problems: dict[int, tuple[_PreparedLasso, np.ndarray]] = {}

    n_steps = end_r - h_start
    next_step = 0
    if checkpoint_path.exists() and not force:
        with checkpoint_path.open("rb") as fp:
            checkpoint = pickle.load(fp)
        tr_matrix = checkpoint["tr_matrix"]
        next_step = checkpoint["next_step"]
    else:
        tr_matrix = np.zeros((frequency_vector_complete.shape[0], n_steps), dtype=complex)

    for time_step in range(next_step, n_steps):
        signal_time = signal_considered[:, time_step]
        complex_opt_r = coarse_solver.solve(signal_time)

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_matrix[position_max_r]

        if position_max_r not in refined_problems:
            refined_min = max(time_max_r - range_refined_down, t_min)
            refined_max = min(time_max_r + range_refined_up, t_max)
            t_matrix_refined, _ = build_t_matrix(
                frequency_vector,
                delta_t_refined,
                refined_min,
                refined_max,
            )
            t_matrix_complete_refined, _ = build_t_matrix(
                frequency_vector_complete,
                delta_t_refined,
                refined_min,
                refined_max,
            )
            refined_problems[position_max_r] = (
                _PreparedLasso(t_matrix_refined, select_subcarriers),
                t_matrix_complete_refined,
            )
        refined_solver, t_matrix_complete_refined = refined_problems[position_max_r]
        complex_opt_r_refined = refined_solver.solve(signal_time)

        position_max_r_refined = np.argmax(abs(complex_opt_r_refined))
        tr_values = np.multiply(t_matrix_complete_refined, complex_opt_r_refined)
        trr_values = np.multiply(
            tr_values,
            np.conj(tr_values[:, position_max_r_refined : position_max_r_refined + 1]),
        )
        tr_matrix[:, time_step] = np.sum(trr_values, axis=1)

        if checkpoint_every > 0 and (time_step + 1) % checkpoint_every == 0:
            _atomic_pickle_dump(
                checkpoint_path,
                {
                    "tr_matrix": tr_matrix,
                    "next_step": time_step + 1,
                    "h_start": h_start,
                    "h_end": h_end_arg,
                },
            )

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
    phase_dir: Path = PHASE_DIR,
) -> None:
    tasks = []
    recording_steps = []
    for mat_path in mat_files:
        stem = mat_path.stem
        with (phase_dir / f"signal_{stem}.txt").open("rb") as fp:
            signal_complete = pickle.load(fp)
        end_r = h_end if h_end != -1 else signal_complete.shape[1]
        n_steps = end_r - h_start
        recording_steps.append((stem, n_steps))

    # Stream-major ordering avoids loading four copies of the same recording
    # concurrently when a large worker pool starts.
    for stream in range(nss * ncore):
        for stem, _ in recording_steps:
            tr_path = phase_dir / f"Tr_vector_{stem}_stream_{stream}.txt"
            if not force and tr_path.exists():
                continue
            tasks.append(
                (stem, stream, h_start, h_end, force, checkpoint_every, phase_dir)
            )

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

    with ProcessPoolExecutor(max_workers=min(jobs, len(tasks))) as executor:
        futures = [executor.submit(_estimate_h_stream, task) for task in tasks]
        with tqdm(
            total=len(futures),
            desc="H-estimation streams",
            unit="stream",
            dynamic_ncols=True,
            bar_format=TQDM_BAR_FORMAT,
        ) as stream_progress:
            for future in as_completed(futures):
                future.result()
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
    show_progress: bool = True,
    phase_dir: Path = PHASE_DIR,
) -> list[Path]:
    """Reconstruct sanitized amplitude/phase matrices using SHARP's logic."""
    stem = mat_path.stem
    subset = mat_path.parent.name
    subset_dir = processed_root / subset
    subset_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    for stream in tqdm(
        range(nss * ncore),
        desc=f"{stem} reconstruct",
        unit="stream",
        leave=False,
        disable=not show_progress,
    ):
        output_path = subset_dir / f"{stem}_stream_{stream}.mat"
        output_paths.append(output_path)
        if output_path.exists() and not force:
            continue

        tr_path = phase_dir / f"Tr_vector_{stem}_stream_{stream}.txt"
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
            disable=not show_progress,
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

        if h_est.shape[1] > 2:
            # The sequential projection preserves the affine component of the
            # first packet, so all interior packets can be solved together.
            errors = phase_before[:, 1:-1] - phase_before[:, :1]
            correction = np.linalg.lstsq(ones_vector.T, errors, rcond=None)[0]
            phase_before[:, 1:-1] -= ones_vector.T @ correction

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
    show_progress: bool = True,
) -> list[dict[str, object]]:
    """Compute SHARP Doppler profiles from reconstructed amplitude/phase."""
    outputs: list[dict[str, object]] = []
    for processed_path in tqdm(
        processed_paths,
        desc="doppler streams",
        unit="stream",
        leave=False,
        disable=not show_progress,
    ):
        subset = processed_path.parent.name
        output_dir = output_root / subset
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{processed_path.stem}.txt"
        if output_path.exists() and not force:
            with output_path.open("rb") as fp:
                existing = pickle.load(fp)
            if not isinstance(existing, np.ndarray) or existing.ndim != 2:
                raise ValueError(f"Invalid existing Doppler trace: {output_path}")
            outputs.append(
                {
                    "path": str(output_path.relative_to(output_root)),
                    "shape": list(existing.shape),
                    "dtype": str(existing.dtype),
                }
            )
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
                "corresponding phase, processed, and output intermediates."
            )
        csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(
            csi_matrix_processed[:, :, 0],
            axis=1,
            keepdims=True,
        )
        csi_matrix_complete = csi_matrix_processed[:, :, 0] * np.exp(1j * csi_matrix_processed[:, :, 1])

        window_count = len(
            range(0, csi_matrix_complete.shape[0] - sample_length, sliding)
        )
        csi_d_profile_array = np.empty((window_count, 100), dtype=np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(
            csi_matrix_complete,
            sample_length,
            axis=0,
        )
        window = hann(sample_length)[None, None, :]
        chunk_size = 4
        with tqdm(
            total=window_count,
            desc=f"{processed_path.stem} doppler",
            unit="win",
            leave=False,
            disable=not show_progress,
        ) as progress:
            for output_start in range(0, window_count, chunk_size):
                output_end = min(output_start + chunk_size, window_count)
                input_start = output_start * sliding
                input_end = output_end * sliding
                csi_matrix_cut = np.nan_to_num(
                    windows[input_start:input_end:sliding],
                )
                csi_doppler_prof = fft(csi_matrix_cut * window, n=100, axis=-1)
                csi_d_profile_array[output_start:output_end] = fftshift(
                    np.sum(
                        csi_doppler_prof.real**2 + csi_doppler_prof.imag**2,
                        axis=1,
                    ),
                    axes=-1,
                )
                progress.update(output_end - output_start)

        csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
        csi_d_profile_array = csi_d_profile_array / csi_d_profile_array_max
        csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_level)] = mt.pow(10, noise_level)

        with output_path.open("wb") as fp:
            pickle.dump(csi_d_profile_array, fp)
        outputs.append(
            {
                "path": str(output_path.relative_to(output_root)),
                "shape": list(csi_d_profile_array.shape),
                "dtype": str(csi_d_profile_array.dtype),
            }
        )
    return outputs


def _postprocess_recording(
    task: tuple[Path, Path, Path, Path, argparse.Namespace, bool],
) -> dict[str, object]:
    mat_path, processed_root, output_root, phase_dir, args, show_progress = task
    processed_paths = reconstruct_streams(
        mat_path,
        processed_root,
        reconstruct_start=args.reconstruct_start,
        reconstruct_end=args.reconstruct_end,
        nss=args.nss,
        ncore=args.ncore,
        force=args.force,
        show_progress=show_progress,
        phase_dir=phase_dir,
    )
    doppler_outputs = compute_doppler(
        processed_paths,
        output_root,
        doppler_start=args.doppler_start,
        doppler_end=args.doppler_end,
        sample_length=args.sample_length,
        sliding=args.sliding,
        noise_level=args.noise_level,
        force=args.force,
        show_progress=show_progress,
    )
    target_shapes = {tuple(item["shape"]) for item in doppler_outputs}
    if len(target_shapes) != 1:
        raise ValueError(f"Generated Doppler stream shapes differ for {mat_path}: {target_shapes}")
    if len(doppler_outputs) != args.nss * args.ncore:
        raise ValueError(
            f"Generated {len(doppler_outputs)} streams for {mat_path}; "
            f"expected {args.nss * args.ncore}"
        )
    with (phase_dir / f"signal_{mat_path.stem}.txt").open("rb") as fp:
        signal_shape = pickle.load(fp).shape
    expected_frames = expected_doppler_frames(signal_shape[1], args)
    target_shape = next(iter(target_shapes))
    if target_shape != (expected_frames, 100):
        raise ValueError(
            f"Generated Doppler shape {target_shape} for {mat_path}; "
            f"expected {(expected_frames, 100)} from the configured preprocessing geometry"
        )
    csi_buff_shape = dict((name, shape) for name, shape, _ in sio.whosmat(mat_path)).get(
        "csi_buff"
    )
    return {
        "scenario": mat_path.parent.name,
        "raw_stem": mat_path.stem,
        "source_raw_path": str(mat_path.resolve()),
        "source_csi_buff_shape": list(csi_buff_shape) if csi_buff_shape else None,
        "cleaned_raw_shape": [
            signal_shape[2],
            signal_shape[0],
            signal_shape[1],
        ],
        "aligned_raw_start": (
            args.preprocess_start
            + args.h_start
            + args.reconstruct_start
            + args.doppler_start
        ),
        "expected_doppler_frames": expected_frames,
        "doppler_streams": doppler_outputs,
    }


def git_revision() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_is_dirty() -> bool | None:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def write_manifest(
    output_root: Path,
    *,
    args: argparse.Namespace,
    subsets: tuple[str, ...],
    recordings: list[dict[str, object]],
) -> None:
    parameters = {
        name: getattr(args, name)
        for name in (
            "nss",
            "ncore",
            "preprocess_start",
            "h_start",
            "h_end",
            "reconstruct_start",
            "reconstruct_end",
            "doppler_start",
            "doppler_end",
            "sample_length",
            "sliding",
            "noise_level",
        )
    }
    manifest = {
        "format_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_revision": git_revision(),
        "git_dirty": git_is_dirty(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "input_root": str(args.input_root.resolve()),
        "phase_root": str(args.phase_root.resolve()),
        "processed_root": str(args.processed_root.resolve()),
        "output_root": str(output_root.resolve()),
        "subsets": list(subsets),
        "parameters": parameters,
        "recordings": recordings,
    }
    temporary = output_root / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    os.replace(temporary, output_root / "manifest.json")


def expected_doppler_frames(signal_time: int, args: argparse.Namespace) -> int:
    h_end = args.h_end if args.h_end != -1 else signal_time
    processed_time = (
        h_end
        - args.h_start
        - args.reconstruct_start
        - args.reconstruct_end
        - args.doppler_start
        - args.doppler_end
    )
    if processed_time <= args.sample_length:
        return 0
    return len(range(0, processed_time - args.sample_length, args.sliding))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=PROJECT_ROOT / "data" / "CSI-80Mhz")
    parser.add_argument("--phase-root", type=Path, default=PHASE_DIR)
    parser.add_argument("--processed-root", type=Path, default=PROJECT_ROOT / "data" / "sharp_processed_pi")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "doppler_traces_pi")
    parser.add_argument(
        "--subsets",
        default=",".join(DEFAULT_PI_SUBSETS),
        help="Comma-separated subset names or family selectors AR, PC, PI, or all.",
    )
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=0,
        help="Packet checkpoint interval; 0 resumes at completed stream boundaries.",
    )
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
    if args.jobs < 1 or args.nss < 1 or args.ncore < 1:
        raise ValueError("--jobs, --nss, and --ncore must be positive")
    if args.checkpoint_every < 0:
        raise ValueError("--checkpoint-every must be non-negative")
    if args.sample_length < 1 or args.sliding < 1:
        raise ValueError("--sample-length and --sliding must be positive")
    selectors = tuple(item.strip() for item in args.subsets.split(",") if item.strip())
    subsets = resolve_subsets(args.input_root, selectors)
    args.phase_root.mkdir(parents=True, exist_ok=True)
    args.processed_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    mat_files = iter_mat_files(args.input_root, subsets, args.include_empty)
    if args.limit is not None:
        mat_files = mat_files[: args.limit]
    if not mat_files:
        raise ValueError("No MAT recordings selected")

    signal_tasks = [
        (
            mat_path,
            args.preprocess_start,
            args.nss,
            args.ncore,
            args.force,
            args.phase_root,
        )
        for mat_path in mat_files
    ]
    signal_jobs = min(args.jobs, 8, len(signal_tasks))
    if signal_jobs == 1:
        for task in tqdm(signal_tasks, desc="Signal preprocessing", unit="trace"):
            _ensure_sharp_signal_task(task)
    else:
        with ProcessPoolExecutor(max_workers=signal_jobs) as executor:
            futures = [executor.submit(_ensure_sharp_signal_task, task) for task in signal_tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Signal preprocessing",
                unit="trace",
            ):
                future.result()

    estimate_h_streams(
        mat_files,
        h_start=args.h_start,
        h_end=args.h_end,
        nss=args.nss,
        ncore=args.ncore,
        force=args.force,
        jobs=args.jobs,
        checkpoint_every=args.checkpoint_every,
        phase_dir=args.phase_root,
    )

    tasks = [
        (
            mat_path,
            args.processed_root,
            args.output_root,
            args.phase_root,
            args,
            args.jobs == 1,
        )
        for mat_path in mat_files
    ]
    if args.jobs == 1:
        recordings = [
            _postprocess_recording(task)
            for task in tqdm(tasks, desc="SHARP traces", unit="trace")
        ]
    else:
        recordings = []
        with ProcessPoolExecutor(max_workers=min(args.jobs, len(tasks))) as executor:
            futures = [executor.submit(_postprocess_recording, task) for task in tasks]
            for future in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="SHARP traces",
                unit="trace",
            ):
                recordings.append(future.result())
        recordings.sort(key=lambda item: str(item["source_raw_path"]))

    write_manifest(
        args.output_root,
        args=args,
        subsets=subsets,
        recordings=recordings,
    )
    print(f"Generated {len(recordings)} aligned recordings under {args.output_root}")


if __name__ == "__main__":
    main()
