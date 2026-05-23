"""SHARP-like Doppler extraction adapted to XRF55 Intel 5300 CSI.

This keeps the original SHARP processing order as closely as possible:
normalize CSI, estimate sparse paths with OSQP, remove the strongest-path
phase reference, reconstruct amplitude/phase, and compute Doppler profiles.
"""

from argparse import ArgumentParser
from pathlib import Path
import math as mt
import pickle
import sys

import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann

from xrf55_csi import read_xrf55_wifi_file, records_to_csi_array


INTEL_5300_SUBCARRIER_INDEXES = np.array(
    [-28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, -1,
      1,   3,   5,   7,   9,  11,  13,  15,  17,  19, 21, 23, 25, 27, 28],
    dtype=float,
)


def xrf55_csi_to_signal_complete(csi: np.ndarray) -> np.ndarray:
    """Return SHARP-style signal_complete as [subcarrier, packet, stream]."""
    if csi.ndim != 4:
        raise ValueError(f"Expected CSI [packet, subcarrier, rx, tx], got {csi.shape}")
    if csi.shape[-1] != 1:
        raise ValueError(f"Expected one Tx stream for XRF55, got {csi.shape[-1]}")

    signal = csi[:, :, :, 0]  # [packet, subcarrier, rx]
    mean_signal = np.mean(np.abs(signal), axis=1, keepdims=True)
    mean_signal[mean_signal == 0] = 1
    signal = signal / mean_signal
    return np.transpose(signal, (1, 0, 2))


def _load_sharp_optimization():
    sharp_dir = Path(__file__).resolve().parents[1] / "third_party" / "sharp_original" / "Python_code"
    sys.path.insert(0, str(sharp_dir))
    from optimization_utility import build_T_matrix, lasso_regression_osqp_fast

    return build_T_matrix, lasso_regression_osqp_fast


def estimate_sharp_tr_matrix(
    signal_complete: np.ndarray,
    stream_idx: int,
    start_packet: int = 0,
    end_packet: int | None = None,
    subcarriers_space: int = 2,
) -> np.ndarray:
    """Estimate SHARP strongest-path-referenced CFR matrix for one XRF55 Rx stream."""
    build_T_matrix, lasso_regression_osqp_fast = _load_sharp_optimization()

    import scipy

    if end_packet is None:
        end_packet = signal_complete.shape[1]

    delta_t = 1e-7
    delta_t_refined = 5e-9
    range_refined_up = 2.5e-7
    range_refined_down = 2e-7

    delta_f = 312.5e3
    frequency_vector_complete = INTEL_5300_SUBCARRIER_INDEXES * delta_f
    frequency_vector = frequency_vector_complete

    t_min = -3e-7
    t_max = 5e-7
    T_matrix, time_matrix = build_T_matrix(frequency_vector, delta_t, t_min, t_max)

    start_subcarrier = 0
    end_subcarrier = frequency_vector.shape[0]
    select_subcarriers = np.arange(start_subcarrier, end_subcarrier, subcarriers_space)

    row_T = int(T_matrix.shape[0] / subcarriers_space)
    col_T = T_matrix.shape[1]
    m = 2 * row_T
    n = 2 * col_T
    In = scipy.sparse.eye(n)
    Im = scipy.sparse.eye(m)
    On = scipy.sparse.csc_matrix((n, n))
    Onm = scipy.sparse.csc_matrix((n, m))
    P = scipy.sparse.block_diag([On, Im, On], format="csc")
    q = np.zeros(2 * n + m)
    A2 = scipy.sparse.hstack([In, Onm, -In])
    A3 = scipy.sparse.hstack([In, Onm, In])
    ones_n_matr = np.ones(n)
    zeros_n_matr = np.zeros(n)
    zeros_nm_matr = np.zeros(n + m)

    signal_considered = signal_complete[:, start_packet:end_packet, stream_idx]
    tr_matrix = np.zeros((frequency_vector_complete.shape[0], end_packet - start_packet), dtype=complex)

    for time_step in range(end_packet - start_packet):
        signal_time = signal_considered[:, time_step]
        complex_opt_r = lasso_regression_osqp_fast(
            signal_time,
            T_matrix,
            select_subcarriers,
            row_T,
            col_T,
            Im,
            Onm,
            P,
            q,
            A2,
            A3,
            ones_n_matr,
            zeros_n_matr,
            zeros_nm_matr,
        )

        position_max_r = np.argmax(abs(complex_opt_r))
        time_max_r = time_matrix[position_max_r]

        T_matrix_refined, _ = build_T_matrix(
            frequency_vector,
            delta_t_refined,
            max(time_max_r - range_refined_down, t_min),
            min(time_max_r + range_refined_up, t_max),
        )

        col_T_refined = T_matrix_refined.shape[1]
        n_refined = 2 * col_T_refined
        In_refined = scipy.sparse.eye(n_refined)
        On_refined = scipy.sparse.csc_matrix((n_refined, n_refined))
        Onm_refined = scipy.sparse.csc_matrix((n_refined, m))
        P_refined = scipy.sparse.block_diag([On_refined, Im, On_refined], format="csc")
        q_refined = np.zeros(2 * n_refined + m)
        A2_refined = scipy.sparse.hstack([In_refined, Onm_refined, -In_refined])
        A3_refined = scipy.sparse.hstack([In_refined, Onm_refined, In_refined])
        ones_n_matr_refined = np.ones(n_refined)
        zeros_n_matr_refined = np.zeros(n_refined)
        zeros_nm_matr_refined = np.zeros(n_refined + m)

        complex_opt_r_refined = lasso_regression_osqp_fast(
            signal_time,
            T_matrix_refined,
            select_subcarriers,
            row_T,
            col_T_refined,
            Im,
            Onm_refined,
            P_refined,
            q_refined,
            A2_refined,
            A3_refined,
            ones_n_matr_refined,
            zeros_n_matr_refined,
            zeros_nm_matr_refined,
        )

        position_max_r_refined = np.argmax(abs(complex_opt_r_refined))
        T_matrix_refined, _ = build_T_matrix(
            frequency_vector_complete,
            delta_t_refined,
            max(time_max_r - range_refined_down, t_min),
            min(time_max_r + range_refined_up, t_max),
        )

        Tr = np.multiply(T_matrix_refined, complex_opt_r_refined)
        Trr = np.multiply(Tr, np.conj(Tr[:, position_max_r_refined:position_max_r_refined + 1]))
        tr_matrix[:, time_step] = np.sum(Trr, axis=1)

    return tr_matrix


def reconstruct_csi_matrix_processed(h_est: np.ndarray) -> np.ndarray:
    """Apply SHARP reconstruction phase cleanup and return [packet, subcarrier, amp/phase]."""
    csi_matrix_processed = np.zeros((h_est.shape[1], h_est.shape[0], 2))
    csi_matrix_processed[:, :, 0] = np.abs(h_est).T

    phase_before = np.unwrap(np.angle(h_est), axis=0)
    ones_vector = np.ones((2, phase_before.shape[0]))
    ones_vector[1, :] = np.arange(0, phase_before.shape[0])

    for tidx in range(1, phase_before.shape[1]):
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
                    phase_before[idx_act + 1:, tidx] -= 2 * mt.pi
                    idx_prec = idx_act
            elif idxs_invert_down.shape[0] > 0:
                idx_act = idxs_invert_down[0]
                if idx_act == idx_prec:
                    stop = True
                else:
                    phase_before[idx_act + 1:, tidx] += 2 * mt.pi
                    idx_prec = idx_act
            else:
                stop = True

    for tidx in range(1, h_est.shape[1] - 1):
        val_prec = phase_before[:, tidx - 1:tidx]
        val_act = phase_before[:, tidx:tidx + 1]
        error = val_act - val_prec
        temp2 = np.linalg.lstsq(ones_vector.T, error, rcond=None)[0]
        phase_before[:, tidx] = phase_before[:, tidx] - (np.dot(ones_vector.T, temp2)).T

    csi_matrix_processed[:, :, 1] = phase_before.T
    return csi_matrix_processed


def compute_doppler_profile(
    csi_matrix_processed: np.ndarray,
    sample_length: int = 51,
    sliding: int = 1,
    noise_level: float = -2,
) -> np.ndarray:
    """Compute SHARP 100-bin Doppler profile from reconstructed amplitude/phase."""
    csi_matrix_processed = csi_matrix_processed.copy()
    csi_matrix_processed[:, :, 0] = csi_matrix_processed[:, :, 0] / np.mean(
        csi_matrix_processed[:, :, 0],
        axis=1,
        keepdims=True,
    )
    csi_matrix_complete = csi_matrix_processed[:, :, 0] * np.exp(1j * csi_matrix_processed[:, :, 1])

    csi_d_profile_list = []
    for i in range(0, csi_matrix_complete.shape[0] - sample_length, sliding):
        csi_matrix_cut = csi_matrix_complete[i:i + sample_length, :]
        csi_matrix_cut = np.nan_to_num(csi_matrix_cut)

        hann_window = np.expand_dims(hann(sample_length), axis=-1)
        csi_matrix_wind = np.multiply(csi_matrix_cut, hann_window)
        csi_doppler_prof = fft(csi_matrix_wind, n=100, axis=0)
        csi_doppler_prof = fftshift(csi_doppler_prof, axes=0)

        csi_d_map = np.abs(csi_doppler_prof * np.conj(csi_doppler_prof))
        csi_d_map = np.sum(csi_d_map, axis=1)
        csi_d_profile_list.append(csi_d_map)

    csi_d_profile_array = np.asarray(csi_d_profile_list)
    csi_d_profile_array_max = np.max(csi_d_profile_array, axis=1, keepdims=True)
    csi_d_profile_array = csi_d_profile_array / csi_d_profile_array_max
    csi_d_profile_array[csi_d_profile_array < mt.pow(10, noise_level)] = mt.pow(10, noise_level)
    return csi_d_profile_array


def compute_sharp_like_doppler_for_file(
    path: str | Path,
    stream_idx: int = 0,
    max_packets: int | None = None,
) -> np.ndarray:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)
    if max_packets is not None:
        csi = csi[:max_packets]
    signal_complete = xrf55_csi_to_signal_complete(csi)
    h_est = estimate_sharp_tr_matrix(signal_complete, stream_idx=stream_idx)
    csi_matrix_processed = reconstruct_csi_matrix_processed(h_est)
    return compute_doppler_profile(csi_matrix_processed)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("path", help="XRF55 raw Wi-Fi .dat or .mat file")
    parser.add_argument("--stream", type=int, default=0, help="Rx stream index")
    parser.add_argument("--max-packets", type=int, default=120, help="Limit packets for a quick smoke test")
    parser.add_argument("--output", default=None, help="Optional pickle output path")
    args = parser.parse_args()

    doppler = compute_sharp_like_doppler_for_file(args.path, args.stream, args.max_packets)
    print(f"doppler shape: {doppler.shape}")
    print(f"doppler min/mean/max: {doppler.min():.4f} / {doppler.mean():.4f} / {doppler.max():.4f}")

    if args.output:
        with Path(args.output).open("wb") as fp:
            pickle.dump(doppler, fp)


if __name__ == "__main__":
    main()
