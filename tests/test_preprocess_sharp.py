from __future__ import annotations

import argparse
import pickle
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import scipy
import scipy.io as sio


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocessing.preprocess_sharp import (  # noqa: E402
    _PreparedLasso,
    build_t_matrix,
    compute_doppler,
    ensure_sharp_signal,
    expected_doppler_frames,
    iter_mat_files,
    resolve_subsets,
)
from optimization_utility import build_T_matrix, lasso_regression_osqp_fast  # noqa: E402
from scripts.convert_csi_doppler_memmap import (  # noqa: E402
    is_legacy_sharp_aligned,
    resolve_scenarios,
)
from wifi_doppler.data.csi_to_sharp_doppler_dataset import (  # noqa: E402
    CsiToSharpDopplerDataset,
    raw_file_key,
    raw_scenario_dir,
)
from wifi_doppler.data.doppler_dataset import parse_trace_filename  # noqa: E402


class GenericSharpPreprocessingTests(unittest.TestCase):
    def test_signal_preprocessing_uses_configured_phase_root(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            mat_path = root / "PC-1a" / "PC1a_W.mat"
            mat_path.parent.mkdir()
            rng = np.random.default_rng(3)
            csi = rng.normal(size=(16, 256)) + 1j * rng.normal(size=(16, 256))
            sio.savemat(mat_path, {"csi_buff": csi})

            phase_root = root / "phase"
            phase_root.mkdir()
            ensure_sharp_signal(
                mat_path,
                preprocess_start=0,
                nss=1,
                ncore=4,
                force=False,
                phase_dir=phase_root,
            )

            with (phase_root / "signal_PC1a_W.txt").open("rb") as fp:
                signal = pickle.load(fp)
            self.assertEqual(signal.shape, (242, 4, 4))
            self.assertTrue(np.isfinite(signal).all())

    def test_optimized_sharp_solver_matches_original(self) -> None:
        frequencies = np.linspace(-40e6, 40e6, 242)
        original_t, original_times = build_T_matrix(
            frequencies,
            1e-7,
            -3e-7,
            5e-7,
        )
        optimized_t, optimized_times = build_t_matrix(
            frequencies,
            1e-7,
            -3e-7,
            5e-7,
        )
        np.testing.assert_array_equal(optimized_t, original_t)
        np.testing.assert_array_equal(optimized_times, original_times)

        selected = np.arange(0, 242, 2)
        row_t, col_t = original_t[selected].shape
        m, n = 2 * row_t, 2 * col_t
        identity_m = scipy.sparse.eye(m)
        identity_n = scipy.sparse.eye(n)
        zeros_n = scipy.sparse.csc_matrix((n, n))
        zeros_nm = scipy.sparse.csc_matrix((n, m))
        p_matrix = scipy.sparse.block_diag(
            [zeros_n, identity_m, zeros_n],
            format="csc",
        )
        a2_matrix = scipy.sparse.hstack([identity_n, zeros_nm, -identity_n])
        a3_matrix = scipy.sparse.hstack([identity_n, zeros_nm, identity_n])
        prepared = _PreparedLasso(optimized_t, selected)
        signals = np.random.default_rng(7).normal(size=(8, 242)).astype(np.complex128)
        for signal in signals:
            original = lasso_regression_osqp_fast(
                signal,
                original_t,
                selected,
                row_t,
                col_t,
                identity_m,
                zeros_nm,
                p_matrix,
                np.zeros(2 * n + m),
                a2_matrix,
                a3_matrix,
                np.ones(n),
                np.zeros(n),
                np.zeros(n + m),
            )
            reproduced = prepared.solve(signal)
            np.testing.assert_array_equal(reproduced, original)

    def test_family_selectors_discover_ar_and_pc_files(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for subset, stems in {
                "AR-1a": ("AR1a_W",),
                "PC-1a": ("PC1a_C1",),
                "PI-1a": ("PI1a_p00", "PI1a_p03"),
            }.items():
                directory = root / subset
                directory.mkdir()
                for stem in stems:
                    (directory / f"{stem}.mat").touch()

            subsets = resolve_subsets(root, ("AR", "PC"))
            files = iter_mat_files(root, subsets, include_empty=False)

            self.assertEqual(subsets, ("AR-1a", "PC-1a"))
            self.assertEqual([path.stem for path in files], ["AR1a_W", "PC1a_C1"])
            self.assertEqual(
                [path.stem for path in iter_mat_files(root, ("PI-1a",), include_empty=False)],
                ["PI1a_p03"],
            )
            self.assertEqual(resolve_scenarios(root, ["AR", "PC"]), ["AR-1a", "PC-1a"])
            (root / "S1b").mkdir()
            self.assertEqual(
                resolve_scenarios(root, ["AR", "PC"]),
                ["AR-1a", "S1b", "PC-1a"],
            )

    def test_legacy_scenarios_map_to_canonical_raw_names(self) -> None:
        self.assertEqual(raw_scenario_dir("S2a"), "AR-1d")
        self.assertEqual(raw_scenario_dir("S6b"), "AR-5b")
        self.assertEqual(
            raw_file_key("S2a", "AR1d_J.mat".removesuffix(".mat")),
            ("S2a", "J", ""),
        )
        self.assertTrue(is_legacy_sharp_aligned(20_000, 18_369))
        self.assertTrue(is_legacy_sharp_aligned(20_000, 18_368))
        self.assertFalse(is_legacy_sharp_aligned(20_000, 18_500))

    def test_compute_doppler_preserves_canonical_subset_and_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            processed_path = root / "processed" / "PC-1a" / "PC1a_W_stream_0.mat"
            processed_path.parent.mkdir(parents=True)
            values = np.zeros((12, 242, 2), dtype=np.float64)
            values[..., 0] = 1
            values[..., 1] = np.arange(12, dtype=np.float64)[:, None] * 0.1
            sio.savemat(processed_path, {"csi_matrix_processed": values})

            outputs = compute_doppler(
                [processed_path],
                root / "doppler",
                doppler_start=1,
                doppler_end=1,
                sample_length=3,
                sliding=1,
                noise_level=-1.2,
                force=False,
            )

            output_path = root / "doppler" / outputs[0]["path"]
            with output_path.open("rb") as fp:
                trace = pickle.load(fp)
            self.assertEqual(outputs[0]["path"], "PC-1a/PC1a_W_stream_0.txt")
            self.assertEqual(trace.shape, (7, 100))
            self.assertTrue(np.isfinite(trace).all())

    def test_expected_frame_count_matches_sharp_loop(self) -> None:
        args = argparse.Namespace(
            h_start=0,
            h_end=-1,
            reconstruct_start=0,
            reconstruct_end=0,
            doppler_start=800,
            doppler_end=800,
            sample_length=31,
            sliding=1,
        )
        self.assertEqual(expected_doppler_frames(20_000, args), 18_369)

    def test_canonical_pc_targets_pair_with_their_raw_mat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "PC-1a"
            doppler_dir = root / "doppler" / "PC-1a"
            raw_dir.mkdir(parents=True)
            doppler_dir.mkdir(parents=True)

            csi = np.ones((40, 256), dtype=np.complex64)
            sio.savemat(raw_dir / "PC1a_W1.mat", {"csi_buff": csi})
            for antenna in range(4):
                with (doppler_dir / f"PC1a_W1_stream_{antenna}.txt").open("wb") as fp:
                    pickle.dump(np.ones((8, 100), dtype=np.float32), fp)

            dataset = CsiToSharpDopplerDataset(
                raw_root=root / "raw",
                doppler_root=root / "doppler",
                scenarios=("PC-1a",),
                split=(0, 1),
                doppler_window_size=3,
                window_stride=3,
                split_guard=0,
                doppler_start=0,
                doppler_sample_length=3,
                num_subcarriers=3,
                target_transform="none",
            )

            self.assertEqual(len(dataset.traces), 1)
            self.assertEqual(dataset.traces[0].label, "W1")
            self.assertEqual(
                parse_trace_filename("PC1a_W1_stream_3.txt"),
                {
                    "scenario": "PC-1a",
                    "label": "W1",
                    "repetition": "",
                    "antenna": 3,
                },
            )


if __name__ == "__main__":
    unittest.main()
