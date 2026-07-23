from __future__ import annotations

import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import warnings

import numpy as np
import scipy.io as sio
import torch
import yaml

from scripts.train_csi_to_doppler import configure_cuda_convolution_backend, validate_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset
from wifi_doppler.data.windowing import WindowIndex
from wifi_doppler.models.csi_to_doppler import CsiToDopplerUNet1D
from wifi_doppler.models.csi_to_doppler_2d import CsiToDopplerUNet2DDecoder
from wifi_doppler.models.csi_to_doppler_builders import build_csi_to_doppler_model
from wifi_doppler.training.distillation import (
    DistillationMetricAccumulator,
    count_recording_batches,
    distillation_loss,
    iter_recording_batches,
    load_training_checkpoint,
    move_batches_to_device,
    prefetch_batches,
    run_distillation_epoch,
    save_training_checkpoint,
)


class FakeRecording:
    def __init__(self, name: str, offset: float):
        self.filename_stem = name
        values = np.arange(4 * 3 * 6, dtype=np.float32).reshape(4, 3, 6) + offset
        self.raw = values.astype(np.complex64) + 1j * (values + 1)
        self.doppler = np.repeat(values[:, :1, :].transpose(0, 2, 1), 5, axis=2)
        self.raw_loads = 0
        self.doppler_loads = 0

    def load_raw(self) -> np.ndarray:
        self.raw_loads += 1
        return self.raw

    def load_doppler(self) -> np.ndarray:
        self.doppler_loads += 1
        return self.doppler

    def clear_cache(self) -> None:
        pass


class FakeDataset:
    def __init__(self, num_views: int = 1):
        self.traces = [FakeRecording("recording_a", 0), FakeRecording("recording_b", 100)]
        self.subcarrier_views = tuple(
            np.asarray(view) for view in ([0, 2], [1, 2])[:num_views]
        )
        self.window_indexes = [
            WindowIndex(0, 0, 3),
            WindowIndex(0, 3, 6),
            WindowIndex(1, 0, 3),
            WindowIndex(1, 3, 6),
        ]

    def __len__(self) -> int:
        return len(self.window_indexes) * len(self.subcarrier_views)

    @staticmethod
    def raw_bounds_for_doppler_window(start: int, end: int) -> tuple[int, int]:
        return start, end


class TinyStudent(torch.nn.Module):
    num_antennas = 4

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        values = inputs[..., 0].mean(dim=2)
        return self.scale * values.unsqueeze(-1).expand(-1, -1, -1, 5)


class DistillationTests(unittest.TestCase):
    def test_cudnn_version_mismatch_falls_back_to_native_cuda(self) -> None:
        mismatch = RuntimeError("cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH")
        original_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            with (
                mock.patch("scripts.train_csi_to_doppler.torch.zeros", return_value=mock.sentinel.inputs),
                mock.patch("scripts.train_csi_to_doppler.torch.ones", return_value=mock.sentinel.weights),
                mock.patch(
                    "scripts.train_csi_to_doppler.torch.nn.functional.conv1d",
                    side_effect=(mismatch, mock.sentinel.output),
                ) as conv1d,
                mock.patch("scripts.train_csi_to_doppler.torch.cuda.synchronize") as synchronize,
                warnings.catch_warnings(record=True) as caught,
            ):
                warnings.simplefilter("always")
                enabled = configure_cuda_convolution_backend(torch.device("cuda"))

            self.assertFalse(enabled)
            self.assertFalse(torch.backends.cudnn.enabled)
            self.assertEqual(conv1d.call_count, 2)
            synchronize.assert_called_once_with(torch.device("cuda"))
            self.assertIn("Mixed cuDNN sublibrary versions", str(caught[0].message))
        finally:
            torch.backends.cudnn.enabled = original_enabled

    def test_unrelated_cudnn_error_is_not_hidden(self) -> None:
        original_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = True
        try:
            with (
                mock.patch("scripts.train_csi_to_doppler.torch.zeros", return_value=mock.sentinel.inputs),
                mock.patch("scripts.train_csi_to_doppler.torch.ones", return_value=mock.sentinel.weights),
                mock.patch(
                    "scripts.train_csi_to_doppler.torch.nn.functional.conv1d",
                    side_effect=RuntimeError("a different CUDA failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "different CUDA failure"):
                    configure_cuda_convolution_backend(torch.device("cuda"))
        finally:
            torch.backends.cudnn.enabled = original_enabled

    def test_mse_and_metrics(self) -> None:
        predictions = torch.tensor([[[[0.0, -1.0, 2.0]], [[1.0, 3.0, 0.0]]]])
        targets = torch.tensor([[[[0.0, 1.0, 1.0]], [[1.0, 1.0, 0.0]]]])
        self.assertAlmostEqual(float(distillation_loss(predictions, targets)), 1.5)

        metrics = DistillationMetricAccumulator(num_antennas=2)
        metrics.update(predictions, targets)
        result = metrics.compute()
        self.assertAlmostEqual(result["mse"], 1.5)
        self.assertAlmostEqual(result["mae"], 5 / 6)
        self.assertAlmostEqual(result["rmse"], np.sqrt(1.5))
        self.assertAlmostEqual(result["negative_fraction"], 1 / 6)
        self.assertAlmostEqual(result["mse_antenna_0"], 5 / 3)
        self.assertAlmostEqual(result["mse_antenna_1"], 4 / 3)

    def test_motion_weighted_wasserstein_uses_doppler_bin_distance(self) -> None:
        target = torch.zeros(1, 1, 1, 5)
        prediction = torch.zeros_like(target)
        target[..., 3] = 1
        prediction[..., 1] = 1

        loss = distillation_loss(
            prediction,
            target,
            name="motion_weighted_wasserstein",
            options={
                "floor": 0,
                "center_half_width": 0,
                "motion_weight": 0,
                "wasserstein_weight": 1,
                "raw_mse_weight": 0,
                "smooth_l1_beta": 0.1,
            },
        )
        active_map_loss = torch.nn.functional.smooth_l1_loss(
            prediction,
            target,
            beta=0.1,
        )
        self.assertAlmostEqual(float(loss - active_map_loss), 0.5, places=6)

    def test_recording_iterator_loads_once_and_covers_all_windows(self) -> None:
        dataset = FakeDataset()
        batches = list(iter_recording_batches(dataset, batch_size=1, shuffle=False, seed=4))
        filenames = [filename for batch in batches for filename in batch.filenames]

        self.assertEqual(count_recording_batches(dataset, batch_size=1), 4)
        self.assertEqual(len(filenames), len(dataset))
        self.assertEqual(len(set(filenames)), len(dataset))
        for recording in dataset.traces:
            self.assertEqual(recording.raw_loads, 1)
            self.assertEqual(recording.doppler_loads, 1)

    def test_recording_iterator_is_deterministic_for_a_seed(self) -> None:
        def order(seed: int) -> list[str]:
            dataset = FakeDataset()
            return [
                filename
                for batch in iter_recording_batches(dataset, batch_size=1, shuffle=True, seed=seed)
                for filename in batch.filenames
            ]

        self.assertEqual(order(8), order(8))
        self.assertNotEqual(order(8), order(9))

    def test_mixed_recording_batches_are_balanced_and_cover_every_view(self) -> None:
        dataset = FakeDataset(num_views=2)
        batches = list(
            iter_recording_batches(
                dataset,
                batch_size=4,
                shuffle=True,
                seed=3,
                recordings_per_batch=2,
            )
        )
        filenames = [filename for batch in batches for filename in batch.filenames]

        self.assertEqual(len(filenames), len(dataset))
        self.assertEqual(len(set(filenames)), len(dataset))
        for batch in batches:
            counts = {
                name: sum(filename.startswith(name) for filename in batch.filenames)
                for name in ("recording_a", "recording_b")
            }
            self.assertEqual(counts, {"recording_a": 2, "recording_b": 2})
        for recording in dataset.traces:
            self.assertEqual(recording.raw_loads, 1)
            self.assertEqual(recording.doppler_loads, 1)

    def test_mixed_recording_iterator_is_deterministic_for_a_seed(self) -> None:
        def order(seed: int) -> list[str]:
            dataset = FakeDataset(num_views=2)
            return [
                filename
                for batch in iter_recording_batches(
                    dataset,
                    batch_size=4,
                    shuffle=True,
                    seed=seed,
                    recordings_per_batch=2,
                )
                for filename in batch.filenames
            ]

        self.assertEqual(order(8), order(8))
        self.assertNotEqual(order(8), order(9))

    def test_mixed_recording_batches_require_memmap_storage(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "csi_to_doppler" / "pi_cross_domain_mse.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["data"]["storage"] = "source"
        with self.assertRaisesRegex(ValueError, "require data.storage=memmap"):
            validate_config(config)
        config["data"]["storage"] = "memmap"
        for invalid in (0, config["training"]["batch_size"] + 1, 1.5):
            config["training"]["recordings_per_batch"] = invalid
            with self.assertRaisesRegex(ValueError, "integer between 1 and batch_size"):
                validate_config(config)

    def test_one_epoch_cpu_smoke(self) -> None:
        dataset = FakeDataset()
        model = TinyStudent()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.scale.detach().clone()
        batches = iter_recording_batches(dataset, batch_size=2, shuffle=True, seed=0)
        batches = prefetch_batches(batches, max_prefetch=2, pin_memory=False)
        batches = move_batches_to_device(batches, device=torch.device("cpu"), cuda_prefetch=False)
        callback_steps = []
        result = run_distillation_epoch(
            model,
            batches,
            device=torch.device("cpu"),
            optimizer=optimizer,
            amp_enabled=False,
            batch_callback=lambda step, metrics: callback_steps.append((step, metrics)),
            batch_callback_every=2,
        )
        self.assertEqual(result.num_samples, len(dataset))
        self.assertEqual(result.global_step, 2)
        self.assertNotEqual(float(before), float(model.scale.detach()))
        self.assertIn("mse", result.metrics)
        self.assertEqual([step for step, _ in callback_steps], [2])

    def test_checkpoint_restores_training_and_rng_state(self) -> None:
        model = TinyStudent()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "latest.pt"
            save_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                epoch=3,
                global_step=12,
                best_metric=0.25,
                patience_counter=2,
                config={"run": {"id": "test"}},
                history=[{"epoch": 3}],
                wandb_run_id="wandb-test",
            )
            expected_random = random.random()
            expected_numpy = float(np.random.random())
            expected_torch = float(torch.rand(()))
            model.scale.data.fill_(99)
            random.seed(99)
            np.random.seed(99)
            torch.manual_seed(99)

            checkpoint = load_training_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                scaler=None,
                map_location="cpu",
            )
            self.assertEqual(checkpoint["epoch"], 3)
            self.assertAlmostEqual(float(model.scale.detach()), 0.5)
            self.assertEqual(random.random(), expected_random)
            self.assertEqual(float(np.random.random()), expected_numpy)
            self.assertEqual(float(torch.rand(())), expected_torch)


class CsiToDopplerModelTests(unittest.TestCase):
    def test_full_2d_decoder_output_shape_and_legacy_builder(self) -> None:
        common = {
            "num_antennas": 4,
            "num_subcarriers": 3,
            "input_parts": 2,
            "output_doppler_bins": 12,
            "output_time": 5,
            "base_channels": 4,
            "mid_channels": 6,
            "bottleneck_channels": 8,
        }
        model = build_csi_to_doppler_model(
            {
                "architecture": "unet2d_decoder",
                **common,
                "decoder_channels": 3,
                "decoder_coarse_bins": 3,
            }
        ).eval()

        with torch.no_grad():
            output = model(torch.randn(2, 4, 3, 7, 2))

        self.assertIsInstance(model, CsiToDopplerUNet2DDecoder)
        self.assertEqual(tuple(output.shape), (2, 4, 5, 12))
        self.assertTrue(any(isinstance(module, torch.nn.Conv2d) for module in model.decoder1.modules()))
        self.assertIsInstance(build_csi_to_doppler_model(common), CsiToDopplerUNet1D)

    def test_full_2d_decoder_config_validation(self) -> None:
        config = yaml.safe_load(
            (PROJECT_ROOT / "configs" / "csi_to_doppler" / "pi_cross_domain_mse.yaml").read_text(
                encoding="utf-8"
            )
        )
        config["model"]["architecture"] = "unknown"
        with self.assertRaisesRegex(ValueError, "model.architecture"):
            validate_config(config)

        config["model"]["architecture"] = "unet2d_decoder"
        config["model"]["decoder_coarse_bins"] = config["model"]["output_doppler_bins"] + 1
        with self.assertRaisesRegex(ValueError, "decoder_coarse_bins"):
            validate_config(config)
        config["model"]["decoder_coarse_bins"] = 25
        config["model"]["decoder_channels"] = 1.5
        with self.assertRaisesRegex(ValueError, "decoder_channels"):
            validate_config(config)


class PairedDatasetTests(unittest.TestCase):
    def test_mat_pickle_pairing_alignment_and_direct_targets(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "PI-1a"
            doppler_dir = root / "doppler" / "PI-1a"
            raw_dir.mkdir(parents=True)
            doppler_dir.mkdir(parents=True)

            packet_values = np.arange(1, 41, dtype=np.float32)[:, None]
            subcarrier_values = np.arange(1, 257, dtype=np.float32)[None, :]
            csi_buff = (packet_values + subcarrier_values).astype(np.complex64) * (1 + 1j)
            sio.savemat(raw_dir / "PI1a_p03.mat", {"csi_buff": csi_buff})
            for antenna in range(4):
                target = np.full((8, 100), antenna + 1, dtype=np.float64)
                with (doppler_dir / f"PI1a_p03_stream_{antenna}.txt").open("wb") as stream:
                    pickle.dump(target, stream)

            dataset = CsiToSharpDopplerDataset(
                raw_root=root / "raw",
                doppler_root=root / "doppler",
                scenarios=("PI-1a",),
                split=(0.0, 1.0),
                doppler_window_size=3,
                window_stride=3,
                split_guard=0,
                doppler_start=0,
                doppler_sample_length=3,
                num_subcarriers=3,
                target_transform="none",
            )
            inputs, targets, filename = dataset[0]
            self.assertEqual(tuple(inputs.shape), (4, 3, 5, 2))
            self.assertEqual(tuple(targets.shape), (4, 3, 100))
            self.assertEqual(filename, "PI-1a_p03_d0-3_view0")
            for antenna in range(4):
                self.assertTrue(torch.all(targets[antenna] == antenna + 1))
            self.assertEqual(dataset.pairing_report()["num_pairs"], 1)

    def test_local_cli_one_epoch_with_wandb_disabled(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw" / "PI-1a"
            doppler_dir = root / "doppler" / "PI-1a"
            raw_dir.mkdir(parents=True)
            doppler_dir.mkdir(parents=True)

            packet_values = np.arange(1, 169, dtype=np.float32)[:, None]
            subcarrier_values = np.arange(1, 257, dtype=np.float32)[None, :]
            for person, offset in (("p03", 0), ("p05", 10)):
                csi_buff = (packet_values + subcarrier_values + offset).astype(np.complex64) * (1 + 1j)
                sio.savemat(raw_dir / f"PI1a_{person}.mat", {"csi_buff": csi_buff})
                for antenna in range(4):
                    target = np.full(
                        (40, 100),
                        0.1 * (antenna + 1) + offset / 100,
                        dtype=np.float32,
                    )
                    with (doppler_dir / f"PI1a_{person}_stream_{antenna}.txt").open("wb") as stream:
                        pickle.dump(target, stream)

            prepared_root = root / "prepared"
            converted = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "convert_csi_doppler_memmap.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--raw-root",
                    str(root / "raw"),
                    "--doppler-root",
                    str(root / "doppler"),
                    "--output-root",
                    str(prepared_root),
                    "--scenarios",
                    "PI-1a",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(converted.returncode, 0, msg=converted.stdout + converted.stderr)
            self.assertTrue((prepared_root / "manifest.json").is_file())

            config = {
                "run": {"id": "smoke", "output_root": str(root / "runs"), "seed": 0},
                "device": "cpu",
                "data": {
                    "storage": "memmap",
                    "raw_root": str(root / "raw"),
                    "doppler_root": str(root / "doppler"),
                    "prepared_root": str(prepared_root),
                    "doppler_window_size": 3,
                    "window_stride": 3,
                    "split_guard": 0,
                    "doppler_start": 0,
                    "doppler_sample_length": 3,
                    "doppler_sliding": 1,
                    "num_subcarriers": 3,
                    "subcarrier_sampling": "fixed_uniform",
                    "num_subcarrier_views": 1,
                    "subcarrier_seed": 0,
                    "target_transform": "none",
                    "splits": {
                        "train": {"scenarios": ["PI-1a"], "interval": [0.0, 0.6]},
                        "source_val": {"scenarios": ["PI-1a"], "interval": [0.6, 0.8]},
                        "target_val": {"scenarios": ["PI-1a"], "interval": [0.6, 0.8]},
                        "target_test": {"scenarios": ["PI-1a"], "interval": [0.8, 1.0]},
                    },
                },
                "model": {
                    "architecture": "unet2d_decoder",
                    "num_antennas": 4,
                    "num_subcarriers": 3,
                    "input_parts": 2,
                    "output_doppler_bins": 100,
                    "output_time": 3,
                    "base_channels": 4,
                    "mid_channels": 6,
                    "bottleneck_channels": 8,
                    "decoder_channels": 2,
                    "decoder_coarse_bins": 25,
                },
                "training": {
                    "loss": "motion_weighted_wasserstein",
                    "loss_options": {
                        "floor": 0.0630957344,
                        "center_half_width": 1,
                        "motion_weight": 4.0,
                        "wasserstein_weight": 0.1,
                        "raw_mse_weight": 0.05,
                        "smooth_l1_beta": 0.1,
                    },
                    "batch_size": 2,
                    "recordings_per_batch": 2,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "amp": False,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
                    "early_stopping_metric": "loss",
                    "log_every_steps": 1,
                    "validation_examples": 0,
                },
                "wandb": {
                    "mode": "disabled",
                    "project": "test",
                    "entity": None,
                    "name": None,
                    "watch": "gradients",
                    "watch_log_frequency": 100,
                },
            }
            config_path = root / "config.yaml"
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_csi_to_doppler.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--config",
                    str(config_path),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, msg=completed.stdout + completed.stderr)
            run_dir = root / "runs" / "smoke"
            self.assertTrue((run_dir / "model.pt").exists())
            self.assertTrue((run_dir / "training" / "latest.pt").exists())
            self.assertTrue((run_dir / "training" / "final_test.json").exists())
            run_record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(run_record["status"], "completed")
            self.assertEqual(run_record["model_key"], "csi_to_doppler_unet2d_decoder_v1")
            self.assertEqual(
                run_record["builder"],
                "wifi_doppler.models.csi_to_doppler_2d.CsiToDopplerUNet2DDecoder",
            )

            config["training"]["epochs"] = 2
            config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
            resumed = subprocess.run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "scripts" / "train_csi_to_doppler.py"),
                    "--project-root",
                    str(PROJECT_ROOT),
                    "--config",
                    str(config_path),
                    "--resume",
                    str(run_dir / "training" / "latest.pt"),
                ],
                cwd=PROJECT_ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(resumed.returncode, 0, msg=resumed.stdout + resumed.stderr)
            latest = torch.load(run_dir / "training" / "latest.pt", weights_only=False)
            self.assertEqual(latest["epoch"], 2)


if __name__ == "__main__":
    unittest.main()
