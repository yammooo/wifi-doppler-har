from __future__ import annotations

import json
from pathlib import Path
import pickle
import random
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import scipy.io as sio
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset
from wifi_doppler.data.windowing import WindowIndex
from wifi_doppler.training.distillation import (
    DistillationMetricAccumulator,
    count_recording_batches,
    distillation_loss,
    iter_recording_batches,
    load_training_checkpoint,
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
    def __init__(self):
        self.traces = [FakeRecording("recording_a", 0), FakeRecording("recording_b", 100)]
        self.subcarrier_views = (np.asarray([0, 2]),)
        self.window_indexes = [
            WindowIndex(0, 0, 3),
            WindowIndex(0, 3, 6),
            WindowIndex(1, 0, 3),
            WindowIndex(1, 3, 6),
        ]

    def __len__(self) -> int:
        return len(self.window_indexes)

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

    def test_one_epoch_cpu_smoke(self) -> None:
        dataset = FakeDataset()
        model = TinyStudent()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        before = model.scale.detach().clone()
        result = run_distillation_epoch(
            model,
            iter_recording_batches(dataset, batch_size=2, shuffle=True, seed=0),
            device=torch.device("cpu"),
            optimizer=optimizer,
            amp_enabled=False,
        )
        self.assertEqual(result.num_samples, len(dataset))
        self.assertEqual(result.global_step, 2)
        self.assertNotEqual(float(before), float(model.scale.detach()))
        self.assertIn("mse", result.metrics)

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
            csi_buff = (packet_values + subcarrier_values).astype(np.complex64) * (1 + 1j)
            sio.savemat(raw_dir / "PI1a_p03.mat", {"csi_buff": csi_buff})
            for antenna in range(4):
                target = np.full((40, 100), 0.1 * (antenna + 1), dtype=np.float32)
                with (doppler_dir / f"PI1a_p03_stream_{antenna}.txt").open("wb") as stream:
                    pickle.dump(target, stream)

            config = {
                "run": {"id": "smoke", "output_root": str(root / "runs"), "seed": 0},
                "device": "cpu",
                "data": {
                    "raw_root": str(root / "raw"),
                    "doppler_root": str(root / "doppler"),
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
                    "num_antennas": 4,
                    "num_subcarriers": 3,
                    "input_parts": 2,
                    "output_doppler_bins": 100,
                    "output_time": 3,
                    "base_channels": 4,
                    "mid_channels": 6,
                    "bottleneck_channels": 8,
                },
                "training": {
                    "loss": "mse",
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "epochs": 1,
                    "amp": False,
                    "early_stopping_patience": 1,
                    "early_stopping_min_delta": 0.0,
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
