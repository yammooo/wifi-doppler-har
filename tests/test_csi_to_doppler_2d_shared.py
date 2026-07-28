from __future__ import annotations

from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from scripts.train_csi_to_doppler import validate_config
from wifi_doppler.models.csi_to_doppler_2d_shared import (
    SharedAntennaCsiToDopplerUNet2D,
)
from wifi_doppler.models.csi_to_doppler_builders import (
    build_csi_to_doppler_model,
    csi_to_doppler_model_metadata,
)


class SharedAntennaUNet2DTests(unittest.TestCase):
    @staticmethod
    def _model(**overrides) -> SharedAntennaCsiToDopplerUNet2D:
        options = {
            "architecture": "unet2d_shared_antenna_full_resolution",
            "num_antennas": 4,
            "num_subcarriers": 5,
            "input_parts": 2,
            "output_doppler_bins": 12,
            "output_time": 9,
            "temporal_context": 3,
            "base_channels": 4,
            "mid_channels": 6,
            "bottleneck_channels": 8,
            "decoder_channels": 4,
        }
        options.update(overrides)
        return build_csi_to_doppler_model(options).eval()

    def test_shared_weights_are_antenna_permutation_equivariant(self) -> None:
        model = self._model()
        inputs = torch.randn(2, 4, 5, 11, 2)
        permutation = torch.tensor([2, 0, 3, 1])

        with torch.no_grad():
            output = model(inputs)
            separate = torch.cat(
                [model(inputs[:, antenna : antenna + 1]) for antenna in range(4)],
                dim=1,
            )
            permuted = model(inputs[:, permutation])
            repeated = model(inputs[:, :1].expand(-1, 4, -1, -1, -1))

        self.assertEqual(tuple(output.shape), (2, 4, 9, 12))
        self.assertEqual(tuple(model(inputs[:, :1]).shape), (2, 1, 9, 12))
        torch.testing.assert_close(output, separate)
        torch.testing.assert_close(permuted, output[:, permutation])
        torch.testing.assert_close(repeated, repeated[:, :1].expand_as(repeated))

    def test_uses_exact_valid_crop_and_only_resizes_time(self) -> None:
        model = self._model(output_time=5, temporal_context=31)
        time_grid = torch.arange(35, dtype=torch.float32).reshape(1, 1, 35, 1)
        torch.testing.assert_close(
            model._crop_valid_time(time_grid).flatten(),
            torch.arange(15, 20, dtype=torch.float32),
        )

        grid = torch.tensor([[[[0.0, 10.0], [2.0, 14.0]]]])
        resized = model._resize_time(grid, 5)
        expected = torch.stack(
            [
                F.interpolate(
                    grid[..., doppler_bin].reshape(1, 1, 2),
                    size=5,
                    mode="linear",
                    align_corners=False,
                ).flatten()
                for doppler_bin in range(2)
            ],
            dim=-1,
        )
        torch.testing.assert_close(resized[0, 0], expected)

        head_input_times: list[int] = []
        handle = model.output_head.register_forward_pre_hook(
            lambda _module, args: head_input_times.append(args[0].shape[2])
        )
        with torch.no_grad():
            model(torch.randn(1, 1, 5, 35, 2))
        handle.remove()
        self.assertEqual(head_input_times, [35])

    def test_group_norm_and_full_resolution_projection(self) -> None:
        model = self._model(num_subcarriers=7)
        self.assertTrue(any(isinstance(module, torch.nn.GroupNorm) for module in model.modules()))
        self.assertFalse(
            any(
                isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d))
                for module in model.modules()
            )
        )
        self.assertEqual(
            model.grid_projection.out_channels,
            model.decoder_channels * model.output_doppler_bins,
        )
        with torch.no_grad():
            output = model(torch.randn(1, 2, 7, 11, 2))
        self.assertEqual(tuple(output.shape), (1, 2, 9, 12))

        with self.assertRaisesRegex(ValueError, "time=11"):
            model(torch.randn(1, 2, 7, 10, 2))

    def test_builder_metadata_and_config_validation(self) -> None:
        config_path = (
            PROJECT_ROOT
            / "configs"
            / "csi_to_doppler"
            / "ar_pc_pi_cross_domain_unet2d_shared_antenna_full_resolution_motion_aware_full_subcarriers.yaml"
        )
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        validate_config(config)
        self.assertEqual(
            csi_to_doppler_model_metadata(config["model"]),
            (
                "csi_to_doppler_unet2d_shared_antenna_full_resolution_v1",
                "wifi_doppler.models.csi_to_doppler_2d_shared."
                "SharedAntennaCsiToDopplerUNet2D",
            ),
        )

        config["model"]["temporal_context"] = 29
        with self.assertRaisesRegex(ValueError, "must match"):
            validate_config(config)
        config["model"]["temporal_context"] = 30
        with self.assertRaisesRegex(ValueError, "positive odd"):
            validate_config(config)


if __name__ == "__main__":
    unittest.main()
