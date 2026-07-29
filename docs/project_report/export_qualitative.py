"""Export one fixed legacy-SHARP test window for the report comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from scripts.evaluate_doppler_classifier_fidelity import affine_correct, fixed_stft
from wifi_doppler.models.csi_to_doppler_builders import build_csi_to_doppler_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording", default="S1a_C")
    parser.add_argument("--start", type=int, default=16920)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device_name = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if device_name == "auto":
        device_name = "cpu"
    device = torch.device(device_name)

    manifest = json.loads((args.prepared_root / "manifest.json").read_text())
    item = next(
        recording
        for recording in manifest["recordings"]
        if recording["filename_stem"] == args.recording
    )
    raw = np.load(args.prepared_root / item["raw_path"], mmap_mode="r")
    target = np.load(args.prepared_root / item["doppler_path"], mmap_mode="r")

    raw_start = 800 + args.start
    raw_window = np.asarray(raw[:, :, raw_start : raw_start + 370])
    target_window = np.asarray(
        target[:, args.start : args.start + 340], dtype=np.float32
    )

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = build_csi_to_doppler_model(checkpoint["config"]["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_input = torch.from_numpy(
        np.stack((raw_window.real, raw_window.imag), axis=-1).astype(np.float32)
    ).unsqueeze(0).to(device)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        enabled=device.type == "cuda",
    ):
        student = model(model_input).float()[0].cpu().numpy()
        affine = fixed_stft(affine_correct(raw_window[None]), device)[0].cpu().numpy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        target=target_window,
        student=student,
        affine=affine,
        recording=args.recording,
        start=args.start,
    )
    print(args.output)


if __name__ == "__main__":
    main()
