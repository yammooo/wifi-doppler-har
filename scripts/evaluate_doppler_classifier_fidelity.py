from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader


LABELS = ("E", "L", "W", "R", "J")
SCENARIOS = ("AR-1a", "AR-1b", "AR-1c")
LEGACY_SCENARIOS = ("S1a", "S1b", "S1c")
FLOOR = 10**-1.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Doppler generators through a frozen SHARP classifier."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/csi_doppler_memmap"),
    )
    parser.add_argument(
        "--legacy-doppler-root",
        type=Path,
        default=Path("data/doppler_traces"),
    )
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--student",
        action="append",
        default=[],
        metavar="NAME=CHECKPOINT",
        help="Best CSI-to-Doppler checkpoint; may be repeated.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", nargs=2, type=float, default=(0.9, 1.0))
    parser.add_argument("--window-stride", type=int, default=30)
    parser.add_argument("--split-guard", type=int, default=31)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def resolve_path(root: Path, path: Path) -> Path:
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_students(values: list[str], root: Path) -> dict[str, Path]:
    students = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected NAME=CHECKPOINT, got {value!r}.")
        name, path = value.split("=", 1)
        if not name or name in students:
            raise ValueError(f"Invalid or duplicate student name: {name!r}.")
        students[name] = resolve_path(root, Path(path))
    return students


def load_classifier(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    from wifi_doppler.models.sharp import MultiAntennaModel, SingleAntennaModel

    model = MultiAntennaModel(SingleAntennaModel()).to(device)
    with torch.inference_mode():
        model.forward_antennas(torch.zeros(1, 4, 340, 100, device=device))
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.eval()


def load_students(
    checkpoint_paths: dict[str, Path],
    device: torch.device,
) -> tuple[dict[str, torch.nn.Module], dict[str, dict[str, Any]]]:
    from wifi_doppler.models.csi_to_doppler_builders import build_csi_to_doppler_model

    models = {}
    metadata = {}
    for name, path in checkpoint_paths.items():
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        config = checkpoint["config"]
        model = build_csi_to_doppler_model(config["model"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        models[name] = model.eval()
        metadata[name] = {
            "checkpoint": str(path),
            "checkpoint_sha256": sha256(path),
            "epoch": checkpoint.get("epoch"),
            "model": config["model"],
            "data": config["data"],
        }
    return models, metadata


def fixed_stft(signal: np.ndarray, device: torch.device, chunk_size: int = 16) -> torch.Tensor:
    """Apply SHARP's fixed 31-packet Doppler tail to complex CSI."""
    values = torch.from_numpy(np.ascontiguousarray(signal)).to(device)
    windows = values.unfold(-1, 31, 1)
    hann = torch.hann_window(31, periodic=False, dtype=torch.float32, device=device)
    output = torch.empty(
        (*signal.shape[:2], windows.shape[-2], 100),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, windows.shape[-2], chunk_size):
        stop = min(start + chunk_size, windows.shape[-2])
        spectrum = torch.fft.fft(windows[..., start:stop, :] * hann, n=100, dim=-1)
        power = torch.fft.fftshift(spectrum.abs().square().sum(dim=2), dim=-1)
        power /= power.amax(dim=-1, keepdim=True).clamp_min(1e-12)
        output[..., start:stop, :] = power.clamp_min(FLOOR)
    return output


def affine_correct(raw: np.ndarray) -> np.ndarray:
    """Remove each packet's affine phase difference from the first packet."""
    phase = np.unwrap(np.angle(raw), axis=2)
    subcarrier = np.arange(raw.shape[2], dtype=np.float32)
    design = np.stack((subcarrier, np.ones_like(subcarrier)), axis=1)
    inverse = np.linalg.pinv(design).astype(np.float32)
    error = phase - phase[..., :1]
    coefficients = np.einsum("ks,bast->bakt", inverse, error, optimize=True)
    correction = np.einsum("sk,bakt->bast", design, coefficients, optimize=True)
    return (np.abs(raw) * np.exp(1j * (phase - correction))).astype(
        np.complex64,
        copy=False,
    )


def classifier_predictions(
    classifier: torch.nn.Module,
    maps: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    maps = maps.float()
    return (
        classifier(maps, fusion="sharp").argmax(dim=1),
        classifier(maps, fusion="sum").argmax(dim=1),
    )


def generate_maps(
    raw_batch: np.ndarray,
    students: dict[str, torch.nn.Module],
    device: torch.device,
    timing: dict[str, dict[str, Any]] | None = None,
) -> dict[str, torch.Tensor]:
    generated: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    generated["raw_fixed_stft"] = fixed_stft(raw_batch, device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    if timing is not None:
        timing["raw_fixed_stft"]["generation_seconds"] += time.perf_counter() - started

    started = time.perf_counter()
    generated["affine_fixed_stft"] = fixed_stft(affine_correct(raw_batch), device)
    torch.cuda.synchronize(device) if device.type == "cuda" else None
    if timing is not None:
        timing["affine_fixed_stft"]["generation_seconds"] += time.perf_counter() - started

    model_input = torch.from_numpy(
        np.stack((raw_batch.real, raw_batch.imag), axis=-1).astype(
            np.float32,
            copy=False,
        )
    ).to(device)
    for name, model in students.items():
        started = time.perf_counter()
        with torch.inference_mode(), torch.autocast(
            device_type=device.type,
            enabled=device.type == "cuda",
        ):
            generated[name] = model(model_input).float()
        torch.cuda.synchronize(device) if device.type == "cuda" else None
        if timing is not None:
            timing[name]["generation_seconds"] += time.perf_counter() - started
    return generated


def estimate_recording_means(
    raw: np.ndarray,
    teacher: np.ndarray,
    students: dict[str, torch.nn.Module],
    batch_size: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Estimate each generator's full-recording mean for SHARP centering."""
    means = {
        "recomputed_sharp": torch.from_numpy(np.asarray(teacher).mean(axis=1)).to(
            device=device,
            dtype=torch.float32,
        )
    }
    sums = {
        name: torch.zeros((4, 100), dtype=torch.float64, device=device)
        for name in ("raw_fixed_stft", "affine_fixed_stft", *students)
    }
    frame_count = 0
    starts = list(range(0, teacher.shape[1] - 340 + 1, 340))
    final_start = teacher.shape[1] - 340
    if not starts or starts[-1] != final_start:
        starts.append(final_start)

    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset : offset + batch_size]
        raw_batch = np.stack(
            [
                np.asarray(raw[:, :, 800 + index : 800 + index + 370])
                for index in batch_starts
            ]
        )
        generated = generate_maps(raw_batch, students, device)
        for name, maps in generated.items():
            sums[name] += maps.double().sum(dim=(0, 2))
        frame_count += len(batch_starts) * 340

    means.update({name: total.float() / frame_count for name, total in sums.items()})
    return means


def new_stats(num_classes: int) -> dict[str, Any]:
    return {
        "count": 0,
        "sharp_correct": 0,
        "sum_correct": 0,
        "teacher_agreement": 0,
        "confusion": np.zeros((num_classes, num_classes), dtype=np.int64),
        "generation_seconds": 0.0,
    }


def update_classification(
    stats: dict[str, Any],
    labels: torch.Tensor,
    sharp_predictions: torch.Tensor,
    sum_predictions: torch.Tensor,
    teacher_predictions: torch.Tensor,
) -> None:
    labels_cpu = labels.cpu().numpy()
    sharp_cpu = sharp_predictions.cpu().numpy()
    stats["count"] += len(labels_cpu)
    stats["sharp_correct"] += int((sharp_predictions == labels).sum())
    stats["sum_correct"] += int((sum_predictions == labels).sum())
    stats["teacher_agreement"] += int((sharp_predictions == teacher_predictions).sum())
    np.add.at(stats["confusion"], (labels_cpu, sharp_cpu), 1)


def finish_stats(stats: dict[str, Any]) -> dict[str, Any]:
    count = stats["count"]
    confusion = stats["confusion"]
    class_counts = confusion.sum(axis=1)
    return {
        "windows": count,
        "sharp_fusion_accuracy": stats["sharp_correct"] / count,
        "sum_fusion_accuracy": stats["sum_correct"] / count,
        "teacher_decision_agreement": stats["teacher_agreement"] / count,
        "confusion_matrix": confusion.tolist(),
        "per_class_accuracy": {
            label: (
                float(confusion[index, index] / class_counts[index])
                if class_counts[index]
                else None
            )
            for index, label in enumerate(LABELS)
        },
        "generation_seconds": stats["generation_seconds"],
        "generation_windows_per_second": (
            count / stats["generation_seconds"]
            if stats["generation_seconds"] > 0
            else None
        ),
    }


def evaluate_legacy_teacher(
    classifier: torch.nn.Module,
    root: Path,
    interval: tuple[float, float],
    stride: int,
    split_guard: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    from wifi_doppler.data.doppler_dataset import DopplerWindowDataset

    dataset = DopplerWindowDataset(
        root,
        scenarios=list(LEGACY_SCENARIOS),
        split=interval,
        window_stride=stride,
        split_guard=split_guard,
        labels=LABELS,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    stats = new_stats(len(LABELS))
    with torch.inference_mode():
        for maps, labels in loader:
            maps = maps.to(device)
            labels = labels.to(device)
            sharp_predictions, sum_predictions = classifier_predictions(classifier, maps)
            update_classification(
                stats,
                labels,
                sharp_predictions,
                sum_predictions,
                sharp_predictions,
            )
    result = finish_stats(stats)
    result["teacher_decision_agreement"] = 1.0
    return result


def evaluate_prepared(
    classifier: torch.nn.Module,
    students: dict[str, torch.nn.Module],
    prepared_root: Path,
    interval: tuple[float, float],
    stride: int,
    split_guard: int,
    batch_size: int,
    device: torch.device,
) -> dict[str, Any]:
    from wifi_doppler.training.distillation import DistillationMetricAccumulator

    manifest = json.loads((prepared_root / "manifest.json").read_text(encoding="utf-8"))
    recordings = [
        item
        for item in manifest["recordings"]
        if item["scenario"] in SCENARIOS and item["label"][:1] in LABELS
    ]
    if not recordings:
        raise ValueError(f"No classifier-compatible AR recordings in {prepared_root}.")

    names = ("recomputed_sharp", "raw_fixed_stft", "affine_fixed_stft", *students)
    stats = {name: new_stats(len(LABELS)) for name in names}
    metrics = {
        name: DistillationMetricAccumulator(
            num_antennas=4,
            motion_floor=FLOOR,
            motion_threshold=0.2,
            center_half_width=5,
        )
        for name in names
    }
    label_to_index = {label: index for index, label in enumerate(LABELS)}

    for recording in recordings:
        raw = np.load(
            prepared_root / recording["raw_path"].replace("\\", "/"),
            mmap_mode="r",
            allow_pickle=False,
        )
        teacher = np.load(
            prepared_root / recording["doppler_path"].replace("\\", "/"),
            mmap_mode="r",
            allow_pickle=False,
        )
        recording_means = estimate_recording_means(
            raw,
            teacher,
            students,
            batch_size,
            device,
        )
        start = int(teacher.shape[1] * interval[0]) + (split_guard if interval[0] else 0)
        stop = int(teacher.shape[1] * interval[1]) - 340
        starts = list(range(start, stop + 1, stride))

        for offset in range(0, len(starts), batch_size):
            batch_starts = starts[offset : offset + batch_size]
            raw_batch = np.stack(
                [np.asarray(raw[:, :, 800 + index : 800 + index + 370]) for index in batch_starts]
            )
            target = torch.from_numpy(
                np.stack([np.asarray(teacher[:, index : index + 340]) for index in batch_starts])
            ).to(device)
            labels = torch.full(
                (len(batch_starts),),
                label_to_index[recording["label"][:1]],
                dtype=torch.long,
                device=device,
            )

            generated: dict[str, torch.Tensor] = {"recomputed_sharp": target}
            generated.update(generate_maps(raw_batch, students, device, stats))

            with torch.inference_mode():
                centered_target = target - recording_means["recomputed_sharp"][:, None, :]
                teacher_predictions, _ = classifier_predictions(
                    classifier,
                    centered_target,
                )
                for name, maps in generated.items():
                    centered_maps = maps - recording_means[name][:, None, :]
                    sharp_predictions, sum_predictions = classifier_predictions(
                        classifier,
                        centered_maps,
                    )
                    update_classification(
                        stats[name],
                        labels,
                        sharp_predictions,
                        sum_predictions,
                        teacher_predictions,
                    )
                    metrics[name].update(maps, target)

    return {
        name: {
            **finish_stats(stats[name]),
            "reconstruction": metrics[name].compute(),
        }
        for name in names
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    sys.path.insert(0, str(project_root / "src"))
    prepared_root = resolve_path(project_root, args.prepared_root)
    legacy_root = resolve_path(project_root, args.legacy_doppler_root)
    classifier_path = resolve_path(project_root, args.classifier_checkpoint)
    output_path = resolve_path(project_root, args.output)
    student_paths = parse_students(args.student, project_root)
    interval = (float(args.interval[0]), float(args.interval[1]))
    if not 0 <= interval[0] < interval[1] <= 1:
        raise ValueError(f"Invalid interval: {interval}")
    if args.window_stride < 1 or args.batch_size < 1:
        raise ValueError("window-stride and batch-size must be positive.")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    classifier = load_classifier(classifier_path, device)
    students, student_metadata = load_students(student_paths, device)
    print(f"device={device}; students={list(students)}", flush=True)

    legacy = evaluate_legacy_teacher(
        classifier,
        legacy_root,
        interval,
        args.window_stride,
        args.split_guard,
        args.batch_size,
        device,
    )
    print(
        f"legacy SHARP: {legacy['sharp_fusion_accuracy']:.4f} "
        f"over {legacy['windows']} windows",
        flush=True,
    )
    methods = evaluate_prepared(
        classifier,
        students,
        prepared_root,
        interval,
        args.window_stride,
        args.split_guard,
        args.batch_size,
        device,
    )
    for name, values in methods.items():
        print(
            f"{name}: accuracy={values['sharp_fusion_accuracy']:.4f} "
            f"agreement={values['teacher_decision_agreement']:.4f}",
            flush=True,
        )

    results = {
        "protocol": {
            "scenarios": list(SCENARIOS),
            "legacy_scenarios": list(LEGACY_SCENARIOS),
            "labels": list(LABELS),
            "interval": list(interval),
            "window_size": 340,
            "window_stride": args.window_stride,
            "split_guard": args.split_guard,
            "raw_context": 31,
            "classifier_training": {
                "source": "notebooks/sharp_reproduction.ipynb",
                "scenarios": list(LEGACY_SCENARIOS),
                "labels": list(LABELS),
                "train_interval": [0.0, 0.6],
                "validation_interval": [0.6, 0.8],
            },
        },
        "classifier": {
            "checkpoint": str(classifier_path),
            "checkpoint_sha256": sha256(classifier_path),
        },
        "students": student_metadata,
        "legacy_sharp": legacy,
        "methods": methods,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
