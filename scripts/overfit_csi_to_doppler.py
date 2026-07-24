from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Overfit a fixed CSI-to-Doppler batch.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser.parse_args()


def select_motion_rich_windows(
    dataset,
    *,
    recording_name: str,
    num_windows: int,
    center_half_width: int,
    floor: float,
) -> list[tuple[int, float]]:
    if num_windows < 1:
        raise ValueError("diagnostic.num_windows must be >= 1.")

    recording_indexes = [
        index for index, trace in enumerate(dataset.traces) if trace.filename_stem == recording_name
    ]
    if len(recording_indexes) != 1:
        available = ", ".join(trace.filename_stem for trace in dataset.traces)
        raise ValueError(
            f"Expected one recording named {recording_name!r}; available recordings: {available}"
        )

    recording_idx = recording_indexes[0]
    doppler = dataset.traces[recording_idx].load_doppler()
    num_bins = doppler.shape[-1]
    center = num_bins // 2
    if not 0 <= center_half_width < center:
        raise ValueError("diagnostic.center_half_width must select a proper Doppler subset.")
    outside_center = np.ones(num_bins, dtype=bool)
    outside_center[center - center_half_width : center + center_half_width + 1] = False

    candidates = []
    for base_idx, window in enumerate(dataset.window_indexes):
        if window.recording_idx != recording_idx:
            continue
        target = doppler[:, window.start : window.end]
        score = float(np.maximum(target[..., outside_center] - floor, 0).mean())
        candidates.append((base_idx, score, window.start, window.end))

    selected: list[tuple[int, float, int, int]] = []
    for candidate in sorted(candidates, key=lambda item: (-item[1], item[2])):
        _, _, start, end = candidate
        if all(end <= other_start or start >= other_end for _, _, other_start, other_end in selected):
            selected.append(candidate)
            if len(selected) == num_windows:
                break

    if len(selected) != num_windows:
        raise ValueError(
            f"Only {len(selected)} non-overlapping windows are available in {recording_name}; "
            f"requested {num_windows}."
        )
    return [(base_idx, score) for base_idx, score, _, _ in sorted(selected, key=lambda item: item[2])]


def build_fixed_batch(dataset, selected: list[tuple[int, float]]):
    from wifi_doppler.training.distillation import DistillationBatch

    if len(dataset.subcarrier_views) != 1:
        raise ValueError("The overfit diagnostic requires data.num_subcarrier_views=1.")
    samples = [dataset[base_idx] for base_idx, _ in selected]
    return DistillationBatch(
        inputs=torch.stack([sample[0] for sample in samples]),
        targets=torch.stack([sample[1] for sample in samples]),
        filenames=tuple(sample[2] for sample in samples),
    )


def diagnostic_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    *,
    center_half_width: int,
    floor: float,
) -> dict[str, float]:
    center = predictions.shape[-1] // 2
    outside_center = torch.ones(predictions.shape[-1], dtype=torch.bool, device=predictions.device)
    outside_center[center - center_half_width : center + center_half_width + 1] = False
    error = predictions - targets
    background = targets <= floor + 1e-6
    leakage = (predictions - floor).clamp_min(0)
    return {
        "mse": float(error.square().mean()),
        "off_center_mse": float(error[..., outside_center].square().mean()),
        "background_leakage": float(leakage[background].mean()) if background.any() else 0.0,
        "max_abs_error": float(error.abs().max()),
        "negative_fraction": float((predictions < 0).float().mean()),
    }


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from scripts.train_csi_to_doppler import (
        add_src_to_path,
        build_datasets,
        configure_cuda_convolution_backend,
        initialize_wandb,
        load_resolved_config,
        log_examples,
        log_model_artifact,
        save_json,
        save_yaml,
        seed_everything,
        select_device,
        utc_now,
    )

    add_src_to_path(project_root)
    from wifi_doppler.models.csi_to_doppler import count_trainable_parameters
    from wifi_doppler.models.csi_to_doppler_builders import build_csi_to_doppler_model
    from wifi_doppler.training.distillation import save_inference_checkpoint

    config = load_resolved_config(args.config.resolve(), args.overrides, project_root)
    diagnostic = config["diagnostic"]
    if config["training"]["loss"] != "mse":
        raise ValueError("The overfit diagnostic requires training.loss=mse.")

    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = select_device(str(config["device"]))
    cudnn_enabled = configure_cuda_convolution_backend(device)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(
            cudnn_enabled and config["training"].get("cudnn_benchmark", True)
        )

    run_id = config["run"]["id"] or datetime.now().strftime("csi-overfit-%Y%m%d-%H%M%S")
    config["run"]["id"] = run_id
    run_dir = Path(config["run"]["output_root"]) / run_id
    training_dir = run_dir / "training"
    if run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    training_dir.mkdir(parents=True)
    save_yaml(training_dir / "resolved_config.yaml", config)

    datasets = build_datasets(config, ("train",))
    dataset = datasets["train"]
    selected = select_motion_rich_windows(
        dataset,
        recording_name=str(diagnostic["recording"]),
        num_windows=int(diagnostic["num_windows"]),
        center_half_width=int(diagnostic["center_half_width"]),
        floor=float(diagnostic["floor"]),
    )
    batch = build_fixed_batch(dataset, selected)
    selected_windows = [
        {"filename": filename, "motion_score": score}
        for filename, (_, score) in zip(batch.filenames, selected)
    ]
    save_json(training_dir / "selected_windows.json", selected_windows)

    inputs = batch.inputs.to(device)
    targets = batch.targets.to(device)
    model = build_csi_to_doppler_model(config["model"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    wandb_run = initialize_wandb(config, run_dir, None)
    if wandb_run is not None:
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")
        wandb_run.summary["model/trainable_parameters"] = count_trainable_parameters(model)
        wandb_run.summary["diagnostic/recording"] = diagnostic["recording"]
        wandb_run.summary["diagnostic/num_windows"] = len(selected)
        wandb_run.config.update({"selected_windows": selected_windows}, allow_val_change=True)

    run_record: dict[str, Any] = {
        "model_run_id": run_id,
        "representation": "raw_csi_to_sharp_doppler_overfit",
        "wandb_run_id": wandb_run.id if wandb_run is not None else None,
        "status": "running",
        "updated_at": utc_now(),
    }
    save_json(run_dir / "run.json", run_record)

    steps = int(diagnostic["steps"])
    log_every = int(diagnostic["log_every_steps"])
    image_every = int(diagnostic["image_every_steps"])
    if min(steps, log_every, image_every) < 1:
        raise ValueError("diagnostic steps and logging intervals must be >= 1.")
    history = []
    started_at = time.perf_counter()

    try:
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                predictions = model(inputs)
                loss = torch.nn.functional.mse_loss(predictions, targets)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if step == 1 or step % log_every == 0 or step == steps:
                with torch.inference_mode():
                    model.train()
                    train_predictions = model(inputs)
                    train_metrics = diagnostic_metrics(
                        train_predictions,
                        targets,
                        center_half_width=int(diagnostic["center_half_width"]),
                        floor=float(diagnostic["floor"]),
                    )
                    model.eval()
                    eval_predictions = model(inputs)
                    eval_metrics = diagnostic_metrics(
                        eval_predictions,
                        targets,
                        center_half_width=int(diagnostic["center_half_width"]),
                        floor=float(diagnostic["floor"]),
                    )

                record = {
                    "global_step": step,
                    "elapsed_seconds": time.perf_counter() - started_at,
                    "train": train_metrics,
                    "eval": eval_metrics,
                }
                history.append(record)
                save_json(training_dir / "history.json", history)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "global_step": step,
                            **{f"train/{key}": value for key, value in train_metrics.items()},
                            **{f"eval/{key}": value for key, value in eval_metrics.items()},
                        }
                    )
                    if step == 1 or step % image_every == 0 or step == steps:
                        examples = [
                            {
                                "filename": batch.filenames[index],
                                "prediction": eval_predictions[index].detach().float().cpu(),
                                "target": targets[index].detach().float().cpu(),
                            }
                            for index in range(min(2, len(batch.filenames)))
                        ]
                        log_examples(wandb_run, "overfit_eval", examples, step)
                print(
                    f"step {step}: train_mse={train_metrics['mse']:.6g} "
                    f"eval_mse={eval_metrics['mse']:.6g} "
                    f"off_center_mse={eval_metrics['off_center_mse']:.6g}"
                )

        model_path = run_dir / "model.pt"
        save_inference_checkpoint(
            model_path,
            model=model,
            epoch=steps,
            metrics={f"eval/{key}": value for key, value in eval_metrics.items()},
            config=config,
        )
        run_record.update(
            {
                "status": "completed",
                "steps": steps,
                "final_train": train_metrics,
                "final_eval": eval_metrics,
                "updated_at": utc_now(),
            }
        )
        save_json(run_dir / "run.json", run_record)
        if wandb_run is not None:
            for key, value in eval_metrics.items():
                wandb_run.summary[f"final_eval/{key}"] = value
            log_model_artifact(
                wandb_run,
                model_path,
                kind="overfit-final",
                metadata={"steps": steps, **eval_metrics},
            )
        print(f"run directory: {run_dir}")
    except Exception:
        run_record.update({"status": "failed", "updated_at": utc_now()})
        save_json(run_dir / "run.json", run_record)
        raise
    finally:
        dataset.clear_cache()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
