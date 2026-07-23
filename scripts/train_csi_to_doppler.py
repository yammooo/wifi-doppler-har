from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import sys
from typing import Any
import warnings

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import torch
import yaml


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CSI-to-SHARP-Doppler student locally.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override an existing YAML key with dotted syntax; may be repeated.",
    )
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def load_resolved_config(config_path: Path, overrides: list[str], project_root: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")
    for override in overrides:
        apply_override(config, override)

    for key in ("raw_root", "doppler_root", "prepared_root"):
        if key not in config["data"]:
            continue
        path = Path(config["data"][key])
        config["data"][key] = str((project_root / path).resolve() if not path.is_absolute() else path.resolve())
    output_root = Path(config["run"]["output_root"])
    config["run"]["output_root"] = str(
        (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    )
    validate_config(config)
    return config


def apply_override(config: dict[str, Any], expression: str) -> None:
    if "=" not in expression:
        raise ValueError(f"Override must use KEY=VALUE syntax: {expression!r}")
    dotted_key, raw_value = expression.split("=", 1)
    parts = dotted_key.split(".")
    target: dict[str, Any] = config
    for part in parts[:-1]:
        if part not in target or not isinstance(target[part], dict):
            raise KeyError(f"Unknown configuration key: {dotted_key}")
        target = target[part]
    if parts[-1] not in target:
        raise KeyError(f"Unknown configuration key: {dotted_key}")
    target[parts[-1]] = yaml.safe_load(raw_value)


def validate_config(config: dict[str, Any]) -> None:
    required_splits = {"train", "source_val", "target_val", "target_test"}
    missing = required_splits - set(config["data"]["splits"])
    if missing:
        raise ValueError(f"Missing data splits: {sorted(missing)}")
    if config["training"]["loss"] not in {"mse", "motion_weighted_wasserstein"}:
        raise ValueError(
            "training.loss must be mse or motion_weighted_wasserstein."
        )
    loss_options = config["training"].get("loss_options", {})
    if not isinstance(loss_options, dict):
        raise ValueError("training.loss_options must be a mapping.")
    if config["training"]["loss"] == "motion_weighted_wasserstein":
        nonnegative_options = (
            "center_half_width",
            "motion_weight",
            "wasserstein_weight",
            "raw_mse_weight",
        )
        if any(float(loss_options.get(key, 0)) < 0 for key in nonnegative_options):
            raise ValueError(f"Loss options {nonnegative_options} must be non-negative.")
        if float(loss_options.get("smooth_l1_beta", 0.1)) <= 0:
            raise ValueError("training.loss_options.smooth_l1_beta must be positive.")
    early_stopping_metric = config["training"].get("early_stopping_metric", "mse")
    if early_stopping_metric not in {"loss", "mse"}:
        raise ValueError("training.early_stopping_metric must be loss or mse.")
    if config["model"]["num_subcarriers"] != config["data"]["num_subcarriers"]:
        raise ValueError("model.num_subcarriers must match data.num_subcarriers.")
    if config["model"]["output_time"] != config["data"]["doppler_window_size"]:
        raise ValueError("model.output_time must match data.doppler_window_size.")
    architecture = config["model"].get("architecture", "unet1d_legacy")
    if architecture not in {"unet1d_legacy", "unet2d_decoder"}:
        raise ValueError("model.architecture must be unet1d_legacy or unet2d_decoder.")
    if architecture == "unet2d_decoder":
        decoder_channels = config["model"].get("decoder_channels")
        coarse_bins = config["model"].get("decoder_coarse_bins")
        if (
            not isinstance(decoder_channels, int)
            or isinstance(decoder_channels, bool)
            or decoder_channels < 1
        ):
            raise ValueError("model.decoder_channels must be an integer >= 1.")
        if (
            not isinstance(coarse_bins, int)
            or isinstance(coarse_bins, bool)
            or not 1 <= coarse_bins <= config["model"]["output_doppler_bins"]
        ):
            raise ValueError(
                "model.decoder_coarse_bins must be an integer between 1 and output_doppler_bins."
            )
    if config["training"]["batch_size"] < 1 or config["training"]["epochs"] < 1:
        raise ValueError("training.batch_size and training.epochs must be >= 1.")
    recordings_per_batch = config["training"].get("recordings_per_batch", 1)
    if (
        not isinstance(recordings_per_batch, int)
        or isinstance(recordings_per_batch, bool)
        or not 1 <= recordings_per_batch <= config["training"]["batch_size"]
    ):
        raise ValueError("training.recordings_per_batch must be an integer between 1 and batch_size.")
    if recordings_per_batch > 1 and config["data"].get("storage", "source") != "memmap":
        raise ValueError("Mixed-recording batches require data.storage=memmap.")
    if config["training"]["early_stopping_patience"] < 1:
        raise ValueError("training.early_stopping_patience must be >= 1.")
    if config["training"]["log_every_steps"] < 1:
        raise ValueError("training.log_every_steps must be >= 1.")
    if config["training"]["validation_examples"] < 0:
        raise ValueError("training.validation_examples must be >= 0.")
    if config["training"].get("prefetch_batches", 0) < 0:
        raise ValueError("training.prefetch_batches must be >= 0.")
    if config["data"].get("storage", "source") not in {"source", "memmap"}:
        raise ValueError("data.storage must be source or memmap.")
    if config["data"].get("storage") == "memmap" and not config["data"].get("prepared_root"):
        raise ValueError("data.prepared_root is required when data.storage=memmap.")
    if config["wandb"]["mode"] not in {"online", "offline", "disabled"}:
        raise ValueError("wandb.mode must be online, offline, or disabled.")
    for split_name, split in config["data"]["splits"].items():
        start, end = split["interval"]
        if not 0 <= start < end <= 1:
            raise ValueError(f"Invalid interval for {split_name}: {split['interval']}")
        if not split["scenarios"]:
            raise ValueError(f"No scenarios configured for {split_name}.")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def configure_cuda_convolution_backend(device: torch.device) -> bool:
    """Probe cuDNN and fall back to PyTorch's native CUDA convolutions if needed."""
    if device.type != "cuda" or not torch.backends.cudnn.enabled:
        return False

    inputs = torch.zeros((1, 1, 8), device=device)
    weights = torch.ones((1, 1, 3), device=device)
    try:
        torch.nn.functional.conv1d(inputs, weights, padding=1)
        torch.cuda.synchronize(device)
    except RuntimeError as error:
        if "CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH" not in str(error):
            raise
        torch.backends.cudnn.enabled = False
        # Confirm that the fallback works now, before dataset construction and W&B setup.
        torch.nn.functional.conv1d(inputs, weights, padding=1)
        torch.cuda.synchronize(device)
        warnings.warn(
            "Mixed cuDNN sublibrary versions were detected. cuDNN has been disabled "
            "for this process; training will continue with native CUDA convolutions.",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    return True


def make_grad_scaler(enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=enabled)


def build_datasets(config: dict[str, Any], split_names: tuple[str, ...]):
    from wifi_doppler.data.csi_to_sharp_doppler_dataset import CsiToSharpDopplerDataset
    from wifi_doppler.data.prepared_csi_doppler_dataset import PreparedCsiToSharpDopplerDataset

    data = config["data"]
    common = {
        "doppler_window_size": data["doppler_window_size"],
        "window_stride": data["window_stride"],
        "split_guard": data["split_guard"],
        "doppler_start": data["doppler_start"],
        "doppler_sample_length": data["doppler_sample_length"],
        "doppler_sliding": data["doppler_sliding"],
        "num_subcarriers": data["num_subcarriers"],
        "subcarrier_sampling": data["subcarrier_sampling"],
        "num_subcarrier_views": data["num_subcarrier_views"],
        "subcarrier_seed": data["subcarrier_seed"],
        "target_transform": data["target_transform"],
        "cache_raw": False,
        "cache_doppler": False,
    }
    if data.get("storage", "source") == "memmap":
        dataset_cls = PreparedCsiToSharpDopplerDataset
        storage = {"prepared_root": data["prepared_root"]}
    else:
        dataset_cls = CsiToSharpDopplerDataset
        storage = {"raw_root": data["raw_root"], "doppler_root": data["doppler_root"]}
    datasets = {}
    for name in split_names:
        split = data["splits"][name]
        datasets[name] = dataset_cls(
            scenarios=split["scenarios"],
            split=tuple(split["interval"]),
            **storage,
            **common,
        )
        if len(datasets[name]) == 0:
            raise ValueError(f"Configured split {name!r} contains no windows.")
    return datasets


def dataset_metadata(datasets: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "windows": len(dataset),
            "recordings": len(dataset.traces),
            "pairing": dataset.pairing_report(),
        }
        for name, dataset in datasets.items()
    }


def prefix_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    return {f"{prefix}/{name}": float(value) for name, value in metrics.items()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def save_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")


def initialize_wandb(config: dict[str, Any], run_dir: Path, resume_id: str | None):
    settings = config["wandb"]
    if settings["mode"] == "disabled":
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B logging is enabled but wandb is not installed. Install requirements.txt "
            "or set wandb.mode=disabled."
        ) from exc

    return wandb.init(
        project=settings["project"],
        entity=settings["entity"],
        name=settings["name"] or config["run"]["id"],
        id=resume_id,
        resume="allow" if resume_id else None,
        mode=settings["mode"],
        config=config,
        dir=str(run_dir / "training"),
        save_code=True,
    )


def log_examples(wandb_run, split_name: str, examples: list[dict[str, Any]], global_step: int) -> None:
    if wandb_run is None or not examples:
        return
    import matplotlib.pyplot as plt
    import wandb

    images = []
    for example in examples:
        target = example["target"].numpy()
        prediction = example["prediction"].numpy()
        num_antennas = target.shape[0]
        fig, axes = plt.subplots(
            num_antennas,
            3,
            figsize=(12, 3 * num_antennas),
            constrained_layout=True,
            squeeze=False,
        )
        for antenna in range(num_antennas):
            shared_min = float(min(target[antenna].min(), prediction[antenna].min()))
            shared_max = float(max(target[antenna].max(), prediction[antenna].max()))
            values = (target[antenna], prediction[antenna], np.abs(prediction[antenna] - target[antenna]))
            titles = ("target", "prediction", "absolute error")
            for column, (value, title) in enumerate(zip(values, titles)):
                kwargs = {} if column == 2 else {"vmin": shared_min, "vmax": shared_max}
                axes[antenna, column].imshow(value, aspect="auto", origin="lower", **kwargs)
                axes[antenna, column].set_title(f"antenna {antenna}: {title}")
                axes[antenna, column].set_xlabel("Doppler bin")
                axes[antenna, column].set_ylabel("time")
        fig.suptitle(example["filename"])
        images.append(wandb.Image(fig, caption=example["filename"]))
        plt.close(fig)
    wandb_run.log({f"{split_name}/examples": images, "global_step": global_step})


def log_model_artifact(wandb_run, path: Path, *, kind: str, metadata: dict[str, Any]) -> None:
    if wandb_run is None:
        return
    import wandb

    artifact = wandb.Artifact(
        name=f"{wandb_run.id}-{kind}",
        type="model",
        metadata=metadata,
    )
    artifact.add_file(str(path))
    wandb_run.log_artifact(artifact, aliases=[kind])


def load_inference_state(path: Path, model: torch.nn.Module, device: torch.device) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def normalized_resume_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = deepcopy(config)
    normalized["data"].pop("storage", None)
    normalized["data"].pop("prepared_root", None)
    for key in (
        "epochs",
        "log_every_steps",
        "validation_examples",
        "prefetch_batches",
        "pin_memory",
        "cuda_prefetch",
        "cudnn_benchmark",
        "allow_tf32",
    ):
        normalized["training"].pop(key, None)
    normalized.pop("wandb", None)
    normalized["run"].pop("id", None)
    return normalized


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.models.csi_to_doppler import count_trainable_parameters
    from wifi_doppler.models.csi_to_doppler_builders import (
        build_csi_to_doppler_model,
        csi_to_doppler_model_metadata,
    )
    from wifi_doppler.training.distillation import (
        iter_recording_batches,
        load_training_checkpoint,
        move_batches_to_device,
        prefetch_batches,
        run_distillation_epoch,
        save_inference_checkpoint,
        save_training_checkpoint,
    )

    config = load_resolved_config(args.config.resolve(), args.overrides, project_root)
    seed = int(config["run"]["seed"])
    seed_everything(seed)
    device = select_device(str(config["device"]))
    cudnn_enabled = configure_cuda_convolution_backend(device)
    amp_enabled = bool(config["training"]["amp"] and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = bool(
            cudnn_enabled and config["training"].get("cudnn_benchmark", True)
        )
        allow_tf32 = bool(config["training"].get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")

    resume_preview = None
    resume_path = args.resume.resolve() if args.resume else None
    if resume_path is not None:
        resume_preview = torch.load(resume_path, map_location="cpu", weights_only=False)
        if normalized_resume_config(config) != normalized_resume_config(resume_preview["config"]):
            raise ValueError(
                "The resume configuration differs from the checkpoint. Only runtime performance/logging "
                "settings, training.epochs, W&B settings, and run.id may change."
            )
        run_id = str(resume_preview["config"]["run"]["id"])
    else:
        run_id = config["run"]["id"] or datetime.now().strftime("csi-doppler-%Y%m%d-%H%M%S")
    config["run"]["id"] = run_id

    run_dir = Path(config["run"]["output_root"]) / run_id
    training_dir = run_dir / "training"
    if resume_path is None and run_dir.exists():
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    training_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(training_dir / "resolved_config.yaml", config)

    backend = f", cuDNN: {cudnn_enabled}" if device.type == "cuda" else ""
    print(f"device: {device} (AMP: {amp_enabled}{backend})")
    print("building paired datasets...")
    datasets = build_datasets(config, ("train", "source_val", "target_val"))
    data_metadata = dataset_metadata(datasets)
    save_json(training_dir / "pairing_reports.json", data_metadata)
    for name, metadata in data_metadata.items():
        print(f"{name}: {metadata['recordings']} recordings, {metadata['windows']} windows")

    model = build_csi_to_doppler_model(config["model"]).to(device)
    model_key, model_builder = csi_to_doppler_model_metadata(config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["training"]["learning_rate"]))
    scaler = make_grad_scaler(amp_enabled)
    start_epoch = 1
    global_step = 0
    best_metric = float("inf")
    patience_counter = 0
    history: list[dict[str, Any]] = []
    resume_wandb_id = None
    if resume_path is not None:
        checkpoint = load_training_checkpoint(
            resume_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_metric = float(checkpoint["best_metric"])
        patience_counter = int(checkpoint["patience_counter"])
        history = list(checkpoint["history"])
        resume_wandb_id = checkpoint.get("wandb_run_id")
        print(f"resumed {resume_path} at epoch {start_epoch}, step {global_step}")

    wandb_run = initialize_wandb(config, run_dir, resume_wandb_id)
    if wandb_run is not None:
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("*", step_metric="global_step")
        if config["wandb"]["watch"]:
            wandb_run.watch(
                model,
                log=config["wandb"]["watch"],
                log_freq=int(config["wandb"]["watch_log_frequency"]),
            )
        wandb_run.summary["model/trainable_parameters"] = count_trainable_parameters(model)
        wandb_run.config.update({"dataset_metadata": data_metadata}, allow_val_change=True)
        for split_name, metadata in data_metadata.items():
            wandb_run.summary[f"data/{split_name}_windows"] = metadata["windows"]

    run_record = {
        "model_run_id": run_id,
        "model_key": model_key,
        "representation": "raw_csi_to_sharp_doppler",
        "builder": model_builder,
        "training_objective": config["training"]["loss"],
        "device": str(device),
        "wandb_run_id": wandb_run.id if wandb_run is not None else resume_wandb_id,
        "status": "running",
        "updated_at": utc_now(),
    }
    save_json(run_dir / "run.json", run_record)

    batch_size = int(config["training"]["batch_size"])
    log_every = int(config["training"]["log_every_steps"])
    max_examples = int(config["training"]["validation_examples"])
    patience = int(config["training"]["early_stopping_patience"])
    min_delta = float(config["training"]["early_stopping_min_delta"])
    early_stopping_metric = str(config["training"].get("early_stopping_metric", "mse"))
    loss_options = config["training"].get("loss_options", {})
    latest_path = training_dir / "latest.pt"
    best_path = run_dir / "model.pt"

    def batches_for(split_name: str, *, shuffle: bool, batch_seed: int):
        host_batches = iter_recording_batches(
            datasets[split_name],
            batch_size=batch_size,
            shuffle=shuffle,
            seed=batch_seed,
            recordings_per_batch=(
                int(config["training"].get("recordings_per_batch", 1)) if shuffle else 1
            ),
        )
        host_batches = prefetch_batches(
            host_batches,
            max_prefetch=int(config["training"].get("prefetch_batches", 0)),
            pin_memory=bool(config["training"].get("pin_memory", True) and device.type == "cuda"),
        )
        return move_batches_to_device(
            host_batches,
            device=device,
            cuda_prefetch=bool(config["training"].get("cuda_prefetch", True)),
        )

    try:
        for epoch in range(start_epoch, int(config["training"]["epochs"]) + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            def log_batch(step: int, metrics: dict[str, float]) -> None:
                if wandb_run is not None and step % log_every == 0:
                    wandb_run.log({**prefix_metrics("train_batch", metrics), "global_step": step})

            train_result = run_distillation_epoch(
                model,
                batches_for("train", shuffle=True, batch_seed=seed + epoch),
                device=device,
                optimizer=optimizer,
                scaler=scaler,
                amp_enabled=amp_enabled,
                loss_name=config["training"]["loss"],
                loss_options=loss_options,
                global_step=global_step,
                batch_callback=log_batch if wandb_run is not None else None,
                batch_callback_every=log_every,
            )
            global_step = train_result.global_step

            evaluations = {}
            for split_name in ("source_val", "target_val"):
                evaluations[split_name] = run_distillation_epoch(
                    model,
                    batches_for(split_name, shuffle=False, batch_seed=seed),
                    device=device,
                    amp_enabled=amp_enabled,
                    loss_name=config["training"]["loss"],
                    loss_options=loss_options,
                    global_step=global_step,
                    max_examples=max_examples,
                )

            target_metric = evaluations["target_val"].metrics[early_stopping_metric]
            improved = target_metric < best_metric - min_delta
            if improved:
                best_metric = target_metric
                patience_counter = 0
            else:
                patience_counter += 1

            epoch_record = {
                "epoch": epoch,
                "global_step": global_step,
                "train": train_result.metrics,
                "source_val": evaluations["source_val"].metrics,
                "target_val": evaluations["target_val"].metrics,
                f"best_target_val_{early_stopping_metric}": best_metric,
                "patience_counter": patience_counter,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if device.type == "cuda":
                epoch_record["gpu_peak_memory_bytes"] = int(torch.cuda.max_memory_allocated(device))
            history.append(epoch_record)
            save_json(training_dir / "history.json", history)

            if improved:
                save_inference_checkpoint(
                    best_path,
                    model=model,
                    epoch=epoch,
                    metrics=prefix_metrics("target_val", evaluations["target_val"].metrics),
                    config=config,
                )
            save_training_checkpoint(
                latest_path,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                epoch=epoch,
                global_step=global_step,
                best_metric=best_metric,
                patience_counter=patience_counter,
                config=config,
                history=history,
                wandb_run_id=wandb_run.id if wandb_run is not None else resume_wandb_id,
            )

            logged = {
                "epoch": epoch,
                "global_step": global_step,
                "learning_rate": optimizer.param_groups[0]["lr"],
                f"early_stopping/best_target_val_{early_stopping_metric}": best_metric,
                "early_stopping/patience_counter": patience_counter,
                **prefix_metrics("train", train_result.metrics),
                **prefix_metrics("source_val", evaluations["source_val"].metrics),
                **prefix_metrics("target_val", evaluations["target_val"].metrics),
            }
            if "gpu_peak_memory_bytes" in epoch_record:
                logged["system/gpu_peak_memory_bytes"] = epoch_record["gpu_peak_memory_bytes"]
            if wandb_run is not None:
                wandb_run.log(logged)
                log_examples(wandb_run, "source_val", evaluations["source_val"].examples, global_step)
                log_examples(wandb_run, "target_val", evaluations["target_val"].examples, global_step)
                log_model_artifact(
                    wandb_run,
                    latest_path,
                    kind="latest",
                    metadata={"epoch": epoch, f"target_val_{early_stopping_metric}": target_metric},
                )
                if improved:
                    log_model_artifact(
                        wandb_run,
                        best_path,
                        kind="best",
                        metadata={"epoch": epoch, f"target_val_{early_stopping_metric}": target_metric},
                    )

            print(
                f"epoch {epoch}: train_loss={train_result.metrics['loss']:.6g} "
                f"source_val_loss={evaluations['source_val'].metrics['loss']:.6g} "
                f"target_val_loss={evaluations['target_val'].metrics['loss']:.6g} "
                f"target_val_mse={evaluations['target_val'].metrics['mse']:.6g} "
                f"best_{early_stopping_metric}={best_metric:.6g}"
            )
            if patience_counter >= patience:
                print(f"early stopping after {patience_counter} evaluations without improvement")
                break

        if not best_path.exists():
            raise RuntimeError(f"Best checkpoint was not created: {best_path}")
        best_checkpoint = load_inference_state(best_path, model, device)
        datasets.update(build_datasets(config, ("target_test",)))
        data_metadata.update(dataset_metadata({"target_test": datasets["target_test"]}))
        save_json(training_dir / "pairing_reports.json", data_metadata)
        if wandb_run is not None:
            wandb_run.summary["data/target_test_windows"] = data_metadata["target_test"]["windows"]
            wandb_run.config.update({"dataset_metadata": data_metadata}, allow_val_change=True)
        test_result = run_distillation_epoch(
            model,
            batches_for("target_test", shuffle=False, batch_seed=seed),
            device=device,
            amp_enabled=amp_enabled,
            loss_name=config["training"]["loss"],
            loss_options=loss_options,
            global_step=global_step,
            max_examples=max_examples,
        )
        final_results = {
            "best_epoch": int(best_checkpoint["epoch"]),
            f"best_target_val_{early_stopping_metric}": best_metric,
            "target_test": test_result.metrics,
        }
        save_json(training_dir / "final_test.json", final_results)
        if wandb_run is not None:
            wandb_run.log({**prefix_metrics("target_test", test_result.metrics), "global_step": global_step})
            log_examples(wandb_run, "target_test", test_result.examples, global_step)
            for key, value in final_results["target_test"].items():
                wandb_run.summary[f"target_test/{key}"] = value
            wandb_run.summary["best_epoch"] = final_results["best_epoch"]
            wandb_run.summary[f"best_target_val_{early_stopping_metric}"] = best_metric

        run_record.update(
            {
                "status": "completed",
                "best_epoch": final_results["best_epoch"],
                f"best_target_val_{early_stopping_metric}": best_metric,
                "target_test": final_results["target_test"],
                "updated_at": utc_now(),
            }
        )
        save_json(run_dir / "run.json", run_record)
        print(f"target_test_mse={test_result.metrics['mse']:.6g}")
        print(f"run directory: {run_dir}")
    except Exception:
        run_record.update({"status": "failed", "updated_at": utc_now()})
        save_json(run_dir / "run.json", run_record)
        raise
    finally:
        for dataset in datasets.values():
            dataset.clear_cache()
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
