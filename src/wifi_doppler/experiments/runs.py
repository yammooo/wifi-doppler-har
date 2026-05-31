from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import shutil
from pathlib import Path
from typing import Any

from wifi_doppler.experiments.artifacts import save_json


def runs_root(project_root: str | Path) -> Path:
    return Path(project_root) / "experiments" / "runs"


def run_dir(project_root: str | Path, model_run_id: str) -> Path:
    return runs_root(project_root) / model_run_id


def run_metadata_path(project_root: str | Path, model_run_id: str) -> Path:
    return run_dir(project_root, model_run_id) / "run.json"


def run_checkpoint_path(project_root: str | Path, model_run_id: str) -> Path:
    return run_dir(project_root, model_run_id) / "model.pt"


def training_dir(project_root: str | Path, model_run_id: str) -> Path:
    return run_dir(project_root, model_run_id) / "training"


def evaluation_path(project_root: str | Path, model_run_id: str, protocol_name: str) -> Path:
    return run_dir(project_root, model_run_id) / "evaluations" / f"{slug(protocol_name)}.json"


def projection_dir(project_root: str | Path, model_run_id: str) -> Path:
    return run_dir(project_root, model_run_id) / "projections"


def projection_record_path(project_root: str | Path, model_run_id: str, projection_name: str) -> Path:
    return projection_dir(project_root, model_run_id) / f"{slug(projection_name)}.json"


def projection_coordinates_path(project_root: str | Path, model_run_id: str, projection_name: str) -> Path:
    return projection_dir(project_root, model_run_id) / f"{slug(projection_name)}.npz"


def comparison_dir(project_root: str | Path, model_run_id: str, comparison_name: str) -> Path:
    return run_dir(project_root, model_run_id) / "comparisons" / slug(comparison_name)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    return "_".join(part for part in safe.split("_") if part) or "item"


def ensure_run(
    project_root: str | Path,
    *,
    model_run_id: str,
    label: str | None,
    model_key: str,
    representation: str,
    builder: str,
    checkpoint_path: str | Path | None = None,
    training_objective: str | None = None,
    episode_style: str | None = None,
    notes: str | None = None,
) -> Path:
    path = run_dir(project_root, model_run_id)
    path.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if checkpoint_path is not None:
        checkpoint = copy_checkpoint(project_root, model_run_id, checkpoint_path)

    metadata = {
        "model_run_id": model_run_id,
        "model_key": model_key,
        "representation": representation,
        "builder": builder,
        "training_objective": training_objective,
        "episode_style": episode_style,
        "notes": notes,
        "checkpoint": checkpoint_fingerprint(checkpoint or run_checkpoint_path(project_root, model_run_id)),
        "updated_at": utc_now(),
    }
    if label is not None:
        metadata["label"] = label
        metadata["short_label"] = label
    if checkpoint_path is not None:
        metadata["source_checkpoint"] = checkpoint_fingerprint(checkpoint_path)
    existing_path = run_metadata_path(project_root, model_run_id)
    if existing_path.exists():
        existing = load_json(existing_path)
        metadata = {**existing, **{key: value for key, value in metadata.items() if value is not None}}
    else:
        metadata.setdefault("label", model_run_id)
        metadata.setdefault("short_label", metadata["label"])
        metadata["created_at"] = metadata["updated_at"]
    save_json(existing_path, metadata)
    return path


def copy_checkpoint(project_root: str | Path, model_run_id: str, checkpoint_path: str | Path) -> Path:
    source = Path(checkpoint_path)
    if not source.exists():
        raise FileNotFoundError(source)
    target = run_checkpoint_path(project_root, model_run_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target


def copy_training_artifacts(
    project_root: str | Path,
    model_run_id: str,
    *,
    history_path: str | Path | None = None,
    curves_path: str | Path | None = None,
) -> dict[str, str]:
    copied = {}
    target_dir = training_dir(project_root, model_run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        copied["history"] = str(_copy_file(history_path, target_dir / "history.json").resolve())
    if curves_path is not None:
        copied["curves"] = str(_copy_file(curves_path, target_dir / "curves.png").resolve())
    return copied


def load_run(project_root: str | Path, model_run_id: str) -> dict[str, Any]:
    return load_json(run_metadata_path(project_root, model_run_id))


def refresh_run_checkpoint(project_root: str | Path, model_run_id: str) -> Path:
    path = run_metadata_path(project_root, model_run_id)
    metadata = load_json(path)
    metadata["checkpoint"] = checkpoint_fingerprint(run_checkpoint_path(project_root, model_run_id))
    metadata["updated_at"] = utc_now()
    save_json(path, metadata)
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    import json

    return json.loads(Path(path).read_text(encoding="utf-8"))


def checkpoint_fingerprint(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": None, "mtime": None, "sha256_16": None}

    checkpoint = Path(path).resolve()
    if not checkpoint.exists():
        return {"path": str(checkpoint), "exists": False, "size_bytes": None, "mtime": None, "sha256_16": None}

    stat = checkpoint.stat()
    return {
        "path": str(checkpoint),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256_16": _file_sha256_prefix(checkpoint),
    }


def _file_sha256_prefix(path: Path, prefix: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:prefix]


def _copy_file(source: str | Path, target: Path) -> Path:
    source_path = Path(source)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != target.resolve():
        shutil.copy2(source_path, target)
    return target
