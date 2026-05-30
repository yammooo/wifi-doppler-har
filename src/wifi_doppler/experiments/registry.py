from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from wifi_doppler.experiments.artifacts import save_json


SCHEMA_VERSION = 1
REQUIRED_ENVELOPE_KEYS = (
    "schema_version",
    "record_type",
    "record_id",
    "created_at",
    "source",
    "model",
    "dataset",
    "protocol",
    "metrics",
    "artifacts",
)


def registry_root(project_root: str | Path) -> Path:
    return Path(project_root) / "experiments" / "registry"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def record_id(record_type: str, stem: str | None = None) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_type = _safe_slug(record_type.replace(".", "_"))
    safe_stem = _safe_slug(stem) if stem else "record"
    return f"{timestamp}_{safe_type}_{safe_stem}_{uuid4().hex[:8]}"


def checkpoint_fingerprint(path: str | Path | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "exists": False, "size_bytes": None, "mtime": None, "sha256_16": None}

    checkpoint_path = Path(path).resolve()
    if not checkpoint_path.exists():
        return {
            "path": str(checkpoint_path),
            "exists": False,
            "size_bytes": None,
            "mtime": None,
            "sha256_16": None,
        }

    stat = checkpoint_path.stat()
    return {
        "path": str(checkpoint_path),
        "exists": True,
        "size_bytes": stat.st_size,
        "mtime": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "sha256_16": _file_sha256_prefix(checkpoint_path),
    }


def make_record(
    *,
    record_type: str,
    source: dict[str, Any],
    model: dict[str, Any] | None = None,
    dataset: dict[str, Any] | None = None,
    protocol: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    artifacts: dict[str, Any] | None = None,
    stem: str | None = None,
    explicit_record_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "schema_version": SCHEMA_VERSION,
        "record_type": record_type,
        "record_id": explicit_record_id or record_id(record_type, stem),
        "created_at": utc_now(),
        "source": source,
        "model": model or {},
        "dataset": dataset or {},
        "protocol": protocol or {},
        "metrics": metrics or {},
        "artifacts": artifacts or {},
    }
    if extra:
        record.update(extra)
    validate_record(record)
    return record


def validate_record(record: dict[str, Any]) -> None:
    missing = [key for key in REQUIRED_ENVELOPE_KEYS if key not in record]
    if missing:
        raise ValueError(f"Record is missing required keys: {missing}")
    if record["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {record['schema_version']}")
    for key in ("record_type", "record_id"):
        if not isinstance(record[key], str) or not record[key]:
            raise ValueError(f"{key} must be a non-empty string.")
    for key in ("source", "model", "dataset", "protocol", "metrics", "artifacts"):
        if not isinstance(record[key], dict):
            raise ValueError(f"{key} must be a dictionary.")


def record_dir(root: str | Path, record: dict[str, Any]) -> Path:
    validate_record(record)
    record_type = record["record_type"]
    if record_type == "evaluation.kshot":
        return (
            registry_root(root)
            / "evaluations"
            / _required_slug(record, "model", "model_run_id")
            / _required_slug(record, "protocol", "name")
        )
    if record_type == "projection.umap":
        return (
            registry_root(root)
            / "projections"
            / _required_slug(record, "model", "model_run_id")
            / _required_slug(record, "protocol", "name")
        )
    if record_type == "comparison.figure":
        return registry_root(root) / "comparisons" / _required_slug(record, "protocol", "name")
    if record_type == "model.training":
        return registry_root(root) / "models" / _required_slug(record, "model", "model_run_id")
    return registry_root(root) / "other" / _safe_slug(record["record_id"])


def record_path(root: str | Path, record: dict[str, Any]) -> Path:
    return record_dir(root, record) / "record.json"


def save_record(root: str | Path, record: dict[str, Any]) -> Path:
    validate_record(record)
    return save_json(record_path(root, record), record)


def load_record(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        record = json.load(f)
    validate_record(record)
    return record


def load_records(paths: list[str | Path]) -> list[dict[str, Any]]:
    return [load_record(path) for path in paths]


def _required_slug(record: dict[str, Any], section: str, key: str) -> str:
    value = record.get(section, {}).get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{record['record_type']} requires {section}.{key}.")
    return _safe_slug(value)


def _file_sha256_prefix(path: Path, prefix: int = 16) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()[:prefix]


def _safe_slug(value: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    safe = "_".join(part for part in safe.split("_") if part)
    return safe or "record"
