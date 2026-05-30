from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


MODEL_ID_MAP = {
    "raw_csi_proto": "raw_csi_mixed_proto",
    "old_proto_featuremap": "doppler_featuremap_proto",
    "softmax_featuremap": "doppler_softmax_baseline",
    "pooled_proto": "doppler_pooled_head_proto",
    "flatten_mlp_proto": "doppler_flatten_mlp_proto",
}

MODEL_LABELS = {
    "raw_csi_mixed_proto": "Raw CSI mixed proto",
    "doppler_featuremap_proto": "Doppler feature-map proto",
    "doppler_softmax_baseline": "Doppler softmax baseline",
    "doppler_pooled_head_proto": "Doppler pooled-head proto",
    "doppler_flatten_mlp_proto": "Doppler flatten-MLP proto",
}


def add_src_to_path(project_root: Path) -> None:
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Migrate legacy K-shot JSONs into the canonical registry layout.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    add_src_to_path(project_root)

    from wifi_doppler.experiments.registry import save_record

    candidates: list[tuple[Path, dict[str, Any]]] = []
    for path in _legacy_json_paths(project_root):
        candidates.extend((path, record) for record in migrate_json(path, project_root))

    kept = _keep_latest_per_model_protocol(candidates)
    if not args.dry_run:
        for record in kept:
            save_record(project_root, record)

    print(f"legacy JSON files scanned: {len(_legacy_json_paths(project_root))}")
    print(f"K-shot registry records {'planned' if args.dry_run else 'written'}: {len(kept)}")
    print(f"duplicates collapsed: {sum(len(r.get('skipped_legacy_duplicates', [])) for r in kept)}")


def _legacy_json_paths(project_root: Path) -> list[Path]:
    registry_root = project_root / "experiments" / "registry"
    return [
        path
        for path in sorted((project_root / "experiments").rglob("*.json"))
        if not path.is_relative_to(registry_root)
    ]


def migrate_json(path: Path, project_root: Path) -> list[dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    records: list[dict[str, Any]] = []
    config = data.get("config", {}) if isinstance(data, dict) else {}
    if isinstance(data, dict) and "results" in data and isinstance(data["results"], dict):
        results = data["results"]
        checkpoints = config.get("checkpoints", {})
        if "mixed_source" in results:
            records.extend(_records_from_result_map(path, results["mixed_source"], config, "mixed_source", checkpoints))
        for domain, result_map in results.get("per_domain", {}).items():
            records.extend(_records_from_result_map(path, result_map, config, f"same_domain_{domain}", checkpoints))
    elif isinstance(data, dict) and "source_results" in data:
        records.extend(_records_from_result_map(path, data["source_results"], config, "mixed_source", config.get("checkpoints", {})))
    elif isinstance(data, dict) and "softmax_fewshot_results" in data:
        result_map = {
            "softmax_featuremap": data["softmax_fewshot_results"],
            "old_proto_featuremap": data.get("proto_fewshot_results", {}),
        }
        records.extend(_records_from_result_map(path, result_map, config, "same_domain_PI-4a", config.get("checkpoints", {})))
    elif isinstance(data, dict) and "raw_proto_fewshot_results" in data:
        result_map = {
            "raw_csi_proto": data["raw_proto_fewshot_results"],
            "old_proto_featuremap": data.get("doppler_featuremap_proto_fewshot_results", {}),
            "softmax_featuremap": data.get("doppler_softmax_fewshot_results", {}),
        }
        records.extend(_records_from_result_map(path, result_map, config, "same_domain_PI-4a", config.get("checkpoints", {})))
    return records


def _records_from_result_map(
    path: Path,
    result_map: dict[str, Any],
    config: dict[str, Any],
    protocol_name: str,
    checkpoints: dict[str, Any],
) -> list[dict[str, Any]]:
    from wifi_doppler.experiments.registry import make_record

    records = []
    for legacy_model_id, results in result_map.items():
        if not _looks_like_kshot_result(results):
            continue
        model_run_id = _canonical_model_id(legacy_model_id, path)
        if model_run_id is None:
            continue
        k_values = sorted(int(k) for k in results.keys())
        normalized = {}
        for k in k_values:
            value = results.get(str(k), results.get(k))
            normalized[str(k)] = {
                "mean": float(value["mean"]),
                "std": float(value["std"]),
                "trials": [float(v) for v in value["trials"]],
            }
        records.append(
            make_record(
                record_type="evaluation.kshot",
                stem=f"{model_run_id}_{protocol_name}",
                source={"mode": "migrated", "legacy_paths": [str(path.resolve())]},
                model={
                    "model_run_id": model_run_id,
                    "model_id": model_run_id,
                    "label": MODEL_LABELS.get(model_run_id, model_run_id),
                    "representation": "raw_csi" if model_run_id.startswith("raw_csi") else "doppler",
                    "checkpoint": {"path": str(checkpoints.get(legacy_model_id)) if checkpoints.get(legacy_model_id) else None},
                },
                dataset=_dataset_from_config(config),
                protocol={"name": protocol_name, "task": "few_shot_person_identification"},
                metrics={"k_values": k_values, "results": normalized},
                artifacts={"legacy_dir": str(path.parent.resolve())},
            )
        )
    return records


def _canonical_model_id(legacy_model_id: str, path: Path) -> str | None:
    if legacy_model_id == "raw_csi_proto" and _is_domain_cross_raw_csi_source(path):
        return None
    return MODEL_ID_MAP.get(legacy_model_id)


def _is_domain_cross_raw_csi_source(path: Path) -> bool:
    text = str(path).lower()
    return "domain_cross_raw_csi" in text or "20260529_193154" in text


def _keep_latest_per_model_protocol(candidates: list[tuple[Path, dict[str, Any]]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[tuple[Path, dict[str, Any]]]] = {}
    for source_path, record in candidates:
        key = (record["model"]["model_run_id"], record["protocol"]["name"])
        grouped.setdefault(key, []).append((source_path, record))

    kept = []
    for records in grouped.values():
        ordered = sorted(records, key=lambda item: (item[0].stat().st_mtime, item[1].get("created_at", "")), reverse=True)
        keep_path, keep = ordered[0]
        skipped = [str(path.resolve()) for path, _ in ordered[1:]]
        if skipped:
            keep["skipped_legacy_duplicates"] = skipped
            keep["source"]["legacy_paths"] = [str(keep_path.resolve())]
        kept.append(keep)
    return sorted(kept, key=lambda record: (record["model"]["model_run_id"], record["protocol"]["name"]))


def _looks_like_kshot_result(value: Any) -> bool:
    if not isinstance(value, dict) or not value:
        return False
    first = next(iter(value.values()))
    return isinstance(first, dict) and {"mean", "std", "trials"}.issubset(first)


def _dataset_from_config(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "train_scenarios": config.get("train_scenarios") or config.get("source_domains"),
        "target_scenarios": config.get("target_scenarios") or config.get("target_domains"),
        "domains": config.get("domains"),
        "persons": config.get("persons"),
        "window_size": config.get("window_size"),
        "window_stride": config.get("window_stride"),
    }


if __name__ == "__main__":
    main()
