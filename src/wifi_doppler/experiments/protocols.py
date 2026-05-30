from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wifi_doppler.data.doppler_dataset import DopplerWindowDataset
from wifi_doppler.data.raw_csi_dataset import RawCsiWindowDataset


DEFAULT_PERSONS = ("p03", "p05", "p06", "p07", "p08", "p09", "p10", "p11", "p12", "p13")
SOURCE_DOMAINS = ("PI-1a", "PI-2a", "PI-3a")
TARGET_DOMAINS = ("PI-4a",)
ALL_DOMAINS = SOURCE_DOMAINS + TARGET_DOMAINS
DEFAULT_K_VALUES = (1, 3, 5, 10, 25, 50, 100)
DEFAULT_ENROLLMENT_SPLIT = (0.0, 0.6)
DEFAULT_QUERY_SPLIT = (0.6, 0.8)


@dataclass(frozen=True)
class KShotProtocol:
    name: str
    enrollment_domains: tuple[str, ...]
    query_domains: tuple[str, ...]
    enrollment_split: tuple[float, float] = DEFAULT_ENROLLMENT_SPLIT
    query_split: tuple[float, float] = DEFAULT_QUERY_SPLIT

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task": "few_shot_person_identification",
            "enrollment_domains": list(self.enrollment_domains),
            "query_domains": list(self.query_domains),
            "enrollment_split": list(self.enrollment_split),
            "query_split": list(self.query_split),
        }


def parse_kshot_protocol(name: str) -> KShotProtocol:
    if name == "mixed_source":
        return KShotProtocol(
            name=name,
            enrollment_domains=SOURCE_DOMAINS,
            query_domains=SOURCE_DOMAINS,
        )

    prefix = "same_domain_"
    if name.startswith(prefix):
        domain = name.removeprefix(prefix)
        if domain not in ALL_DOMAINS:
            raise ValueError(f"Unknown same-domain protocol {name!r}; expected one of {ALL_DOMAINS}.")
        return KShotProtocol(name=name, enrollment_domains=(domain,), query_domains=(domain,))

    raise ValueError(
        f"Unknown protocol {name!r}. Use 'mixed_source' or 'same_domain_PI-1a'..."
    )


def default_protocol_names() -> list[str]:
    return ["mixed_source"] + [f"same_domain_{domain}" for domain in ALL_DOMAINS]


def protocol_dataset_metadata(
    *,
    representation: str,
    data_root: str | Path,
    persons: tuple[str, ...] = DEFAULT_PERSONS,
    window_size: int = 340,
    window_stride: int = 30,
    split_guard: int = 31,
) -> dict[str, Any]:
    return {
        "representation": representation,
        "data_root": str(Path(data_root).resolve()),
        "persons": list(persons),
        "window_size": window_size,
        "window_stride": window_stride,
        "split_guard": split_guard,
    }


def build_kshot_datasets(
    *,
    project_root: str | Path,
    representation: str,
    protocol: KShotProtocol,
    persons: tuple[str, ...] = DEFAULT_PERSONS,
    window_size: int = 340,
    window_stride: int = 30,
    split_guard: int = 31,
):
    root = Path(project_root)
    if representation == "raw_csi":
        data_root = root / "data" / "raw_csi_traces_pi"
        dataset_cls = RawCsiWindowDataset
        kwargs = {"flatten_channels": True, "cache_traces": True}
    elif representation == "doppler":
        data_root = root / "data" / "doppler_traces_pi"
        dataset_cls = DopplerWindowDataset
        kwargs = {}
    else:
        raise ValueError(f"Unknown representation: {representation!r}")

    enrollment_dataset = dataset_cls(
        data_root,
        scenarios=list(protocol.enrollment_domains),
        split=protocol.enrollment_split,
        window_size=window_size,
        window_stride=window_stride,
        split_guard=split_guard,
        labels=persons,
        **kwargs,
    )
    query_dataset = dataset_cls(
        data_root,
        scenarios=list(protocol.query_domains),
        split=protocol.query_split,
        window_size=window_size,
        window_stride=window_stride,
        split_guard=split_guard,
        labels=persons,
        **kwargs,
    )
    metadata = protocol_dataset_metadata(
        representation=representation,
        data_root=data_root,
        persons=persons,
        window_size=window_size,
        window_stride=window_stride,
        split_guard=split_guard,
    )
    metadata.update(
        {
            "enrollment_windows": len(enrollment_dataset),
            "query_windows": len(query_dataset),
        }
    )
    return enrollment_dataset, query_dataset, metadata
