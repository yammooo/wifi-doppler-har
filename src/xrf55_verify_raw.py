"""Smoke checks for XRF55 raw Wi-Fi CSI files."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from xrf55_csi import read_xrf55_wifi_file, records_to_csi_array


def summarize_file(path: Path) -> None:
    records = read_xrf55_wifi_file(path, strict=False)
    csi = records_to_csi_array(records)
    timestamps = np.array([record.timestamp_low for record in records], dtype=np.int64)
    diffs = np.diff(timestamps)
    amp = np.abs(csi)
    phase = np.angle(csi)

    print(f"file: {path}")
    print(f"records: {len(records)}")
    print(f"csi_shape [packet, subcarrier, rx, tx]: {csi.shape}")
    print(f"nrx: {sorted({record.nrx for record in records})}")
    print(f"ntx: {sorted({record.ntx for record in records})}")
    print(f"amp min/mean/max: {amp.min():.3f} / {amp.mean():.3f} / {amp.max():.3f}")
    print(f"phase min/mean/max: {phase.min():.3f} / {phase.mean():.3f} / {phase.max():.3f}")
    if len(diffs):
        print(
            "timestamp diff us min/median/max: "
            f"{diffs.min()} / {int(np.median(diffs))} / {diffs.max()}"
        )


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        default="data/XRF55_rawdata/WiFi/Scene_1/lb/01/01_01_01.dat",
        help="Path to one XRF55 raw Wi-Fi .dat file.",
    )
    args = parser.parse_args()
    summarize_file(Path(args.path))


if __name__ == "__main__":
    main()
