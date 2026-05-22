"""Dataset indexing helpers for XRF55 raw Wi-Fi CSI."""

from collections import Counter
from pathlib import Path
import re

from recordings import XRF55RawRecording


XRF55_FILE_PATTERN = re.compile(
    r"^(?P<subject>\d{2})_(?P<action>\d{2})_(?P<repetition>\d{2})\.(?P<ext>dat|mat)$"
)


def scan_xrf55_raw_wifi(root: str | Path = "data/XRF55_rawdata/WiFi") -> list[XRF55RawRecording]:
    """Scan XRF55 raw Wi-Fi files into typed recording objects."""
    root = Path(root)
    recordings: list[XRF55RawRecording] = []

    for path in sorted(root.glob("Scene_*/*/*/*")):
        if not path.is_file():
            continue

        match = XRF55_FILE_PATTERN.match(path.name)
        if not match:
            continue

        receiver = path.parent.parent.name
        scene = path.parent.parent.parent.name
        subject_dir = path.parent.name
        parts = match.groupdict()
        if subject_dir != parts["subject"]:
            raise ValueError(f"Subject directory and filename disagree: {path}")

        recordings.append(
            XRF55RawRecording(
                scene=scene,
                receiver=receiver,
                subject=parts["subject"],
                action=parts["action"],
                repetition=parts["repetition"],
                path=path,
            )
        )

    return recordings


if __name__ == "__main__":
    recordings = scan_xrf55_raw_wifi()
    print(f"Raw Wi-Fi recordings: {len(recordings)}")
    print("By scene/receiver:")
    for key, count in sorted(Counter((r.scene, r.receiver) for r in recordings).items()):
        print(f"  {key[0]} {key[1]}: {count}")

    first = recordings[0]
    print(f"First recording: {first.recording_id}")
    print(f"Path: {first.path}")
    print(f"CSI shape: {first.load_csi().shape}")
