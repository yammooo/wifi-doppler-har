from pathlib import Path
import shutil
import zipfile


DEFAULT_DOPPLER_DRIVE_FILE_ID = "1vGHldHZVb9hQ1_YfFqZHqF8Ty1DQZEea"
EXPECTED_DOPPLER_SUBDIRS = (
    "S1a",
    "S1b",
    "S1c",
    "S2a",
    "S2b",
    "S3a",
    "S4a",
    "S4b",
    "S5a",
    "S6a",
    "S6b",
    "S7a",
)


def has_doppler_traces(path: Path) -> bool:
    """Return True when the expected SHARP Doppler trace layout is present."""
    path = Path(path)
    return path.exists() and all((path / subdir).is_dir() for subdir in EXPECTED_DOPPLER_SUBDIRS)


def download_doppler_zip(
    output_path: Path,
    drive_file_id: str = DEFAULT_DOPPLER_DRIVE_FILE_ID,
) -> Path:
    """Download the precomputed Doppler traces zip from Google Drive."""
    try:
        import gdown
    except ImportError as exc:
        raise ImportError("Install download dependency first: pip install gdown") from exc

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(id=drive_file_id, output=str(output_path), quiet=False, fuzzy=True)

    if not output_path.is_file():
        raise FileNotFoundError(f"Download did not create expected file: {output_path}")

    return output_path


def extract_doppler_zip(zip_path: Path, target_dir: Path, overwrite: bool = False) -> Path:
    """Extract doppler_traces.zip so target_dir contains S1a, S1b, ..., S7a."""
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    data_dir = target_dir.parent
    extract_tmp = data_dir / "_doppler_extract_tmp"

    if target_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Target already exists: {target_dir}")
        shutil.rmtree(target_dir)

    if extract_tmp.exists():
        shutil.rmtree(extract_tmp)
    extract_tmp.mkdir(parents=True)

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_tmp)

        extracted_doppler = extract_tmp / "doppler_traces"
        if not extracted_doppler.is_dir():
            matches = [path for path in extract_tmp.rglob("doppler_traces") if path.is_dir()]
            if not matches:
                raise FileNotFoundError("The zip file does not contain a doppler_traces folder.")
            extracted_doppler = matches[0]

        target_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted_doppler), str(target_dir))
    finally:
        if extract_tmp.exists():
            shutil.rmtree(extract_tmp)

    if not has_doppler_traces(target_dir):
        raise RuntimeError(f"Extraction finished, but expected folders are missing under {target_dir}")

    return target_dir


def ensure_doppler_traces(
    data_dir: Path = Path("data"),
    drive_file_id: str = DEFAULT_DOPPLER_DRIVE_FILE_ID,
    zip_name: str = "doppler_traces.zip",
) -> Path:
    """Ensure data/doppler_traces exists, downloading and extracting it when missing."""
    data_dir = Path(data_dir)
    doppler_dir = data_dir / "doppler_traces"

    if has_doppler_traces(doppler_dir):
        print(f"Doppler traces already available: {doppler_dir}")
        return doppler_dir

    zip_path = data_dir / zip_name
    print("Doppler traces missing. Downloading doppler_traces.zip...")
    download_doppler_zip(zip_path, drive_file_id=drive_file_id)

    print("Extracting into data/doppler_traces...")
    doppler_dir = extract_doppler_zip(zip_path, doppler_dir)
    print(f"Doppler traces ready: {doppler_dir}")
    return doppler_dir


if __name__ == "__main__":
    ensure_doppler_traces()
