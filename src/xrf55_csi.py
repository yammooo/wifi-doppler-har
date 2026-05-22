"""Intel 5300 CSI parser helpers for XRF55 raw Wi-Fi .dat files."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat


@dataclass(frozen=True)
class CsiRecord:
    """One Intel 5300 CSI record."""

    timestamp_low: int
    bfee_count: int
    nrx: int
    ntx: int
    rssi: tuple[int, int, int]
    noise: int
    agc: int
    antenna_sel: int
    payload_len: int
    fake_rate_n_flags: int
    csi: np.ndarray


def _int8(value: int) -> int:
    return value - 256 if value >= 128 else value


def _read_signed_byte_from_bits(buf: bytes, bit_index: int) -> int:
    byte_index = bit_index // 8
    shift = bit_index % 8
    raw = (buf[byte_index] >> shift) | (buf[byte_index + 1] << (8 - shift))
    value = raw & 0xFF
    return _int8(value)


def _parse_bfee_payload(payload: bytes) -> CsiRecord:
    nrx = payload[8]
    ntx = payload[9]
    csi_buf = payload[20:]

    csi = np.zeros((30, nrx, ntx), dtype=np.complex64)
    bit_index = 0
    for subcarrier_idx in range(30):
        bit_index += 3
        for rx_idx in range(nrx):
            for tx_idx in range(ntx):
                real = _read_signed_byte_from_bits(csi_buf, bit_index)
                imag = _read_signed_byte_from_bits(csi_buf, bit_index + 8)
                csi[subcarrier_idx, rx_idx, tx_idx] = complex(real, imag)
                bit_index += 16

    return CsiRecord(
        timestamp_low=int.from_bytes(payload[0:4], "little"),
        bfee_count=int.from_bytes(payload[4:6], "little"),
        nrx=nrx,
        ntx=ntx,
        rssi=tuple(_int8(payload[i]) for i in (10, 11, 12)),
        noise=_int8(payload[13]),
        agc=payload[14],
        antenna_sel=payload[15],
        payload_len=int.from_bytes(payload[16:18], "little"),
        fake_rate_n_flags=int.from_bytes(payload[18:20], "little"),
        csi=csi,
    )


def read_intel5300_dat(path: str | Path, strict: bool = True) -> list[CsiRecord]:
    """Read Intel 5300 CSI Tool .dat records from an XRF55 raw Wi-Fi file."""
    data = Path(path).read_bytes()
    records: list[CsiRecord] = []
    pos = 0

    while pos + 3 <= len(data):
        field_len = int.from_bytes(data[pos : pos + 2], "big")
        code = data[pos + 2]
        record_end = pos + 2 + field_len

        if field_len <= 1 or record_end > len(data):
            if not strict:
                break
            raise ValueError(f"Invalid record at byte {pos}: field_len={field_len}")

        payload = data[pos + 3 : record_end]
        if code == 0xBB:
            records.append(_parse_bfee_payload(payload))

        pos = record_end

    if strict and pos != len(data):
        raise ValueError(f"Trailing bytes after parsing {path}: parsed={pos}, total={len(data)}")

    return records


def read_intel5300_mat(path: str | Path) -> list[CsiRecord]:
    """Read already-parsed Intel 5300 CSI records from XRF55 .mat files."""
    mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    data = np.atleast_1d(mat["data"])
    records: list[CsiRecord] = []

    for entry in data:
        csi = np.asarray(entry.csi)
        if csi.shape == (entry.Nrx, 30):
            csi = csi.T[:, :, None]
        elif csi.shape == (entry.Ntx, entry.Nrx, 30):
            csi = np.moveaxis(csi, -1, 0).transpose(0, 2, 1)
        else:
            raise ValueError(f"Unexpected CSI shape in {path}: {csi.shape}")

        records.append(
            CsiRecord(
                timestamp_low=int(entry.timestamp_low),
                bfee_count=int(entry.bfee_count),
                nrx=int(entry.Nrx),
                ntx=int(entry.Ntx),
                rssi=(int(entry.rssi_a), int(entry.rssi_b), int(entry.rssi_c)),
                noise=int(entry.noise),
                agc=int(entry.agc),
                antenna_sel=0,
                payload_len=0,
                fake_rate_n_flags=int(entry.rate),
                csi=csi.astype(np.complex64, copy=False),
            )
        )

    return records


def read_xrf55_wifi_file(path: str | Path, strict: bool = True) -> list[CsiRecord]:
    """Read one XRF55 Wi-Fi CSI file, dispatching by extension."""
    path = Path(path)
    if path.suffix == ".dat":
        return read_intel5300_dat(path, strict=strict)
    if path.suffix == ".mat":
        return read_intel5300_mat(path)
    raise ValueError(f"Unsupported XRF55 Wi-Fi file extension: {path.suffix}")


def records_to_csi_array(records: list[CsiRecord]) -> np.ndarray:
    """Stack records as [packet, subcarrier, rx, tx]."""
    if not records:
        raise ValueError("Cannot stack an empty CSI record list")
    return np.stack([record.csi for record in records], axis=0)
