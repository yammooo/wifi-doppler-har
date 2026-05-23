"""Shared Doppler utilities for XRF55 raw Wi-Fi experiments."""

from pathlib import Path
import math as mt

import numpy as np
from scipy.fftpack import fft, fftshift
from scipy.signal.windows import hann


ACTION_NAMES = {
    "01": "carrying weight",
    "02": "mopping the floor",
    "03": "cutting",
    "04": "wearing hat",
    "05": "using a phone",
    "06": "throw something",
    "07": "put something on the table",
    "08": "put on clothing",
    "09": "picking",
    "10": "drinking",
    "11": "smoking",
    "12": "eating",
    "13": "brushing teeth",
    "14": "blow dry hair",
    "15": "brush hair",
    "16": "shake hands",
    "17": "hugging",
    "18": "hand something to someone",
    "19": "kick someone",
    "20": "hit someone with something",
    "21": "choke someone's neck",
    "22": "push someone",
    "23": "body weight squats",
    "24": "tai chi",
    "25": "boxing",
    "26": "weightlifting",
    "27": "hula hooping",
    "28": "jump rope",
    "29": "jumping jack",
    "30": "high leg lift",
    "31": "waving",
    "32": "clap hands",
    "33": "fall on the floor",
    "34": "jumping",
    "35": "running",
    "36": "sitting down",
    "37": "standing up",
    "38": "turning",
    "39": "walking",
    "40": "stretch oneself",
    "41": "pat on shoulder",
    "42": "playing erhu",
    "43": "playing ukulele",
    "44": "playing drum",
    "45": "stomping",
    "46": "shaking head",
    "47": "nodding",
    "48": "draw circles",
    "49": "draw a cross",
    "50": "pushing",
    "51": "pulling",
    "52": "swipe left",
    "53": "swipe right",
    "54": "swipe up",
    "55": "swipe down",
}


def temporal_fft_profile(
    matrix: np.ndarray,
    sample_length: int = 51,
    sliding: int = 1,
    n_fft: int = 100,
    noise_level: float = -2,
) -> np.ndarray:
    """Compute SHARP-style temporal FFT power, summing over feature columns."""
    profiles = []
    for start in range(0, matrix.shape[0] - sample_length, sliding):
        cut = np.nan_to_num(matrix[start:start + sample_length])
        windowed = cut * np.expand_dims(hann(sample_length), axis=-1)
        spectrum = fftshift(fft(windowed, n=n_fft, axis=0), axes=0)
        power = np.abs(spectrum * np.conj(spectrum))
        profiles.append(np.sum(power, axis=1))

    profile = np.asarray(profiles)
    profile_max = np.max(profile, axis=1, keepdims=True)
    profile_max[profile_max == 0] = 1
    profile = profile / profile_max
    profile[profile < mt.pow(10, noise_level)] = mt.pow(10, noise_level)
    return profile


def find_trial(
    root: str | Path,
    scene: str,
    receiver: str,
    subject: str,
    action: str,
    repetition: str,
) -> Path:
    path = Path(root) / scene / receiver / subject / f"{subject}_{action}_{repetition}.dat"
    if path.exists():
        return path

    mat_path = path.with_suffix(".mat")
    if mat_path.exists():
        return mat_path

    raise FileNotFoundError(path)


def mask_center_bins(doppler: np.ndarray, bins: int) -> np.ndarray:
    if bins <= 0:
        return doppler

    masked = doppler.copy()
    center = masked.shape[1] // 2
    start = max(0, center - bins)
    end = min(masked.shape[1], center + bins + 1)
    masked[:, start:end] = np.min(masked)
    return masked
