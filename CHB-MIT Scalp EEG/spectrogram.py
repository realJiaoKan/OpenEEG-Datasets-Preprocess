"""
Postprocess stage (Spectrogram) — Convert the EEG data to spectrograms using STFT.
"""

import numpy as np
import os
from pathlib import Path
import torch
import shutil

from settings import PROCESSED_DATA_PATH, SAMPLE_RATE, REMOVE_BANDS, USE_LOG_POWER

IN_DIR = PROCESSED_DATA_PATH / "Interictal-Preictal" / "EEG"
OUT_DIR = PROCESSED_DATA_PATH / "Interictal-Preictal" / "Spectrogram"

OUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def process_patient(npz_path: Path, out_path: Path):
    # Load data from .npz file
    data = np.load(npz_path)
    X_time = torch.from_numpy(data["X"]).to(device=DEVICE, dtype=torch.float32)
    y = data["y"]
    N, C, T = X_time.shape

    # We mask some frequency bands to remove noise such as 60Hz power line noise.
    freqs = np.fft.rfftfreq(T, d=1.0 / SAMPLE_RATE)
    mask = np.ones_like(freqs, dtype=bool)
    mask[freqs == 0] = False
    for low, high in REMOVE_BANDS:
        mask[(freqs >= low) & (freqs <= high)] = False
    mask = torch.from_numpy(mask).to(device=DEVICE)

    X_flat = X_time.reshape(N * C, T)  # (N * C, T)

    # STFT
    n_fft = T
    hop_length = T
    win_length = T
    window = torch.hann_window(win_length, device=DEVICE)

    spec_c = torch.stft(
        X_flat,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        return_complex=True,
        center=False,
        pad_mode="reflect",
    )  # (N * C, freq_bins, 1)

    # Calculate the absolute value of the complex spectrum
    spec_flat = spec_c.abs()
    if USE_LOG_POWER:
        spec_flat = spec_flat.add(1e-8).log10().mul(20)

    spec = spec_flat.view(
        N, C, spec_flat.size(1), spec_flat.size(2)
    )  # (N, C, freq_bins, 1)

    # Remove masked frequencies
    spec = spec[:, :, mask, :]

    # Reshape
    spec = spec.view(N, 1, C, -1)  # (N, C as image, C as EEG, F as frequency bins)

    # Save to specified output path
    spec_np = spec.cpu().numpy()
    np.savez_compressed(out_path, X=spec_np, y=y)


def main():
    # Walk through input directory recursively
    for root, dirs, files in os.walk(IN_DIR):
        for filename in files:
            src = Path(root) / filename
            rel = src.relative_to(IN_DIR)
            dst = OUT_DIR / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if src.suffix == ".npz":
                process_patient(src, dst)
            else:
                # Copy non-npz files (e.g., stats.json)
                shutil.copy2(src, dst)


if __name__ == "__main__":
    main()
