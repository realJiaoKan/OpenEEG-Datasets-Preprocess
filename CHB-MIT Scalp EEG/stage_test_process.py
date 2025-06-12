"""
Stage test — Create the dataset for each whole record,
with the original order and labels.
"""

import pickle
import numpy as np
from tqdm import tqdm
import gc

from settings import PROCESSED_DATA_PATH, SAMPLE_RATE, CHANNELS, WINDOW_SIZE_SEC

IN_DIR = PROCESSED_DATA_PATH / "Stage 1"
OUT_DIR = PROCESSED_DATA_PATH / "Stage Test"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def slice_segment(wave: np.ndarray):
    """
    Slice a segment into windows
    """
    # Calculate the length of a windows in the unit of samples
    win_samples = int(WINDOW_SIZE_SEC * SAMPLE_RATE)

    # Sclice the segment into windows
    total = wave.shape[1]
    windows = []
    for start in range(0, total - win_samples + 1, win_samples):
        end = start + win_samples
        window = wave[:, start:end]
        windows.append(window)

    return windows


def main():
    all_files = list(IN_DIR.glob("*.pkl"))

    # Iterate through each file and slice all segments
    for pkl_path in tqdm(all_files, desc="Processing files", unit="file", position=0):
        X, y = [], []

        # Load the record
        with open(pkl_path, "rb") as f:
            record = pickle.load(f)

        # Process each segment sequentially
        for seg in tqdm(
            record["segments"],
            desc=f"Segments in {pkl_path.name}",
            unit="seg",
            position=1,
            leave=False,
        ):
            # Stack channel data
            wave = np.stack([seg["data"][ch] for ch in CHANNELS], axis=0)

            # Slice into windows using fixed window size
            windows = slice_segment(wave)

            # Normalize and collect
            for win in windows:
                norm_win = (win - win.mean(axis=1, keepdims=True)) / (
                    win.std(axis=1, keepdims=True) + 1e-8
                )
                X.append(norm_win)
                y.append(seg["label"])

        X = np.array(X, dtype=np.float32)
        y = np.array(y)
        np.savez_compressed(OUT_DIR / f"{pkl_path.stem}.npz", X=X, y=y)

        del record
        gc.collect()


if __name__ == "__main__":
    main()
