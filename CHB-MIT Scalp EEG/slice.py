"""
Second stage processing — Slice the EEG waveforms of all groups of each patient,
and save them in different directories according to the label.
"""

import pickle
import numpy as np
from tqdm import tqdm
import gc
import json

from settings import (
    PROCESSED_DATA_PATH,
    SAMPLE_RATE,
    CHANNELS,
    WINDOW_SIZE_SEC,
    LABELS_STEP_SIZE_SEC,
    LABELS_TO_SLICE,
)

IN_DIR = PROCESSED_DATA_PATH / "Label & Splited"
OUT_DIR = PROCESSED_DATA_PATH / "Sliced"

SUMMARY_FILE = PROCESSED_DATA_PATH / "sliced_summary.json"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def slice_segment(wave: np.ndarray, step_sec: float):
    """
    Slice a segment into windows
    """
    # Calculate the length of a windows in the unit of samples
    win_samples = int(WINDOW_SIZE_SEC * SAMPLE_RATE)
    step_samples = int(step_sec * SAMPLE_RATE)

    # Sclice the segment into windows
    total = wave.shape[1]
    windows = []
    for start in range(0, total - win_samples + 1, step_samples):
        end = start + win_samples
        window = wave[:, start:end]
        windows.append(window)

    return windows


def main():
    summary = {}

    all_files = list(IN_DIR.glob("*.pkl"))
    stat = {}

    with tqdm(total=len(all_files), unit="file", position=0) as pbar:
        for pkl_path in all_files:
            pbar.set_description(f"File: {pkl_path.name}")

            # Load the pkl file
            with open(pkl_path, "rb") as f:
                record = pickle.load(f)

            pid = record["metadata"]["subject_id"]
            gk = pkl_path.stem.split("_", 1)[1]
            curr_offset = 0
            # Slice the segments
            with tqdm(
                record["segments"],
                desc=f"Segments in {pkl_path.name}",
                unit="seg",
                position=1,
                leave=False,
            ) as segs:
                # Initialize the stats for the current patient
                pid = record["metadata"]["subject_id"]
                stat[pid] = {} if pid not in stat else stat[pid]

                for seg in segs:
                    # Determine segment offsets
                    start_offset = curr_offset
                    seg_len = len(seg["data"][CHANNELS[0]])
                    end_offset = start_offset + seg_len
                    curr_offset = end_offset

                    # Only process the segments with the current label
                    if seg["label"] not in LABELS_TO_SLICE:
                        del seg["data"]
                        continue

                    # Update the stats for the current label
                    stat[pid][seg["label"]] = stat[pid].get(seg["label"], 0) + 1

                    segs.set_postfix_str(f"Stacking...")
                    wave = np.stack([seg["data"][ch] for ch in CHANNELS], axis=0)
                    del seg["data"]

                    segs.set_postfix_str(f"Slicing...")
                    windows = slice_segment(wave, LABELS_STEP_SIZE_SEC[seg["label"]])

                    segs.set_postfix_str(f"Aggregating...")
                    aggregated = []  # Aggregated segments for the current label
                    for win in windows:
                        # Normalization
                        norm_win = (win - win.mean(axis=1, keepdims=True)) / (
                            win.std(axis=1, keepdims=True) + 1e-8
                        )

                        aggregated.append(
                            {
                                "data": {
                                    ch: norm_win[i].tolist()
                                    for i, ch in enumerate(CHANNELS)
                                },
                            }
                        )

                    # Save the aggregated segments to a new pkl file
                    out_pkl = (
                        OUT_DIR
                        / f"{pid}"
                        / f"{seg['label']}"
                        / f"{stat[pid][seg['label']]}.pkl"
                    )
                    out_pkl.parent.mkdir(parents=True, exist_ok=True)
                    with open(out_pkl, "wb") as wf:
                        pickle.dump(aggregated, wf, protocol=pickle.HIGHEST_PROTOCOL)

                    start_sec = int(start_offset / SAMPLE_RATE)
                    end_sec = int(end_offset / SAMPLE_RATE)
                    summary.setdefault(pid, {}).setdefault(gk, {})
                    label = seg["label"]
                    summary[pid][gk].setdefault(label, {})
                    file_idx = stat[pid][label]
                    summary[pid][gk][label][str(file_idx)] = (start_sec, end_sec)

                    del windows, aggregated
                    gc.collect()

            del record
            gc.collect()
            pbar.update(1)

    # Sort the summary dict by patient, group key, label, and file index
    sorted_summary = {}
    for pid in sorted(summary):
        sorted_summary[pid] = {}
        for gk in sorted(summary[pid]):
            sorted_summary[pid][gk] = {}
            for label in sorted(summary[pid][gk]):
                sorted_summary[pid][gk][label] = {}
                # file indices are strings, convert to int for correct ordering
                for file_idx in sorted(summary[pid][gk][label], key=lambda x: int(x)):
                    sorted_summary[pid][gk][label][file_idx] = summary[pid][gk][label][
                        file_idx
                    ]
    summary = sorted_summary
    with open(SUMMARY_FILE, "w") as sf:
        json.dump(summary, sf, indent=4)


if __name__ == "__main__":
    main()
