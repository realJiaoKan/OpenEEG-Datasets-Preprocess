"""
First stage preprocessing — Concatenate, fill missing (blank), label,
and split the EDF waveforms of all groups (group) of each patient according to the new label system,
and save them.
"""

import orjson as json
from pathlib import Path
import numpy as np
import mne
from tqdm import tqdm
import pandas as pd
import pickle

from settings import (
    RAW_DATA_PATH,
    PROCESSED_DATA_PATH,
    SAMPLE_RATE,
    CHANNELS,
    PREICTAL_TRANSITION_START,
    PREICTAL_TRANSITION_END,
    PREICTAL_START,
    PREICTAL_END,
    SPH_START,
    SPH_END,
    POSTICTAL_TRANSITION_START,
    POSTICTAL_TRANSITION_END,
)

TIME_INFO = PROCESSED_DATA_PATH / "files_summary.json"
OUT_DIR = PROCESSED_DATA_PATH / "Label & Splited"

OUT_DIR.mkdir(parents=True, exist_ok=True)


def make_label_array(total_samples, seizure_events, data_windows):
    """
    Generate a label array for the given seizure events and data windows.
    By default, all samples are labeled as "interictal".
    The function will label the data by the order of labels, by doing so,
    the label with the highest priority will be assigned first.
    The detailed priority order is as follows:
    0. (Default) interictal
    1. preictal_transition_window
    2. preictal
    3. postictal_transition_window
    4. sph
    5. ictal
    6. blank
    Bigger the index, higher the priority.
    The output is a numpy array of shape (total_samples,) with the labels.
    Though its a little bit inefficitent to store label for each single sample,
    it is easier to process and more flexible for future use :)
    """
    # Initialize labels with "interictal"
    labels = np.array(["interictal"] * total_samples, dtype=object)

    # Label data windows in order of priority
    label_windows = [
        (
            "preictal_transition_window",
            lambda t0, t1: (
                t0 + PREICTAL_TRANSITION_START,
                t0 + PREICTAL_TRANSITION_END,
            ),
        ),
        ("preictal", lambda t0, t1: (t0 + PREICTAL_START, t0 + PREICTAL_END)),
        (
            "postictal_transition_window",
            lambda t0, t1: (
                t1 + POSTICTAL_TRANSITION_START,
                t1 + POSTICTAL_TRANSITION_END,
            ),
        ),
        ("sph", lambda t0, t1: (t0 + SPH_START, t0 + SPH_END)),
        ("ictal", lambda t0, t1: (t0, t1)),
    ]

    for label_name, window_fn in label_windows:
        for t0, t1 in seizure_events:
            start_sec, end_sec = window_fn(t0, t1)
            i0 = max(0, int(start_sec * SAMPLE_RATE))
            i1 = min(total_samples, int(end_sec * SAMPLE_RATE))
            labels[i0:i1] = label_name

    # Calculate blank intervals caused by gaps between original files
    sorted_wins = sorted(
        data_windows, key=lambda x: x[0]
    )  # Should already be sorted, but just in case
    blank_intervals = []
    prev_end = 0.0
    for ws, we in sorted_wins:
        if ws > prev_end:
            blank_intervals.append((prev_end, ws))
        prev_end = max(prev_end, we)

    # Add the last blank interval if needed
    if prev_end * SAMPLE_RATE < total_samples:
        blank_intervals.append((prev_end, total_samples / SAMPLE_RATE))

    # Label blank intervals
    for t0, t1 in blank_intervals:
        i0 = max(0, int(t0 * SAMPLE_RATE))
        i1 = min(total_samples, int(t1 * SAMPLE_RATE))
        labels[i0:i1] = "blank"

    return labels


def split_segments_by_label(wave: np.ndarray, labels: np.ndarray):
    """
    Based on the label array, split the wave into segments,
    by considering each continuous identical labels as a single segment.
    Output:
        segments — List[Dict]
            {
                'label': str,
                'data':  {
                    channel1: [...],
                    channel2: [...],
                    ...
                }
            }
    """
    assert wave.shape[0] == len(CHANNELS)
    assert wave.shape[1] == labels.shape[0]

    segments = []

    T = wave.shape[1]
    idx = 0
    while idx < T:
        lab = labels[idx]
        j = idx + 1
        while j < T and labels[j] == lab:
            j += 1
        segment_data = {ch: wave[ci, idx:j].tolist() for ci, ch in enumerate(CHANNELS)}
        segments.append({"label": lab, "data": segment_data})
        idx = j

    return segments


def process_patient(pid, info, time_info):
    """
    Process a single patient by reading the EDF files,
    concatenating the data, labeling it, and saving the segments.
    """
    for gk, ginfo in tqdm(
        time_info.items(),
        desc=f"Processing groups for {pid}",
        total=len(time_info),
        position=1,
        leave=False,
    ):
        out = {
            "metadata": {
                "subject_id": pid,
                "gender": info.get("gender", ""),
                "age": info.get("age", ""),
            },
            "segments": {},
        }

        # Calculate the length of this group in unit of samples
        max_end = max(v["end_relative"] for v in ginfo["files"].values())
        total_samples = int(max_end * SAMPLE_RATE)

        # The data we need to concatenate from all the files in this group
        wave = np.zeros((len(CHANNELS), total_samples), dtype=float)
        ictal_events = []

        # Process each file in the group and concatenate the data
        for fname, finfo in tqdm(
            ginfo["files"].items(),
            desc=f" Group {gk}",
            total=len(ginfo["files"]),
            position=2,
            leave=False,
        ):
            start_samp = int(finfo["start_relative"] * SAMPLE_RATE)
            edf_path = Path(RAW_DATA_PATH) / pid / fname
            raw = mne.io.read_raw_edf(
                edf_path,
                include=CHANNELS,
                preload=True,
                verbose="ERROR",
            )

            # Some files have duplicated TP-P8 channels, and by reading the file using mne,
            # the duplicated channels are named as T8-P8-0 and T8-P8-1.
            # We need to remove the duplicated TP-P8 and rename the remaining one.
            raw.drop_channels(["T8-P8-1"], on_missing="ignore")
            try:
                raw.rename_channels({"T8-P8-0": "T8-P8"})
            except ValueError:
                pass
                # Ignore if the channel is not found, in this case,
                # it means the channel is not duplicated.

            # Sort the channels to match the order in CHANNELS
            raw.reorder_channels(CHANNELS)

            # Make sure every file has the same sample rate and channel names being selected in the same order
            assert (
                raw.info["sfreq"] == SAMPLE_RATE
            ), f"Sample rate mismatch for {edf_path}"
            assert (
                raw.info["ch_names"] == CHANNELS
            ), f"Channel name mismatch for {edf_path}"

            # Copy the wave data from EDF file
            dat = raw.get_data()
            L = dat.shape[1]
            wave[:, start_samp : start_samp + L] = dat

            # Add the ictal events of this file to the list
            for ict in ginfo.get("ictals", []):
                ictal_events.append(ict)

        # Generate labels for the concatenated data
        file_windows = [
            (finfo["start_relative"], finfo["end_relative"])
            for finfo in ginfo["files"].values()
        ]
        labels = make_label_array(total_samples, ictal_events, file_windows)

        # Split the data into segments based on the labels
        segs = split_segments_by_label(wave, labels)
        out["segments"] = segs

        # Save the segments to a file
        with open(OUT_DIR / f"{pid}_{gk}.pkl", "wb") as f:
            pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    tinfo = json.loads(TIME_INFO.read_text(encoding="utf-8"))

    # Read the patient related information
    info_path = next(Path(RAW_DATA_PATH).glob("SUBJECT-INFO*"))
    df = pd.read_csv(info_path, sep=r"\s+", engine="python")
    df["Case"] = df["Case"].str.strip()
    subj_info = {
        row["Case"]: {"gender": row["Gender"], "age": float(row["Age"])}
        for _, row in df.iterrows()
    }

    # Run the processing for each patient
    for pid, info in tqdm(
        subj_info.items(),
        total=len(subj_info),
        desc="Processing patients",
        position=0,
        leave=True,
    ):
        if pid not in tinfo:
            continue
        process_patient(pid, info, tinfo[pid])


if __name__ == "__main__":
    main()
