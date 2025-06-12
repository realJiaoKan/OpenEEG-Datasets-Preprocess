"""
Plot the EEG waveform with label color bands for files processed by `stage_1_process.py`,
to check the labeling and segmenting process.
"""

import pickle
import numpy as np
import plotly.graph_objs as go
from math import ceil

from settings import (
    PROCESSED_DATA_PATH,
    SAMPLE_RATE,
    CHANNELS,
    MAX_POINTS,
    LABEL_COLOR_MAP,
)

PLOTTING_FILE = PROCESSED_DATA_PATH / "Label & Splited" / "chb16_group_1.pkl"


def plot_eeg_waveform_with_labels(pkl_path):
    """Concatenate segments and plot EEG waveform with label color band."""
    # Load the pkl file
    with open(pkl_path, "rb") as f:
        out = pickle.load(f)
    segments = out["segments"]
    sample_counts = [len(seg["data"][CHANNELS[0]]) for seg in segments]
    total_samples = sum(sample_counts)

    # Concatenate segments and create the label array
    wave = np.zeros((len(CHANNELS), total_samples), dtype=float)
    labels = []
    idx = 0
    for seg, count in zip(segments, sample_counts):
        for ci, ch in enumerate(CHANNELS):
            wave[ci, idx : idx + count] = seg["data"][ch]
        labels.extend([seg["label"]] * count)
        idx += count
    labels = np.array(labels)

    # Downsample the data for plotting
    step = max(1, int(ceil(total_samples / MAX_POINTS)))
    t_full = np.arange(total_samples) / SAMPLE_RATE
    t_ds = t_full[::step]
    wave_ds = wave[:, ::step] * 1e6  # Convert the unit to µV

    # Compute the ranges for plotting
    y_min, y_max = wave_ds.min(), wave_ds.max()
    band_h = (y_max - y_min) * 0.05
    y0, y1 = y_min - 1.5 * band_h, y_min - 0.5 * band_h  # For the color band

    # Create the figure
    fig = go.Figure()

    # Wavefroms
    for ci, ch in enumerate(CHANNELS):
        fig.add_trace(go.Scatter(x=t_ds, y=wave_ds[ci], mode="lines", name=ch))

    # Color band for labels
    start_time = 0.0
    for seg, count in zip(segments, sample_counts):
        seg_dur = count / SAMPLE_RATE
        color = LABEL_COLOR_MAP.get(seg["label"], "rgba(150,150,150,0.5)")
        fig.add_shape(
            type="rect",
            xref="x",
            yref="y",
            x0=start_time,
            x1=start_time + seg_dur,
            y0=y0,
            y1=y1,
            fillcolor=color,
            line=dict(width=0),
        )
        start_time += seg_dur

    # Layout settings
    fig.update_layout(
        title="EEG Full Waveform with Labels",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude (µV)",
        showlegend=True,
    )

    fig.show(config={"responsive": True})


if __name__ == "__main__":
    plot_eeg_waveform_with_labels(PLOTTING_FILE)
