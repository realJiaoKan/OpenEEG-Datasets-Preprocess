"""
Plot the EEG waveform with label color bands for files processed by `stage_test_process.py`,
to check the labeling and segmenting process.
"""

import numpy as np
import plotly.graph_objs as go
from math import ceil

from settings import (
    PROCESSED_DATA_PATH,
    SAMPLE_RATE,
    CHANNELS,
    WINDOW_SIZE_SEC,
    MAX_POINTS,
    LABEL_COLOR_MAP,
)

PLOTTING_FILE = PROCESSED_DATA_PATH / "Stage Test" / "chb01_group_3.npz"

LENGTH_FROM_START = None  # Number of samples to plot from the start


def plot_test_waveform_with_labels(npz_path):
    """Concatenate processed windows and plot normalized EEG waveform with label color bands."""
    # Load the npz file
    data = np.load(npz_path)

    # Concatenate segments
    X = data["X"] if LENGTH_FROM_START is None else data["X"][:LENGTH_FROM_START]
    y = data["y"] if LENGTH_FROM_START is None else data["y"][:LENGTH_FROM_START]
    N, C, T = X.shape
    wave = X.transpose(1, 0, 2).reshape(C, N * T)
    total_samples = wave.shape[1]

    # Downsample the data for plotting
    step = max(1, int(ceil(total_samples / MAX_POINTS)))
    t_full = np.arange(total_samples) / SAMPLE_RATE
    t_ds = t_full[::step]
    wave_ds = wave[:, ::step]

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
    # Combine the continuous labels for better performance,
    # or it will be VERY slow to plot
    run_start = 0
    current_label = y[0]
    for i, label in enumerate(y):
        if label != current_label:
            x0 = run_start * WINDOW_SIZE_SEC
            x1 = i * WINDOW_SIZE_SEC
            color = LABEL_COLOR_MAP.get(current_label, "rgba(150,150,150,0.5)")
            fig.add_shape(
                type="rect",
                xref="x",
                yref="y",
                x0=x0,
                x1=x1,
                y0=y0,
                y1=y1,
                fillcolor=color,
                line=dict(width=0),
            )
            run_start = i
            current_label = label

    # Add the last band
    x0 = run_start * WINDOW_SIZE_SEC
    x1 = len(y) * WINDOW_SIZE_SEC
    color = LABEL_COLOR_MAP.get(current_label, "rgba(150,150,150,0.5)")
    fig.add_shape(
        type="rect",
        xref="x",
        yref="y",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        fillcolor=color,
        line=dict(width=0),
    )

    # Layout settings
    fig.update_layout(
        title="Processed EEG Waveform with Labels (Stage Test)",
        xaxis_title="Time (s)",
        yaxis_title="Normalized Amplitude",
        showlegend=True,
    )

    fig.show(config={"responsive": True})


if __name__ == "__main__":
    plot_test_waveform_with_labels(PLOTTING_FILE)
