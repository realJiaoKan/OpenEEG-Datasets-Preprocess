"""
Draw a single-channel stitched EEG labeling figure for the first seizure in
chb01_03.edf.

Each displayed phase uses one equal-length real EEG snippet so the overall
figure stays compact while still showing all annotated stages.
"""

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import mne
import numpy as np

from settings import (
    POSTICTAL_TRANSITION_END,
    POSTICTAL_TRANSITION_START,
    PREICTAL_END,
    PREICTAL_START,
    PREICTAL_TRANSITION_END,
    PREICTAL_TRANSITION_START,
    RAW_DATA_PATH,
    SAMPLE_RATE,
    SPH_END,
    SPH_START,
)

PATIENT_ID = "chb01"
EDF_NAME = "chb01_03.edf"
CHANNEL_NAME = "F8-T8"
SEIZURE_INDEX = 0
DISPLAY_DURATION_SEC = 18.0
OUTPUT_PATH = Path(__file__).with_suffix(".pdf")
SEIZURE_MAP_PATH = Path(__file__).parent / "Data" / "seizure_map.json"

# The display titles and colors follow the reference figure requested by the user.
DISPLAY_PHASES = [
    {
        "interval_key": "preictal_transition_window",
        "title": "Preictal",
        "color": "#eef6f1",
        "anchor": "end",
    },
    {
        "interval_key": "preictal",
        "title": "Preictal Transition",
        "color": "#fbf1e5",
        "anchor": "end",
    },
    {
        "interval_key": "sph",
        "title": "SPH",
        "color": "#f8e0df",
        "anchor": "end",
    },
    {
        "interval_key": "ictal",
        "title": "Ictal",
        "color": "#f5d3dc",
        "anchor": "start",
    },
    {
        "interval_key": "postictal_transition_window",
        "title": "Postictal",
        "color": "#e8def6",
        "anchor": "start",
    },
]


def load_first_seizure(file_name: str, seizure_index: int = 0) -> tuple[float, float]:
    """Load the selected seizure onset and offset from seizure_map.json."""
    with SEIZURE_MAP_PATH.open("r", encoding="utf-8") as f:
        seizure_map = json.load(f)

    seizure_events = seizure_map[file_name]
    if not seizure_events:
        raise ValueError(f"No seizure events found for {file_name}.")
    if seizure_index >= len(seizure_events):
        raise IndexError(
            f"seizure_index={seizure_index} is out of range for {file_name}."
        )

    seizure_start, seizure_end = seizure_events[seizure_index]
    return float(seizure_start), float(seizure_end)


def build_intervals(
    seizure_start: float,
    seizure_end: float,
    file_duration: float,
) -> dict[str, tuple[float, float]]:
    """Build phase intervals in absolute seconds within the EDF file."""
    intervals = {
        "preictal_transition_window": (
            max(0.0, seizure_start + PREICTAL_TRANSITION_START),
            max(0.0, seizure_start + PREICTAL_TRANSITION_END),
        ),
        "preictal": (
            max(0.0, seizure_start + PREICTAL_START),
            max(0.0, seizure_start + PREICTAL_END),
        ),
        "sph": (
            max(0.0, seizure_start + SPH_START),
            max(0.0, seizure_start + SPH_END),
        ),
        "ictal": (
            max(0.0, seizure_start),
            min(file_duration, seizure_end),
        ),
        "postictal_transition_window": (
            min(file_duration, seizure_end + POSTICTAL_TRANSITION_START),
            min(file_duration, seizure_end + POSTICTAL_TRANSITION_END),
        ),
    }

    for label_name, (start_sec, end_sec) in intervals.items():
        if end_sec <= start_sec:
            raise ValueError(
                f"Interval {label_name} is empty after clipping: {start_sec:.2f}-{end_sec:.2f}s."
            )
        if end_sec - start_sec < DISPLAY_DURATION_SEC:
            raise ValueError(
                f"Interval {label_name} is shorter than {DISPLAY_DURATION_SEC:.1f}s."
            )
    return intervals


def choose_snippet_bounds(
    start_sec: float, end_sec: float, anchor: str
) -> tuple[float, float]:
    """Select an equal-length snippet from a longer interval."""
    if anchor == "end":
        return end_sec - DISPLAY_DURATION_SEC, end_sec
    if anchor == "start":
        return start_sec, start_sec + DISPLAY_DURATION_SEC
    if anchor == "center":
        center = 0.5 * (start_sec + end_sec)
        half = 0.5 * DISPLAY_DURATION_SEC
        return center - half, center + half
    raise ValueError(f"Unknown anchor: {anchor}")


def load_stitched_channel(
    raw: mne.io.BaseRaw,
    intervals: dict[str, tuple[float, float]],
) -> tuple[list[dict], np.ndarray]:
    """Load one real EEG snippet from each phase and stitch them for display."""
    snippets = []
    all_values = []
    display_cursor = 0.0

    for spec in DISPLAY_PHASES:
        snippet_start, snippet_end = choose_snippet_bounds(
            *intervals[spec["interval_key"]],
            spec["anchor"],
        )
        sample_start = int(round(snippet_start * SAMPLE_RATE))
        sample_end = int(round(snippet_end * SAMPLE_RATE))
        data_uv = (
            raw.get_data(
                picks=[CHANNEL_NAME],
                start=sample_start,
                stop=sample_end,
            )[0]
            * 1e6
        )
        data_uv = data_uv - np.median(data_uv)

        time_axis = display_cursor + np.arange(data_uv.size, dtype=float) / SAMPLE_RATE
        snippets.append(
            {
                "title": spec["title"],
                "color": spec["color"],
                "display_start": display_cursor,
                "display_end": display_cursor + DISPLAY_DURATION_SEC,
                "time_axis": time_axis,
                "data_uv": data_uv,
            }
        )
        all_values.append(data_uv)
        display_cursor += DISPLAY_DURATION_SEC

    return snippets, np.concatenate(all_values)


def configure_plot_style() -> None:
    """Use a Times New Roman styled serif look similar to the reference figure."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def add_scale_bar(
    ax: plt.Axes,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> None:
    """Draw a 100 µV scale bar near the lower-left corner."""
    scale_uv = 100.0
    x_pos = x_min + 0.014 * (x_max - x_min)
    y_pos = y_min + 0.14 * (y_max - y_min)
    ax.plot([x_pos, x_pos], [y_pos, y_pos + scale_uv], color="black", lw=1.2, zorder=6)
    ax.text(
        x_pos + 0.01 * (x_max - x_min),
        y_pos + 0.5 * scale_uv,
        "100 µV",
        va="center",
        ha="left",
        fontsize=13,
    )


def add_phase_annotations(
    ax_header: plt.Axes, ax_wave: plt.Axes, snippets: list[dict]
) -> None:
    """Draw stitched phase spans and titles over the stitched display axis."""
    for snippet in snippets:
        ax_wave.axvspan(
            snippet["display_start"],
            snippet["display_end"],
            color=snippet["color"],
            alpha=0.32,
            zorder=0,
        )
        ax_header.axvspan(
            snippet["display_start"],
            snippet["display_end"],
            color=snippet["color"],
            alpha=0.78,
            zorder=1,
        )
        ax_wave.axvline(snippet["display_start"], color="#c8c8c8", lw=0.7, zorder=2)
        ax_header.axvline(snippet["display_start"], color="#d2d2d2", lw=0.7, zorder=2)
        ax_header.text(
            0.5 * (snippet["display_start"] + snippet["display_end"]),
            0.5,
            snippet["title"],
            transform=ax_header.get_xaxis_transform(),
            ha="center",
            va="center",
            fontsize=16,
        )

    ax_wave.axvline(snippets[-1]["display_end"], color="#c8c8c8", lw=0.7, zorder=2)
    ax_header.axvline(snippets[-1]["display_end"], color="#d2d2d2", lw=0.7, zorder=2)


def plot_labeling_sample(save_path: Path = OUTPUT_PATH) -> Path:
    """Create and save the stitched single-channel labeling sample figure."""
    configure_plot_style()

    edf_path = RAW_DATA_PATH / PATIENT_ID / EDF_NAME
    seizure_start, seizure_end = load_first_seizure(EDF_NAME, SEIZURE_INDEX)
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    if CHANNEL_NAME not in raw.ch_names:
        raise ValueError(f"Channel {CHANNEL_NAME} not found in {edf_path.name}.")

    file_duration = raw.n_times / raw.info["sfreq"]
    intervals = build_intervals(seizure_start, seizure_end, file_duration)
    snippets, all_values = load_stitched_channel(raw, intervals)

    y_abs = np.quantile(np.abs(all_values), 0.995)
    y_lim = max(220.0, float(np.ceil((1.7 * y_abs) / 10.0) * 10.0))
    y_min, y_max = -y_lim, y_lim
    x_min, x_max = 0.0, snippets[-1]["display_end"]

    fig = plt.figure(figsize=(13.2, 2.65), dpi=220)
    gs = fig.add_gridspec(2, 1, height_ratios=[0.16, 0.84], hspace=0.0)
    ax_header = fig.add_subplot(gs[0])
    ax_wave = fig.add_subplot(gs[1], sharex=ax_header)

    add_phase_annotations(ax_header, ax_wave, snippets)
    for snippet in snippets:
        ax_wave.plot(
            snippet["time_axis"], snippet["data_uv"], color="black", lw=0.65, zorder=4
        )

    ax_wave.set_xlim(x_min, x_max)
    ax_wave.set_ylim(y_min, y_max)
    ax_wave.set_xlabel("Time (s)", fontsize=16)
    ax_wave.set_yticks([])
    ax_wave.tick_params(axis="x", labelsize=13)
    ax_wave.xaxis.set_major_locator(MultipleLocator(5))
    ax_wave.xaxis.set_minor_locator(MultipleLocator(2.5))
    ax_wave.grid(
        which="major", axis="x", linestyle=(0, (3, 3)), color="#d9d9d9", linewidth=0.55
    )
    ax_wave.grid(
        which="minor", axis="x", linestyle=(0, (2, 4)), color="#ececec", linewidth=0.45
    )
    ax_wave.axhline(0.0, color="#d7d7d7", lw=0.5, linestyle=(0, (3, 3)), zorder=1)

    ax_header.set_ylim(0.0, 1.0)
    ax_header.set_yticks([])
    ax_header.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    for side in ["top", "right", "left", "bottom"]:
        ax_header.spines[side].set_visible(False)

    ax_wave.spines["top"].set_visible(False)
    ax_wave.spines["right"].set_visible(False)
    ax_wave.spines["left"].set_linewidth(0.85)
    ax_wave.spines["bottom"].set_linewidth(0.85)

    ax_wave.text(
        -0.02,
        0.5,
        CHANNEL_NAME,
        transform=ax_wave.transAxes,
        ha="right",
        va="center",
        fontsize=16,
    )
    add_scale_bar(ax_wave, x_min, x_max, y_min, y_max)

    fig.subplots_adjust(left=0.08, right=0.995, top=0.985, bottom=0.19, hspace=0.0)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return save_path


if __name__ == "__main__":
    saved_path = plot_labeling_sample()
    print(f"Saved figure to: {saved_path}")
