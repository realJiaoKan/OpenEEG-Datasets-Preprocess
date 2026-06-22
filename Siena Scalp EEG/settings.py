from pathlib import Path

# Environment Settings
RANDOM_SEED = 42

# File Paths
ROOT_PATH = Path(__file__).parent
DATA_PATH = ROOT_PATH / "Data"
RAW_DATA_PATH = DATA_PATH / "Raw"
PROCESSED_DATA_PATH = DATA_PATH / "Processed"
SUBJECT_INFO_FILE = DATA_PATH / "subject_info.csv"

# EEG Metadata
SAMPLE_RATE = 512  # Hz

# Processing Parameters
CHANNELS = [
    # Left lateral chain, anterior to posterior
    "F9",
    "F7",
    "Fc5",
    "T3",
    "Cp5",
    "T5",

    # Left parasagittal chain, anterior to posterior
    "Fp1",
    "F3",
    "Fc1",
    "C3",
    "Cp1",
    "P3",
    "O1",

    # Right parasagittal chain, anterior to posterior
    "Fp2",
    "F4",
    "Fc2",
    "C4",
    "Cp2",
    "P4",
    "O2",

    # Right lateral chain, anterior to posterior
    "F10",
    "F8",
    "Fc6",
    "T4",
    "Cp6",
    "T6",

    # Midline chain, anterior to posterior
    "Fz",
    "Cz",
    "Pz",
]  # Ordered

# MNE reads Siena EDF channel names with an "EEG " prefix and occasional
# capitalization differences. The processed data keeps the clean names above.
EDF_CHANNEL_RENAMES = {f"EEG {ch}": ch for ch in CHANNELS}
EDF_CHANNEL_RENAMES.update(
    {
        "EEG FP2": "Fp2",
        "EEG CZ": "Cz",
    }
)
EDF_CHANNELS = list(EDF_CHANNEL_RENAMES.keys())

WINDOW_SIZE_SEC = 5.0  # s

LABELS_STEP_SIZE_SEC = {
    "interictal": WINDOW_SIZE_SEC,
    "preictal_transition_window": WINDOW_SIZE_SEC,
    "postictal_transition_window": WINDOW_SIZE_SEC,
    "preictal": 1,
    "sph": WINDOW_SIZE_SEC,
    "ictal": WINDOW_SIZE_SEC,
}  # s

LABELS_TO_SLICE = [
    "preictal",
    # "ictal",
    # "sph",
    "interictal",
    # "preictal_transition_window",
    # "postictal_transition_window",
]

LABELS_TO_TRAIN = {
    "interictal": 0.3,  # The value means the ratio of how many samples to be selected
    "preictal": 1,
    # "ictal": 1,
    # "sph": 1,
    # "preictal_transition_window": 1,
    # "postictal_transition_window": 1,
}  # The order here also reflects the label number in the dataset

DATASETS = {
    "PN00": ["PN00"],
    "PN01": ["PN01"],
    "PN03": ["PN03"],
    "PN05": ["PN05"],
    "PN06": ["PN06"],
    "PN07": ["PN07"],
    "PN09": ["PN09"],
    "PN10": ["PN10"],
    "PN11": ["PN11"],
    "PN12": ["PN12"],
    "PN13": ["PN13"],
    "PN14": ["PN14"],
    "PN16": ["PN16"],
    "PN17": ["PN17"],
}

# Labeling Parameters
PREICTAL_TRANSITION_START = -2.5 * 3600  # s
PREICTAL_TRANSITION_END = -0.5 * 3600  # s

PREICTAL_START = -30 * 60  # s
PREICTAL_END = -5 * 60  # s

SPH_START = -5 * 60  # s
SPH_END = 0  # s

POSTICTAL_TRANSITION_START = 0  # s
POSTICTAL_TRANSITION_END = 1 * 3600  # s

# Plotting Parameters
MAX_POINTS = 1000

LABEL_COLOR_MAP = {
    "interictal": "rgba(0,200,0,0.3)",  # Green
    "preictal": "rgba(255,165,0,0.6)",  # Orange
    "preictal_transition_window": "rgba(255,200,50,0.5)",  # Yellow-orange
    "ictal": "rgba(255,0,0,0.6)",  # Red
    "postictal_transition_window": "rgba(0,0,255,0.5)",  # Blue
    "sph": "rgba(128,0,128,0.5)",  # Purple
    "blank": "rgba(200,200,200,0.3)",  # Gray
}

# Postprocessing Parameters
## Spectrogram
REMOVE_BANDS = [(47, 53), (97, 103), (147, 153), (197, 203), (247, 253)]

USE_LOG_POWER = True
