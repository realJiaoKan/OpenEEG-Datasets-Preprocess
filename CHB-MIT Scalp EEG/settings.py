from pathlib import Path

# Environment Settings
RANDOM_SEED = 42

# File Paths
ROOT_PATH = Path(__file__).parent
RAW_DATA_PATH = ROOT_PATH / "Data" / "Raw"
PROCESSED_DATA_PATH = ROOT_PATH / "Data" / "Processed"

# EEG Metadata
SAMPLE_RATE = 256  # Hz

# Processing Parameters
CHANNELS = [
    "FP1-F7",
    "F7-T7",
    "T7-P7",
    "P7-O1",
    "FP1-F3",
    "F3-C3",
    "C3-P3",
    "P3-O1",
    "FP2-F4",
    "F4-C4",
    "C4-P4",
    "P4-O2",
    "FP2-F8",
    "F8-T8",
    "T8-P8",
    "P8-O2",
    "FZ-CZ",
    "CZ-PZ",
]  # Ordered

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
    "chb01": ["chb01"],
    "chb02": ["chb02"],
    "chb03": ["chb03"],
    "chb04": ["chb04"],
    "chb05": ["chb05"],
    "chb06": ["chb06"],
    "chb07": ["chb07"],
    "chb08": ["chb08"],
    "chb09": ["chb09"],
    "chb10": ["chb10"],
    "chb11": ["chb11"],
    "chb14": ["chb14"],
    "chb15": ["chb15"],
    "chb16": ["chb16"],
    "chb17": ["chb17"],
    "chb18": ["chb18"],
    "chb19": ["chb19"],
    "chb20": ["chb20"],
    "chb21": ["chb21"],
    "chb22": ["chb22"],
    "chb23": ["chb23"],
    # "combined": ["chb01", "chb02"],
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
MAX_POINTS = 4000

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
REMOVE_BANDS = [(57, 63), (117, 123)]

USE_LOG_POWER = True
