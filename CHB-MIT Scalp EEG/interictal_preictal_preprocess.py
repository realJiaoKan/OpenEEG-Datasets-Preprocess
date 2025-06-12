import pickle
import numpy as np
import gc
import json

from settings import (
    RANDOM_SEED,
    PROCESSED_DATA_PATH,
    CHANNELS,
    WINDOW_SIZE_SEC,
    LABELS_STEP_SIZE_SEC,
)

IN_DIR = PROCESSED_DATA_PATH / "Sliced"
OUT_DIR = PROCESSED_DATA_PATH / "Interictal-Preictal" / "Raw"

OUT_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = PROCESSED_DATA_PATH / "sliced_summary.json"
FOLD_SUMMARY_FILE = PROCESSED_DATA_PATH / "Interictal-Preictal" / "fold_summary.json"

np.random.seed(RANDOM_SEED)


def combine_interictal():
    for pid_dir in IN_DIR.iterdir():
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        inter_dir = pid_dir / "interictal"
        if not inter_dir.exists():
            continue
        windows_list = []
        for pkl_path in inter_dir.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                aggregated = pickle.load(f)
            for win in aggregated:
                data = np.stack([win["data"][ch] for ch in CHANNELS], axis=0)
                windows_list.append(data)
        if windows_list:
            X = np.stack(windows_list, axis=0)
            out_pid = OUT_DIR / pid
            out_pid.mkdir(parents=True, exist_ok=True)
            out_file = out_pid / "interictal_combined.npz"
            np.savez_compressed(out_file, X=X)
            del windows_list, X
            gc.collect()


def process_preictal():
    required_sec = 15 * 60
    step_sec = LABELS_STEP_SIZE_SEC.get("preictal", WINDOW_SIZE_SEC)
    # Load sliced summary and prepare fold summary
    with open(SUMMARY_FILE, "r") as sf:
        summary_data = json.load(sf)
    fold_summary = {}
    for pid_dir in IN_DIR.iterdir():
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        fold_summary[pid] = {}
        pre_dir = pid_dir / "preictal"
        if not pre_dir.exists():
            continue
        out_pid = OUT_DIR / pid
        out_pid.mkdir(parents=True, exist_ok=True)
        counter = 0
        for pkl_path in pre_dir.glob("*.pkl"):
            with open(pkl_path, "rb") as f:
                aggregated = pickle.load(f)
            n = len(aggregated)
            duration = (n - 1) * step_sec + WINDOW_SIZE_SEC
            if duration < required_sec:
                continue
            data_list = [
                np.stack([win["data"][ch] for ch in CHANNELS], axis=0)
                for win in aggregated
            ]
            X = np.stack(data_list, axis=0)
            out_file = out_pid / f"preictal_{counter}.npz"
            np.savez_compressed(out_file, X=X)

            # Record time info from sliced summary
            file_idx = int(pkl_path.stem)
            time_tuple = None
            for gk, gk_data in summary_data.get(pid, {}).items():
                times = gk_data.get("preictal", {})
                if str(file_idx) in times:
                    time_tuple = tuple(times[str(file_idx)])
                    break
            if time_tuple is not None:
                fold_summary[pid][counter] = {"group": gk, "time": time_tuple}

            counter += 1
            del aggregated, data_list, X
            gc.collect()

    sorted_summary = {}
    for pid in sorted(fold_summary):
        sorted_summary[pid] = {}
        for idx in sorted(fold_summary[pid].keys()):
            sorted_summary[pid][str(idx)] = fold_summary[pid][idx]
    # Save sorted fold summary for preictal
    with open(FOLD_SUMMARY_FILE, "w") as ff:
        json.dump(sorted_summary, ff, indent=4)


def main():
    combine_interictal()
    process_preictal()


if __name__ == "__main__":
    main()
