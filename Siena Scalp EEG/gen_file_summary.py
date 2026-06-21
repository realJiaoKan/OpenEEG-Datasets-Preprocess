"""
File summary - Parse Siena seizure text files and EDF headers to extract file
start/end times, group files, compute relative times, and integrate ictal
segments into the patient info.
"""

import json
import re

from gen_seizure_map import (
    format_seconds,
    parse_seizure_list,
    read_edf_metadata,
)
from settings import DATA_PATH, PROCESSED_DATA_PATH, RAW_DATA_PATH, SAMPLE_RATE

SEIZURE_FILE = DATA_PATH / "seizure_map.json"
PROC_FILE = PROCESSED_DATA_PATH / "files_summary.json"

PROC_FILE.parent.mkdir(parents=True, exist_ok=True)


def natural_key(name):
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


def ordered_edf_files(patient_dir, file_records):
    ordered = []
    seen = set()
    for file_name in file_records:
        edf_path = patient_dir / file_name
        if edf_path.exists() and file_name not in seen:
            ordered.append(edf_path)
            seen.add(file_name)

    for edf_path in sorted(patient_dir.glob("*.edf"), key=lambda p: natural_key(p.name)):
        if edf_path.name not in seen:
            ordered.append(edf_path)
            seen.add(edf_path.name)

    return ordered


def parse_summary_text(patient_dir):
    """
    Parse Siena text/header metadata into the same grouped structure used by CHB.
    EDF headers are used for recording start and duration because several Siena
    text files contain small registration-time typos.
    """
    pid = patient_dir.name
    summary_file = patient_dir / f"Seizures-list-{pid}.txt"
    text = summary_file.read_text(encoding="utf-8", errors="replace")
    _, file_records = parse_seizure_list(text, pid)

    entries = []
    prev_abs_end = None
    prev_end_day = 0

    for edf_path in ordered_edf_files(patient_dir, file_records):
        start_seconds, duration_seconds, sfreq = read_edf_metadata(edf_path)
        assert sfreq == SAMPLE_RATE, f"Sample rate mismatch for {edf_path}"

        day = prev_end_day
        abs_start = day * 86400 + start_seconds
        if prev_abs_end is not None and abs_start <= prev_abs_end:
            day += 1
            abs_start = day * 86400 + start_seconds

        abs_end = abs_start + duration_seconds
        entries.append(
            {
                "name": edf_path.name,
                "start_str": format_seconds(start_seconds),
                "end_str": format_seconds(start_seconds + duration_seconds),
                "abs_start": abs_start,
                "abs_end": abs_end,
            }
        )
        prev_abs_end = abs_end
        prev_end_day = int(abs_end // 86400)

    # Group entries with max gap <= 2h (7200s)
    groups = []
    current = [entries[0]]
    for e in entries[1:]:
        if e["abs_start"] - current[-1]["abs_end"] <= 7200:
            current.append(e)
        else:
            groups.append(current)
            current = [e]
    groups.append(current)

    # Build result
    result = {}
    for idx, grp in enumerate(groups, start=1):
        group_key = f"group_{idx}"
        result[group_key] = {}
        base = grp[0]["abs_start"]
        for e in grp:
            result[group_key][e["name"]] = {
                "start_original": e["start_str"],
                "end_original": e["end_str"],
                "start_relative": e["abs_start"] - base,
                "end_relative": e["abs_end"] - base,
            }

    return result


def main():
    seizure_map = json.loads(SEIZURE_FILE.read_text(encoding="utf-8"))

    all_data = {}

    for patient_dir in sorted(
        RAW_DATA_PATH.iterdir(), key=lambda p: p.name
    ):  # Make sure the result are well ordered
        if not patient_dir.is_dir():
            continue

        print(f"Processing {patient_dir.name}...")

        summary_file = patient_dir / f"Seizures-list-{patient_dir.name}.txt"
        assert summary_file.exists()

        parsed = parse_summary_text(patient_dir)

        # Integrate ictals into patient info
        patient_info = {}

        for group_key, files in parsed.items():
            ictals = []
            for fname, finfo in files.items():
                file_ictals = seizure_map.get(fname, [])
                offset = finfo["start_relative"]
                for seg in file_ictals:
                    ictals.append([offset + seg[0], offset + seg[1]])
            patient_info[group_key] = {"files": files, "ictals": ictals}

        all_data[patient_dir.name] = patient_info

    # Save to JSON
    with PROC_FILE.open("w", encoding="utf-8") as fp:
        json.dump(all_data, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
