"""
Seizure map - Parse Siena seizure text files and save per-EDF ictal intervals.

The output matches the CHB seizure_map.json format:
{
    "PN00-1.edf": [[start_sec, end_sec], ...],
    ...
}
"""

import csv
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import mne

from settings import DATA_PATH, RAW_DATA_PATH, SAMPLE_RATE, SUBJECT_INFO_FILE

SEIZURE_FILE = DATA_PATH / "seizure_map.json"

FILENAME_FIXES = {
    "PN01.edf": "PN01-1.edf",
    "PNO6-1.edf": "PN06-1.edf",
    "PNO6-2.edf": "PN06-2.edf",
    "PNO6-4.edf": "PN06-4.edf",
    "PN11-.edf": "PN11-1.edf",
}

# PN00-3 has a typo in the text file: the seizure end hour is written as 19,
# while the recording ends at 18:57. The intended one-minute seizure is 18:28-18:29.
SEIZURE_TIME_FIXES = {
    ("PN00", "PN00-3.edf", "18:28:29", "19:29:29"): ("18:28:29", "18:29:29"),
}


@dataclass
class SeizureEvent:
    file_name: str
    start: tuple[int, int, int]
    end: tuple[int, int, int]


def normalize_file_name(name: str) -> str:
    m = re.search(r"[A-Za-z0-9.-]+\.edf", name)
    if not m:
        raise ValueError(f"Cannot parse EDF file name from: {name!r}")
    file_name = m.group(0).strip()
    return FILENAME_FIXES.get(file_name, file_name)


def parse_time_value(value: str) -> tuple[int, int, int]:
    value = re.sub(r"(?<=\d)\s+(?=\d[.:])", "", value.strip())
    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{1,2})\s*[:.]\s*(\d{1,2})", value)
    if not m:
        raise ValueError(f"Cannot parse time from: {value!r}")
    h, mm, ss = map(int, m.groups())
    return h, mm, ss


def format_hms(time_value: tuple[int, int, int]) -> str:
    h, mm, ss = time_value
    return f"{h:02d}:{mm:02d}:{ss:02d}"


def format_seconds(seconds: int) -> str:
    seconds %= 86400
    h = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{h:02d}:{mm:02d}:{ss:02d}"


def seconds_from_hms(time_value: tuple[int, int, int]) -> int:
    h, mm, ss = time_value
    return h * 3600 + mm * 60 + ss


def seconds_after_start(time_value: tuple[int, int, int], start_seconds: int) -> int:
    seconds = seconds_from_hms(time_value) - start_seconds
    while seconds < 0:
        seconds += 86400
    return seconds


def apply_time_fix(
    pid: str,
    file_name: str,
    start: tuple[int, int, int],
    end: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    fixed = SEIZURE_TIME_FIXES.get((pid, file_name, format_hms(start), format_hms(end)))
    if fixed is None:
        return start, end
    return parse_time_value(fixed[0]), parse_time_value(fixed[1])


def parse_seizure_list(
    text: str,
    pid: str,
) -> tuple[list[SeizureEvent], dict[str, dict[str, tuple[int, int, int]]]]:
    events = []
    file_records = {}
    current_file = None
    seizure_start = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        lower = line.lower()
        if lower.startswith("file name:"):
            current_file = normalize_file_name(line.split(":", 1)[1])
            file_records.setdefault(current_file, {})
            seizure_start = None
            continue

        if lower.startswith("registration start time:"):
            if current_file is None:
                raise ValueError(f"Registration start without file in {pid}: {line!r}")
            file_records.setdefault(current_file, {})["start"] = parse_time_value(
                line.split(":", 1)[1]
            )
            continue

        if lower.startswith("registration end time:"):
            if current_file is None:
                raise ValueError(f"Registration end without file in {pid}: {line!r}")
            file_records.setdefault(current_file, {})["end"] = parse_time_value(
                line.split(":", 1)[1]
            )
            continue

        if lower.startswith("seizure start time:") or lower.startswith("start time:"):
            seizure_start = parse_time_value(line.split(":", 1)[1])
            continue

        if lower.startswith("seizure end time:") or lower.startswith("end time:"):
            if current_file is None or seizure_start is None:
                raise ValueError(f"Seizure end without start in {pid}: {line!r}")
            seizure_end = parse_time_value(line.split(":", 1)[1])
            seizure_start, seizure_end = apply_time_fix(
                pid, current_file, seizure_start, seizure_end
            )
            events.append(SeizureEvent(current_file, seizure_start, seizure_end))
            seizure_start = None

    return events, file_records


@lru_cache(maxsize=None)
def read_edf_metadata(edf_path: Path) -> tuple[int, int, float]:
    raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
    meas_date = raw.info.get("meas_date")
    if meas_date is None:
        raise ValueError(f"Missing measurement date in {edf_path}")
    start_seconds = (
        meas_date.hour * 3600 + meas_date.minute * 60 + meas_date.second
    )
    duration_seconds = int(round(raw.n_times / raw.info["sfreq"]))
    return start_seconds, duration_seconds, float(raw.info["sfreq"])


def load_expected_seizure_counts() -> dict[str, int]:
    with SUBJECT_INFO_FILE.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, skipinitialspace=True)
        return {
            row["patient_id"]: int(row["number_seizures"])
            for row in reader
            if row.get("patient_id")
        }


def main():
    seizure_map = {
        edf_path.name: []
        for edf_path in sorted(RAW_DATA_PATH.glob("PN*/*.edf"), key=lambda p: p.name)
    }
    parsed_counts = {}

    for patient_dir in sorted(RAW_DATA_PATH.iterdir(), key=lambda p: p.name):
        if not patient_dir.is_dir():
            continue

        pid = patient_dir.name
        seizure_list = patient_dir / f"Seizures-list-{pid}.txt"
        assert seizure_list.exists()

        text = seizure_list.read_text(encoding="utf-8", errors="replace")
        events, _ = parse_seizure_list(text, pid)
        parsed_counts[pid] = len(events)

        for event in events:
            edf_path = patient_dir / event.file_name
            if not edf_path.exists():
                raise FileNotFoundError(edf_path)

            start_seconds, duration_seconds, sfreq = read_edf_metadata(edf_path)
            assert sfreq == SAMPLE_RATE, f"Sample rate mismatch for {edf_path}"

            start_offset = seconds_after_start(event.start, start_seconds)
            end_offset = seconds_after_start(event.end, start_seconds)
            while end_offset < start_offset:
                end_offset += 86400

            if end_offset > duration_seconds:
                raise ValueError(
                    f"Seizure exceeds EDF duration for {edf_path}: "
                    f"{start_offset}-{end_offset}s > {duration_seconds}s"
                )

            seizure_map[event.file_name].append([start_offset, end_offset])

    expected_counts = load_expected_seizure_counts()
    for pid, expected in expected_counts.items():
        parsed = parsed_counts.get(pid, 0)
        if parsed != expected:
            raise ValueError(
                f"Seizure count mismatch for {pid}: parsed {parsed}, expected {expected}"
            )

    for events in seizure_map.values():
        events.sort(key=lambda x: x[0])

    with SEIZURE_FILE.open("w", encoding="utf-8") as fp:
        json.dump(seizure_map, fp, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
