"""
File summary — Parse the summary text file to extract file names and their start/end times,
then group them and compute absolute times, and integrate ictal segments into the patient info.
"""

import re
import json

from settings import RAW_DATA_PATH, PROCESSED_DATA_PATH

SEIZURE_FILE = PROCESSED_DATA_PATH / "seizure_map.json"
PROC_FILE = PROCESSED_DATA_PATH / "files_summary.json"

PROC_FILE.parent.mkdir(parents=True, exist_ok=True)


def parse_summary_text(text):
    """
    Parse the summary text file to extract file names and their start/end times.
    The start and end times are converted to absolute times, and the entries are
    grouped based on a maximum gap of 2 hours (7200 seconds) between them.
    Each group is assigned a unique key, and the start and end times are
    represented as relative times from the start of the group.
    The function returns a dictionary with the following structure:
    {
        "group_1": {
            "file_name_1": {
                "start_original": "original_start_time",
                "end_original": "original_end_time",
                "start_relative": relative_start_time,
                "end_relative": relative_end_time
            },
            ...
        },
        ...
    }
    """
    # Extract entries
    pattern = re.compile(
        r"File Name: (?P<name>.+?)\s+File Start Time: (?P<start>[\d:]+)\s+File End Time: (?P<end>[\d:]+)",
        re.MULTILINE,
    )
    entries = []
    for m in pattern.finditer(text):
        entries.append(
            {
                "name": m.group("name").strip(),
                "start_str": m.group("start").strip(),
                "end_str": m.group("end").strip(),
            }
        )

    # Parse h:m:s
    def parse_hms(hms):
        h, mm, ss = map(int, hms.split(":"))
        return h, mm, ss

    # Compute absolute times with cross-day inference
    prev_abs_end = None
    prev_end_day = 0
    for e in entries:
        sh, sm, ss = parse_hms(e["start_str"])
        sec_start = (sh % 24) * 3600 + sm * 60 + ss
        day = prev_end_day
        abs_start = day * 86400 + sec_start
        if prev_abs_end is not None and abs_start <= prev_abs_end:
            day += 1
            abs_start = day * 86400 + sec_start

        eh, em, es = parse_hms(e["end_str"])
        sec_end = (eh % 24) * 3600 + em * 60 + es
        day_offset = eh // 24
        abs_end = (day + day_offset) * 86400 + sec_end

        e["abs_start"] = abs_start
        e["abs_end"] = abs_end
        prev_abs_end = abs_end
        prev_end_day = day + day_offset

    # Sort by abs_start
    entries.sort(key=lambda x: x["abs_start"])

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

        summary_file = patient_dir / f"{patient_dir.name}-summary.txt"
        assert summary_file.exists()

        text = summary_file.read_text(encoding="utf-8")
        parsed = parse_summary_text(text)

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
