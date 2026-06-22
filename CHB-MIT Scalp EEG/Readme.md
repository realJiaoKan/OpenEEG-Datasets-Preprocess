# [CHB-MIT Scalp EEG Database](https://physionet.org/content/chbmit/1.0.0/)

This directory is the CHB-MIT adapter for the shared
[Interictal-Preictal pipeline](../Readme.md). Read the root README first for
the shared label priority, slicing, balancing, NPZ contracts, and the meaning
of every stage below. This document records choices that are specific to CHB.

## Raw Data Layout

The scripts expect:

```text
CHB-MIT Scalp EEG/
  Data/
    Raw/
      SUBJECT-INFO...                 # whitespace-separated subject metadata
      chbNN/
        chbNN-summary.txt             # recording file start/end times
        *.edf
    seizure_map.json                  # curated annotation input
```

`gen_file_summary.py` writes `Data/Processed/files_summary.json`. It parses
the `File Name`, `File Start Time`, and `File End Time` entries in each
`chbNN-summary.txt`, infers midnight rollover from sequential files, and starts
a new group whenever the gap from the preceding recording exceeds two hours.
Within each group it converts every file's time bounds into seconds relative to
the group start, then adds the matching ictal intervals from
`Data/seizure_map.json`.

### Curated Seizure Map

Unlike Siena, this repository does not generate CHB seizure annotations from a
script. `Data/seizure_map.json` is a required manually prepared input. Its
contract is:

```json
{
  "chb01_03.edf": [[2996, 3036]],
  "chb01_04.edf": []
}
```

Every EDF filename must have an entry; intervals are `[start_sec, end_sec]`
relative to that EDF's start. Update and review this map before regenerating
`files_summary.json`; errors here directly change label windows and fold
membership.

## CHB-Specific Channel Curation

CHB EDF files do not expose a uniform montage across all patients and files.
The raw data included in this workspace was manually screened before this
pipeline so that every retained EDF can supply all 18 configured bipolar
derivations. The waveform stage asserts this exact order for every file:

```text
FP1-F7, F7-T7, T7-P7, P7-O1,
FP1-F3, F3-C3, C3-P3, P3-O1,
FP2-F4, F4-C4, C4-P4, P4-O2,
FP2-F8, F8-T8, T8-P8, P8-O2,
FZ-CZ, CZ-PZ
```

These are **18 bipolar derivations**, not 18 individual electrode positions.
Do not compare their channel axis directly with Siena's 29 single-position
channels.

Some CHB EDFs contain duplicated `TP-P8` data that MNE exposes as `T8-P8-0`
and `T8-P8-1`. `group_label_and_split.py` removes `T8-P8-1`, renames
`T8-P8-0` to `T8-P8`, and only then applies the fixed channel order. A missing
or differently named required channel fails the stage assertion deliberately;
it must be resolved by a documented curation decision rather than silently
zero-filling or reordering data.

`settings.DATASETS` lists candidate IDs but is not consulted by the current
processing scripts. The actual cohort is determined by the CHB raw
directories and the corresponding summary/seizure-map entries.

## Sampling And Spectrogram Details

- Sampling rate: **256 Hz**.
- Time-domain window shape: `(N, 18, 1280)` because 5 seconds x 256 Hz = 1280.
- The power-line environment is 60 Hz. Spectrograms remove DC and
  `[57, 63]` Hz plus `[117, 123]` Hz before log-amplitude conversion.
- Frequency resolution is 0.2 Hz; final spectrogram shape is
  `(N, 1, 18, 578)`.

## Run Order

The seizure map must already exist. Then run:

```bash
./.venv/bin/python "CHB-MIT Scalp EEG/gen_file_summary.py"
./.venv/bin/python "CHB-MIT Scalp EEG/group_label_and_split.py"
./.venv/bin/python "CHB-MIT Scalp EEG/slice.py"
./.venv/bin/python "CHB-MIT Scalp EEG/interictal_preictal_preprocess.py"
./.venv/bin/python "CHB-MIT Scalp EEG/interictal_preictal_postprocess.py"
./.venv/bin/python "CHB-MIT Scalp EEG/spectrogram.py"
```

## Current Processed Cohort And Exclusions

The curated raw tree currently contains 16 patients:

```text
chb01 chb02 chb03 chb05 chb07 chb08 chb09 chb10
chb14 chb16 chb17 chb18 chb19 chb20 chb21 chb23
```

This is the post-curation CHB cohort, not the complete public CHB-MIT subject
list. Subjects absent from this tree were excluded before these scripts ran
because the required uniform 18-channel input was not retained for this
project. The repository does not contain a more granular per-subject rejection
log; do not infer a clinical exclusion reason from absence alone.

For the current outputs, all 16 curated subjects have both accepted preictal
and interictal windows, so all reach the final EEG and Spectrogram stages. The
generated `files_summary.json` contains 63 recording groups and 90 ictal
events; the final tree contains 58 balanced folds. Fold counts by patient are:

```text
chb01:5  chb02:3  chb03:3  chb05:3  chb07:3  chb08:5
chb09:4  chb10:7  chb14:5  chb16:3  chb17:2  chb18:4
chb19:2  chb20:3  chb21:3  chb23:3
```

These counts are properties of the current curated source and annotation map;
they should be regenerated rather than hard-coded by downstream consumers.
