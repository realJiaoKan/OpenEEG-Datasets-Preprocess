# [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)

This directory adapts Siena to the shared
[Interictal-Preictal pipeline](../Readme.md). Read the root README for the
common label priority, slicing, final-fold construction, final file contracts, and stage
semantics. This document describes the source-specific parsing and curation
required by Siena.

## Raw Data Layout

The scripts expect the PhysioNet data in this layout:

```text
Siena Scalp EEG/
  Data/
    subject_info.csv
    Raw/
      PNxx/
        Seizures-list-PNxx.txt
        *.edf
```

`subject_info.csv` supplies patient ID, sex, age, and the expected number of
seizures. The waveform stage copies sex and age into its intermediate pickle
metadata. The CSV seizure count is also used to validate the parsed annotations
before any waveform processing starts.

## Siena Seizure Annotation Adapter

Run `gen_seizure_map.py` before all other stages. It parses every
`Seizures-list-PNxx.txt` and writes:

```text
Data/seizure_map.json
```

The result uses the same CHB-compatible contract:

```json
{
  "PN01-1.edf": [[start_sec, end_sec]],
  "PN01-2.edf": []
}
```

All intervals are seconds relative to the actual EDF start, and every EDF is
included even when it has no seizure. The parser accepts both `:` and `.` time
separators, removes whitespace accidentally embedded inside time values, and
supports both `Seizure Start/End Time` and `Start/End Time` labels.

### Known Source Corrections

The text files contain several filename and timestamp errors. These are handled
explicitly in `gen_seizure_map.py`, rather than corrected manually in the raw
data:

| Source text value | EDF used by the pipeline | Reason |
| --- | --- | --- |
| `PN01.edf` | `PN01-1.edf` | Text filename omits the recording suffix. |
| `PNO6-1.edf`, `PNO6-2.edf`, `PNO6-4.edf` | `PN06-1.edf`, `PN06-2.edf`, `PN06-4.edf` | Letter `O` is used instead of digit `0`. |
| `PN11-.edf` | `PN11-1.edf` | Recording suffix is missing. |
| PN00-3 seizure `18:28:29` to `19:29:29` | `18:28:29` to `18:29:29` | The stated end hour exceeds the EDF end (`18:57`); the intended one-minute event is used. |

For every parsed event the script checks that the corrected EDF exists, its
sampling rate is 512 Hz, and the event lies within the recording duration. It
then verifies each patient's parsed seizure count against `subject_info.csv`.
A mismatch fails rather than silently creating a partial map.

## File Times And Grouping

`gen_file_summary.py` writes:

```text
Data/Processed/files_summary.json
```

Siena seizure-list registration times have small errors in some records. The
script uses the EDF header measurement time and actual sample-derived duration
as the authoritative recording start/end metadata, while using the text-file
order to order known files. EDFs not mentioned in the text are appended using a
natural filename order. Sequential header times are rolled across midnight,
then recordings separated by more than two hours start a new group. Seizure
intervals from `seizure_map.json` are converted to group-relative coordinates.

This header-based treatment is intentional: do not replace it with the raw
registration-time fields from the text files without independently resolving
their typos.

## Siena Channel Handling

Siena is processed at **512 Hz** and has 29 single-position EEG channels. They
are not CHB's 18 bipolar derivations. MNE reads Siena channels with an `EEG `
prefix; it also exposes `EEG FP2` and `EEG CZ` in uppercase in some files. The
waveform stage normalizes those names, then asserts and applies this exact
spatial order:

```text
F9, F7, Fc5, T3, Cp5, T5,
Fp1, F3, Fc1, C3, Cp1, P3, O1,
Fp2, F4, Fc2, C4, Cp2, P4, O2,
F10, F8, Fc6, T4, Cp6, T6,
Fz, Cz, Pz
```

The order is left lateral (anterior to posterior), left parasagittal,
right parasagittal, right lateral, then midline. It preserves spatial locality
for channel-as-image models. Do not rename Siena channels to CHB bipolar names
or drop them to 18 channels unless a separately validated montage transform is
introduced.

## Sampling And Spectrogram Details

- Sampling rate: **512 Hz**.
- Time-domain window shape: `(N, 29, 2560)` because 5 seconds x 512 Hz = 2560.
- The power-line environment is 50 Hz. Spectrograms remove DC and the five
  harmonics `[47, 53]`, `[97, 103]`, `[147, 153]`, `[197, 203]`, and
  `[247, 253]` Hz before log-amplitude conversion.
- Frequency resolution remains 0.2 Hz because the waveform window is 5
  seconds. The final spectrogram shape is `(N, 1, 29, 1125)`.
- Final folds retain all interictal and preictal windows; configure loss weights
  during training if class imbalance needs to be compensated.

## Run Order

Run all stages in order after placing raw data under `Data/Raw/`:

```bash
./.venv/bin/python "Siena Scalp EEG/gen_seizure_map.py"
./.venv/bin/python "Siena Scalp EEG/gen_file_summary.py"
./.venv/bin/python "Siena Scalp EEG/group_label_and_split.py"
./.venv/bin/python "Siena Scalp EEG/slice.py"
./.venv/bin/python "Siena Scalp EEG/interictal_preictal_preprocess.py"
./.venv/bin/python "Siena Scalp EEG/interictal_preictal_postprocess.py"
./.venv/bin/python "Siena Scalp EEG/spectrogram.py"
```

The first two commands generate `seizure_map.json` and `files_summary.json`.
The root README explains the inputs and outputs of every later stage.

## Current Processed Cohort And Exclusions

The current source tree contains 14 patients:

```text
PN00 PN01 PN03 PN05 PN06 PN07 PN09 PN10 PN11 PN12 PN13 PN14 PN16 PN17
```

The generated `files_summary.json` contains 32 groups and 47 ictal events. All
14 patients are processed through labelling, slicing, and preictal extraction.
The final EEG/Spectrogram dataset contains only these 7 patients:

```text
PN01 PN03 PN06 PN07 PN12 PN13 PN14
```

The other seven patients are not manually deleted and should not be interpreted
as clinical exclusions. They do have accepted preictal segments, but the
current shared labeling and slicing rules produce no
`interictal_combined.npz` for them. Since final folds require paired
interictal and preictal data, `interictal_preictal_postprocess.py`
skips them:

| Patient | Accepted preictal segments | Final-stage reason |
| --- | ---: | --- |
| PN00 | 1 | No combined interictal windows. |
| PN05 | 3 | No combined interictal windows. |
| PN09 | 3 | No combined interictal windows. |
| PN10 | 7 | No combined interictal windows. |
| PN11 | 1 | No combined interictal windows. |
| PN16 | 2 | No combined interictal windows. |
| PN17 | 2 | No combined interictal windows. |

The final 20 folds are distributed as follows:

```text
PN01:2  PN03:2  PN06:5  PN07:1  PN12:3  PN13:3  PN14:4
```

`settings.DATASETS` lists candidate IDs but is not read by the current run
scripts. Discover the actual final cohort from the `EEG/` or
`Spectrogram/` directory instead of hard-coding this list.
