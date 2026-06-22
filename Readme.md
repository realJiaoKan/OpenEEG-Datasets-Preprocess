# Open EEG Datasets Preprocess

This repository preprocesses scalp EEG datasets into a common
Interictal-Preictal classification format. The dataset-specific raw-data
adapters live in the individual dataset directories; from waveform grouping
onward, the processing contract is the same.

## Supported Datasets

- [CHB-MIT Scalp EEG](CHB-MIT%20Scalp%20EEG/Readme.md)
- [Siena Scalp EEG](Siena%20Scalp%20EEG/Readme.md)

The dependencies are listed in `requirements.txt`. The scripts expect the raw
EDF files to be placed in the dataset-specific directory layout described in
the corresponding README.

## Shared Objective

The current pipeline only creates the binary **Interictal-Preictal** dataset.
Ictal, seizure-prediction-horizon (SPH), and transition windows are generated
as labels during the first stage so that temporal exclusions are explicit, but
they are not exported as model inputs by the current configuration.

Each retained example is a 5-second, per-channel z-scored EEG window. Final
folds are balanced binary datasets:

- `0`: interictal
- `1`: preictal

`fold_i.npz` samples are randomly shuffled. Their array order must not be
treated as chronological order.

## Common Configuration

The two datasets use the same time-domain and labeling configuration. Sampling
rate, selected channels, source file parsing, and power-line frequency masks
are dataset-specific and documented in each dataset README.

| Setting | Value | Meaning |
| --- | --- | --- |
| Window length | 5 seconds | Every exported EEG example has this duration. |
| Interictal step | 5 seconds | Non-overlapping interictal windows. |
| Preictal step | 1 second | Preictal windows overlap by 4 seconds. |
| Preictal transition | seizure onset -2.5 h to -0.5 h | Explicitly excluded from the binary dataset. |
| Preictal | seizure onset -30 min to -5 min | Positive class candidate interval. |
| SPH | seizure onset -5 min to onset | Explicitly excluded from the binary dataset. |
| Ictal | annotated onset to annotated end | Explicitly excluded from the binary dataset. |
| Postictal transition | seizure end to +1 h | Explicitly excluded from the binary dataset. |
| Minimum accepted preictal segment | 15 minutes | Shorter continuous preictal segments are not made into folds. |
| Interictal training ratio setting | 0.3 | Present in `settings.py`, but the current scripts do not read it. Final folds use the balancing procedure described below. |
| Random seed | 42 | Used in the Interictal-Preictal preparation and balancing stages. |

## Label Construction And Priority

The first waveform stage constructs a timeline for every recording group.
Files with an inter-file gap of at most two hours belong to the same group. EDF
waveforms are placed at their group-relative offsets. Gaps are represented by
zero-filled waveform space only as an intermediate representation and are
always labelled `blank`, so they never enter the final dataset.

Every sample initially receives `interictal`. Labels are written in the order
below; a later row takes priority when intervals overlap:

| Priority | Label | Interval for seizure `[t0, t1]` |
| --- | --- | --- |
| 0 | `interictal` | Default for recorded samples not covered below. |
| 1 | `preictal_transition_window` | `[t0 - 2.5 h, t0 - 0.5 h]` |
| 2 | `preictal` | `[t0 - 30 min, t0 - 5 min]` |
| 3 | `postictal_transition_window` | `[t1, t1 + 1 h]` |
| 4 | `sph` | `[t0 - 5 min, t0]` |
| 5 | `ictal` | `[t0, t1]` |
| 6 | `blank` | Any interval without a source EDF recording. |

The implementation clips all label windows to the group boundaries. It splits
the waveform whenever the label changes, then only slices `interictal` and
`preictal` segments. Each retained 5-second window is normalized independently
per channel using `(x - mean) / (std + 1e-8)`.

## Processing Stages

Run the commands from the dataset README in the stated order. Later stages
consume the files written by earlier stages.

### 0. Seizure Map And File Summary

The pipeline requires two pieces of temporal metadata before waveform loading:

1. `seizure_map.json`: maps each EDF filename to a list of seizure intervals
   measured in seconds relative to that EDF's start. EDF files with no seizure
   must still be present with an empty list.
2. `files_summary.json`: records each patient's groups, files, group-relative
   file bounds, and group-relative ictal intervals.

The raw-data adapter determines how these files are obtained. For example, one
dataset may provide a manually curated seizure map, while another may parse
per-patient seizure text files. This difference is documented locally.

### 1. `group_label_and_split.py`

**Input:** `files_summary.json`, EDF files, and subject metadata.

**Output:** `Processed/Label & Splited/<patient>_group_<n>.pkl`.

Each pickle contains:

```text
{
  "metadata": {"subject_id": ..., "gender": ..., "age": ...},
  "segments": [
    {"label": <label>, "data": {<channel>: [samples...], ...}},
    ...
  ]
}
```

The stage asserts the configured sampling rate and exact post-cleaning channel
order for every EDF. It retains all labels in the pickle so subsequent
experiments can deliberately enable other label classes.

### 2. `slice.py`

**Input:** `Label & Splited/*.pkl`.

**Output:**

- `Processed/Sliced/<patient>/interictal/<index>.pkl`
- `Processed/Sliced/<patient>/preictal/<index>.pkl`
- `Processed/sliced_summary.json`

Every output pickle is a list of normalized, fixed-length window dictionaries.
`sliced_summary.json` maps each saved segment index back to its patient, group,
label, and group-relative start/end seconds. Only `interictal` and `preictal`
are sliced by the current settings.

### 3. `interictal_preictal_preprocess.py`

**Input:** `Sliced/` and `sliced_summary.json`.

**Output:** `Processed/Interictal-Preictal/Raw/` and
`Processed/Interictal-Preictal/fold_summary.json`.

All a patient's interictal windows are concatenated into
`<patient>/interictal_combined.npz` with key `X`. Every continuous preictal
segment that lasts at least 15 minutes becomes `<patient>/preictal_<i>.npz`,
also with key `X`. `fold_summary.json` maps each accepted preictal index to its
source group and group-relative segment time.

At this point there are no class labels inside the NPZ files. A patient can
have preictal files but no `interictal_combined.npz`; such a patient cannot
produce a balanced final binary fold.

### 4. `interictal_preictal_postprocess.py`

**Input:** `Interictal-Preictal/Raw/`.

**Output:** `Interictal-Preictal/EEG/<patient>/fold_<i>.npz` and `stats.json`.

For a patient with `k` accepted preictal files, the stage shuffles all
interictal windows and partitions them into `k` disjoint parts. Each part is
paired with one preictal file. The larger class is sampled without replacement
down to the smaller class, the two classes are concatenated, labelled, and
shuffled.

Final EEG fold contract:

```text
X: (N, C, T)  # float64, normalized time-domain windows
y: (N,)       # int64; 0=interictal, 1=preictal
```

`stats.json` stores the per-fold count of each class. The stage skips a patient
if it has no preictal file or no combined interictal file.

### 5. `spectrogram.py`

**Input:** `Interictal-Preictal/EEG/`.

**Output:** `Interictal-Preictal/Spectrogram/` with the same patient/fold tree.

The script converts each 5-second channel independently using one Hann-window
STFT: `n_fft = win_length = hop_length = T`, `center=False`. Therefore each
window produces one temporal STFT column, which is flattened into the frequency
axis. It removes DC and the configured power-line bands, then applies
`20 * log10(abs(STFT) + 1e-8)`.

Final spectrogram fold contract:

```text
X: (N, 1, C, F)  # float32; singleton image channel, EEG-channel axis, frequency axis
y: (N,)          # copied unchanged from the corresponding EEG fold
```

`stats.json` is copied unchanged from the EEG tree. `F` depends on sample rate
and the dataset-specific frequency mask.

## Re-running Safely

The scripts create output directories but do not clean them. When raw files,
seizure annotations, settings, or channel selections change, remove the
generated `Processed/` tree and any generated `seizure_map.json` before a full
rerun. Otherwise, stale folds or files from a prior run may remain alongside
new outputs.

Generated EDF-derived data (`*.edf`, `*.pkl`, `*.npz`) is ignored by Git. Keep
the source scripts, settings, README files, and annotation-generation logic
under version control.
