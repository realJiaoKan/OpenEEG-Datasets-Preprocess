# [Siena Scalp EEG Database](https://physionet.org/content/siena-scalp-eeg/1.0.0/)

Interictal-Preictal preprocessing follows the CHB-MIT pipeline:

```bash
./.venv/bin/python "Siena Scalp EEG/gen_seizure_map.py"
./.venv/bin/python "Siena Scalp EEG/gen_file_summary.py"
./.venv/bin/python "Siena Scalp EEG/group_label_and_split.py"
./.venv/bin/python "Siena Scalp EEG/slice.py"
./.venv/bin/python "Siena Scalp EEG/interictal_preictal_preprocess.py"
./.venv/bin/python "Siena Scalp EEG/interictal_preictal_postprocess.py"
./.venv/bin/python "Siena Scalp EEG/spectrogram.py"
```

The first two commands generate `Datasets/Data/seizure_map.json` and
`Datasets/Data/Processed/files_summary.json`.

`settings.CHANNELS` uses a CHB-like spatial order: left lateral chain, left
parasagittal chain, right parasagittal chain, right lateral chain, then midline;
each chain is ordered from anterior to posterior.
