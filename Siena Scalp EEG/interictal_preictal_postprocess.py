import numpy as np
import json

from settings import RANDOM_SEED, PROCESSED_DATA_PATH

IN_DIR = PROCESSED_DATA_PATH / "Interictal-Preictal" / "Raw"
OUT_DIR = PROCESSED_DATA_PATH / "Interictal-Preictal" / "EEG"

OUT_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(RANDOM_SEED)


def main():
    for pid_dir in IN_DIR.iterdir():
        if not pid_dir.is_dir():
            continue
        pid = pid_dir.name
        inter_file = pid_dir / "interictal_combined.npz"
        pre_files = sorted(pid_dir.glob("preictal_*.npz"))
        k = len(pre_files)
        if k == 0 or not inter_file.exists():
            continue

        # Load interictal data
        inter_data = np.load(inter_file)["X"]
        n_inter = inter_data.shape[0]

        # Shuffle and split interictal windows
        indices = np.arange(n_inter)
        np.random.shuffle(indices)
        parts = np.array_split(indices, k)
        stats = {}
        out_pid_dir = OUT_DIR / pid
        out_pid_dir.mkdir(parents=True, exist_ok=True)

        for i, pre_file in enumerate(pre_files):
            pre_data = np.load(pre_file)["X"]
            n_pre = pre_data.shape[0]
            inter_idx = parts[i]

            # Sample to balance counts
            if len(inter_idx) > n_pre:
                sel_inter = np.random.choice(inter_idx, n_pre, replace=False)
            else:
                sel_inter = inter_idx
            if n_pre > len(inter_idx):
                sel_pre = np.random.choice(n_pre, len(inter_idx), replace=False)
            else:
                sel_pre = np.arange(n_pre)
            X_inter = inter_data[sel_inter]
            X_pre = pre_data[sel_pre]

            # Combine and label
            X = np.concatenate([X_inter, X_pre], axis=0)
            y = np.concatenate(
                [np.zeros(len(X_inter), dtype=int), np.ones(len(X_pre), dtype=int)],
                axis=0,
            )

            # Shuffle combined data
            perm = np.random.permutation(len(y))
            X_shuffled = X[perm]
            y_shuffled = y[perm]

            # Save fold
            fold_file = out_pid_dir / f"fold_{i}.npz"

            # Save fold
            np.savez_compressed(fold_file, X=X_shuffled, y=y_shuffled)
            stats[f"fold_{i}"] = {"interictal": len(X_inter), "preictal": len(X_pre)}

        # Save stats for patient
        stats_file = out_pid_dir / "stats.json"
        with open(stats_file, "w", encoding="utf-8") as sf:
            json.dump(stats, sf, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
