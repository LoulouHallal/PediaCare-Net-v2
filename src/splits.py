"""Historical subject-level split used by every thesis experiment.

This function is intentionally kept byte-for-byte equivalent in algorithmic
logic to the implementation supplied from the original workflow.  Do not
replace the RNG, sorting behaviour, fractions, or rounding without creating a
new experimental protocol.
"""

import numpy as np


def subject_level_split(meta: np.ndarray, seed: int = 42,
                        train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Split subjects (not windows) into train/validation/test masks.

    All windows belonging to one subject remain in exactly one split.
    With the 244-subject thesis cohort and seed 42 this yields 170/36/38
    train/validation/test subjects.
    """
    subjects = np.unique(meta)
    rng = np.random.default_rng(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_subj = set(subjects[:n_train])
    val_subj = set(subjects[n_train:n_train + n_val])
    test_subj = set(subjects[n_train + n_val:])

    print(f"Subject split: train={len(train_subj)} | "
          f"val={len(val_subj)} | test={len(test_subj)} "
          f"(total={n})")

    train_mask = np.isin(meta, list(train_subj))
    val_mask = np.isin(meta, list(val_subj))
    test_mask = np.isin(meta, list(test_subj))

    return train_mask, val_mask, test_mask
