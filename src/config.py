"""
config.py
=========
Portable paths and experiment constants for PediaCare-Net v2.

The original research workspace lived under Google Drive.  The public
repository must not depend on that layout, so every default path is now
repository-relative and may be overridden with environment variables.

No scientific constants are changed here: split seed, horizons, window
length, metric definitions, and evaluation constraints are preserved.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str, default: Path) -> Path:
    """Return an expanded absolute path from an environment override."""
    value = os.environ.get(name)
    path = Path(value).expanduser() if value else Path(default)
    return path.resolve()


# ─── ROOTS ────────────────────────────────────────────────────────────────────

# By default this is the repository root (the parent of src/).  A Colab/Drive
# user can still override it, but a normal `git clone` needs no configuration.
_REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _env_path("PEDIACARE_PROJECT_ROOT", _REPO_ROOT)

SRC = PROJECT_ROOT / "src"
DATA_ROOT = _env_path("PEDIACARE_DATA_DIR", PROJECT_ROOT / "data")
RESULTS = _env_path("PEDIACARE_RESULTS_DIR", PROJECT_ROOT / "results")
DATA_DERIVED = _env_path("PEDIACARE_DERIVED_DIR", PROJECT_ROOT / "data_derived")

# ─── DATA INPUTS ──────────────────────────────────────────────────────────────

# These files are deliberately NOT distributed with the repository.  See
# data/README.md for provenance/licensing notes and the expected layout.
DATA = {
    # Historical window file used by the original classical-ML experiments and
    # as the exact 244-subject cohort source when rebuilding windows.
    "metabonet": _env_path(
        "PEDIACARE_METABONET_WINDOWS",
        DATA_ROOT / "external" / "metabonet_windows_pediatric.npz",
    ),

    # Optional external-validation datasets.
    "t1duom": _env_path(
        "PEDIACARE_T1DUOM_WINDOWS",
        DATA_ROOT / "external" / "t1duom_windows.npz",
    ),
    "ohio": _env_path(
        "PEDIACARE_OHIO_WINDOWS",
        DATA_ROOT / "external" / "ohio_windows.npz",
    ),
}

RAW_DATA = {
    "metabonet": _env_path(
        "PEDIACARE_METABONET_RAW",
        DATA_ROOT / "raw" / "metabonet_public.parquet",
    ),
}

# Optional PRIVATE split-reference file. It is intentionally not distributed
# because subject identifiers may be governed by the dataset's access terms.
# Authorized users can point PEDIACARE_SPLIT_REFERENCE to a local JSON file
# containing validation_subjects/test_subjects for exact assignment checks.
SPLIT_REFERENCE = _env_path(
    "PEDIACARE_SPLIT_REFERENCE",
    DATA_ROOT / "private_split_reference.json",
)

# ─── PROJECT LAYOUT ───────────────────────────────────────────────────────────

RQ1 = RESULTS / "RQ1_balancing"
RQ2 = RESULTS / "RQ2_models"
RQ3 = RESULTS / "RQ3_xai"

RQ2_STAGES = {
    "stage1_ml": RQ2 / "stage1_classical_ml",
    "stage2_dl": RQ2 / "stage2_classical_dl",
    "stage3_tcn": RQ2 / "stage3_tcn",
    "stage4_transformer": RQ2 / "stage4_transformer",
    "stage5_hybrid": RQ2 / "stage5_hybrid",
}

ALL_DIRS = [SRC, RESULTS, DATA_DERIVED, RQ1, RQ2, RQ3] + list(RQ2_STAGES.values())

# ─── EXPERIMENT CONSTANTS ─────────────────────────────────────────────────────

HORIZONS = [15, 30, 60, 120]

# MUST stay 42: this is the historical subject split used throughout the thesis.
SPLIT_SEED = 42

MIN_RECALL = 0.80
CALIB_MIN_RECALL = 0.85

METRICS = ["auroc", "auprc", "ppv", "recall", "f1", "specificity"]

N_BOOT = 10_000
N_BOOT_POOLED = 500

# Historical materialised MetaboNet file: channels 4-11 were all-zero, so
# classical summary features use the four live channels by default.
LIVE_CHANNELS = 4
N_CHANNELS = 12
WINDOW_LEN = 60

FEATURE_NAMES = [
    "glucose", "bolus_decay", "carbs_active", "basal_dose",
    "motion_intensity", "step_count", "active_kcal", "met",
    "sleep_light_frac", "sleep_rem_frac", "sleep_deep_frac",
    "sleep_awake_frac",
]


def ensure_dirs():
    """Create writable repository output directories."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)
    return ALL_DIRS


def check_data():
    """Report availability of configured data inputs."""
    return {
        **{name: path.exists() for name, path in DATA.items()},
        **{f"raw_{name}": path.exists() for name, path in RAW_DATA.items()},
    }


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print("Creating output directories...")
    for d in ensure_dirs():
        print(f"  {d}")
    print("\nData availability:")
    all_paths = {**DATA, **{f"raw_{k}": v for k, v in RAW_DATA.items()}}
    for name, ok in check_data().items():
        print(f"  {'OK     ' if ok else 'MISSING'} {name:>14}  ->  {all_paths[name]}")
