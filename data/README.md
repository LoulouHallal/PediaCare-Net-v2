# Data setup

The large patient-level data files used by this thesis are **not stored in the Git repository**. The code remains public and portable, while raw and derived data are obtained separately according to the source-data access and redistribution terms.

## 1. Original MetaboNet data

Official source:

- **MetaboNet data portal:** https://metabo-net.org/data

The thesis preprocessing pipeline uses the standardized MetaboNet parquet file:

```text
metabonet_public.parquet
```

Place it at:

```text
data/raw/metabonet_public.parquet
```

or point the code to another location with `PEDIACARE_METABONET_RAW`.

The raw parquet must expose the fields consumed by `src/build_windows.py`, including `id`, `date`, `CGM`, `basal`, `bolus`, `carbs`, and `insulin`.

## 2. Pediatric cohort reference

The historical thesis cohort contains **244 pediatric subjects**. Reconstruction of the original cohort selection confirmed the rule:

```text
age_first < 18 years
```

The exact cohort used by the historical experiments is preserved in:

```text
metabonet_windows_pediatric.npz
```

For the released preprocessing code, this file is used as the source of the exact 244 subject identifiers. `src/build_windows.py` then returns to `metabonet_public.parquet`, selects those subjects, and rebuilds the causal compact representation from the raw records. The old feature values inside `metabonet_windows_pediatric.npz` are therefore **not** the source of the final `windows_stride6.npz` feature values.

Default location:

```text
data/external/metabonet_windows_pediatric.npz
```

## 3. Thesis-derived data package

A separate data package may be used to avoid regenerating the large derived arrays from the raw parquet.

**Download:** `https://drive.google.com/drive/folders/16LMOLM-NG2w6ktN8AZbZaO2q4pkPjLsL`


Expected package contents:

| File | Purpose |
|---|---|
| `metabonet_windows_pediatric.npz` | Historical pediatric window artifact preserving the exact 244-subject cohort used by the thesis. |
| `windows_stride6.npz` | Main causally preprocessed stride-6 representation used for training. |
| `windows_stride6_meta.json` | Counts, positive rates, preprocessing settings, and other metadata for the stride-6 representation. |
| `windows_eval.npz` | Dense stride-1 representation used for validation/test evaluation. |
| `windows_eval_meta.json` | Metadata associated with the dense evaluation representation. |
| `raw_treatment_stride6.npz` | Raw/absolute basal, bolus, and carbohydrate values aligned to the processed timeline for diagnostics and explanation workflows. |
| `patient_context_stride6.npz` | Experimental 48-hour patient-context feature set; not part of the final seven-channel TSL-GRU input. |
| `nadir_targets_stride6.npz` | Experimental auxiliary future-glucose-nadir targets. |
| `features_stride6.npy` | Fixed-length summary features used by the classical machine-learning path. |
| `cgm_timeline_metabonet.parquet` | Real-unit CGM timeline with timestamps used for event-level/monitoring analyses. |
| `cgm_timeline_metabonet_coverage.json` | Per-subject CGM coverage and monitoring statistics for the timeline file. |

## 4. Two supported reproduction routes

### Route A — reproduce preprocessing from the original data

Required inputs:

```text
data/raw/metabonet_public.parquet
data/external/metabonet_windows_pediatric.npz
```

Then run:

```bash
python src/build_windows.py --stride 6 --tag stride6
python src/build_windows.py --stride 1 --tag eval
```

This produces the final causal training/evaluation representations under `data_derived/`.

### Route B — start from the thesis-derived representations

Download the derived-data package and place at minimum these files in `data_derived/`:

```text
data_derived/windows_stride6.npz
data_derived/windows_stride6_meta.json
data_derived/windows_eval.npz
data_derived/windows_eval_meta.json
```

The model experiments can then be run without rebuilding the windows from the raw parquet. Other files from the package are required only by the experiment families that use them.

## 5. Environment-variable overrides

Data do not have to be copied into the repository. Paths can be configured with:

```bash
export PEDIACARE_METABONET_RAW=/path/to/metabonet_public.parquet
export PEDIACARE_METABONET_WINDOWS=/path/to/metabonet_windows_pediatric.npz
export PEDIACARE_T1DUOM_WINDOWS=/path/to/t1duom/windows.npz
export PEDIACARE_OHIO_WINDOWS=/path/to/ohio/windows.npz
export PEDIACARE_DERIVED_DIR=/path/to/generated/windows
```

This is useful in Google Colab/Drive and keeps a normal Git clone portable.

## 6. Historical split verification

The public repository preserves the deterministic subject-level split algorithm and verifies the expected **170 train / 36 validation / 38 test** counts for a 244-subject cohort.

If an authorized reviewer has a private reference file containing `validation_subjects` and `test_subjects`, point the code to it with:

```bash
export PEDIACARE_SPLIT_REFERENCE=/secure/path/private_split_reference.json
```

`common.verify_split()` will then compare the exact assignments when the full 244-subject cohort is loaded. Keep private split-reference files outside Git.
