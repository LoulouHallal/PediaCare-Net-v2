# Data setup

The pediatric patient data used by this thesis are **not distributed in this repository**. Do not commit raw records, large derived arrays, or private Google Drive files unless you have explicit permission and the dataset licence permits redistribution.

## Expected layout

```text
data/
├── raw/
│   └── metabonet_public.parquet
├── external/
│   ├── metabonet_windows_pediatric.npz
│   ├── t1duom_windows.npz          # optional
│   └── ohio_windows.npz            # optional
└── private_split_reference.json    # optional, local-only; never commit
```

`metabonet_windows_pediatric.npz` is the historical window file used by the original classical-ML path and, in `build_windows.py`, as the exact source of the 244-subject cohort IDs. The raw parquet is then rebuilt into the causal compact representation used by the final deep-learning experiments.

The raw parquet is expected to expose the columns used by `src/build_windows.py`: `id`, `date`, `CGM`, `basal`, `bolus`, `carbs`, and `insulin`.

## Environment-variable overrides

You do not have to copy data into the repository. Paths can be configured with:

```bash
export PEDIACARE_METABONET_RAW=/path/to/metabonet_public.parquet
export PEDIACARE_METABONET_WINDOWS=/path/to/metabonet_windows_pediatric.npz
export PEDIACARE_T1DUOM_WINDOWS=/path/to/t1duom/windows.npz
export PEDIACARE_OHIO_WINDOWS=/path/to/ohio/windows.npz
export PEDIACARE_DERIVED_DIR=/path/to/generated/windows
```

This is useful in Google Colab/Drive and also keeps a normal Git clone portable.

## Historical split verification

The public repository preserves the exact deterministic split algorithm and verifies the expected 170/36/38 subject counts. It intentionally does **not** publish historical subject identifier lists.

If an authorized reviewer has a private reference file containing `validation_subjects` and `test_subjects`, point the code to it with:

```bash
export PEDIACARE_SPLIT_REFERENCE=/secure/path/private_split_reference.json
```

`common.verify_split()` will then compare the exact assignments when the full 244-subject cohort is loaded. Keep this file outside Git or at `data/private_split_reference.json`, which is ignored by `.gitignore`.
