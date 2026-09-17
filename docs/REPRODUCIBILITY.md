# Reproducibility guide

## 1. Clone and install

```bash
git clone <repository-url>
cd PediaCare-Net-v2
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-smoke.txt
```

The thesis environment recorded Python 3.13.15 and PyTorch 2.11.0+cu128. The minimal smoke-test requirements use the same recorded PyTorch/NumPy/scikit-learn versions. PyTorch 2.11.0 provides CPython 3.13 wheels; GPU reproduction still depends on the machine's NVIDIA driver/CUDA setup.

## 2. Verify the code without patient data

```bash
python scripts/repository_audit.py
python tests/smoke_test.py
```

A successful run ends with:

```text
PediaCare-Net smoke test: PASSED
```

The smoke test verifies repository-relative paths, the historical 170/36/38 split logic on a synthetic 244-subject cohort, the actual 7-channel TSL-GRU implementation, its 15,974 trainable parameters, and a finite `(batch, 4)` forward pass for 60 timesteps.

## 3. Install the full experiment environment and configure restricted data

```bash
pip install -r requirements.txt
```

Follow `data/README.md`. The default expected files are:

```text
data/raw/metabonet_public.parquet
data/external/metabonet_windows_pediatric.npz
```

They can instead remain anywhere on disk by setting `PEDIACARE_METABONET_RAW` and `PEDIACARE_METABONET_WINDOWS`.

Check configured paths:

```bash
python src/config.py
```

## 4. Rebuild causal windows

The thesis uses 60 observations (5 hours at 5-minute sampling), a 288-reading warm-up, causal expanding normalization, and subject-preserving windows. Build the training representation with stride 6 and the evaluation representation with stride 1:

```bash
python src/build_windows.py --stride 6 --tag stride6
python src/build_windows.py --stride 1 --tag eval
```

The outputs are written to `data_derived/` by default and are intentionally ignored by Git.

## 5. Quick real-data execution check

After `windows_stride6.npz` exists:

```bash
python scripts/quick_realdata_test.py --tag stride6 --train 128 --val 64
```

This performs one optimizer step on real training windows and a held-out validation forward pass. It is only an execution/integrity check; its tiny-sample loss is not a thesis result.

## 6. RQ1 — class imbalance

The rebuilt classical path supports the final label sets and standard balancing methods:

```bash
python src/stage1_v2.py --model rf --balance all --labels any --tag stride6
python src/stage1_v2.py --model xgboost --balance all --labels any --tag stride6
python src/stage1_v2.py --collect
```

Deep sequence balancing/loss experiments are run through `stage2_deep.py`, for example:

```bash
python src/stage2_deep.py --model gru --loss weighted_bce --balance none --labels any --tag stride6 --seed 42
python src/stage2_deep.py --model lstm --loss weighted_bce --balance none --labels any --tag stride6 --seed 42
```

## 7. RQ2 — architecture progression

Deep baselines:

```bash
python src/stage2_deep.py --model all --loss weighted_bce --balance none --labels any --tag stride6 --seed 42
```

Advanced baselines:

```bash
python src/stage345_advanced.py --model core --loss weighted_bce --balance none --labels any --tag stride6 --seed 42
```

Threshold-aware representation:

```bash
python src/ta_gru.py --model all --labels any --tag stride6 --seed 42
```

Absolute glucose state/rate extension:

```bash
python src/absolute_state.py --model all --tag stride6 --seed 42
```

Proposed TSL-GRU and frozen four-arm ablation:

```bash
python src/tsl_gru.py --model all --labels any --tag stride6 --seed 42
python src/tsl_gru.py --model all --labels any --tag stride6 --seed 43
python src/tsl_gru.py --model all --labels any --tag stride6 --seed 44
python src/tsl_gru.py --collect
```

Mechanism intervention on trained TSL-GRU checkpoints:

```bash
python src/tsl_intervene.py --variant tsl_gru --labels any --tag stride6 --seed 42
```

## 8. Efficiency and deployment

Computational-cost benchmark:

```bash
python src/efficiency_benchmark.py --only gru_ta_7ch,tsl_gru
```

To regenerate the broad architecture table, omit `--only`. Several comparison architecture definitions remain in `src/` because the efficiency table imports them directly.

Deployment benchmark with eager and compiled PyTorch paths:

```bash
python src/deployment_benchmark.py --only gru_ta_7ch,tsl_gru
```

`torch.compile` results are hardware/software dependent. The scientific claims about parameter count and analytic MACs are architecture properties; latency must be reported together with the device/runtime on which it was measured.

## 9. RQ3 — final XAI v3.2.1

XAI requires the trained checkpoint (not committed to Git) at the naming/location expected by `src/tsl_gru.py`, plus the derived real-data arrays.

Preflight:

```bash
python src/run_xai_v3_2_1.py --models tsl_gru --horizon 30 --check_only
```

Small verification run:

```bash
python src/run_xai_v3_2_1.py --models tsl_gru --horizon 30 --n_pop 20 --n_case 3 --n_heatmaps 3 --ig_steps 8 --n_iter 30 --n_cf 4
```

Full thesis configuration:

```bash
python src/run_xai_v3_2_1.py --models tsl_gru --horizon 30 --n_pop 2000 --n_case 0 --n_heatmaps 6 --ig_steps 50 --n_iter 200 --n_cf 6
```

Counterfactual outputs are model-sensitivity / what-if explanations. They are not causal clinical treatment recommendations.

## 10. Split identity

The historical split function is now local in `src/splits.py`; no old `PediaCare-Net` repository is imported. The algorithm uses `np.unique`, `np.random.default_rng(42)`, then 70% / 15% integer-floor slicing. With 244 subjects it yields 170 train, 36 validation, and 38 test subjects.

The public repository intentionally omits historical subject identifier lists. Authorized reviewers may set `PEDIACARE_SPLIT_REFERENCE` to a private JSON reference containing `validation_subjects` and `test_subjects`; `common.verify_split()` will then check exact assignments in addition to the 170/36/38 counts.

## 11. What is intentionally not in Git

Do not commit raw pediatric records, generated multi-GB window arrays, checkpoints, resumable `.ckpt` files, private Drive content, patient/subject identifier lists, per-subject JSON outputs, or transient XAI plots. These are excluded by `.gitignore` or intentionally pruned from the public release.
