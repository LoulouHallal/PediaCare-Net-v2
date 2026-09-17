# PediaCare-Net v2

**PediaCare-Net v2** is the reproducibility repository for a Master's thesis on early pediatric hypoglycemia prediction from continuous glucose monitoring, insulin, and carbohydrate time series. The repository contains the preprocessing pipeline, class-imbalance experiments, classical/deep/advanced baselines, the threshold-aware representation, the proposed **Trajectory-Scaled Light GRU (TSL-GRU)**, efficiency/deployment benchmarks, and the final XAI v3.2.1 workflow.

> Patient data and trained checkpoints are intentionally not distributed here. See [`data/README.md`](data/README.md).

## Fastest verification for an instructor

After cloning the repository:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-smoke.txt
python scripts/repository_audit.py
python tests/smoke_test.py
```

The final line should be:

```text
PediaCare-Net smoke test: PASSED
```

This test needs **no patient data**. It imports the real final TSL-GRU code, instantiates the 7-channel / 60-timestep model, verifies the historical 244-subject split logic, confirms the model has **15,974 trainable parameters**, and checks a finite four-horizon forward pass.

> **Research-use only:** this repository is not a medical device and must not be used for clinical diagnosis or treatment decisions. See [`DISCLAIMER.md`](DISCLAIMER.md).

## Thesis protocol at a glance

- Pediatric cohort: **244 subjects**, split at subject level into **170 train / 36 validation / 38 test** with seed 42.
- CGM sampling: 5 minutes; input window: **60 observations = 5 hours**.
- Prediction horizons: **15, 30, 60, and 120 minutes**.
- Final headline event target: any future glucose value **< 70 mg/dL** within the horizon; the code also retains a persistence/episode label set for sensitivity analysis.
- Preprocessing: causal expanding per-subject normalization, warm-up of 288 observations, subject/segment boundary preservation, and bounded normalized values in the final workflow.
- Representation progression: 5 base channels → 7 threshold-aware channels → optional 9-channel absolute-state extension.

The central representation result is cross-architecture: at 15 minutes, the stored thesis table reports GRU mean AUPRC **0.7700 → 0.8594** when threshold-aware features are added; TCN changes **0.7590 → 0.8562** under the same representation change. The 9-channel GRU extension reaches **0.8640** in that representation table.

For the proposed cell, the three-seed table reports **GRU+TA: 41,188 parameters, AUPRC 0.8594 ± 0.0022** and **TSL-GRU: 15,974 parameters, AUPRC 0.8599 ± 0.0062** at 15 minutes. The thesis therefore treats predictive performance as tied while reporting about **61% fewer parameters**. The computational-cost table reports approximately **2.29 MMAC/window** for GRU+TA and **0.79 MMAC/window** for TSL-GRU.

## Repository structure

```text
PediaCare-Net-v2/
├── src/                    # final pipeline + architecture definitions used by benchmarks
├── tests/                  # dataset-free smoke test
├── scripts/                # lightweight verification helpers
├── docs/                   # reproduction and experiment documentation
├── data/                   # data-placement instructions; no patient measurements
├── data_derived/           # generated locally; Git-ignored
├── results/                # aggregate thesis tables only; no per-subject outputs
└── research_experiments/   # archived pilots, diagnostics, patches, older XAI
```

## Portable paths

The original research workspace used `/content/drive/MyDrive/...`. That dependency has been removed. Paths now default to the cloned repository and can be overridden with environment variables, e.g.:

```bash
export PEDIACARE_METABONET_RAW=/path/to/metabonet_public.parquet
export PEDIACARE_METABONET_WINDOWS=/path/to/metabonet_windows_pediatric.npz
```

Google Colab remains supported: clone/place the repository anywhere in Drive and either use the repository-relative `data/` layout or set the variables above.

## Reproducing the experiments

For the full experiment pipeline, install `requirements.txt` after the smoke test. The full command sequence is documented in [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md). The experiment/script map is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

Typical progression:

```bash
# preprocess
python src/build_windows.py --stride 6 --tag stride6
python src/build_windows.py --stride 1 --tag eval

# deep baseline
python src/stage2_deep.py --model gru --loss weighted_bce --balance none --labels any --tag stride6 --seed 42

# threshold-aware representation
python src/ta_gru.py --model all --labels any --tag stride6 --seed 42

# proposed TSL-GRU
python src/tsl_gru.py --model all --labels any --tag stride6 --seed 42

# efficiency / deployment
python src/efficiency_benchmark.py --only gru_ta_7ch,tsl_gru
python src/deployment_benchmark.py --only gru_ta_7ch,tsl_gru

# final XAI v3.2.1 (requires trained checkpoint)
python src/run_xai_v3_2_1.py --models tsl_gru --horizon 30 --check_only
```

## Reproducibility safeguards

The repository deliberately preserves the scientific protocol rather than "cleaning" it into a different experiment. In particular, the horizons, seed, subject-level split algorithm, 60-step history, threshold-aware channel construction, TSL-GRU implementation, and aggregate thesis tables are unchanged.

For privacy and dataset-governance reasons, public files do **not** contain the historical patient/subject identifier lists or per-subject JSON outputs. Authorized users can provide a private split-reference file through `PEDIACARE_SPLIT_REFERENCE` for exact assignment verification. No raw pediatric data, large generated arrays, checkpoints, private Drive files, or patient identifiers should be pushed to GitHub.

## License and citation

The research code is provided under the permissive [MIT License](LICENSE). It allows reuse, modification, redistribution, and commercial use provided the copyright and licence notice are retained; the software is supplied without warranty. External datasets are **not** relicensed by this repository and remain governed by their own access/redistribution terms.

Academic users can cite the project using [`CITATION.cff`](CITATION.cff). The first-release checklist and suggested GitHub metadata are in [`docs/GITHUB_RELEASE.md`](docs/GITHUB_RELEASE.md).
