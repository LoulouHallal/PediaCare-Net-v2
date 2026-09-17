# GitHub release checklist

## Recommended repository metadata

**Repository name:** `PediaCare-Net-v2`

**Description:**
> Reproducibility code for a Master's thesis on pediatric hypoglycemia prediction: threshold-aware time-series representation, TSL-GRU, benchmarking, deployment efficiency, and XAI.

**Suggested topics:**
`pytorch`, `time-series`, `healthcare-ai`, `diabetes`, `hypoglycemia`, `continuous-glucose-monitoring`, `gru`, `explainable-ai`, `reproducible-research`, `deep-learning`

## Before the first public push

Run from the repository root:

```bash
python scripts/repository_audit.py
python tests/smoke_test.py
```

Confirm that no raw data, derived arrays, checkpoints, private split-reference
files, or patient identifiers are staged:

```bash
git status --short
git ls-files | grep -Ei '\.(pt|pth|ckpt|npy|npz|parquet)$' && echo "CHECK THESE FILES" || true
git grep -nE 'IOBP|private_split_reference' -- ':!docs/GITHUB_RELEASE.md' || true
```

## Initial Git commands

Create an empty GitHub repository first, without adding a README or licence in
the GitHub UI. Then run:

```bash
git init
git branch -M main
git add .
git commit -m "release: PediaCare-Net v2 thesis reproducibility code"
git remote add origin https://github.com/<USERNAME>/PediaCare-Net-v2.git
git push -u origin main
```

Replace `<USERNAME>` with the GitHub account or organization that will own the
repository.

## First release

After the push and after the GitHub Actions smoke test is green:

```bash
git tag -a v1.0.0 -m "PediaCare-Net v2 thesis reproducibility release"
git push origin v1.0.0
```

On GitHub, create a release from `v1.0.0`. A suitable release title is:

> PediaCare-Net v2 — Thesis Reproducibility Release

Suggested release note:

> Initial thesis reproducibility release. Includes portable preprocessing,
> class-imbalance experiments, baseline/advanced model benchmarks,
> threshold-aware representation experiments, the proposed TSL-GRU,
> efficiency/deployment benchmarks, final XAI v3.2.1 code, aggregate thesis
> tables, and dataset-free verification tests. Restricted patient data and
> trained checkpoints are intentionally excluded.

## Visibility

If dataset/provider or university publication rules have not yet been checked,
create the repository as **private** first. It can be switched to public after
those permissions are confirmed. The code is licensed under MIT; external
patient data are not covered by that licence.
