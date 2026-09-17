# Public repository audit

This cleaned snapshot was prepared from the filtered thesis workspace. The scientific result files were preserved; the changes are repository engineering and documentation changes rather than experimental changes.

## Completed checks

- Replaced the hardcoded Google Drive project root with repository-relative defaults and environment-variable overrides.
- Removed the runtime import from the old `PediaCare-Net` repository.
- Copied the historical `subject_level_split` logic into `src/splits.py` without changing its RNG, fractions, or integer-floor split logic.
- Removed public subject-identifier lists and per-subject JSON outputs. Exact assignment checking remains available through an optional private `PEDIACARE_SPLIT_REFERENCE` file for authorized reviewers.
- Updated raw MetaboNet path references to use portable configuration.
- Moved 46 pilot/diagnostic/patch/legacy-XAI Python files out of the main `src/` path into `research_experiments/`.
- Added a dataset-free smoke test and a small real-data end-to-end execution test.
- Added README, dependency files, data instructions, reproduction commands, experiment map, `.gitignore`, MIT code licence, research-use disclaimer, citation metadata, GitHub Actions smoke test, and a release checklist.
- Checked that no `.pt`, `.pth`, `.ckpt`, `.npy`, `.npz`, or `.parquet` data/model artifact is present in this public snapshot.
- `python -m compileall src tests scripts` passes.
- `python tests/smoke_test.py` passes and confirms 15,974 TSL-GRU parameters and output shape `(4, 4)` for a synthetic `(4, 60, 7)` input.
- The analytic benchmark code independently returns 41,188 parameters / 2.294656 MMAC for GRU+TA and 15,974 / 0.789376 MMAC for TSL-GRU.

## What cannot be re-run in this snapshot alone

The filtered ZIP intentionally contains no raw pediatric dataset, generated multi-GB window arrays, or trained checkpoints. Therefore full preprocessing, performance reproduction, the real-data quick test, and final XAI cannot be executed until the authorized data and/or checkpoint files are supplied locally.

The exact 170 training IDs cannot be listed from this filtered archive alone because the archive does not contain the full 244-subject metadata array. The deterministic split logic is preserved locally; authorized reviewers can additionally supply a private assignment reference if required.

## Licence and privacy note

The repository uses the MIT licence for the research code. Dataset licensing is separate and no patient/subject identifier lists are included in the public release. Aggregate thesis tables are retained, while per-subject result JSONs are excluded.
