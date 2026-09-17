# Research experiments archive

This directory preserves exploratory code that informed the thesis but is **not** part of the recommended public reproduction path.

- `pilots/` — pilot runners and earlier proposed recurrent variants.
- `diagnostics/` — one-off diagnostics, error analyses, treatment/context probes, and supporting scripts.
- `legacy_xai/` — superseded XAI runners/engines. The final thesis XAI path is `src/run_xai_v3_2_1.py` + `src/xai_engine_v3_2.py`.
- `patches/` — historical patch scripts retained only for provenance.
- `tuning/` — exploratory sweeps and optimization scripts.

The files are retained so the research trail is auditable, but they may assume the historical working directory or require `PYTHONPATH=src`. They are not required for the smoke test or the main reproduction commands in `docs/REPRODUCIBILITY.md`.
