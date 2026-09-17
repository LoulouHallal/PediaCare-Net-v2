# Experiment map

This document separates the **final thesis reproduction path** from exploratory research code.

## Final / thesis-facing path

| Purpose | Script(s) | Main output |
|---|---|---|
| Portable configuration and fixed split | `src/config.py`, `src/splits.py`, `src/common.py` | paths, subject masks, metrics |
| Rebuild causal windows | `src/build_windows.py` | `data_derived/windows_stride6.npz`, `windows_eval.npz` |
| RQ1 / classical imbalance | `src/stage1_v2.py`, `src/balancing.py`, `src/imbalance_ensembles.py` | classical model JSON summaries |
| Deep baselines | `src/stage2_deep.py` | GRU/LSTM results |
| Advanced baselines | `src/stage345_advanced.py` | TCN/Transformer/Hybrid results |
| Threshold-aware representation | `src/ta_gru.py` | 5-channel vs 7-channel evidence |
| Absolute-state extension | `src/absolute_state.py` | 7-channel vs 9-channel evidence |
| Proposed recurrent cell | `src/tsl_gru.py` | four-arm TSL ablation and checkpoints |
| Mechanism intervention | `src/tsl_intervene.py` | trained-cell intervention results |
| Computational cost | `src/efficiency_benchmark.py` | `results/tables/T10_efficiency.*` |
| Deployment latency | `src/deployment_benchmark.py` | deployment benchmark CSV |
| Final XAI | `src/run_xai_v3_2_1.py`, `src/xai_engine_v3_2.py` | population/patient IG + counterfactual sensitivity |

The final headline tables use the `any` label set: a positive label means at least one future CGM value is below 70 mg/dL within the requested horizon. The preprocessing code also retains a persistence/episode label set for controlled label-definition analyses.

## Representation progression

The final representation analysis keeps architecture, split, loss, and protocol fixed while changing only inputs:

- 5 channels: glucose, basal, bolus, carbohydrates, carbohydrate-recording indicator.
- 7 channels: add threshold proximity and downward glucose velocity.
- 9 channels: add absolute glucose state and absolute rate/change.

The stored representation table reports 15-minute GRU AUPRC 0.7700 for 5 channels, 0.8594 with the two threshold-aware channels, and 0.8640 after the absolute-state extension. The TCN replication changes from 0.7590 to 0.8562 when the threshold-aware representation is added.

## TSL-GRU comparison

The stored three-seed comparison uses the same 7-channel threshold-aware inputs. GRU+TA has 41,188 trainable parameters and mean AUPRC 0.8594 at 15 minutes; TSL-GRU has 15,974 parameters and mean AUPRC 0.8599. The project therefore treats predictive performance as tied while emphasizing the 61% parameter reduction. The efficiency table records approximately 2.29 MMAC/window for GRU+TA and 0.79 MMAC/window for TSL-GRU.

## Research archive

`research_experiments/` contains old pilots, diagnostics, patches, tuning utilities, and superseded XAI implementations. They are retained for provenance, not for the recommended verification workflow.
