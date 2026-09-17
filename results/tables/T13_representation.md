# Representation, not architecture

Same architectures, same split, same loss, same protocol. Only the
input representation differs.

    5 channels   glucose, basal, bolus, carbs, carbs_observed
    7 channels   + ta_proximity_c, ta_downslope_vplus
    9 channels   + glucose_absolute, rate_absolute

## h = 15 min

| family | 5-ch AUPRC | +TA (7-ch) | Δ TA | +abs (9-ch) | Δ abs | Δ total |
|---|---|---|---|---|---|---|
| GRU | 0.7700 ± 0.0041 (3) | 0.8594 ± 0.0022 (3) | **+0.0894** | 0.8640 | +0.0046 | **+0.0940** |
| TCN | 0.7590 ± 0.0031 (3) | 0.8562 | **+0.0971** | — | — | — |
| LSTM | 0.7657 ± 0.0050 (3) | — | — | — | — | — |
| Transformer | 0.7558 ± 0.0039 (3) | — | — | — | — | — |
| Hybrid | 0.7463 ± 0.0063 (3) | — | — | — | — | — |

## h = 30 min

| family | 5-ch AUPRC | +TA (7-ch) | Δ TA | +abs (9-ch) | Δ abs | Δ total |
|---|---|---|---|---|---|---|
| GRU | 0.6760 ± 0.0022 (3) | 0.7476 ± 0.0009 (3) | **+0.0716** | 0.7528 | +0.0052 | **+0.0768** |
| TCN | 0.6669 ± 0.0022 (3) | 0.7433 | **+0.0765** | — | — | — |
| LSTM | 0.6753 ± 0.0046 (3) | — | — | — | — | — |
| Transformer | 0.6632 ± 0.0026 (3) | — | — | — | — | — |
| Hybrid | 0.6587 ± 0.0009 (3) | — | — | — | — | — |

## h = 60 min

| family | 5-ch AUPRC | +TA (7-ch) | Δ TA | +abs (9-ch) | Δ abs | Δ total |
|---|---|---|---|---|---|---|
| GRU | 0.5704 ± 0.0028 (3) | 0.6271 ± 0.0011 (3) | **+0.0567** | 0.6313 | +0.0042 | **+0.0609** |
| TCN | 0.5650 ± 0.0003 (3) | 0.6233 | **+0.0583** | — | — | — |
| LSTM | 0.5713 ± 0.0034 (3) | — | — | — | — | — |
| Transformer | 0.5623 ± 0.0020 (3) | — | — | — | — | — |
| Hybrid | 0.5606 ± 0.0007 (3) | — | — | — | — | — |

## h = 120 min

| family | 5-ch AUPRC | +TA (7-ch) | Δ TA | +abs (9-ch) | Δ abs | Δ total |
|---|---|---|---|---|---|---|
| GRU | 0.4905 ± 0.0020 (3) | 0.5323 ± 0.0007 (3) | **+0.0418** | 0.5365 | +0.0042 | **+0.0460** |
| TCN | 0.4881 ± 0.0015 (3) | 0.5307 | **+0.0426** | — | — | — |
| LSTM | 0.4914 ± 0.0025 (3) | — | — | — | — | — |
| Transformer | 0.4860 ± 0.0019 (3) | — | — | — | — | — |
| Hybrid | 0.4859 ± 0.0011 (3) | — | — | — | — | — |

Values are per-subject mean AUPRC; ± is the standard deviation and
(n) the number of seeds where more than one was run.

## Reading this

Threshold-aware features improve the GRU by +0.0894 AUPRC at h=15.
They improve the TCN by +0.0971 — a replication across a different
computational family, which is what makes the finding a property of the
representation rather than of one model.

For comparison, the GRU's own run-to-run spread over five seeds is
0.00207 mean AUPRC, and no architectural modification tested in this
project exceeded it.

Under causal per-patient z-scoring, 70 mg/dL maps to a different
normalised value for every child and drifts as their running
statistics update, so the network cannot locate the clinical
threshold. TA restores it in units of that child's own variability.


# Within the TA representation: proposed cells vs their baseline

Once TA is supplied, the architecture comparison becomes fair —
every arm below reads the same 7 channels. This is the table the
TSL-GRU efficiency claim rests on: it must TIE on AUPRC for the
parameter reduction to mean anything.

| model | params | AUPRC h=15 | AUPRC h=30 | AUPRC h=60 | AUPRC h=120 | Δ vs GRU+TA (h=15) |
|---|---|---|---|---|---|---|
| GRU + TA | 41,188 | 0.8594 ± 0.0022 | 0.7476 ± 0.0009 | 0.6271 ± 0.0011 | 0.5323 ± 0.0007 | — |
| TSL-GRU | 15,974 | 0.8599 ± 0.0062 | 0.7485 ± 0.0048 | 0.6288 ± 0.0034 | 0.5342 ± 0.0029 | +0.0005 |
| TSL-static | 15,460 | 0.8573 ± 0.0035 | 0.7458 ± 0.0022 | 0.6252 ± 0.0023 | 0.5306 ± 0.0017 | -0.0020 |
| LiGRU-TA | 28,324 | 0.8588 | 0.7455 | 0.6252 | 0.5312 | -0.0006 |
| TCN + TA | 91,620 | 0.8562 | 0.7433 | 0.6233 | 0.5307 | -0.0032 |
| DMS-TCN | 31,988 | 0.8572 | 0.7434 | 0.6246 | 0.5315 | -0.0021 |
| MSD-TCN | 76,708 | 0.8504 | 0.7397 | 0.6202 | 0.5263 | -0.0090 |

TSL-GRU differs from GRU+TA by +0.0005 AUPRC at h=15, inside the GRU's own 5-seed spread of 0.00207 — a tie, which is the claim.
It does so with 15,974 parameters against 41,188, 61% fewer.

All arms share hidden 64, 2 layers, dropout 0.2, Adam at 1e-3,
batch 512 and ReduceLROnPlateau with patience 6. Only the
recurrent cell differs, so the parameter reduction comes from
cell design and not from a narrower model. No per-architecture
tuning was performed; the shared settings were chosen on the
GRU baseline, which is the conservative direction.
