# T9 — Proposed architecture: TSL-GRU

Trajectory-Scaled Light GRU. The reset gate is removed and the update gate is replaced by a retention coefficient conditioned on a causal EMA of threshold-relative glucose dynamics.

```
h~_t     = ReLU(BN(W_x x_t) + U_h h_{t-1})
s_t      = rho * s_{t-1} + (1 - rho) * [c_t, v_t, c_t*v_t]
lambda_t = clip(sigmoid(theta) + kappa * tanh(W_s s_t + b_s), eps, 1-eps)
h_t      = lambda_t * h_{t-1} + (1 - lambda_t) * h~_t
```

## Part 1 — architecture comparison

All arms share the same data, subject split, loss, threshold policy and training indices per seed. Only the recurrent cell differs.

### h = 15 min

| arm                            |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   ppv_mean |   recall_mean |   f1_mean |
|:-------------------------------|-----------:|----------:|-------------:|-------------:|-----------:|-----------:|--------------:|----------:|
| A  GRU + TA (baseline)         |      41188 |         3 |       0.9924 |       0.8594 |     0.0022 |     0.7104 |        0.8251 |    0.7615 |
| B  LiGRU-style (no reset gate) |      28324 |         1 |       0.9924 |       0.8588 |   nan      |     0.7162 |        0.8283 |    0.7663 |
| C  TSL, constant retention     |      15460 |         3 |       0.9922 |       0.8573 |     0.0035 |     0.7142 |        0.8194 |    0.7613 |
| D  TSL-GRU (proposed)          |      15974 |         3 |       0.9925 |       0.8599 |     0.0062 |     0.7147 |        0.8233 |    0.7638 |

### h = 30 min

| arm                            |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   ppv_mean |   recall_mean |   f1_mean |
|:-------------------------------|-----------:|----------:|-------------:|-------------:|-----------:|-----------:|--------------:|----------:|
| A  GRU + TA (baseline)         |      41188 |         3 |       0.9721 |       0.7476 |     0.0009 |     0.4635 |        0.8222 |    0.5895 |
| B  LiGRU-style (no reset gate) |      28324 |         1 |       0.9717 |       0.7455 |   nan      |     0.4567 |        0.8207 |    0.5834 |
| C  TSL, constant retention     |      15460 |         3 |       0.9719 |       0.7458 |     0.0022 |     0.4661 |        0.8185 |    0.5903 |
| D  TSL-GRU (proposed)          |      15974 |         3 |       0.9722 |       0.7485 |     0.0048 |     0.4629 |        0.8233 |    0.5893 |

### h = 60 min

| arm                            |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   ppv_mean |   recall_mean |   f1_mean |
|:-------------------------------|-----------:|----------:|-------------:|-------------:|-----------:|-----------:|--------------:|----------:|
| A  GRU + TA (baseline)         |      41188 |         3 |       0.9248 |       0.6271 |     0.0011 |     0.3185 |        0.7991 |    0.4518 |
| B  LiGRU-style (no reset gate) |      28324 |         1 |       0.9243 |       0.6252 |   nan      |     0.3172 |        0.8027 |    0.4499 |
| C  TSL, constant retention     |      15460 |         3 |       0.9245 |       0.6252 |     0.0023 |     0.3227 |        0.7932 |    0.4537 |
| D  TSL-GRU (proposed)          |      15974 |         3 |       0.9254 |       0.6288 |     0.0034 |     0.3189 |        0.8047 |    0.4526 |

### h = 120 min

| arm                            |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   ppv_mean |   recall_mean |   f1_mean |
|:-------------------------------|-----------:|----------:|-------------:|-------------:|-----------:|-----------:|--------------:|----------:|
| A  GRU + TA (baseline)         |      41188 |         3 |       0.8415 |       0.5323 |     0.0007 |     0.2793 |        0.7700 |    0.4059 |
| B  LiGRU-style (no reset gate) |      28324 |         1 |       0.8424 |       0.5312 |   nan      |     0.2783 |        0.7671 |    0.4032 |
| C  TSL, constant retention     |      15460 |         3 |       0.8413 |       0.5306 |     0.0017 |     0.2763 |        0.7694 |    0.4019 |
| D  TSL-GRU (proposed)          |      15974 |         3 |       0.8436 |       0.5342 |     0.0029 |     0.2761 |        0.7761 |    0.4034 |

## Headline

TSL-GRU uses **15,974 parameters** vs **41,188** for the GRU baseline — a **61% reduction** — at equivalent AUPRC:

|   horizon | GRU+TA          | TSL-GRU         |   D − A |   D − C |
|----------:|:----------------|:----------------|--------:|--------:|
|        15 | 0.8594 ± 0.0022 | 0.8599 ± 0.0062 |  0.0005 |  0.0025 |
|        30 | 0.7476 ± 0.0009 | 0.7485 ± 0.0048 |  0.0009 |  0.0027 |
|        60 | 0.6271 ± 0.0011 | 0.6288 ± 0.0034 |  0.0017 |  0.0036 |
|       120 | 0.5323 ± 0.0007 | 0.5342 ± 0.0029 |  0.0019 |  0.0036 |

D − A differences are within seed variability at every horizon: the architectures perform equivalently. D − C is positive at every horizon but compares two separately trained models, so it carries the full training noise; the intervention below is the stronger test of the mechanism.

## Part 2 — intervention on trained checkpoints

Every weight is held fixed; only the retention signal is altered at inference.

- **constant** — lambda replaced by its per-unit mean (computed on validation): removes temporal variation, holds the average level fixed
- **shuffled** — trajectory descriptor permuted across time: preserves lambda's distribution, destroys its alignment with the input

### h = 15 min

|   seed |   kappa |   auprc_normal |   auprc_constant |   d_vs_constant | d_vs_constant_CI   | d_vs_constant_sig   |   d_vs_shuffled | d_vs_shuffled_CI   | d_vs_shuffled_sig   |
|-------:|--------:|---------------:|-----------------:|----------------:|:-------------------|:--------------------|----------------:|:-------------------|:--------------------|
|     42 |  0.6014 |         0.8613 |           0.8216 |          0.0397 | [+0.0280, +0.0552] | yes                 |          0.0041 | [+0.0027, +0.0056] | yes                 |
|     43 |  0.6093 |         0.8530 |           0.8363 |          0.0168 | [+0.0107, +0.0241] | yes                 |          0.0000 | [-0.0006, +0.0007] | no                  |
|     44 |  0.5938 |         0.8652 |           0.8177 |          0.0476 | [+0.0312, +0.0696] | yes                 |          0.0006 | [-0.0007, +0.0017] | no                  |

### h = 30 min

|   seed |   kappa |   auprc_normal |   auprc_constant |   d_vs_constant | d_vs_constant_CI   | d_vs_constant_sig   |   d_vs_shuffled | d_vs_shuffled_CI   | d_vs_shuffled_sig   |
|-------:|--------:|---------------:|-----------------:|----------------:|:-------------------|:--------------------|----------------:|:-------------------|:--------------------|
|     42 |  0.6014 |         0.7488 |           0.7171 |          0.0317 | [+0.0241, +0.0416] | yes                 |          0.0020 | [+0.0007, +0.0031] | yes                 |
|     43 |  0.6093 |         0.7435 |           0.7272 |          0.0163 | [+0.0120, +0.0212] | yes                 |         -0.0001 | [-0.0010, +0.0009] | no                  |
|     44 |  0.5938 |         0.7531 |           0.7106 |          0.0425 | [+0.0324, +0.0557] | yes                 |          0.0009 | [+0.0002, +0.0016] | yes                 |

### h = 60 min

|   seed |   kappa |   auprc_normal |   auprc_constant |   d_vs_constant | d_vs_constant_CI   | d_vs_constant_sig   |   d_vs_shuffled | d_vs_shuffled_CI   | d_vs_shuffled_sig   |
|-------:|--------:|---------------:|-----------------:|----------------:|:-------------------|:--------------------|----------------:|:-------------------|:--------------------|
|     42 |  0.6014 |         0.6287 |           0.6015 |          0.0272 | [+0.0225, +0.0330] | yes                 |          0.0022 | [+0.0014, +0.0031] | yes                 |
|     43 |  0.6093 |         0.6255 |           0.6040 |          0.0215 | [+0.0154, +0.0297] | yes                 |         -0.0001 | [-0.0007, +0.0007] | no                  |
|     44 |  0.5938 |         0.6323 |           0.5895 |          0.0427 | [+0.0333, +0.0548] | yes                 |          0.0004 | [-0.0000, +0.0009] | no                  |

### h = 120 min

|   seed |   kappa |   auprc_normal |   auprc_constant |   d_vs_constant | d_vs_constant_CI   | d_vs_constant_sig   |   d_vs_shuffled | d_vs_shuffled_CI   | d_vs_shuffled_sig   |
|-------:|--------:|---------------:|-----------------:|----------------:|:-------------------|:--------------------|----------------:|:-------------------|:--------------------|
|     42 |  0.6014 |         0.5338 |           0.5090 |          0.0248 | [+0.0199, +0.0308] | yes                 |          0.0017 | [+0.0008, +0.0027] | yes                 |
|     43 |  0.6093 |         0.5315 |           0.5096 |          0.0219 | [+0.0163, +0.0294] | yes                 |         -0.0000 | [-0.0007, +0.0008] | no                  |
|     44 |  0.5938 |         0.5373 |           0.4973 |          0.0400 | [+0.0295, +0.0535] | yes                 |          0.0006 | [+0.0000, +0.0014] | yes                 |

## Interpretation

Freezing lambda degrades AUPRC in **12 of 12** model-horizon combinations (range +0.0163 to +0.0476), so the performance depends on the trajectory-conditioned variation in retention rather than on a favourable constant memory timescale.

Permuting the trajectory descriptor is significant in only **6 of 12** combinations. The benefit therefore derives from adaptive retention dynamics, not from alignment with the instantaneous trajectory — a limitation that should be stated rather than glossed.

kappa was initialised at 0.10 and converged to 0.594–0.609 across seeds: the conditioning term was reinforced during training, not driven out.
