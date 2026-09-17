# GRU absolute state

**gru_ta** (41,188 params, 7 channels) versus **gru_ta_abs** (41,572 params, 9 channels).

Per-subject means. Deltas carry 95% CIs from a paired subject-level cluster bootstrap; `sig` marks intervals that exclude zero.

Note that significance and practical size are separate questions: a small delta whose interval excludes zero is a real effect that happens to be small.

## h = 15 min  (base rate 0.0382)

| metric      |   before |   after |   delta | CI                 | sig   |   change_% |
|:------------|---------:|--------:|--------:|:-------------------|:------|-----------:|
| auroc       |   0.9926 |  0.9924 | -0.0001 | [-0.0003, +0.0001] | no    |    -0.0104 |
| auprc       |   0.8597 |  0.8662 |  0.0065 | [+0.0002, +0.0149] | yes   |     0.7603 |
| ppv         |   0.7171 |  0.7332 |  0.0160 | [+0.0101, +0.0225] | yes   |     2.2369 |
| recall      |   0.8302 |  0.8246 | -0.0055 | [-0.0134, +0.0013] | no    |    -0.6672 |
| f1          |   0.7680 |  0.7750 |  0.0070 | [+0.0022, +0.0120] | yes   |     0.9163 |
| specificity |   0.9888 |  0.9896 |  0.0008 | [+0.0004, +0.0011] | yes   |     0.0791 |

Constraint met: 28/38 -> 25/38   |   worst-subject recall 0.702 -> 0.695

## h = 30 min  (base rate 0.0534)

| metric      |   before |   after |   delta | CI                 | sig   |   change_% |
|:------------|---------:|--------:|--------:|:-------------------|:------|-----------:|
| auroc       |   0.9719 |  0.9722 |  0.0003 | [-0.0002, +0.0008] | no    |     0.0348 |
| auprc       |   0.7482 |  0.7537 |  0.0055 | [+0.0007, +0.0117] | yes   |     0.7350 |
| ppv         |   0.4602 |  0.4835 |  0.0232 | [+0.0166, +0.0299] | yes   |     5.0499 |
| recall      |   0.8254 |  0.8130 | -0.0124 | [-0.0174, -0.0079] | yes   |    -1.4965 |
| f1          |   0.5876 |  0.6030 |  0.0154 | [+0.0099, +0.0208] | yes   |     2.6140 |
| specificity |   0.9499 |  0.9543 |  0.0044 | [+0.0032, +0.0058] | yes   |     0.4659 |

Constraint met: 27/38 -> 25/38   |   worst-subject recall 0.571 -> 0.571

## h = 60 min  (base rate 0.0816)

| metric      |   before |   after |   delta | CI                 | sig   |   change_% |
|:------------|---------:|--------:|--------:|:-------------------|:------|-----------:|
| auroc       |   0.9242 |  0.9245 |  0.0004 | [-0.0007, +0.0014] | no    |     0.0399 |
| auprc       |   0.6277 |  0.6316 |  0.0039 | [+0.0008, +0.0072] | yes   |     0.6160 |
| ppv         |   0.3124 |  0.3341 |  0.0217 | [+0.0164, +0.0272] | yes   |     6.9385 |
| recall      |   0.8075 |  0.7909 | -0.0167 | [-0.0225, -0.0112] | yes   |    -2.0644 |
| f1          |   0.4471 |  0.4651 |  0.0179 | [+0.0134, +0.0225] | yes   |     4.0055 |
| specificity |   0.8515 |  0.8640 |  0.0124 | [+0.0091, +0.0159] | yes   |     1.4600 |

Constraint met: 22/38 -> 21/38   |   worst-subject recall 0.595 -> 0.561

## h = 120 min  (base rate 0.1314)

| metric      |   before |   after |   delta | CI                 | sig   |   change_% |
|:------------|---------:|--------:|--------:|:-------------------|:------|-----------:|
| auroc       |   0.8407 |  0.8428 |  0.0021 | [-0.0001, +0.0047] | no    |     0.2504 |
| auprc       |   0.5328 |  0.5363 |  0.0036 | [+0.0009, +0.0069] | yes   |     0.6734 |
| ppv         |   0.2722 |  0.2897 |  0.0175 | [+0.0127, +0.0229] | yes   |     6.4324 |
| recall      |   0.7788 |  0.7557 | -0.0231 | [-0.0318, -0.0151] | yes   |    -2.9706 |
| f1          |   0.3998 |  0.4138 |  0.0140 | [+0.0097, +0.0188] | yes   |     3.4986 |
| specificity |   0.7037 |  0.7309 |  0.0272 | [+0.0189, +0.0364] | yes   |     3.8681 |

Constraint met: 21/38 -> 17/38   |   worst-subject recall 0.505 -> 0.488

## Summary

16 of 24 metric-horizon combinations improved with a 95% CI excluding zero.
