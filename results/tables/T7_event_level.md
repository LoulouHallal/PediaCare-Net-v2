# T7 — Event-level comparison

Episodes: >= 3 consecutive CGM readings < 70 mg/dL. Alarms are grouped with a refractory period equal to the prediction horizon and matched to an episode if they fall in the warning window before onset. CIs are subject-level cluster bootstrap.

`event_f1` is the harmonic mean of episode recall and alarm precision. AUROC, AUPRC and specificity have no event-level equivalent: there is no scored population and no well-defined true-negative episode.

Runs on a different label set are excluded: episode-onset labels define a different prediction task with a different base rate, so they do not belong in the same comparison.

## Selected operating points

**Caveat:** each model's threshold was chosen independently on validation, so these rows sit at different points on their own curves. Use the matched-budget table below for a like-for-like model comparison.

| model                             |   horizon |   threshold | persistence   |   episode_recall |   fa_per_day |   alarm_precision |   event_f1 |   median_lead_min |   window_ppv_same_thr |
|:----------------------------------|----------:|------------:|:--------------|-----------------:|-------------:|------------------:|-----------:|------------------:|----------------------:|
| gru__weighted_bce__none__any__h30 |        30 |      0.9000 | 1of1          |           0.8617 |       2.2885 |            0.3032 |     0.4486 |           14.2763 |                0.5435 |
| rf__none__any__h30                |        30 |      0.2500 | 1of1          |           0.8912 |       2.9536 |            0.2688 |     0.4130 |           13.4868 |                0.4605 |
| bagging__none__any__h30           |        30 |      0.2500 | 1of1          |           0.9026 |       3.2870 |            0.2476 |     0.3886 |           13.6184 |                0.4475 |
| xgboost__none__any__h30           |        30 |      0.2500 | 1of1          |           0.8743 |       3.0606 |            0.2491 |     0.3878 |           12.8947 |                0.4594 |

## Episode recall at a matched false-alarm budget

Interpolated from each model's validation Pareto front. This is the fair comparison: same alarm burden for every model.

| model                             |   recall@1.0FA |   recall@2.0FA |   recall@3.0FA |   recall@4.0FA |
|:----------------------------------|---------------:|---------------:|---------------:|---------------:|
| bagging__none__any__h30           |         0.5242 |         0.7255 |         0.8453 |         0.9022 |
| gru__weighted_bce__none__any__h30 |         0.6210 |         0.7016 |         0.8287 |         0.8967 |
| rf__none__any__h30                |         0.5370 |         0.7384 |         0.8463 |         0.9018 |
| xgboost__none__any__h30           |         0.4998 |         0.7084 |         0.8213 |         0.8806 |
