# T8 — Results aggregated across seeds (any labels)

Each row is one configuration; `auprc_sd` and `auprc_range` are computed over independent training runs that differ only in the random seed.

**How to read this:** two configurations are only distinguishable if the gap between their `auprc_mean` values exceeds the `auprc_range` of either. Configurations closer than that are tied, however they happen to be ordered here.

## h = 15 min

| model            | method            |   stage |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   auprc_range |   ppv_mean |   recall_mean |   f1_mean |
|:-----------------|:------------------|--------:|-----------:|----------:|-------------:|-------------:|-----------:|--------------:|-----------:|--------------:|----------:|
| gru              | weighted_bce+none |       2 |      40804 |         3 |       0.9888 |       0.7700 |     0.0041 |        0.0083 |     0.3562 |        0.9455 |    0.4741 |
| lstm             | weighted_bce+none |       2 |      53668 |         3 |       0.9888 |       0.7657 |     0.0050 |        0.0098 |     0.3596 |        0.9437 |    0.4765 |
| gru              | bce+random_over   |       2 |      40804 |         1 |       0.9883 |       0.7615 |   nan      |      nan      |     0.3645 |        0.9367 |    0.4804 |
| tcn              | weighted_bce+none |       2 |     104548 |         3 |       0.9883 |       0.7590 |     0.0031 |        0.0060 |     0.3506 |        0.9451 |    0.4704 |
| gru              | focal+none        |       2 |      40804 |         1 |       0.9886 |       0.7579 |   nan      |      nan      |     0.3619 |        0.9440 |    0.4760 |
| lstm             | bce+random_over   |       2 |      53668 |         1 |       0.9881 |       0.7565 |   nan      |      nan      |     0.3471 |        0.9469 |    0.4668 |
| transformer      | weighted_bce+none |       2 |     104676 |         3 |       0.9878 |       0.7558 |     0.0039 |        0.0070 |     0.3405 |        0.9485 |    0.4597 |
| lstm             | bce+none          |       2 |      53668 |         1 |       0.9884 |       0.7526 |   nan      |      nan      |     0.3590 |        0.9477 |    0.4761 |
| gru              | bce+none          |       2 |      40804 |         1 |       0.9886 |       0.7521 |   nan      |      nan      |     0.3598 |        0.9477 |    0.4757 |
| hybrid           | weighted_bce+none |       2 |     213156 |         3 |       0.9876 |       0.7463 |     0.0063 |        0.0121 |     0.3585 |        0.9393 |    0.4782 |
| bagging          | none              |       1 |          0 |         1 |       0.9844 |       0.7200 |   nan      |      nan      |     0.3047 |        0.9419 |    0.4217 |
| rf               | none              |       1 |          0 |         1 |       0.9835 |       0.7122 |   nan      |      nan      |     0.3090 |        0.9455 |    0.4278 |
| xgboost          | random_over       |       1 |          0 |         1 |       0.9826 |       0.6973 |   nan      |      nan      |     0.3072 |        0.9346 |    0.4257 |
| xgboost          | class_weight      |       1 |          0 |         1 |       0.9831 |       0.6942 |   nan      |      nan      |     0.3003 |        0.9472 |    0.4217 |
| xgboost          | none              |       1 |          0 |         1 |       0.9831 |       0.6942 |   nan      |      nan      |     0.3003 |        0.9472 |    0.4217 |
| rf               | random_over       |       1 |          0 |         1 |       0.9833 |       0.6927 |   nan      |      nan      |     0.3111 |        0.9461 |    0.4302 |
| xgboost          | random_under      |       1 |          0 |         1 |       0.9829 |       0.6895 |   nan      |      nan      |     0.2987 |        0.9451 |    0.4170 |
| balanced_bagging | none              |       1 |          0 |         1 |       0.9840 |       0.6877 |   nan      |      nan      |     0.3063 |        0.9502 |    0.4226 |
| rf               | random_under      |       1 |          0 |         1 |       0.9826 |       0.6789 |   nan      |      nan      |     0.2997 |        0.9480 |    0.4165 |
| rf               | class_weight      |       1 |          0 |         1 |       0.9826 |       0.6745 |   nan      |      nan      |     0.3041 |        0.9486 |    0.4231 |
| xgboost          | smote_then_under  |       1 |          0 |         1 |       0.9814 |       0.6676 |   nan      |      nan      |     0.2957 |        0.9395 |    0.4146 |
| easy_ensemble    | none              |       1 |          0 |         1 |       0.9822 |       0.6668 |   nan      |      nan      |     0.2840 |        0.9462 |    0.3976 |
| xgboost          | smote             |       1 |          0 |         1 |       0.9808 |       0.6605 |   nan      |      nan      |     0.2945 |        0.9427 |    0.4139 |
| adaboost         | none              |       1 |          0 |         1 |       0.9820 |       0.6560 |   nan      |      nan      |     0.2809 |        0.9511 |    0.3958 |
| balanced_rf      | none              |       1 |          0 |         1 |       0.9814 |       0.6555 |   nan      |      nan      |     0.2968 |        0.9471 |    0.4147 |
| xgboost          | adasyn            |       1 |          0 |         1 |       0.9806 |       0.6507 |   nan      |      nan      |     0.2887 |        0.9413 |    0.4066 |
| rf               | adasyn            |       1 |          0 |         1 |       0.9817 |       0.6361 |   nan      |      nan      |     0.3058 |        0.9475 |    0.4244 |
| rf               | smote             |       1 |          0 |         1 |       0.9820 |       0.6358 |   nan      |      nan      |     0.3062 |        0.9474 |    0.4245 |
| rf               | smote_then_under  |       1 |          0 |         1 |       0.9808 |       0.6149 |   nan      |      nan      |     0.3035 |        0.9432 |    0.4214 |
| bagged_adaboost  | none              |       1 |          0 |         1 |       0.9526 |       0.4406 |   nan      |      nan      |     0.2342 |        0.9452 |    0.3450 |
| rusboost         | none              |       1 |          0 |         1 |       0.9612 |       0.3933 |   nan      |      nan      |     0.2339 |        0.9453 |    0.3446 |

## h = 30 min

| model            | method            |   stage |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   auprc_range |   ppv_mean |   recall_mean |   f1_mean |
|:-----------------|:------------------|--------:|-----------:|----------:|-------------:|-------------:|-----------:|--------------:|-----------:|--------------:|----------:|
| gru              | weighted_bce+none |       2 |      40804 |         3 |       0.9670 |       0.6760 |     0.0022 |        0.0044 |     0.2978 |        0.9142 |    0.4208 |
| lstm             | weighted_bce+none |       2 |      53668 |         3 |       0.9669 |       0.6753 |     0.0046 |        0.0088 |     0.2998 |        0.9128 |    0.4224 |
| gru              | bce+random_over   |       2 |      40804 |         1 |       0.9664 |       0.6727 |   nan      |      nan      |     0.2982 |        0.9128 |    0.4216 |
| gru              | focal+none        |       2 |      40804 |         1 |       0.9667 |       0.6716 |   nan      |      nan      |     0.2962 |        0.9159 |    0.4176 |
| tcn              | weighted_bce+none |       2 |     104548 |         3 |       0.9657 |       0.6669 |     0.0022 |        0.0041 |     0.2959 |        0.9094 |    0.4193 |
| gru              | bce+none          |       2 |      40804 |         1 |       0.9666 |       0.6653 |   nan      |      nan      |     0.2996 |        0.9150 |    0.4219 |
| lstm             | bce+random_over   |       2 |      53668 |         1 |       0.9654 |       0.6652 |   nan      |      nan      |     0.2870 |        0.9183 |    0.4116 |
| lstm             | bce+none          |       2 |      53668 |         1 |       0.9658 |       0.6639 |   nan      |      nan      |     0.2942 |        0.9140 |    0.4167 |
| transformer      | weighted_bce+none |       2 |     104676 |         3 |       0.9642 |       0.6632 |     0.0026 |        0.0048 |     0.2788 |        0.9173 |    0.4015 |
| hybrid           | weighted_bce+none |       2 |     213156 |         3 |       0.9650 |       0.6587 |     0.0009 |        0.0016 |     0.2969 |        0.9074 |    0.4215 |
| bagging          | none              |       1 |          0 |         1 |       0.9574 |       0.6249 |   nan      |      nan      |     0.2456 |        0.9183 |    0.3632 |
| rf               | none              |       1 |          0 |         1 |       0.9571 |       0.6248 |   nan      |      nan      |     0.2465 |        0.9193 |    0.3654 |
| rf               | random_over       |       1 |          0 |         1 |       0.9572 |       0.6124 |   nan      |      nan      |     0.2483 |        0.9209 |    0.3674 |
| balanced_bagging | none              |       1 |          0 |         1 |       0.9580 |       0.6089 |   nan      |      nan      |     0.2457 |        0.9231 |    0.3631 |
| xgboost          | random_under      |       1 |          0 |         1 |       0.9561 |       0.6077 |   nan      |      nan      |     0.2388 |        0.9238 |    0.3579 |
| xgboost          | class_weight      |       1 |          0 |         1 |       0.9559 |       0.6072 |   nan      |      nan      |     0.2374 |        0.9190 |    0.3560 |
| xgboost          | none              |       1 |          0 |         1 |       0.9559 |       0.6072 |   nan      |      nan      |     0.2374 |        0.9190 |    0.3560 |
| xgboost          | random_over       |       1 |          0 |         1 |       0.9556 |       0.6047 |   nan      |      nan      |     0.2317 |        0.9217 |    0.3495 |
| rf               | class_weight      |       1 |          0 |         1 |       0.9571 |       0.6039 |   nan      |      nan      |     0.2477 |        0.9215 |    0.3663 |
| rf               | random_under      |       1 |          0 |         1 |       0.9568 |       0.6026 |   nan      |      nan      |     0.2421 |        0.9218 |    0.3599 |
| balanced_rf      | none              |       1 |          0 |         1 |       0.9564 |       0.5916 |   nan      |      nan      |     0.2406 |        0.9230 |    0.3583 |
| easy_ensemble    | none              |       1 |          0 |         1 |       0.9545 |       0.5831 |   nan      |      nan      |     0.2242 |        0.9248 |    0.3377 |
| adaboost         | none              |       1 |          0 |         1 |       0.9536 |       0.5816 |   nan      |      nan      |     0.2234 |        0.9246 |    0.3373 |
| xgboost          | smote_then_under  |       1 |          0 |         1 |       0.9522 |       0.5780 |   nan      |      nan      |     0.2231 |        0.9239 |    0.3395 |
| rf               | smote             |       1 |          0 |         1 |       0.9558 |       0.5774 |   nan      |      nan      |     0.2469 |        0.9212 |    0.3654 |
| xgboost          | smote             |       1 |          0 |         1 |       0.9510 |       0.5695 |   nan      |      nan      |     0.2258 |        0.9237 |    0.3419 |
| rf               | adasyn            |       1 |          0 |         1 |       0.9556 |       0.5662 |   nan      |      nan      |     0.2482 |        0.9217 |    0.3666 |
| xgboost          | adasyn            |       1 |          0 |         1 |       0.9513 |       0.5611 |   nan      |      nan      |     0.2266 |        0.9205 |    0.3425 |
| rf               | smote_then_under  |       1 |          0 |         1 |       0.9546 |       0.5567 |   nan      |      nan      |     0.2443 |        0.9224 |    0.3622 |
| bagged_adaboost  | none              |       1 |          0 |         1 |       0.9008 |       0.3534 |   nan      |      nan      |     0.0510 |        1.0000 |    0.0953 |
| rusboost         | none              |       1 |          0 |         1 |       0.9117 |       0.3062 |   nan      |      nan      |     0.1627 |        0.9313 |    0.2628 |

## h = 60 min

| model            | method            |   stage |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   auprc_range |   ppv_mean |   recall_mean |   f1_mean |
|:-----------------|:------------------|--------:|-----------:|----------:|-------------:|-------------:|-----------:|--------------:|-----------:|--------------:|----------:|
| lstm             | weighted_bce+none |       2 |      53668 |         3 |       0.9173 |       0.5713 |     0.0034 |        0.0066 |     0.2430 |        0.8871 |    0.3627 |
| gru              | weighted_bce+none |       2 |      40804 |         3 |       0.9169 |       0.5704 |     0.0028 |        0.0051 |     0.2428 |        0.8852 |    0.3624 |
| gru              | focal+none        |       2 |      40804 |         1 |       0.9170 |       0.5697 |   nan      |      nan      |     0.2435 |        0.8880 |    0.3625 |
| gru              | bce+random_over   |       2 |      40804 |         1 |       0.9150 |       0.5691 |   nan      |      nan      |     0.2377 |        0.8848 |    0.3567 |
| gru              | bce+none          |       2 |      40804 |         1 |       0.9171 |       0.5663 |   nan      |      nan      |     0.2461 |        0.8832 |    0.3656 |
| tcn              | weighted_bce+none |       2 |     104548 |         3 |       0.9155 |       0.5650 |     0.0003 |        0.0007 |     0.2430 |        0.8806 |    0.3631 |
| lstm             | bce+random_over   |       2 |      53668 |         1 |       0.9151 |       0.5649 |   nan      |      nan      |     0.2343 |        0.8923 |    0.3538 |
| lstm             | bce+none          |       2 |      53668 |         1 |       0.9164 |       0.5649 |   nan      |      nan      |     0.2427 |        0.8873 |    0.3626 |
| transformer      | weighted_bce+none |       2 |     104676 |         3 |       0.9135 |       0.5623 |     0.0020 |        0.0039 |     0.2312 |        0.8902 |    0.3494 |
| hybrid           | weighted_bce+none |       2 |     213156 |         3 |       0.9153 |       0.5606 |     0.0007 |        0.0014 |     0.2473 |        0.8742 |    0.3682 |
| rf               | none              |       1 |          0 |         1 |       0.9026 |       0.5322 |   nan      |      nan      |     0.2107 |        0.8917 |    0.3244 |
| bagging          | none              |       1 |          0 |         1 |       0.9027 |       0.5320 |   nan      |      nan      |     0.2082 |        0.8942 |    0.3214 |
| balanced_bagging | none              |       1 |          0 |         1 |       0.9044 |       0.5263 |   nan      |      nan      |     0.2098 |        0.8963 |    0.3231 |
| rf               | random_over       |       1 |          0 |         1 |       0.9029 |       0.5253 |   nan      |      nan      |     0.2136 |        0.8880 |    0.3276 |
| rf               | random_under      |       1 |          0 |         1 |       0.9030 |       0.5213 |   nan      |      nan      |     0.2095 |        0.8924 |    0.3229 |
| xgboost          | class_weight      |       1 |          0 |         1 |       0.9029 |       0.5213 |   nan      |      nan      |     0.2096 |        0.8899 |    0.3232 |
| xgboost          | none              |       1 |          0 |         1 |       0.9029 |       0.5213 |   nan      |      nan      |     0.2096 |        0.8899 |    0.3232 |
| rf               | class_weight      |       1 |          0 |         1 |       0.9033 |       0.5211 |   nan      |      nan      |     0.2121 |        0.8918 |    0.3261 |
| xgboost          | random_over       |       1 |          0 |         1 |       0.9016 |       0.5199 |   nan      |      nan      |     0.2047 |        0.8956 |    0.3182 |
| balanced_rf      | none              |       1 |          0 |         1 |       0.9028 |       0.5169 |   nan      |      nan      |     0.2094 |        0.8929 |    0.3227 |
| xgboost          | random_under      |       1 |          0 |         1 |       0.9022 |       0.5156 |   nan      |      nan      |     0.2100 |        0.8917 |    0.3242 |
| rf               | smote             |       1 |          0 |         1 |       0.9023 |       0.5083 |   nan      |      nan      |     0.2159 |        0.8883 |    0.3300 |
| rf               | adasyn            |       1 |          0 |         1 |       0.9024 |       0.5049 |   nan      |      nan      |     0.2151 |        0.8895 |    0.3292 |
| adaboost         | none              |       1 |          0 |         1 |       0.8931 |       0.5000 |   nan      |      nan      |     0.1870 |        0.9006 |    0.2952 |
| xgboost          | smote             |       1 |          0 |         1 |       0.8961 |       0.4977 |   nan      |      nan      |     0.2056 |        0.8868 |    0.3185 |
| easy_ensemble    | none              |       1 |          0 |         1 |       0.8947 |       0.4942 |   nan      |      nan      |     0.1946 |        0.8938 |    0.3036 |
| xgboost          | smote_then_under  |       1 |          0 |         1 |       0.8965 |       0.4939 |   nan      |      nan      |     0.2049 |        0.8863 |    0.3173 |
| xgboost          | adasyn            |       1 |          0 |         1 |       0.8954 |       0.4919 |   nan      |      nan      |     0.2037 |        0.8873 |    0.3161 |
| rf               | smote_then_under  |       1 |          0 |         1 |       0.9011 |       0.4880 |   nan      |      nan      |     0.2124 |        0.8928 |    0.3261 |
| bagged_adaboost  | none              |       1 |          0 |         1 |       0.8192 |       0.3282 |   nan      |      nan      |     0.0773 |        1.0000 |    0.1402 |
| rusboost         | none              |       1 |          0 |         1 |       0.8306 |       0.2713 |   nan      |      nan      |     0.1523 |        0.8863 |    0.2487 |

## h = 120 min

| model            | method            |   stage |   n_params |   n_seeds |   auroc_mean |   auprc_mean |   auprc_sd |   auprc_range |   ppv_mean |   recall_mean |   f1_mean |
|:-----------------|:------------------|--------:|-----------:|----------:|-------------:|-------------:|-----------:|--------------:|-----------:|--------------:|----------:|
| lstm             | weighted_bce+none |       2 |      53668 |         3 |       0.8319 |       0.4914 |     0.0025 |        0.0047 |     0.2311 |        0.8548 |    0.3485 |
| gru              | weighted_bce+none |       2 |      40804 |         3 |       0.8309 |       0.4905 |     0.0020 |        0.0036 |     0.2329 |        0.8509 |    0.3498 |
| gru              | focal+none        |       2 |      40804 |         1 |       0.8318 |       0.4900 |   nan      |      nan      |     0.2373 |        0.8465 |    0.3536 |
| lstm             | bce+none          |       2 |      53668 |         1 |       0.8320 |       0.4882 |   nan      |      nan      |     0.2317 |        0.8524 |    0.3489 |
| gru              | bce+none          |       2 |      40804 |         1 |       0.8311 |       0.4881 |   nan      |      nan      |     0.2360 |        0.8487 |    0.3531 |
| tcn              | weighted_bce+none |       2 |     104548 |         3 |       0.8304 |       0.4881 |     0.0015 |        0.0030 |     0.2324 |        0.8490 |    0.3502 |
| gru              | bce+random_over   |       2 |      40804 |         1 |       0.8256 |       0.4870 |   nan      |      nan      |     0.2282 |        0.8496 |    0.3444 |
| transformer      | weighted_bce+none |       2 |     104676 |         3 |       0.8292 |       0.4860 |     0.0019 |        0.0036 |     0.2269 |        0.8569 |    0.3435 |
| hybrid           | weighted_bce+none |       2 |     213156 |         3 |       0.8310 |       0.4859 |     0.0011 |        0.0021 |     0.2362 |        0.8418 |    0.3551 |
| lstm             | bce+random_over   |       2 |      53668 |         1 |       0.8275 |       0.4854 |   nan      |      nan      |     0.2251 |        0.8560 |    0.3419 |
| rf               | none              |       1 |          0 |         1 |       0.8205 |       0.4664 |   nan      |      nan      |     0.2165 |        0.8614 |    0.3313 |
| rf               | random_over       |       1 |          0 |         1 |       0.8205 |       0.4654 |   nan      |      nan      |     0.2170 |        0.8621 |    0.3319 |
| rf               | random_under      |       1 |          0 |         1 |       0.8207 |       0.4636 |   nan      |      nan      |     0.2164 |        0.8634 |    0.3313 |
| bagging          | none              |       1 |          0 |         1 |       0.8170 |       0.4629 |   nan      |      nan      |     0.2105 |        0.8656 |    0.3240 |
| xgboost          | class_weight      |       1 |          0 |         1 |       0.8204 |       0.4628 |   nan      |      nan      |     0.2099 |        0.8745 |    0.3258 |
| xgboost          | none              |       1 |          0 |         1 |       0.8204 |       0.4628 |   nan      |      nan      |     0.2099 |        0.8745 |    0.3258 |
| rf               | class_weight      |       1 |          0 |         1 |       0.8212 |       0.4609 |   nan      |      nan      |     0.2175 |        0.8661 |    0.3326 |
| balanced_bagging | none              |       1 |          0 |         1 |       0.8187 |       0.4594 |   nan      |      nan      |     0.2144 |        0.8664 |    0.3286 |
| balanced_rf      | none              |       1 |          0 |         1 |       0.8211 |       0.4584 |   nan      |      nan      |     0.2181 |        0.8623 |    0.3328 |
| xgboost          | random_over       |       1 |          0 |         1 |       0.8176 |       0.4568 |   nan      |      nan      |     0.2052 |        0.8747 |    0.3195 |
| rf               | adasyn            |       1 |          0 |         1 |       0.8202 |       0.4568 |   nan      |      nan      |     0.2195 |        0.8615 |    0.3348 |
| rf               | smote             |       1 |          0 |         1 |       0.8193 |       0.4562 |   nan      |      nan      |     0.2179 |        0.8616 |    0.3329 |
| xgboost          | random_under      |       1 |          0 |         1 |       0.8186 |       0.4546 |   nan      |      nan      |     0.2086 |        0.8688 |    0.3235 |
| rf               | smote_then_under  |       1 |          0 |         1 |       0.8187 |       0.4481 |   nan      |      nan      |     0.2158 |        0.8692 |    0.3311 |
| xgboost          | smote             |       1 |          0 |         1 |       0.8118 |       0.4478 |   nan      |      nan      |     0.2085 |        0.8611 |    0.3226 |
| xgboost          | smote_then_under  |       1 |          0 |         1 |       0.8114 |       0.4455 |   nan      |      nan      |     0.2067 |        0.8719 |    0.3211 |
| adaboost         | none              |       1 |          0 |         1 |       0.8011 |       0.4439 |   nan      |      nan      |     0.1970 |        0.8644 |    0.3070 |
| xgboost          | adasyn            |       1 |          0 |         1 |       0.8090 |       0.4413 |   nan      |      nan      |     0.2077 |        0.8553 |    0.3202 |
| easy_ensemble    | none              |       1 |          0 |         1 |       0.8003 |       0.4351 |   nan      |      nan      |     0.2004 |        0.8614 |    0.3104 |
| rusboost         | none              |       1 |          0 |         1 |       0.7633 |       0.3376 |   nan      |      nan      |     0.1796 |        0.8674 |    0.2849 |
| bagged_adaboost  | none              |       1 |          0 |         1 |       0.7179 |       0.3018 |   nan      |      nan      |     0.1242 |        1.0000 |    0.2144 |
