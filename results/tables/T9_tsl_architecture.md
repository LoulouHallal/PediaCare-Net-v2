|   horizon | arm                            | variant    |   n_params |   n_seeds |   auprc_mean |   auprc_sd |   auroc_mean |   ppv_mean |   recall_mean |   f1_mean |
|----------:|:-------------------------------|:-----------|-----------:|----------:|-------------:|-----------:|-------------:|-----------:|--------------:|----------:|
|        15 | A  GRU + TA (baseline)         | gru_ta     |      41188 |         3 |       0.8594 |     0.0022 |       0.9924 |     0.7104 |        0.8251 |    0.7615 |
|        30 | A  GRU + TA (baseline)         | gru_ta     |      41188 |         3 |       0.7476 |     0.0009 |       0.9721 |     0.4635 |        0.8222 |    0.5895 |
|        60 | A  GRU + TA (baseline)         | gru_ta     |      41188 |         3 |       0.6271 |     0.0011 |       0.9248 |     0.3185 |        0.7991 |    0.4518 |
|       120 | A  GRU + TA (baseline)         | gru_ta     |      41188 |         3 |       0.5323 |     0.0007 |       0.8415 |     0.2793 |        0.7700 |    0.4059 |
|        15 | B  LiGRU-style (no reset gate) | ligru_ta   |      28324 |         1 |       0.8588 |   nan      |       0.9924 |     0.7162 |        0.8283 |    0.7663 |
|        30 | B  LiGRU-style (no reset gate) | ligru_ta   |      28324 |         1 |       0.7455 |   nan      |       0.9717 |     0.4567 |        0.8207 |    0.5834 |
|        60 | B  LiGRU-style (no reset gate) | ligru_ta   |      28324 |         1 |       0.6252 |   nan      |       0.9243 |     0.3172 |        0.8027 |    0.4499 |
|       120 | B  LiGRU-style (no reset gate) | ligru_ta   |      28324 |         1 |       0.5312 |   nan      |       0.8424 |     0.2783 |        0.7671 |    0.4032 |
|        15 | C  TSL, constant retention     | tsl_static |      15460 |         3 |       0.8573 |     0.0035 |       0.9922 |     0.7142 |        0.8194 |    0.7613 |
|        30 | C  TSL, constant retention     | tsl_static |      15460 |         3 |       0.7458 |     0.0022 |       0.9719 |     0.4661 |        0.8185 |    0.5903 |
|        60 | C  TSL, constant retention     | tsl_static |      15460 |         3 |       0.6252 |     0.0023 |       0.9245 |     0.3227 |        0.7932 |    0.4537 |
|       120 | C  TSL, constant retention     | tsl_static |      15460 |         3 |       0.5306 |     0.0017 |       0.8413 |     0.2763 |        0.7694 |    0.4019 |
|        15 | D  TSL-GRU (proposed)          | tsl_gru    |      15974 |         3 |       0.8599 |     0.0062 |       0.9925 |     0.7147 |        0.8233 |    0.7638 |
|        30 | D  TSL-GRU (proposed)          | tsl_gru    |      15974 |         3 |       0.7485 |     0.0048 |       0.9722 |     0.4629 |        0.8233 |    0.5893 |
|        60 | D  TSL-GRU (proposed)          | tsl_gru    |      15974 |         3 |       0.6288 |     0.0034 |       0.9254 |     0.3189 |        0.8047 |    0.4526 |
|       120 | D  TSL-GRU (proposed)          | tsl_gru    |      15974 |         3 |       0.5342 |     0.0029 |       0.8436 |     0.2761 |        0.7761 |    0.4034 |