# Computational cost

Device: cuda, batch 512, sequence 60.

| model          |   channels |   params |   MMACs_per_window |   infer_us_per_window |   train_ms_per_batch |   peak_MB |   epoch_min_est |
|:---------------|-----------:|---------:|-------------------:|----------------------:|---------------------:|----------:|----------------:|
| gru_fused_5ch  |          5 |    40804 |               2.27 |                  6.69 |                 8.32 |    318.17 |            0.11 |
| lstm_fused_5ch |          5 |    53668 |               3.03 |                  6.29 |                 8.29 |    326.77 |            0.11 |
| gru_fused_9ch  |          9 |    41572 |               2.32 |                  3.61 |                 5.04 |    320.68 |            0.07 |
| gru_ta_7ch     |          7 |    41188 |               2.29 |                 83.62 |               106.23 |    157.85 |            1.38 |
| tsl_gru        |          7 |    15974 |               0.79 |                 89.33 |               135.49 |    142.34 |            1.76 |
| ar_rhu         |          9 |    22665 |               1.34 |                 81.90 |               142.48 |    103.09 |            1.86 |
| ar_rhu_h88     |          9 |    41721 |               2.48 |                 82.12 |               133.60 |    133.20 |            1.74 |
| ctf_ru         |          5 |    41551 |               2.47 |                145.75 |               230.96 |   1346.22 |            3.01 |
| drs_gru        |          9 |    41878 |               2.37 |                134.21 |               183.95 |    182.94 |            2.40 |
| pac_gru        |          9 |    40934 |               2.33 |                134.77 |               222.15 |    155.95 |            2.89 |
| kew_entmax     |          9 |    41808 |               2.38 |                571.63 |               588.70 |    530.25 |            7.67 |
| kew_softmax    |          9 |    41808 |               2.38 |                 68.01 |               115.79 |    102.88 |            1.51 |
| adew           |          9 |    54182 |               3.09 |                110.09 |               213.79 |    176.48 |            2.78 |
| xgru           |          9 |    40928 |               2.34 |                387.69 |               495.21 |    190.79 |            6.45 |
| cmr            |          9 |    40141 |               1.51 |                245.21 |               498.68 |    320.43 |            6.49 |
| tcn_ta         |          7 |    91620 |               5.27 |                 11.98 |                18.15 |    251.02 |            0.24 |
| msd_tcn        |          7 |    76708 |              27.58 |                 19.77 |                31.79 |    293.32 |            0.41 |
| dms_tcn        |          7 |    31988 |              24.89 |                 23.39 |                35.34 |    334.45 |            0.46 |

## Notes

- **gru_fused_5ch** — nn.GRU, the 5-channel stage-2 baseline
- **lstm_fused_5ch** — nn.LSTM
- **gru_fused_9ch** — nn.GRU + TA + absolute — the project's best model
- **gru_ta_7ch** — GRU + TA, the arm TSL-GRU was matched against
- **tsl_gru** — trajectory-scaled retention, 61% fewer params
- **ar_rhu** — anticipatory/reactive split
- **ar_rhu_h88** — capacity-matched
- **ctf_ru** — transport over a 20-state clinical grid
- **drs_gru** — signed recurrent history
- **pac_gru** — prediction-anchored scaling + coupled gates
- **kew_entmax** — keep/erase/write simplex, entmax
- **kew_softmax** — same, dense normalisation
- **adew** — anchored factorised erase
- **xgru** — exponential gating + matrix memory
- **cmr** — FiLM + modular + routing
- **tcn_ta** — TCN + TA
- **msd_tcn** — multi-scale dilated TCN
- **dms_tcn** — dynamic multi-scale TCN, 65% fewer params
