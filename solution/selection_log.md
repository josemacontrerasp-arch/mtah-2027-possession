# Selection log

Every experiment that informed what is submitted, in order, with the environment it was judged on
and the decision. External games only; no decision uses the provided match (see
`PREREGISTRATION.md`).

| date | experiment | judged on | result | decision |
|---|---|---|---|---|
| 5 Nov | v1–v4 development (features, GRUs, ensemble, constrained HMM, robust decoder) | DFL LOMO, `standard` + `hard` profiles | v4 0.813 / 0.794 | submitted v4 (SHA `628a4628…`) |
| 7 Nov | *Exploratory, before pre-registration:* LightGBM trained on calibrated vs randomised degradation | `standard`, `hard`, `rnd` (one draw) | calibrated 0.792 / 0.765 / 0.645; randomised 0.762 / 0.756 / 0.745 | motivated the clean-room candidate v6 and the pre-registration; not itself a decision |
| 7 Nov | Pre-registration of selection rules (commit `bac40f6`) | — | — | rules fixed before the results below |
| 7 Nov | Kick-off detector thresholds re-derived on external kick-offs (rule 3) | 17 DFL after-goal kick-offs × 2 randomised trackers | external best: 33/34 found, median error 8 frames; previous values 32/34, 11 frames | adopted the external values (DFL check: 17/17 found, median +5 frames, direction 17/17); provided match: found in both halves |
