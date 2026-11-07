# Pre-registration — selection rules fixed before the numbers they govern

*Written 7 Nov 2026, before any clean-environment result below was computed. Every decision about
what is submitted follows these rules. Nothing is chosen by looking at the provided match.*

## Why

The organisers' 3-Nov email: training, and tuning of any parameter, must use external data only.
Our v4 submission's degradation simulator was matched to aggregate, unlabelled statistics of the
provided tracking — a grey area under that sentence. The clean-room candidate (v6) removes it.

## Candidates

| | training degradation | everything else |
|---|---|---|
| **v4** (submitted, SHA-256 `628a4628…fafa55`) | `degrade.py`, calibrated to unlabelled statistics of the provided ATD | LightGBM + 2×2 GRUs, constrained HMM |
| **v6** (clean room) | `degrade_random.py`: a different tracker per copy, drawn from first-principles priors; no value from the provided match | same features, same model family, decoder re-selected as below |

## Evaluation environments

Decisions use **clean environments only** — degradations with no value taken from the provided
match, never used for training:

* `rnd` — one tracker per external match drawn from the `degrade_random` priors (seed 3101 + i);
* `rnd2` — an independent second draw (seed 4101 + i);
* `rndh` — the harsher half of the same priors (more, longer ball gaps; looser coupling of ball
  loss to stoppages; more noise) (seed 5101 + i).

The older `standard` and `hard` profiles were built from provided-ATD statistics. They are
**reported for information only and used for no decision.**

## Metric and procedure

* Score: the official spellF1 (`evaluation/possession_metrics.py`), mean over the 7 external
  matches, averaged over the three clean environments.
* Decoder parameters are chosen by **nested** leave-one-match-out: choose on 6 matches (all three
  environments), score the 7th, repeat. The nested score is the one compared below.

## Rules

1. **v4 → v6.** Submit v6 if its nested clean score is at least v4's nested clean score − 0.01.
   (Compliance has priority; v6 is the default unless it is clearly worse.)
2. **Decoder upgrades** (shorter turnover minimum k_opp ∈ {1.5, 2.0, 2.5, 3.0} s; minimum stoppage
   length ∈ {0, 2, 3} s, pre-kick-off stoppage exempt; per-member temperature calibration): each
   adopted only if the nested clean score improves by ≥ +0.005, at least 5 of 7 matches improve,
   and the predicted/true spell-count ratio moves by ≤ 5%.
3. **Kick-off detector.** Thresholds are frozen at the values that pass external validation on the
   17 after-goal kick-offs of the DFL games (`solution/kickoff_check.py`); if a re-derivation on
   external kick-offs changes them, the externally chosen values are used. Whatever they then
   produce on the provided match is accepted without inspection beyond "found / not found".
4. **Checks on the provided match** are limited to runtime sanity (row count, no NaN, the official
   validator, byte-identical regeneration in a fresh CPU-only environment) and one public-window
   pipeline check of the final version, which decides nothing.

## Log

Results and decisions are appended to `solution/selection_log.md` as they are made.
