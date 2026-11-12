# Pre-registration — selection rules fixed before the numbers they govern

*Written 7 Nov 2026, before any clean-environment result below was computed. Every decision about
what is submitted follows these rules. Nothing is chosen by looking at the provided match.*

## Why

The organisers' 3-Nov email: training, and tuning of any parameter, must use external data only.
Our v4 candidate's degradation simulator was matched to aggregate, unlabelled statistics of the
provided tracking — a grey area under that sentence. The clean-room candidate (v6) removes it.

## Candidates

| | training degradation | everything else |
|---|---|---|
| **v4** (candidate at the time, never uploaded; SHA-256 `628a4628…fafa55`) | `degrade.py`, calibrated to unlabelled statistics of the provided ATD | LightGBM + 2×2 GRUs, constrained HMM |
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

---

## Addendum, 12 Nov 2026 — one further candidate (v7)

*Written and committed before any v7 number was computed.*

**Why.** The final review of v6 found that its training used four simulated trackers (one per
seed), each applied to all seven external games in both orientations — not "a different tracker
per copy" as the candidates table above says. The technical document was corrected to describe
what v6 really does. v7 tests the design as originally described.

**Candidate v7.** Identical to v6 except that each of the 56 training copies (7 games × 2
orientations × 4 draws) is degraded by its own tracker, drawn from the same `degrade_random`
priors (`MTAH_TRACKERS=percopy`; tracker seed derived from game, orientation and draw index, and
disjoint from the evaluation seeds). Same features, same model hyper-parameters and training
length, same ensemble family (LightGBM share 0.3 / 0.5 / 0.7, GRU split 70:30), same decoder grid,
same clean environments, same nested procedure.

5. **v6 → v7.** v6 stays the submission unless **all** of the following hold, in which case v7
   replaces it:
   * v7's nested clean score is at least v6's + 0.005 (v6: 0.774, recomputed with the same script
     for a paired comparison);
   * v7 is better on at least 5 of the 7 matches (per-match nested scores);
   * in none of the three clean environments does the nested score drop by more than 0.003.
6. **One conditional follow-up (v7b).** Only if rule 5 adopts v7: the same design with eight
   trackers per game and orientation (112 copies), LightGBM on all rows and the GRUs trained for
   the same number of gradient steps as v7 (15 epochs instead of 30). v7b replaces v7 only under
   the three criteria of rule 5, measured against v7. If rule 5 does not adopt v7, v7b is not run.
7. **No other training variant** is tried before the deadline, and no model or decoder change is
   made after 14 Nov.

Rule 4 is unchanged: on the provided match a new final version gets runtime sanity checks only.
The public-window pipeline check was spent on v6 and is **not** repeated for v7 or v7b — their
prediction code is the v6 code; only trained weights and the selected decoder setting differ.
