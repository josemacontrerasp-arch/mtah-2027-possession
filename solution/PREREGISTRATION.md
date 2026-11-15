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

---

## Addendum, 15 Nov 2026 — more labelled external data (v8), amending rule 7

*Written and committed before any v8 number was computed.*

**Why.** Rule 7 closed the search on 14 Nov. On 15 Nov the team decided to test one more
candidate, which changes the training data and nothing else: SkillCorner's open data (20 A-League
2024/25 matches, MIT licence). Its tracking is a real automatic tracker (broadcast video, 10 fps),
so its gaps are real tracker failures rather than simulated ones, and its phases of play give
team possession. The descriptive statistics of these labels were looked at before this addendum
(per match: label shares, about 100 stoppages, spell counts and lengths); no model was trained.

**Candidate v8.** Identical to v6 — same features, model hyper-parameters and training settings,
the same four shared DFL trackers (seeds 11–14), ensemble family, decoder grid, clean
environments and nested procedure — except that every fold, and the final models, also train on
the 20 SkillCorner matches (`solution/external_skillcorner.py`):

* **tracking:** SkillCorner detections only (its extrapolated positions are dropped), resampled to
  25 fps by linear interpolation between two consecutive detections; speeds from position
  differences; coordinates moved to the corner origin;
* **labels:** a frame inside a phase of play belongs to that phase's team in possession; a frame of
  a period outside every phase is dead ball (SkillCorner: phases exist only while the ball is in
  play); frames before a period's first phase or after its last are unlabelled; then the same
  controlled-possession relabel as the DFL labels (2 s, `solution/relabel.py`);
* **copies:** each match in both orientations and two 5 Hz frame offsets (seeds 11 and 12), with
  no simulated degradation — the broadcast tracker's own gaps are the degradation;
* SkillCorner matches are **never evaluated**: the nested score is computed on the same 7 DFL
  matches and three clean environments as v6, so the comparison is paired.

8. **v6 → v8.** The three criteria of rule 5, with v8 in place of v7: v8's nested clean score is
   at least v6's + 0.005 (v6: 0.7736); v8 is better on at least 5 of the 7 matches; no clean
   environment drops by more than 0.003. Otherwise v6 stays the submission.
9. **Rule 7, amended.** v8 is the only exception: no other variant (other copy counts, weights,
   label rules or SkillCorner subsets) is tried. If rule 8 adopts v8 but its final models, file
   and checks (byte-identical regeneration, the three-system self-test, preflight) are not all
   complete by 17 Nov 2026, 12:00 CET, v6 is submitted.

Rule 4 is unchanged; as for v7, the public-window pipeline check is not repeated for v8.
