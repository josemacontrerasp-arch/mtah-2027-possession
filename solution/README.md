# Frame-level possession from automatic tracking — technical document

*More than a Hack 2027, Phase 1. Final version of 5 November 2026.*

## 1. Approach

Possession is predicted in two steps.

1. **Per-frame probabilities** of *home in possession*, *away in possession* or *dead ball* come
   from an ensemble of three models fed the same collective tracking features: a LightGBM
   classifier (weight 0.5) and two bidirectional GRUs that read two-minute stretches at 5 Hz
   (weights 0.35 and 0.15; one sees only raw per-frame features and learns its own temporal
   context, the other also sees hand-built context windows).
2. **A constrained HMM** decoded with Viterbi turns probabilities into possessions. Three states
   (H, A, D) with switching costs, plus the definitional constraint that a possession *won from
   the opponent* lasts at least 2.5 s (a restart possession can be any length). Its settings
   are those with the best average over a standard and a deliberately harsher external benchmark.

**Everything that is learned or tuned is fitted on external games only** — seven Bundesliga
matches of the DFL open dataset, degraded so they look like the provided automatic tracking. The
provided match is used only as input. Model choice, ensemble weights and decoder parameters were
all selected by leave-one-match-out cross-validation on those external games.

Why this design: the ranking metric scores whole possessions (spellF1), so the output has to be
temporally coherent; the hard cases are open-play turnovers and the long stretches where the
tracker has lost the ball, which need context of tens of seconds; and the provided ground truth
may not be used for training or tuning, so the only honest way to fit anything is on other games.

## 2. Assets used and dropped

| asset | status | how |
|---|---|---|
| tactical ATD | **used** | all features; own loader (`solution/atd.py`), slot order from the EPTS DataFormatSpecification |
| broadcast ATD + homography | used for alignment, then **dropped** from the submission | aligned to the tactical timeline from tracking alone (one offset per half, 4014 / 26920 frames, median ball disagreement 0.8 m). Filling tactical ball gaps raised coverage 59.3% → 63.1% but shifts the input away from what the models were calibrated on; the "broadcast cut away" stoppage signal was informative (self-supervised likelihood ratio 5.4) but at a weighting we could defend it changed 0.01% of frames |
| Smart Tagging | parsed, **dropped** | team hints agree with the models' confident frames (LR ≈ 4–13), but conservative fusion changed 0.04% of frames and stronger fusion cannot be validated without labels |
| tactical homography | **dropped** | the camera frames the team block (centre ≈ players' mean x, 3 m), not the ball (12 m) |
| videos | not used | |
| public 5-minute ground truth | **pipeline check only** | run once per frozen version to check alignment and format; never used to train, tune or choose |

## 3. Training data

| dataset | licence | labels | who labelled | how much |
|---|---|---|---|---|
| DFL open dataset — Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12:195, doi:10.6084/m9.figshare.28196177 | CC BY 4.0 | per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) | the league's official data provider (DFL / Sportec Solutions) | 7 matches, 1.0 M frames at 25 Hz |

No data about the provided game other than the provided assets was used. Downloaded but not used:
Metrica Sports sample data, SkillCorner open data (MIT).

### Making external games look like the provided automatic tracking (`solution/degrade.py`)

| property of the provided ATD (graded frames, measured without labels) | degraded DFL |
|---|---|
| ball visible 59.3%, in long blocks (median missing run 268 frames) | 57%, same run-length shapes |
| P(ball visible) falls from 0.83 to 0.12 as the share of players standing still rises | reproduced, mean abs. error 0.043 over 8 bins |
| relative ball–nearest-player speed p10/p50/p90: 0.6 / 3.7 / 12.9 m/s | 0.9 / 3.2 / 13.1 m/s |
| players tracked per team p10/p50/p90: 9 / 11 / 11 | 9 / 11 / 11 |

Ingredients: live-play ball visibility from a renewal process with the ATD's run-length
distributions; during stoppages the ball is hidden except, before 80% of restarts, a placed ball
for 2–10 s; loss and re-acquisition jittered around the stoppage (−1 s / +2 s); whole aerial
episodes (z > 3 m) dropped with probability 0.35; low-pass position noise; light per-player and
bursty group player drop-outs. DFL freezes the ball during every stoppage (100% of dead frames,
< 1% of live frames) — an artefact a real tracker never shows — so the frozen ball is removed.
These parameters match **unlabelled input statistics** of the provided tracking; no ground truth
and no video are involved. Each degraded game is also used with home and away swapped.

Sanity check of the simulated setting: the official baseline algorithm scores spellF1 0.32 /
macroF1 0.74 on degraded DFL, against 0.24 / 0.67 published for it on the provided match (under
naive random drop-out it scored 0.19 / 0.45).

## 4. Possession rules implemented

* **Controlled possession.** The challenge: a change of possession needs the opponent to gain
  *controlled* possession; touches, deflections and aerial duels without control do not count,
  and a contested or loose ball stays with the last team in possession. External labels flip on
  every touch (DFL spells: median 3.7 s, 30% under 2 s). We convert them (`solution/relabel.py`):
  a spell that interrupts the other team's possession and lasts under **2 s** ("a player keeps
  the ball for a few seconds") reverts to that team. The constant comes from the text.
* **Restarts.** After a stoppage there is always a clear possessor: spells that start from a dead
  ball are never relabelled, and the decoder imposes no minimum length on them.
* **Minimum turnover length.** In the decoder, a possession won from the opponent must last at
  least 2.5 s (chosen by cross-validation among 0–5 s).
* **Dead ball** is its own state. Frames of a half's play window before its kick-off are `D`
  (the second half's graded window opens about two minutes before the kick). The kick-off is
  **detected from the tracking** (`solution/kickoff.py`): the one moment of a half when both teams
  stand entirely in their own halves, taking the formation that breaks into play (mean player
  speed rising). The same formation gives each team's attacking direction. No frame number of the
  match is written in the code. The detector's thresholds are generic physical values, and it is
  **validated on external games** (`solution/kickoff_check.py`): on all 17 kick-offs after goals
  in the seven DFL matches, on randomly degraded tracking, it finds 16, with a median error of
  4 frames (0.16 s), 94% within one second and the attacking direction right every time. On the
  provided match it lands 2 and 14 frames from the kick-offs quoted in the documentation.
* No `X` on graded frames: abstaining costs macroF1 and gains nothing under spellF1.

## 5. Features (`solution/features.py`)

All collective and orientation-normalised (home attacks +x); nothing is keyed to a track slot,
because slots are not people. Per frame: ball position, speed and distance to the boundary;
nearest home and away player distances, counts within 3 m / 10 m, relative ball–player speed;
team speeds, share running / standing, centroids, spreads, extreme positions, share goal-side
of the ball, centroid velocities; ball-control evidence at 1.0 / 1.5 / 2.5 m with the last and
next controlling team and time since/until; ball visibility history; the ball's path
interpolated across gaps of up to 15 s and which team lines it; the pass receiver (first player
ahead of a moving ball inside a cone); which team reaches the ball's position 0.6 s ahead first;
and past / future / centred windows of 1–8 s over the key signals. 184 features.

## 6. Results on external data (leave-one-match-out, 7 folds)

| | spellF1 | macroF1 |
|---|---|---|
| official baseline algorithm, same degraded games | 0.322 | 0.741 |
| LightGBM, argmax per frame | ~0.43 | 0.94 |
| LightGBM + HMM | 0.791 | 0.937 |
| + minimum turnover length | (included above) | |
| GRU alone (raw features) | 0.796 | 0.936 |
| ensemble LightGBM + 2 GRUs + constrained HMM | 0.809 | 0.942 |
| **submission: LightGBM + 2 GRUs × 2 seeds + constrained HMM, robust decoder** | **0.813** | **0.943** |
| same, on the stress benchmark (harsher simulated tracker, §8) | 0.794 | 0.931 |
| same, decoder tuned on 6 matches and scored on the 7th (nested) | ≈ 0.807 | |

Per match 0.79–0.85. Dropped after cross-validation: second-stage stacking (0.765 vs 0.781);
adding an older-feature model to the ensemble (no gain); a larger LightGBM with twice the
degraded copies and 800 rounds (0.789 vs 0.791 alone, +0.001 in the ensemble — noise).

**Pipeline check of the frozen submission** (public 5-minute window; official scorer, run once):
spellF1 0.607, macroF1 0.722 (every baseline lower), team accuracy 0.95 where both agree the ball
is alive, alive/dead F1 0.91, frames aligned. Bootstrap 5–95% interval 0.41–0.79; the scorer itself reports that gaps below ~0.37 are
within noise on 17 possessions, so this window cannot rank versions and was not used to.

**Reproducibility:** `python -m solution.predict` regenerates the submission byte for byte —
checked in our development environment and in a freshly created CPU-only environment installed
from `requirements-solution.txt`, with every cache deleted and rebuilt from the raw match files
(SHA-256 `628a46283946051c7939c91bc227fbb8e7a370c7bd0cdaac2a44fd77cfeafa55`). Probabilities are
rounded to 4 decimals before decoding and the GRUs run on CPU, so floating-point noise across
machines cannot change the file. LightGBM retraining is bit-for-bit deterministic too (re-running
the cross-validation reproduced identical out-of-fold files); GRU retraining on a GPU is not
bit-exact, which is why the trained weights are committed.

Where it still loses (diagnostics on held-out games): true possessions under 3 s are recovered
(IoU ≥ 0.5) 53% of the time against 93–97% for possessions over 12 s, and possessions between
two opponent possessions (open-play turnovers) 71% against 90% between two stoppages.

**These are external-data numbers.** The provided match differs (automatic tracking in reality
rather than in simulation; a ground truth from a different provider's definition), so the score
on it will be lower; how much lower is unknowable without its labels.

## 7. Limitations and what we would do with more time

* The simulated degradation ties ball loss to stoppages more cleanly than a real tracker does,
  so dead-ball detection on the provided match is likely the weakest part: externally the
  onset error is a 0.12 s median, on real automatic tracking it will be worse, and a missed
  stoppage merges the possessions either side of it. Match-only stoppage evidence (the broadcast
  cutting away, LR 5.4) exists but could not be weighted without labels.
* The possession definition is converted with one constant (2 s); the provided ground truth's
  exact notion of control may differ.
* Short possessions after turnovers are the main loss; better pass/receiver modelling or a
  learned ball-trajectory model would target them.
* More external games — especially automatic tracking from another provider — would make the
  simulation less important.

## 8. Rules compliance

* No hand labels, no manual corrections, no human-produced data about the provided game.
* Training and tuning on external games only; the public 5-minute window is a pipeline check.
* No frame numbers or other constants tied to specific moments of the provided game: kick-offs and
  attacking directions are detected; the half windows are the organisers' grading windows
  (`games.py`), used as published.
* **AI assistance, and what it never saw.** We used an AI coding assistant to write and review
  code. The provided data was never uploaded to it or to any other AI service: all processing ran
  locally, and only aggregate numbers (counts, rates, quantiles, metric scores) were ever printed
  during development — never frames, rows, positions or labels.
* **One judgement call, stated openly.** The degradation that makes external games resemble the
  provided tracking (§3) matches a handful of *aggregate, unlabelled* statistics of the provided
  ATD (ball-visible share, run lengths of missing ball, visibility vs share of players standing,
  players tracked per team). No ground truth and no video are involved; it is unsupervised
  matching of the input distribution, in the spirit of "using as input the automatic tracking
  data". Should the organisers consider it out of bounds, these parameters can be set from first
  principles instead (`solution/degrade.py`). Nothing that is *trained or tuned* — models,
  ensemble weights, decoder — sees anything but external games.
* External data licences checked (CC BY 4.0 for the data used).

### Stress test

The external benchmark ties ball loss to stoppages more cleanly than a real tracker does. We
therefore also scored every fold on a deliberately harsher degradation (`degrade.HARD`: ball lost
up to 3 s early / 4 s late around stoppages, slow re-acquisition, ball visible through more of
each stoppage, lower live-play visibility) and chose decoder settings on the average of both
benchmarks. LightGBM alone: 0.792 standard → 0.765 stress; the best decoder settings are the
same on both, so the choice is not an artefact of the easy benchmark. Ensemble: see §6.

## 9. Reproduction

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-solution.txt --extra-index-url https://download.pytorch.org/whl/cpu
# match bundle unpacked into data/ (DATA.md)
python -m solution.predict --name <team_name>      # committed models; CPU; 1.5-7 min
python evaluation/validate_submission.py submissions/<team_name>.csv
```

Retraining from external data (optional; deterministic for LightGBM):

```bash
python -m solution.download_external                                  # DFL open data, 2.6 GB
python -m solution.train cv && python -m solution.train final        # LightGBM
SEQ_EPOCHS=30 SEQ_BASE_ONLY=1 SEQ_NAME=gru_base python -m solution.seq final
SEQ_EPOCHS=30 SEQ_NAME=gru_full python -m solution.seq final
SEQ_SEED=8 SEQ_EPOCHS=30 SEQ_BASE_ONLY=1 SEQ_NAME=gru_base_s8 python -m solution.seq final
SEQ_SEED=8 SEQ_EPOCHS=30 SEQ_NAME=gru_full_s8 python -m solution.seq final
# decoder: MTAH_STRESS=1 for every CV run, solution.ensemble for the blend, then
python -m solution.robust_tune <standard oof> <stress oof> --out decoder_robust.json
```
