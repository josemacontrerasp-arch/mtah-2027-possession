# Frame-level possession from automatic tracking — technical document

*More than a Hack 2027, Phase 1. Version v6 ("clean room"), selected 7 November 2026; document updated 13 November 2026.*

## 1. Approach

Possession is predicted in two steps.

1. **Per-frame probabilities** of *home in possession*, *away in possession* or *dead ball* from an
   ensemble fed the same collective tracking features: a LightGBM classifier (weight 0.3) and
   four bidirectional GRUs at 5 Hz, trained on two-minute stretches and reading each half whole
   at prediction (0.7 in total: two seeds of a
   GRU on raw per-frame features, 0.245 each, and two seeds of a GRU that also sees hand-built
   context windows, 0.105 each).
2. **A constrained HMM** decoded with Viterbi turns probabilities into possessions: three states
   (H, A, D) with switching costs, plus the definitional constraint that a possession *won from
   the opponent* lasts at least 2.5 s (a restart possession may be any length).

**Nothing in the pipeline is trained, tuned, calibrated or fitted on the provided match.** Every
model is trained on seven Bundesliga games of the DFL open dataset. Because those games come from
clean league tracking, they are degraded by randomly drawn simulated automatic trackers (domain
randomisation) from broad first-principles priors — no value is taken from the provided match. Ensemble weighting and decoder settings were chosen by nested
leave-one-match-out on *clean* simulated environments, under rules written and committed before
the results existed (`PREREGISTRATION.md`; every decision in `selection_log.md`).

Why this design: the ranking metric scores whole possessions, so the output must be temporally
coherent; the hard cases (open-play turnovers, long stretches without a tracked ball) need tens
of seconds of context; and since we may not learn from the provided match, the model must work
on *any* plausible automatic tracker rather than on one simulation of it.

## 2. Assets used and dropped

| asset | status | how |
|---|---|---|
| tactical ATD | **used** | all features; own loader (`atd.py`), slot order from the EPTS DataFormatSpecification |
| broadcast ATD + homography | tried, **dropped** | aligned to the tactical timeline from tracking alone (one offset per half); its ball fills and "cut-away to replay" stoppage signal were informative but could not be weighted without labels |
| Smart Tagging | tried, **dropped** | team hints agree with confident model frames, but cannot be weighted without labels |
| tactical homography | tried, **dropped** | the camera frames the team block, not the ball |
| videos | not used | |
| public 5-minute ground truth | **pipeline check only** | one check of the final version; it decided nothing |

## 3. Training data

| dataset | licence | labels | labelled by | amount |
|---|---|---|---|---|
| DFL open dataset — Bassek, Rein, Weber, Memmert (2025), *Scientific Data* 12:195, doi:10.6084/m9.figshare.28196177 | CC BY 4.0 | per-frame `BallStatus` (alive/dead) and `BallPossession` (home/away) | the league's official data provider (DFL / Sportec Solutions) | 7 matches, 1.0 M frames at 25 Hz |

No other labelled data and no human-produced data about the provided game.

### Domain randomisation (`degrade_random.py`)

Four simulated trackers are drawn per training run (seeds 11–14), and each is applied to every
external game in both orientations (home and away swapped) — 56 degraded copies. The trackers come
from broad priors chosen from first principles, ranging from a tracker that almost never loses the
ball to one that loses it most of the time:

* ball visibility in live play: alternating visible / missing runs, log-normal, medians 80–1000
  and 20–500 frames;
* stoppages: the ball hidden with probability 0.2–1.0, a placed ball before restarts, loss
  0–3 s early and 0–4 s late around the stoppage, 0–3 s re-acquisition after the restart;
  DFL's frozen ball during stoppages (an artefact: 100% of dead frames) replaced by a carried
  path to the restart spot;
* lofted balls (z > 3 m) dropped with probability 0–0.8;
* players: 1–12% missing each, bursty group losses, smooth position noise 0.05–0.45 m, speed noise.

The statistics the organisers publish in DATA.md (ball visible on 57–59% of frames, 20.5
tracked objects) fall inside these ranges; they were used for orientation only.

**Why not calibrate the simulator to the provided tracking (our v4)?** Matching aggregate,
unlabelled statistics of the provided ATD is arguably compatible with "use the tracking as input",
but the 3-Nov email says parameters "MUST" be tuned on external data. It also made the model
fragile: the calibrated model fell from 0.79 to 0.65 spellF1 on a simulated tracker it had not
been calibrated to, while the randomised one held 0.75 (§6).

## 4. Possession rules implemented

* **Controlled possession.** A change of possession needs the opponent to gain *controlled*
  possession; touches, deflections and aerial duels without control do not count, and a contested
  or loose ball stays with the last team. External labels flip on every touch (DFL spells: median
  3.7 s, 30% under 2 s), so before training they are converted (`relabel.py`): a spell that
  interrupts the other team's possession and lasts under **2 s** ("a player keeps the ball for a
  few seconds") reverts to that team; spells starting from a restart are kept. The constant comes
  from the definition text.
* **Restarts** always have a clear possessor: no minimum length after a stoppage.
* **Minimum turnover length 2.5 s** in the decoder (chosen on clean environments; shorter
  minima, a minimum stoppage length and turnovers ending in a stoppage were tested and rejected
  by the pre-registered rule).
* **Dead ball** is its own state. Frames of a half's play window before its kick-off are `D`.
  Kick-offs are **detected from the tracking** (`kickoff.py`): the formation of both teams fully
  in their own halves that breaks into play. **All eleven of its constants were chosen on external
  kick-offs** by coordinate search (`kickoff_check.py rederive_all`): the 17 kick-offs after goals
  in the DFL games, under two randomised trackers each, with the search window opening at the
  kick, one minute before and two minutes before (the layouts of the published grading windows) —
  102 of 102 found, median error 8 frames. The plain check (`kickoff_check.py`) finds 17/17 with
  the attacking direction right every time. The same formation gives each team's attacking
  direction. No frame number of the provided match appears in the pipeline.
* No `X` on graded frames.

## 5. Features (`features.py`)

Collective and orientation-normalised (home attacks +x); nothing keyed to a track slot. Per
frame: ball position, speed, distance to the boundary; nearest home / away player distances,
counts within 3 m and 10 m, relative ball–player speed; team speeds, shares running / standing,
centroids, spreads, extremes, share goal-side of the ball, centroid velocities; ball-control
evidence at 1.0 / 1.5 / 2.5 m with last / next controlling team and time since / until; ball
visibility history; the ball's path interpolated across gaps up to 15 s and which team lines it;
pass receiver (first player ahead of a moving ball inside a cone); which team reaches the ball's
position 0.6 s ahead first; past / future / centred windows of 1–8 s. 184 features.

## 6. Results (external games, leave-one-match-out)

**Clean environments** (simulated trackers with no value from the provided match, never used in
training: two independent draws from the priors and their harsher half), decoder and ensemble
weighting chosen on six matches and scored on the seventh:

| pipeline | `rnd` | `rnd2` | `rndh` (harsh) | **nested** |
|---|---|---|---|---|
| official baseline algorithm (for scale, same games) | | | | ≈ 0.32\* |
| v4 — calibrated to provided-ATD statistics | 0.707 | 0.745 | 0.586 | 0.679 |
| **v6 — clean room (submitted)** | **0.776** | **0.783** | **0.761** | **0.774** |
| v7 — its own tracker for every training copy (not adopted) | 0.782 | 0.787 | 0.762 | 0.777 |

v6 is better than v4 on all seven held-out matches. \*on the earlier calibrated benchmark.

**v7, tested and not adopted.** v6 draws four simulated trackers and applies each to every game;
v7 gives each of the 56 training copies its own. The rule for switching was written and committed
before v7 was trained (`PREREGISTRATION.md`, addendum of 12 Nov): at least +0.005 nested, better
on at least 5 of the 7 matches, no environment worse by more than 0.003. v7 gained +0.0035 and was
better on 4 of 7 matches — within the noise of seven games — so v6 stays. (LightGBM alone: 0.735
for v6, 0.733 for v7.)

The split between the two GRU kinds (70:30) was re-checked on the clean environments against
50:50 and 90:10: differences ≤ 0.001 in-sample and a lower nested score with the extra choice, so
it was kept.

Tested and **not adopted** (rule: ≥ +0.005 nested): shorter or longer turnover minima, a minimum
stoppage length, turnovers ending in a stoppage (nested 0.771); time-varying switch costs from
posterior jumps (0.773); second-stage stacking; a larger LightGBM; extra ensemble members.

Where it still loses (external diagnostics): short possessions after turnovers, and the exact
timing of turnovers (±1 s).

**Pipeline check** of the final file on the public 5-minute window (the only time v6 was scored
there; it decided nothing): spellF1 0.676, macroF1 0.906, alive/dead F1 0.972, team accuracy 0.95
where both agree the ball is alive; frames aligned. Seventeen possessions cannot rank versions.

## 7. Limitations

* The possession definition is converted with one constant (2 s); the provided ground truth's
  notion of control (Metrica's corrected-tracking ownership) may differ.
* Turnover timing and very short possessions are the main remaining loss.
* The randomised priors are wide by design; a tracker outside them (for example systematic
  position offsets) is not covered.
* With more time: more external games, ideally automatic tracking with possession labels from a
  second provider; a boundary-event model for turnovers.

## 8. Rules compliance

* No hand labels, manual corrections or human-produced data about the provided game.
* **Training and tuning on external games only.** No parameter of the pipeline — simulator,
  models, ensemble weights, decoder, kick-off detector — was fitted to, measured on, or chosen by
  looking at the provided match. The selection rules were committed before the results
  (`PREREGISTRATION.md`, commit `bac40f6`; addendum for the v7 candidate, commit `f0c5e55`) and
  every decision is logged (`selection_log.md`).
* Checks on the provided match: runtime sanity (row count, validator, byte-identical
  regeneration), numerical-robustness checks that print hashes and counts only (regeneration
  under emulated CPUs, noise injection, equality of the feature table between a fresh parse and
  the cache) and one pipeline check of the final version on the public window.
* **AI assistance.** An AI coding assistant was used to write and review code. The provided data
  was never uploaded to it or to any other AI service: everything ran locally and only aggregate
  numbers (counts, rates, metric scores) were ever printed during development.
* Feature design and model hyper-parameters were developed in v1–v4 against external benchmarks
  that included the calibrated simulator. v6 retrains every model under randomised degradation and
  re-selects ensemble weighting, decoder and kick-off constants on clean environments only.
* History, for transparency: an earlier version (v4, never submitted) calibrated its simulator to
  aggregate, unlabelled statistics of the provided tracking and set its kick-off thresholds while
  looking at aggregate diagnostics of the provided match. v6 replaces both, and the scripts that
  did it (`stats_visibility.py`, `domain_check.py`, `degrade.py`'s calibrated profile) are kept as
  history but are not part of the submission chain.
* Licences: DFL data CC BY 4.0 (cited above; the committed models are trained on it — see
  `NOTICE`); starter code and our code MIT.

## 9. Reproduction

```bash
uv venv .venv --python 3.12
uv pip install --python .venv -r requirements-solution.txt -c requirements-solution.lock.txt
source .venv/bin/activate              # Windows: .venv\Scripts\activate
# match bundle unpacked into data/ (DATA.md)
python -m solution.predict --name <team_name>          # committed models, CPU, ~3 min
python evaluation/validate_submission.py submissions/<team_name>.csv
python -m solution.repro_check --fresh-cache           # regenerates and compares the SHA-256
python -m solution.preflight --name <team_name>        # every scriptable check of SUBMISSION.md
```

Plain `pip` works too (`python -m pip install -r requirements-solution.txt -c
requirements-solution.lock.txt` in a Python 3.12 environment). Everything installs from PyPI;
the lock file pins every indirect dependency, and both commands were checked to build exactly the
environment it lists. `MTAH_DATA_DIR` may point at the bundle instead of unpacking it into
`data/`.

**The same file on any machine.** The submission's SHA-256 is
`3333bf64350e9a915e1c61fd4871875266f56cf40246ac06b0fe19f243b202e1` (`models/submission.sha256`).
The chain is built so that floating-point differences between machines do not reach the file:

* **features** use only correctly rounded operations (add, multiply, divide, square root) in a
  fixed order, on arrays whose memory layout is the same on every code path — no maths-library
  call;
* LightGBM prediction is deterministic; the GRUs run on CPU, single thread, in **double
  precision**;
* the blended probabilities are put on a **1e-4 grid** and the decoder works in **exact integer
  arithmetic** (log-probabilities and costs in units of 1e-6 nat; every Viterbi score is an exact
  integer), so the labels are a pure function of the gridded probabilities, with ties broken in a
  fixed order. The file can change only if a blended probability crosses a rounding boundary.

Measured with `solution/noise_check.py` (counts and distances only): of the 476,571 blended
probabilities, the one closest to a rounding boundary is 1.5e-11 away. Running again with the
maths libraries limited to older instruction sets — `MKL_ENABLE_INSTRUCTIONS=SSE4_2`,
`ATEN_CPU_CAPABILITY=default`, `NPY_DISABLE_CPU_FEATURES="AVX2 FMA3"`, `NUMBA_CPU_NAME=generic` —
moves the probabilities by at most 6.7e-16, more than 20,000 times less, and leaves the feature
table bit-identical. Injected relative noise of 1e-10 changed no grid value in 200 trials; 1e-7,
the level of single precision, changed about 7 grid values per trial and never the file. The
steps that still depend on the platform (the GRU arithmetic, one interpolation, two exponentials)
act at the 1e-15 level, behind the grid.

Also checked: environments built from scratch with `uv` and with `pip`; a fresh clone with every
cache rebuilt from the raw files (`solution/preflight.py` repeats this); six emulated CPU
configurations, the same file every time. What we could not run is other hardware — only x86-64
Windows — so `python -m solution.repro_check --fresh-cache` checks any machine in one command.
On macOS, LightGBM needs `brew install libomp`.

**Without the match data**, `python -m solution.selftest` (about 20 s) runs the whole chain —
kick-off detection, features, the five models, blending, grid and decoder — on a synthetic match
built from a fixed seed with exactly rounded arithmetic, and compares every stage with the values
recorded on our machine (`models/selftest_reference.json`): features, LightGBM scores, grid values
and labels must be bit-identical, GRU probabilities equal within 1e-12.
`.github/workflows/selftest.yml` installs the environment as above and runs it on Linux, macOS
and Windows.

Retraining from external data (optional). LightGBM training is bit-for-bit deterministic:
`solution.train final` rebuilds the committed model file exactly (checked). GPU GRU training is
not, which is why trained weights are committed:

```bash
python -m solution.download_external                                     # DFL, 2.6 GB
MTAH_DEGRADE=random MTAH_SEEDS=11,12,13,14 MTAH_MODEL_NAME=frame_lgbm_v6 python -m solution.train final
for s in 7 8; do  # raw-feature GRU and full-feature GRU, two seeds each
  MTAH_DEGRADE=random MTAH_SEEDS=11,12,13,14 SEQ_SEED=$s SEQ_EPOCHS=30 SEQ_BASE_ONLY=1 SEQ_NAME=gru_base$([ $s = 8 ] && echo _s8)_v6 python -m solution.seq final
  MTAH_DEGRADE=random MTAH_SEEDS=11,12,13,14 SEQ_SEED=$s SEQ_EPOCHS=30 SEQ_NAME=gru_full$([ $s = 8 ] && echo _s8)_v6 python -m solution.seq final
done
# selection: cross-validate every member with MTAH_EVALS=rnd,rnd2,rndh (train cv / seq cv), blend
# each LightGBM share w in {0.3, 0.5, 0.7} with solution.ensemble into oofc_v6_ens<w>_<env>, then
#   python -m solution.clean_select ens_v6_final --save decoder_v6.json \
#     --cand w0.3=oofc_v6_ens0.3_rnd,oofc_v6_ens0.3_rnd2,oofc_v6_ens0.3_rndh (same for w0.5, w0.7) \
#     --grid '{"switch_cost":[3,4,6,8,12],"to_dead_cost":[16,32,48],"from_dead_cost":[16,32,48],"min_turnover_s":[2.5],"dead_bias":[-0.5,0.0,0.5]}'
# the v7 candidate (not adopted) is the same recipe with MTAH_TRACKERS=percopy; two selections are
# compared under the pre-registered criteria with
#   python -m solution.clean_select --compare <candidate name> <incumbent name>
```
