"""One in-memory representation shared by the provided match and every external match.

Every loader (provided ATD, DFL open data, Metrica sample data, ...) produces a `Match`, so the
feature code, the model and the decoder never know where a game came from. That is what lets us
train and tune exclusively on external games and then run the identical code on BAR-ATH.

Conventions
-----------
* 25 frames per second; frame i of the arrays is frame i of the source's timeline.
* Pitch coordinates in metres, origin at a corner, x along the length in [0, L], y in [0, W].
* NaN marks a missing observation (ball or player) -- never 0.
* team: 0 = home, 1 = away.  Labels: 0 = H, 1 = A, 2 = D, 3 = X (no label / not graded).
* `attack_dir[half]` is +1 when the HOME team attacks towards x = L in that half, else -1.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

H, A, D, X = 0, 1, 2, 3
LABEL_CHARS = np.array(["H", "A", "D", "X"])


@dataclass
class Match:
    name: str
    fps: int
    pitch: tuple                      # (length, width) in metres
    ball_xy: np.ndarray               # (n, 2) float32, NaN where missing
    ball_s: np.ndarray                # (n,)   float32 provider speed, NaN where missing
    players_xy: np.ndarray            # (n, P, 2) float32, NaN where missing
    players_s: np.ndarray             # (n, P) float32 provider speed (m/s), NaN where missing
    team: np.ndarray                  # (P,) int8, 0 home / 1 away
    half: np.ndarray                  # (n,) int8, 1 / 2 inside a half's play window, 0 outside
    attack_dir: Dict[int, int] = field(default_factory=dict)
    labels: Optional[np.ndarray] = None   # (n,) int8 H/A/D/X, external games only
    ball_z: Optional[np.ndarray] = None   # (n,) float32, external games that have it
    meta: dict = field(default_factory=dict)

    @property
    def n(self):
        return len(self.half)

    def half_blocks(self):
        """[(half, start, stop)] half-open frame spans of each half's play window."""
        out = []
        for h in (1, 2):
            idx = np.flatnonzero(self.half == h)
            if len(idx):
                out.append((h, int(idx[0]), int(idx[-1]) + 1))
        return out


def estimate_attack_dir(match, kickoff_frames):
    """Home attacking direction per half from the kick-off formation.

    At a kick-off every player stands in his own half, so the side of halfway the home team's
    centroid sits on tells us which goal it defends. Averaged over the first two seconds after the
    kick-off frame to be robust to a single bad frame. Uses positions only, no labels.
    """
    L = match.pitch[0]
    out = {}
    for h, frame in kickoff_frames.items():
        lo, hi = max(0, frame - 25), min(match.n, frame + 25)
        xs = match.players_xy[lo:hi, :, 0]
        home = np.nanmean(xs[:, match.team == 0])
        away = np.nanmean(xs[:, match.team == 1])
        if not np.isfinite(home) or not np.isfinite(away):
            raise ValueError("cannot estimate attack direction for half {}".format(h))
        # Home defends the low-x goal (so attacks +x) when its centroid is the lower one.
        out[h] = 1 if home < away else -1
    match.attack_dir = out
    return out
