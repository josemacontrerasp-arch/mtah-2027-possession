"""Smart Tagging (Metrica's automatic tactical tagging) as possession evidence.

Each tagged phase says something about who has the ball:
* attacking phases (BUILD UP, PROGRESSION, CREATING CHANCES, ATTACKING TRANSITION, BALL IN FINAL
  THIRD, BALL IN THE BOX, PLAYERS IN THE BOX, LONG BALL) are tagged with the team IN possession;
* defending phases (DEFENDING IN * THIRD) and DEFENSIVE TRANSITION are tagged with the team OUT
  of possession, so they point at the other side;
* SET PIECES mark a restart inside a ~5 s window: dead before it, the tagged side after it.

The tagging is automatic, not ground truth. How far to trust each kind of tag is estimated
without any labels: by agreement with the frame classifier where that classifier is confident
(see `reliability`). Prints aggregates only.
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import asset_paths  # noqa: E402
from solution.match import H, A  # noqa: E402

FPS = 25
IN_POSSESSION = {"build up", "progression", "creating chances", "attacking transition",
                 "ball in final third", "ball in the box", "players in the box", "long ball"}
OUT_OF_POSSESSION = {"defending in attacking third", "defending in middle third",
                     "defending in defensive third", "defensive transition"}


def _side(team_text):
    t = (team_text or "").strip().lower()
    if "barcelona" in t:
        return H
    if "atletic" in t or "athletic" in t:
        return A
    return None


def load_tags(path=None):
    path = path or asset_paths()["smart_tagging"]
    out = []
    for inst in ET.parse(path).iter("instance"):
        tags = {lab.findtext("group"): lab.findtext("text") for lab in inst.findall("label")}
        out.append(dict(code=(inst.findtext("code") or "").strip().lower(),
                        start=int(round(float(inst.findtext("start")) * FPS)),
                        end=int(round(float(inst.findtext("end")) * FPS)),
                        side=_side(tags.get("Team")),
                        type=(tags.get("Type") or "").strip().lower()))
    return out


def team_evidence(tags, n):
    """Per frame: +1 where tags point at H, -1 at A (summed over overlapping tags), and the code
    family that produced it, for reliability estimation."""
    ev = {"in": np.zeros(n), "out": np.zeros(n), "set": np.zeros(n)}
    for t in tags:
        if t["side"] is None:
            continue
        sign = 1.0 if t["side"] == H else -1.0
        a, b = max(0, t["start"]), min(n, t["end"])
        if t["code"] in IN_POSSESSION:
            ev["in"][a:b] += sign
        elif t["code"] in OUT_OF_POSSESSION:
            ev["out"][a:b] -= sign
        elif t["code"] == "set pieces":
            # the restart is inside the window; possession after it is the tagged side
            mid = (a + b) // 2
            ev["set"][mid:b] += sign
    return ev


def reliability(ev, prob, alive_min=0.8, conf_min=0.9):
    """Agreement of each tag family with the frame classifier where the classifier is confident.

    Returns {family: (agreement, n_frames)}. No labels involved: the classifier was trained on
    external games only.
    """
    alive = (prob[:, 0] + prob[:, 1]) >= alive_min
    p_home = prob[:, 0] / np.clip(prob[:, 0] + prob[:, 1], 1e-6, None)
    confident = alive & ((p_home >= conf_min) | (p_home <= 1 - conf_min))
    out = {}
    for fam, e in ev.items():
        sel = confident & (e != 0)
        if sel.sum() == 0:
            out[fam] = (float("nan"), 0)
            continue
        agree = np.mean((e[sel] > 0) == (p_home[sel] >= 0.5))
        out[fam] = (float(agree), int(sel.sum()))
    return out


if __name__ == "__main__":
    tags = load_tags()
    codes = {}
    for t in tags:
        codes[t["code"]] = codes.get(t["code"], 0) + 1
    print("instances {}, with a side {}, codes {}".format(
        len(tags), sum(t["side"] is not None for t in tags), len(codes)))
