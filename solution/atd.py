"""Fast loader for the provided Metrica FIFA/EPTS ATD (tactical or broadcast) -> Match.

The raw file is perfectly regular -- "<frame>:<P groups of x,y,s>:<x,y,s>" on every line -- so
the whole file is parsed in one C-level pass by pandas after turning both separators into commas.
The result is cached as .npz next to the data (data/ is gitignored, so the cache never ships).

Slot order is taken from the DataFormatSpecification's PlayerChannelRef sequence, which is what
actually defines column order in the raw file, and each slot's team from its Player element.
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from games import GAME, GAMES, asset_paths, half_video_blocks  # noqa: E402
from solution.match import Match, estimate_attack_dir  # noqa: E402

CACHE_VERSION = 2


def _local(tag):
    return tag.split("}")[-1]


def parse_meta(meta_path):
    """(team_of_slot (P,) int8 0=home/1=away, home_team_id, away_team_id, frame_rate)."""
    root = ET.parse(meta_path).getroot()
    teams = [el.get("id") for el in root.iter() if _local(el.tag) == "Team"]
    if len(teams) != 2:
        raise ValueError("expected 2 teams in metadata, found {}".format(len(teams)))
    home_id = teams[0]  # FIFA/EPTS convention: first team listed is the home side

    player_team = {el.get("id"): el.get("teamId")
                   for el in root.iter() if _local(el.tag) == "Player"}
    channel = {el.get("id"): (el.get("playerId"), el.get("channelId"))
               for el in root.iter() if _local(el.tag) == "PlayerChannel"}
    refs = [el.get("playerChannelId")
            for el in root.iter() if _local(el.tag) == "PlayerChannelRef"]
    if len(refs) % 3:
        raise ValueError("player channel refs not a multiple of 3")

    slots = []
    for i in range(0, len(refs), 3):
        trio = [channel[r] for r in refs[i:i + 3]]
        players = {p for p, _ in trio}
        kinds = [c for _, c in trio]
        if len(players) != 1 or kinds != ["x", "y", "s"]:
            raise ValueError("unexpected channel layout in slot {}".format(i // 3))
        slots.append(players.pop())
    team_of_slot = np.array([0 if player_team[p] == home_id else 1 for p in slots], dtype=np.int8)

    rate = None
    for el in root.iter():
        if _local(el.tag) == "FrameRate" and el.text:
            rate = float(el.text)
    return team_of_slot, home_id, teams[1], rate


def parse_raw(raw_path, n_slots):
    """(frame_id (n,), players (n, P, 3), ball (n, 3)) as float32 with NaN for missing."""
    with open(raw_path, "r") as fh:
        text = fh.read()
    text = text.replace(":", ",").replace(";", ",")
    n_cols = 1 + 3 * n_slots + 3
    table = pd.read_csv(io.StringIO(text), header=None, dtype=np.float64, engine="c",
                        na_values=["NaN", "nan"], keep_default_na=False)
    del text
    if table.shape[1] != n_cols:
        raise ValueError("raw file has {} columns, expected {}".format(table.shape[1], n_cols))
    values = table.values
    frame_id = values[:, 0].astype(np.int64)
    # pandas hands back column-major memory; make the arrays row-major (C-contiguous) so a fresh
    # parse and a load from the cache have the same layout. NumPy picks its summation order from
    # the layout, so without this the two could differ in the last bit of a few features.
    players = np.ascontiguousarray(values[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3),
                                   dtype=np.float32)
    ball = np.ascontiguousarray(values[:, 1 + 3 * n_slots:], dtype=np.float32)
    return frame_id, players, ball


def _cache_path(source):
    return os.path.join(ROOT, "data", "cache", "{}_{}_atd_v{}.npz".format(GAME, source,
                                                                          CACHE_VERSION))


def load_atd(source="tactical", use_cache=True, game=GAME, kickoff="detect"):
    """The provided ATD of one source as a Match (labels=None: we never load ground truth here)."""
    cache = _cache_path(source)
    paths = asset_paths(game, source=source)
    L, W = GAMES[game]["pitch"]

    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        frame_id, team = z["frame_id"], z["team"]
        players, ball = np.ascontiguousarray(z["players"]), np.ascontiguousarray(z["ball"])
    else:
        team, _, _, _ = parse_meta(paths["atd_meta"])
        frame_id, players, ball = parse_raw(paths["atd_raw"], len(team))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, frame_id=frame_id, players=players, ball=ball, team=team)

    n = len(frame_id)
    if not np.array_equal(frame_id, np.arange(1, n + 1)):
        raise ValueError("ATD frame ids are not 1..n -- alignment assumption broken")

    scale = np.array([L, W], dtype=np.float32)
    players_xy = players[:, :, :2] * scale
    ball_xy = ball[:, :2] * scale

    half = np.zeros(n, dtype=np.int8)
    if source == "tactical":
        for h, first, last in half_video_blocks(game):
            half[first:last + 1] = h

    m = Match(name="{}_{}".format(game, source), fps=GAMES[game]["fps"], pitch=(L, W),
              ball_xy=ball_xy, ball_s=ball[:, 2].copy(), players_xy=players_xy,
              players_s=players[:, :, 2].copy(), team=np.asarray(team, dtype=np.int8),
              half=half, meta={"source": source})
    if source == "tactical":
        if isinstance(kickoff, dict):
            # Only used by the detector's self-check (solution/kickoff.py), never by the pipeline.
            estimate_attack_dir(m, kickoff)
        else:
            # Kick-offs and attacking directions detected from the tracking itself.
            from solution.kickoff import detect_all
            found = detect_all(m)
            if any(k is None for k, _ in found.values()):
                raise ValueError("kick-off not found in the tracking for every half")
            m.attack_dir = {h: d for h, (k, d) in found.items()}
            m.meta["kickoff"] = {h: k for h, (k, d) in found.items()}
    return m


if __name__ == "__main__":
    # Aggregate sanity summary only -- no frame-level values are printed.
    for src in ("tactical", "broadcast"):
        m = load_atd(src)
        vis = np.isfinite(m.ball_xy[:, 0])
        n_tracked = np.isfinite(m.players_xy[:, :, 0]).sum(axis=1)
        graded = m.half > 0
        print("{:<10} frames {:>7}  slots {:>2} (home {}, away {})  ball visible {:.1%}"
              .format(src, m.n, len(m.team), int((m.team == 0).sum()), int((m.team == 1).sum()),
                      vis.mean()))
        if graded.any():
            print("           graded frames: ball visible {:.1%}, mean tracked objects {:.1f}, "
                  "attack_dir {}".format(vis[graded].mean(), n_tracked[graded].mean(),
                                         m.attack_dir))
