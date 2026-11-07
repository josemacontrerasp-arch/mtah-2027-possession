"""Where does the decoded output lose spellF1 on held-out EXTERNAL games? (DFL only.)

    python -m solution.diagnose [stage]

Reports per-IoU F1, a boundary-error breakdown for matched possessions, which true possessions
are missed (by duration and by how they start/end), and alive/dead timing errors.
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

from solution.decode import decode  # noqa: E402
from solution.match import H, A, D, X  # noqa: E402
from solution.tune import load_oof, spaced, blocks_of, MODEL_DIR  # noqa: E402


def segs(codes, runs):
    return pm.segments(codes, runs)


def main(stage=1):
    data = load_oof(stage)
    name = "decoder.json" if stage == 1 else "decoder_stack.json"
    params = json.load(open(os.path.join(MODEL_DIR, name)))["params"]
    per_iou = {t: [] for t in pm.SPELL_IOUS}
    start_err, end_err = [], []
    missed_len, hit_len = [], []
    miss_ctx = {"D->X->D": 0, "D->X->T": 0, "T->X->D": 0, "T->X->T": 0}
    hit_ctx = dict.fromkeys(miss_ctx, 0)
    dead_on, dead_off = [], []
    for mid, (prob, labels, half) in data.items():
        pred = decode(np.nan_to_num(prob, nan=1.0 / 3), blocks_of(half), len(half), **params)
        gt, pr = spaced(labels, half), spaced(pred, half)
        runs = pm.valid_runs(gt)
        r = pm.spell_scores(gt, pr, runs)
        for t in pm.SPELL_IOUS:
            per_iou[t].append(r["per_iou"][t]["f1"])
        g_sp = pm.possession_spells(gt, runs)
        p_sp = pm.possession_spells(pr, runs)
        ov = pm.spell_overlaps(g_sp, p_sp)
        best = {}
        for iou, gi, pj in ov:
            if gi not in best:
                best[gi] = (iou, pj)
        allseg = segs(gt, runs)
        seg_index = {(a, b): k for k, (a, b, _) in enumerate(allseg)}
        for gi, (a, b, k) in enumerate(g_sp):
            L = (b - a) / 25.0
            k_idx = seg_index[(a, b)]
            before = allseg[k_idx - 1][2] if k_idx > 0 else D
            after = allseg[k_idx + 1][2] if k_idx + 1 < len(allseg) else D
            ctx = "{}->X->{}".format("D" if before == D else "T", "D" if after == D else "T")
            if gi in best and best[gi][0] >= 0.5:
                hit_len.append(L)
                hit_ctx[ctx] += 1
                pa, pb, _ = p_sp[best[gi][1]]
                start_err.append((pa - a) / 25.0)
                end_err.append((pb - b) / 25.0)
            else:
                missed_len.append(L)
                miss_ctx[ctx] += 1
        # dead-ball onset / offset timing: nearest predicted D-boundary to each true one
        for (a, b, k) in allseg:
            if k != D:
                continue
            pd_starts = [s for s, e, kk in segs(pr, runs) if kk == D]
            pd_ends = [e for s, e, kk in segs(pr, runs) if kk == D]
            if pd_starts:
                dead_on.append(min((s - a for s in pd_starts), key=abs) / 25.0)
                dead_off.append(min((e - b for e in pd_ends), key=abs) / 25.0)
    print("decoder", params)
    print("F1 by IoU threshold (mean over matches):",
          {t: round(float(np.mean(v)), 3) for t, v in per_iou.items()})
    se, ee = np.array(start_err), np.array(end_err)
    print("matched @IoU>=0.5: n={}  start err median {:+.2f}s  |p50| {:.2f}s |p90| {:.2f}s ; "
          "end err median {:+.2f}s |p50| {:.2f}s |p90| {:.2f}s".format(
              len(se), np.median(se), np.median(np.abs(se)), np.percentile(np.abs(se), 90),
              np.median(ee), np.median(np.abs(ee)), np.percentile(np.abs(ee), 90)))
    ml, hl = np.array(missed_len), np.array(hit_len)
    for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 25), (25, 1e9)):
        nm = int(((ml >= lo) & (ml < hi)).sum())
        nh = int(((hl >= lo) & (hl < hi)).sum())
        print("  true possessions {:>4}-{:<4}s: recall@0.5 {:.2f}  (missed {}, hit {})".format(
            lo, hi if hi < 1e9 else "inf", nh / max(1, nm + nh), nm, nh))
    print("recall@0.5 by context (before->possession->after, T = other team):")
    for c in miss_ctx:
        tot = miss_ctx[c] + hit_ctx[c]
        print("  {:<8}  n {:>4}  recall {:.2f}".format(c, tot, hit_ctx[c] / max(1, tot)))
    do, df = np.array(dead_on), np.array(dead_off)
    print("dead-ball onset error median {:+.2f}s |p50| {:.2f}s |p90| {:.2f}s ; restart error "
          "median {:+.2f}s |p50| {:.2f}s |p90| {:.2f}s".format(
              np.median(do), np.median(np.abs(do)), np.percentile(np.abs(do), 90),
              np.median(df), np.median(np.abs(df)), np.percentile(np.abs(df), 90)))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
