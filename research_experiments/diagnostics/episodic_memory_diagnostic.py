"""
episodic_memory_diagnostic.py
===============================
Does a child's own earlier hypoglycaemic precursors carry predictive
signal that the current fixed-window model cannot access?

THE HYPOTHESIS
--------------
Every model in this project reads a 60-reading window and nothing else.
Once a child's earlier episode falls outside that window it is gone. The
48-hour patient-context experiment did not test this: it compressed
history into summary statistics (variability, time-below-range, episode
counts) and lost the SHAPE of the precursor trajectory entirely. That
experiment came back negative, which does not settle whether the
trajectories themselves would help.

This asks: at prediction time t, if we compare the current window against
the precursor trajectories of that child's OWN previously completed
episodes, does the similarity predict what happens next?

STRICT CAUSALITY
----------------
An episode enters the memory only once it has ENDED, and only if it ended
before the current window begins. Nothing from the current window, the
prediction horizon, or the future can enter. `verify_causality()` checks
this directly by corrupting future readings and confirming earlier
similarity scores are unchanged.

Memory is also per-subject: a test child is compared only against their
own history, never another child's, so no information crosses the split.

WHAT IS COMPARED
----------------
A precursor is the 60 readings immediately before an episode onset,
represented as a z-scored shape so that similarity captures the
TRAJECTORY PATTERN rather than the absolute level -- absolute level is
already available to the model through the TA and absolute channels, and
including it here would just re-measure something known.

Three similarity statistics per window: maximum, mean of the top three,
and the count of precursors above a threshold.

THE DECISIVE TEST
-----------------
Similarity being higher before events is necessary but not sufficient --
it could simply track "glucose is falling", which the model already sees.
So the diagnostic also fits a logistic regression of the outcome on the
model's own predicted probability, then asks whether adding the
similarity score improves it. That isolates information the model does
NOT already have.

    similarity adds no lift over the model score
        -> the architecture cannot exploit what it cannot distinguish;
           stop before building an episodic-memory model

    similarity adds lift, especially on the missed events
        -> a demonstrable source of information unavailable to the
           baseline, which is what every previous architecture attempt
           lacked

Usage:
    python episodic_memory_diagnostic.py --horizon 30
    python episodic_memory_diagnostic.py --horizon 30 --verify
"""

import json
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import config
import common
from build_windows import episode_onsets, HYPO_MGDL

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "episodic_memory"

PRECURSOR_LEN = 60          # readings before onset, matching the model window
MIN_EPISODES = 2            # a child needs at least this many priors
SIM_THRESHOLD = 0.7

ABS_DIR = config.RESULTS / "RQ2_models" / "absolute_state"


# ─── PRECURSOR EXTRACTION ─────────────────────────────────────────────────────

def episode_bounds(g):
    """
    Onsets and ends using the consensus definition already used to build
    the labels: onset at the first of >= 3 consecutive readings below 70,
    ending only after >= 3 consecutive readings at or above 70.
    """
    onsets = episode_onsets(g)
    ends = []
    for o in onsets:
        e = o
        run = 0
        while e < len(g) - 1:
            e += 1
            run = run + 1 if g[e] >= HYPO_MGDL else 0
            if run >= 3:
                break
        ends.append(e)
    return np.asarray(onsets, dtype=np.int64), np.asarray(ends, dtype=np.int64)


def shape_of(seg):
    """
    z-scored trajectory shape.

    Absolute level is deliberately removed: the model already receives it
    through the TA and absolute channels, so leaving it in would make the
    similarity re-measure information that is not missing.
    """
    s = seg.astype(np.float64)
    m, sd = s.mean(), s.std()
    return ((s - m) / max(sd, 1e-6)).astype(np.float32)


def build_memory(g_sub):
    """Precursor shapes and the index at which each episode ended."""
    on, en = episode_bounds(g_sub)
    shapes, ready = [], []
    for o, e in zip(on, en):
        if o < PRECURSOR_LEN:
            continue
        shapes.append(shape_of(g_sub[o - PRECURSOR_LEN:o]))
        ready.append(e)                       # usable only after this index
    if not shapes:
        return np.zeros((0, PRECURSOR_LEN), np.float32), np.zeros(0, np.int64)
    return np.stack(shapes), np.asarray(ready, dtype=np.int64)


def similarity(cur, mem):
    """Correlation between z-scored shapes, in [-1, 1]."""
    if len(mem) == 0:
        return np.zeros(0, np.float32)
    return (mem @ cur) / PRECURSOR_LEN


# ─── SCORING ──────────────────────────────────────────────────────────────────

def score_windows(bw, idx):
    """
    For each window, similarity to that subject's earlier precursors.

    Causal by construction: only episodes whose END index precedes the
    window's START are eligible.
    """
    ends = bw.starts[idx] + bw.window_len - 1
    subj = bw.reading_subject[ends]
    out = {k: np.full(len(idx), np.nan, np.float32)
           for k in ["max_sim", "top3_sim", "n_above", "n_memory"]}

    for si in np.unique(subj):
        rmask = np.flatnonzero(bw.reading_subject == si)
        if len(rmask) < PRECURSOR_LEN * 2:
            continue
        base = rmask[0]
        g_sub = bw.raw_gluc[rmask]
        mem, ready = build_memory(g_sub)
        if len(mem) == 0:
            continue

        sel = np.flatnonzero(subj == si)
        for k in sel:
            s_local = bw.starts[idx[k]] - base
            e_local = s_local + bw.window_len - 1
            usable = ready < s_local                 # ENDED before the window
            out["n_memory"][k] = int(usable.sum())
            if usable.sum() < MIN_EPISODES:
                continue
            cur = shape_of(g_sub[max(0, e_local - PRECURSOR_LEN + 1):e_local + 1])
            if len(cur) < PRECURSOR_LEN:
                continue
            sims = similarity(cur, mem[usable])
            out["max_sim"][k] = float(sims.max())
            out["top3_sim"][k] = float(np.sort(sims)[-3:].mean())
            out["n_above"][k] = float((sims > SIM_THRESHOLD).sum())
    return out


def verify_causality(bw, n_subj=3, seed=0):
    """Corrupt the future; earlier similarity scores must be unchanged."""
    rng = np.random.default_rng(seed)
    print("Causality check: corrupting future readings must not change "
          "earlier scores\n")
    for _ in range(n_subj):
        si = int(rng.integers(0, len(bw.subjects)))
        rmask = np.flatnonzero(bw.reading_subject == si)
        if len(rmask) < 1500:
            continue
        g = bw.raw_gluc[rmask].copy()
        cut = int(len(g) * 0.6)

        m1, r1 = build_memory(g)
        g2 = g.copy(); g2[cut:] = 400.0
        m2, r2 = build_memory(g2)

        keep1 = r1 < cut
        keep2 = r2 < cut
        same = (keep1.sum() == keep2.sum() and
                np.array_equal(m1[keep1], m2[keep2]))
        print(f"  subject {bw.subjects[si]}: {int(keep1.sum())} precursors "
              f"before the cut | identical after corruption: {same}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)

    if args.verify:
        verify_causality(bw)
        return

    zb = ABS_DIR / "gru_ta_abs__s42_probs.npz"
    if not zb.exists():
        print(f"  missing {zb}")
        return
    z = np.load(zb)
    idx = z["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p = z["test"][:, j]

    print(f"h={args.horizon} | {len(idx):,} test windows | base rate {y.mean():.4f}")
    print(f"scoring similarity to each child's earlier precursors...")
    S = score_windows(bw, idx)

    ok = ~np.isnan(S["max_sim"])
    print(f"\n  windows with >= {MIN_EPISODES} usable prior episodes: "
          f"{ok.sum():,} / {len(idx):,} ({100*ok.mean():.1f}%)")
    print(f"  memory size where available: mean "
          f"{np.nanmean(S['n_memory'][ok]):.1f} precursors")

    if ok.sum() < 300:
        print("  too few windows with usable memory")
        return

    yy, pp = y[ok], p[ok]
    res = {"horizon": args.horizon, "n_usable": int(ok.sum()),
           "base_rate": float(yy.mean())}

    # ---- does similarity differ before events? ----------------------------
    print(f"\n{'#'*92}")
    print("# 1. Is similarity higher before events?")
    print(f"{'#'*92}")
    print(f"\n  {'statistic':>12} {'negatives':>12} {'positives':>12} "
          f"{'delta':>10} {'AUROC':>8}")
    for k in ["max_sim", "top3_sim", "n_above"]:
        v = S[k][ok]
        a, b = v[yy == 0].mean(), v[yy == 1].mean()
        auc = roc_auc_score(yy, v)
        print(f"  {k:>12} {a:>12.4f} {b:>12.4f} {b-a:>+10.4f} {auc:>8.4f}")
        res[k] = {"neg": float(a), "pos": float(b), "auroc": float(auc)}

    # ---- the decisive test -------------------------------------------------
    print(f"\n{'#'*92}")
    print("# 2. Does it add anything BEYOND the model's own prediction?")
    print("#    (higher similarity before events could just track 'falling',")
    print("#     which the model already sees)")
    print(f"{'#'*92}")

    lp = np.log(np.clip(pp, 1e-6, 1 - 1e-6) /
                (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    feats = np.column_stack([S["max_sim"][ok], S["top3_sim"][ok],
                             S["n_above"][ok]])

    base = LogisticRegression(max_iter=1000).fit(lp.reshape(-1, 1), yy)
    both = LogisticRegression(max_iter=1000).fit(
        np.column_stack([lp, feats]), yy)
    pb = base.predict_proba(lp.reshape(-1, 1))[:, 1]
    pj = both.predict_proba(np.column_stack([lp, feats]))[:, 1]

    ap_b, ap_j = average_precision_score(yy, pb), average_precision_score(yy, pj)
    au_b, au_j = roc_auc_score(yy, pb), roc_auc_score(yy, pj)
    print(f"\n  {'model':>26} {'AUPRC':>9} {'AUROC':>9}")
    print(f"  {'model score alone':>26} {ap_b:>9.4f} {au_b:>9.4f}")
    print(f"  {'+ episodic similarity':>26} {ap_j:>9.4f} {au_j:>9.4f}")
    print(f"  {'lift':>26} {ap_j-ap_b:>+9.4f} {au_j-au_b:>+9.4f}")

    rng = np.random.default_rng(0)
    reps = []
    for _ in range(200):
        b = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[b])) < 2:
            continue
        reps.append(average_precision_score(yy[b], pj[b]) -
                    average_precision_score(yy[b], pb[b]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    print(f"  {'95% CI on the lift':>26} [{lo:+.4f}, {hi:+.4f}]"
          f"{'*' if (lo > 0 or hi < 0) else ''}")
    res["lift"] = {"auprc_base": float(ap_b), "auprc_joint": float(ap_j),
                   "delta": float(ap_j - ap_b), "lo": float(lo),
                   "hi": float(hi)}

    # ---- and on the events the model misses? ------------------------------
    thr = np.quantile(pp, 1 - yy.mean() * 3)
    miss = (pp < thr) & (yy == 1)
    det = (pp >= thr) & (yy == 1)
    if miss.sum() > 30:
        print(f"\n{'#'*92}")
        print(f"# 3. The {int(miss.sum()):,} events this model misses")
        print(f"{'#'*92}")
        print(f"\n  {'statistic':>12} {'detected':>12} {'missed':>12} {'delta':>10}")
        for k in ["max_sim", "top3_sim", "n_above"]:
            v = S[k][ok]
            print(f"  {k:>12} {v[det].mean():>12.4f} {v[miss].mean():>12.4f} "
                  f"{v[miss].mean()-v[det].mean():>+10.4f}")
        res["missed"] = {k: {"detected": float(S[k][ok][det].mean()),
                             "missed": float(S[k][ok][miss].mean())}
                         for k in ["max_sim", "top3_sim", "n_above"]}

    print(f"\n{'#'*92}\nREADING THIS\n{'#'*92}")
    d = ap_j - ap_b
    if lo > 0 and d >= 0.010:
        print(f"\n  Episodic similarity adds {d:+.4f} AUPRC on top of the")
        print("  model's own score, with a CI excluding zero. That is")
        print("  information the fixed-window architecture cannot reach,")
        print("  which is exactly what every previous architecture attempt")
        print("  lacked. Worth building.")
    elif lo > 0:
        print(f"\n  A small but real lift ({d:+.4f}). The information exists")
        print("  but is thin; an architecture built on it would likely")
        print("  produce a correspondingly small gain.")
    else:
        print(f"\n  No lift beyond the model's own prediction ({d:+.4f}).")
        print("  Similarity to a child's earlier precursors tells us nothing")
        print("  the current window does not already convey, so an episodic-")
        print("  memory architecture has nothing to exploit.")

    common.save_result(OUT_DIR, f"_episodic_h{args.horizon}", res)
    print(f"\n✓ Saved -> {OUT_DIR / f'_episodic_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
