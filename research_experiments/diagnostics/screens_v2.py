"""
screens_v2.py
===============
Corrected re-run of the innovation and interaction screens.

TWO ERRORS IN THE FIRST VERSIONS
--------------------------------
1. WRONG METRIC. They computed AUPRC pooled over all ~410k test windows.
   The project's primary metric is the PER-SUBJECT MEAN. A subject with
   40,000 windows dominates the pooled curve; one with 5,000 barely
   registers, and subjects differ in prevalence. That is the whole reason
   the screens reported h=15 0.8944 / h=30 0.7923 while the primary table
   reported 0.8649 / 0.7528. Reproduced synthetically: +0.0369 from
   aggregation alone.

2. WRONG BOOTSTRAP UNIT. They resampled individual windows. Windows from
   the same subject are dependent, and neighbouring windows share most of
   their input sequence, so this is pseudoreplication. Measured on
   synthetic data with a true zero effect: window-level CI width 0.00008
   against subject-level 0.01170 -- 143x too narrow. The "+0.0002, CI
   [+0.0001, +0.0003], significant" result from the innovation screen is
   an artefact of that.

Everything here aggregates per subject and resamples subjects.

WHAT IS AND IS NOT BEING TESTED
-------------------------------
Both screens ask whether some extra quantity improves ranking beyond the
trained GRU's own score, evaluated the way the project evaluates.

The interaction screen tests products of the CURRENT input channels,
x_t * x_t. It does NOT test the x_t * h_{t-1} state interaction that a
multiplicative recurrent cell would compute, because h_{t-1} is a learned
function of the whole preceding window: two windows with identical x_t but
different histories are indistinguishable to every x*x feature and
different under x*h. A null here therefore does not rule out
state-interaction cells. It only says current-channel products carry
nothing extra.

NO FIXED "FLOOR"
----------------
The earlier +0.005 figure came from one stacking procedure on one dataset
and is not a general property of AUPRC. Reported here instead: the
subject-level paired difference with its confidence interval, plus how
many of the 38 subjects individually improve. A real effect should be
positive for most subjects, not driven by a few.

Usage:
    python screens_v2.py --horizon 30
    python screens_v2.py --horizon 30 --which interaction
"""

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "screens_v2"


def per_subject_auprc(y, p, subj, min_pos=1):
    """The project's primary metric: mean AUPRC over subjects."""
    out = {}
    for s in np.unique(subj):
        m = subj == s
        if y[m].sum() < min_pos or y[m].sum() == m.sum():
            continue
        out[s] = average_precision_score(y[m], p[m])
    return out


def paired_subject_delta(y, p_base, p_new, subj, n_boot=2000, seed=0):
    """
    Per-subject paired difference, resampled at the SUBJECT level.
    Subjects with no positives are excluded once, by a rule fixed here,
    not per-model.
    """
    a = per_subject_auprc(y, p_base, subj)
    b = per_subject_auprc(y, p_new, subj)
    keys = sorted(set(a) & set(b))
    d = np.array([b[k] - a[k] for k in keys])
    rng = np.random.default_rng(seed)
    reps = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return {"base": float(np.mean([a[k] for k in keys])),
            "new": float(np.mean([b[k] for k in keys])),
            "delta": float(d.mean()), "lo": float(lo), "hi": float(hi),
            "n_subjects": len(keys),
            "n_improved": int((d > 0).sum()),
            "median": float(np.median(d))}


def stack(lp, extra, y, subj, seed=0):
    """
    Fit the stacker with SUBJECT-DISJOINT folds, so the reported gain is
    out-of-fold rather than in-sample. Fitting and scoring the same
    windows is what let redundant predictors look like +0.005.
    """
    rng = np.random.default_rng(seed)
    subs = np.unique(subj)
    fold = {s: i % 4 for i, s in enumerate(rng.permutation(subs))}
    fa = np.array([fold[s] for s in subj])
    pb = np.zeros(len(y)); pn = np.zeros(len(y))
    Xj = np.column_stack([lp, extra])
    mu, sd = Xj.mean(0), Xj.std(0) + 1e-9
    Xz = (Xj - mu) / sd
    for f in range(4):
        tr, te = fa != f, fa == f
        if len(np.unique(y[tr])) < 2:
            continue
        pb[te] = LogisticRegression(max_iter=2000).fit(
            lp[tr].reshape(-1, 1), y[tr]).predict_proba(
            lp[te].reshape(-1, 1))[:, 1]
        pn[te] = LogisticRegression(max_iter=2000).fit(
            Xz[tr], y[tr]).predict_proba(Xz[te])[:, 1]
    return pb, pn


def causal_innovation(g, sub_ids, order=3):
    """One-step linear extrapolation residual; fixed-weight filter."""
    from numpy.lib.stride_tricks import sliding_window_view
    k = int(order)
    x = np.arange(k, dtype=np.float64)
    xb = (k - 1) / 2.0
    w = 1.0 / k + (x - xb) * (k - xb) / ((x - xb) @ (x - xb))
    g = g.astype(np.float64)
    pred = np.full(len(g), np.nan)
    for si in np.unique(sub_ids):
        m = np.flatnonzero(sub_ids == si)
        if len(m) <= k:
            continue
        gg = g[m]
        out = np.full(len(gg), np.nan)
        out[k:] = sliding_window_view(gg, k)[:-1] @ w
        pred[m] = out
    return g - pred


def report(name, r):
    sig = "*" if (r["lo"] > 0 or r["hi"] < 0) else " "
    print(f"\n  {name}")
    print(f"    GRU alone (per-subject mean AUPRC) {r['base']:.4f}")
    print(f"    + features                         {r['new']:.4f}")
    print(f"    paired subject delta               {r['delta']:+.4f} "
          f"[{r['lo']:+.4f}, {r['hi']:+.4f}]{sig}")
    print(f"    median across subjects             {r['median']:+.4f}")
    print(f"    subjects improved                  "
          f"{r['n_improved']}/{r['n_subjects']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--probs", default=None)
    ap.add_argument("--which", default="both",
                    choices=["both", "innovation", "interaction"])
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    from pathlib import Path
    zb = Path(args.probs) if args.probs else None
    if zb is None:
        for c in [config.RESULTS / "RQ2_models" / "adew_gru" / "gru__s42_probs.npz",
                  config.RESULTS / "RQ2_models" / "drs_gru" / "gru_abs__s42_probs.npz"]:
            if c.exists():
                zb = c; break
    if zb is None or not zb.exists():
        print("no saved GRU probabilities found")
        return

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    from ta_gru import attach_ta
    from absolute_state import attach_absolute
    attach_ta(bw, verbose=False)
    attach_absolute(bw, verbose=False)

    z = np.load(zb)
    idx = z["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p = z["test"][:, j]
    ends = bw.starts[idx] + bw.window_len - 1
    subj = bw.reading_subject[ends]

    print(f"{'#'*88}")
    print(f"# CORRECTED SCREENS — per-subject AUPRC, subject-level bootstrap")
    print(f"#   h={args.horizon}, {zb.parent.name}/{zb.name}")
    print(f"{'#'*88}")

    base_ps = per_subject_auprc(y, p, subj)
    print(f"\n  {len(np.unique(subj))} test subjects, "
          f"{len(base_ps)} with >=1 positive at this horizon")
    print(f"  GRU per-subject mean AUPRC: {np.mean(list(base_ps.values())):.4f}")
    print(f"  (pooled over all windows would give "
          f"{average_precision_score(y, p):.4f} — a different quantity)")

    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    results = {}

    if args.which in ("both", "innovation"):
        print(f"\n\n{'='*88}\n1. INNOVATION — does prediction error add anything?"
              f"\n{'='*88}")
        eps = causal_innovation(bw.raw_gluc, bw.reading_subject, 3)
        seg = np.stack([eps[np.maximum(ends - k, 0)] for k in range(12)], 1)
        X = np.column_stack([eps[ends], np.abs(eps[ends]),
                             np.nanmean(seg, 1), np.nanmax(np.abs(seg), 1),
                             np.nanstd(seg, 1)])
        ok = np.isfinite(X).all(1)
        pb, pn = stack(lp[ok], X[ok], y[ok], subj[ok])
        r = paired_subject_delta(y[ok], pb, pn, subj[ok], args.n_boot)
        report("innovation features", r)
        results["innovation"] = r

    if args.which in ("both", "interaction"):
        print(f"\n\n{'='*88}\n2. INTERACTIONS — do current-channel products "
              f"add anything?\n{'='*88}")
        X1 = bw.timeline[ends].astype(np.float64)
        pr = [X1[:, a] * X1[:, b]
              for a in range(X1.shape[1]) for b in range(a, X1.shape[1])]
        X2 = np.column_stack(pr)
        ok = np.isfinite(X1).all(1) & np.isfinite(X2).all(1)
        pb, pn = stack(lp[ok], np.column_stack([X1[ok], X2[ok]]),
                       y[ok], subj[ok])
        r = paired_subject_delta(y[ok], pb, pn, subj[ok], args.n_boot)
        report(f"{X1.shape[1]} channels + {X2.shape[1]} products", r)
        results["interaction"] = r

        # a redundancy control: the SAME stacker given only a monotone
        # transform of the GRU score, which cannot add information
        pb2, pn2 = stack(lp[ok], (lp[ok] ** 3).reshape(-1, 1), y[ok], subj[ok])
        rc = paired_subject_delta(y[ok], pb2, pn2, subj[ok], args.n_boot)
        report("CONTROL: a monotone transform of the GRU score "
               "(must be ~0)", rc)
        results["redundancy_control"] = rc

    print(f"\n\n{'#'*88}\nREADING THIS\n{'#'*88}\n")
    print("  These are out-of-fold, subject-aggregated, subject-bootstrapped")
    print("  numbers, so they are directly comparable to the project's")
    print("  primary metric — unlike the first versions of these screens.\n")
    for k, r in results.items():
        if k == "redundancy_control":
            continue
        d, frac = r["delta"], r["n_improved"] / max(r["n_subjects"], 1)
        if r["lo"] > 0 and d >= 0.005 and frac >= 0.6:
            print(f"  {k}: {d:+.4f}, CI above zero, {r['n_improved']}/"
                  f"{r['n_subjects']} subjects improved. Real unused signal.")
        elif r["lo"] > 0:
            print(f"  {k}: {d:+.4f}, CI above zero but small "
                  f"({r['n_improved']}/{r['n_subjects']} subjects). Weak.")
        else:
            print(f"  {k}: {d:+.4f}, CI crosses zero. No evidence of unused "
                  f"signal.")
    print("\n  The interaction screen covers x*x only. A state-interaction")
    print("  cell computes x*h, which distinguishes windows with identical")
    print("  current inputs but different histories. This screen cannot")
    print("  speak to that.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, f"_screens_v2_h{args.horizon}",
                       {"horizon": args.horizon, "probs": str(zb),
                        **{k: v for k, v in results.items()}})
    print(f"\n✓ Saved -> {OUT_DIR / f'_screens_v2_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
