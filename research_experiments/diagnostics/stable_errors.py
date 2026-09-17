"""
stable_errors.py  --  5-seed error structure, persistence, and the
                      legacy-vs-consensus endpoint question
====================================================================

No training. Joins the five saved baseline seeds in cmr_gru/ to the raw
glucose trace.

WHY FIVE SEEDS
--------------
A single-seed false positive may just be that seed. An error made by all
five is structural. The 5-seed baseline range is 0.00207, so seed-level
disagreement is real and worth separating out before drawing conclusions
about what "the GRU" gets wrong.

WHY PERSISTENCE
---------------
build_windows defines two endpoints:

    labels_any  (legacy)   1 if ANY future reading < 70
    labels      (consensus, called PRIMARY in the builder)
                           episode onset: >= 3 consecutive readings < 70,
                           ending only after >= 3 consecutive >= 70

Training has used labels_any throughout. Under that endpoint a single
noisy reading at 69 creates a positive, which is exactly where Phase 2
found 90.1% of false positives sitting (nadir 70-80, median 72). This
script scores the SAME predictions against BOTH endpoints to see how much
of the measured error is an artifact of the legacy definition.

That comparison is diagnostic, not a result: the two endpoints have
different prevalence (4.18% vs 1.26% at h=15) and their AUPRC values are
not comparable to each other or to anything reported so far.

    python stable_errors.py --tag stride6
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import config  # noqa: E402

HYPO = 70.0
RECENT_N = 6
CHUNK = 100_000
SEEDS = [42, 43, 44, 45, 46]


def window_end(bw):
    for c in (bw.starts + bw.window_len - 1, bw.starts + bw.window_len):
        if np.array_equal(bw.raw_gluc[c], bw.raw_gluc_at_pred):
            return c
    raise AssertionError("cannot reproduce raw_gluc_at_pred")


def future_stats(bw, end_idx, n_steps):
    """nadir, count below, fraction below, longest run below, time to a run of 3."""
    n = end_idx.size
    nadir = np.full(n, np.nan, np.float32)
    n_low = np.zeros(n, np.int16)
    max_run = np.zeros(n, np.int16)
    t_pers = np.full(n, np.nan, np.float32)
    valid = np.zeros(n, bool)
    limit = bw.raw_gluc.shape[0] - 1
    off = np.arange(1, n_steps + 1, dtype=np.int64)

    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        e = end_idx[a:b]
        fi = e[:, None] + off[None, :]
        oob = fi > limit
        fic = np.clip(fi, 0, limit)
        same = ((bw.reading_subject[fic] == bw.reading_subject[e][:, None]) &
                (bw.seg[fic] == bw.seg[e][:, None]) & ~oob)
        v = bw.raw_gluc[fic].astype(np.float32)
        v[~same] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nadir[a:b] = np.nanmin(v, axis=1)
        valid[a:b] = np.isfinite(nadir[a:b])

        low = np.where(np.isnan(v), False, v < HYPO)
        n_low[a:b] = low.sum(axis=1)

        run = np.zeros(b - a, np.int16)
        best = np.zeros(b - a, np.int16)
        first3 = np.full(b - a, -1, np.int32)
        for j in range(n_steps):
            run = np.where(low[:, j], run + 1, 0)
            best = np.maximum(best, run)
            hit = (run == 3) & (first3 < 0)
            first3[hit] = j
        max_run[a:b] = best
        tp = np.full(b - a, np.nan, np.float32)
        ok3 = first3 >= 0
        # onset is 2 samples before the third consecutive low
        tp[ok3] = (first3[ok3] - 1) * bw.sample_min
        t_pers[a:b] = tp

    frac = np.divide(n_low, n_steps, dtype=np.float32)
    return nadir, n_low, frac, max_run, t_pers, valid


def slope_accel(tail):
    t = np.arange(tail.shape[1], dtype=float)
    tc = t - t.mean()
    sl = ((tail - tail.mean(1, keepdims=True)) * tc).sum(1) / (tc ** 2).sum()
    h = tail.shape[1] // 2
    def _s(x):
        tt = np.arange(x.shape[1], dtype=float); tt -= tt.mean()
        return ((x - x.mean(1, keepdims=True)) * tt).sum(1) / (tt ** 2).sum()
    return sl, _s(tail[:, h:]) - _s(tail[:, :h])


def _bin(v, edges, labels):
    return pd.cut(v, bins=edges, labels=labels, right=False,
                  include_lowest=True)


def rate_table(df, by, title):
    """FP rate among negatives, FN rate among positives, per bin."""
    rows = []
    for b, g in df.groupby(by, observed=True):
        pos, neg = g[g.y_true == 1], g[g.y_true == 0]
        rows.append({
            by: b, "n": len(g),
            "n_pos": len(pos),
            "FP_rate": (neg.outcome == "FP").mean() if len(neg) else np.nan,
            "FN_rate": (pos.outcome == "FN").mean() if len(pos) else np.nan,
            "mean_prob": g.p_mean.mean(),
            "seed_sd": g.p_sd.mean(),
        })
    t = pd.DataFrame(rows)
    return f"\n{title}\n" + t.to_string(index=False,
                                        float_format=lambda x: f"{x:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--dir", default=None)
    ap.add_argument("--horizons", default="all")
    ap.add_argument("--outdir", default="results/error_analysis")
    a = ap.parse_args()

    root = a.dir or str(config.RESULTS / "RQ2_models" / "cmr_gru")
    bw = common.load_built(a.tag, label_set="any")
    end_all = window_end(bw)

    # ---- load the five seeds ------------------------------------------
    P, idx = [], None
    for s in SEEDS:
        f = os.path.join(root, f"gru__s{s}_probs.npz")
        if not os.path.exists(f):
            sys.exit(f"missing {f}")
        z = np.load(f)
        if idx is None:
            idx = z["idx_val"]
        elif not np.array_equal(idx, z["idx_val"]):
            sys.exit(f"seed {s} has a different idx_val -- not comparable")
        P.append(z["val"])
    P = np.stack(P)                        # (5, n, 4)
    print(f"{P.shape[0]} seeds | {P.shape[1]:,} windows | {P.shape[2]} horizons")

    end = end_all[idx]
    subj = np.asarray(bw.meta)[idx].astype(str)
    Y_any = bw._labels_any[idx]
    Y_con = bw._labels[idx]

    off = np.arange(-(RECENT_N - 1), 1, dtype=np.int64)
    tail = np.empty((idx.size, RECENT_N), np.float32)
    for i in range(0, idx.size, CHUNK):
        j = min(i + CHUNK, idx.size)
        tail[i:j] = bw.raw_gluc[end[i:j][:, None] + off[None, :]]
    sl, ac = slope_accel(tail)
    g_now = tail[:, -1]

    hz = bw.horizons if a.horizons == "all" else [int(x) for x in a.horizons.split(",")]
    os.makedirs(a.outdir, exist_ok=True)
    report = []

    for k, h in enumerate(bw.horizons):
        if h not in hz:
            continue
        steps = h // bw.sample_min
        nadir, n_low, frac, max_run, t_pers, valid = future_stats(bw, end, steps)

        y = Y_any[:, k].astype(int)
        yc = Y_con[:, k].astype(int)
        p = P[:, :, k]                                  # (5, n)
        p_mean, p_sd = p.mean(0), p.std(0)

        # per-seed thresholds, then per-seed outcomes
        thr = np.array([common.find_threshold(y, p[s])[0] for s in range(len(SEEDS))])
        pred = (p >= thr[:, None]).astype(np.int8)
        n_fp = ((pred == 1) & (y[None, :] == 0)).sum(0)
        n_fn = ((pred == 0) & (y[None, :] == 1)).sum(0)

        thr_m = common.find_threshold(y, p_mean)[0]
        pm = (p_mean >= thr_m).astype(int)
        outcome = np.where((pm == 1) & (y == 1), "TP",
                  np.where((pm == 1) & (y == 0), "FP",
                  np.where((pm == 0) & (y == 1), "FN", "TN")))

        df = pd.DataFrame({
            "subject_id": subj, "y_true": y, "y_consensus": yc,
            "p_mean": p_mean, "p_sd": p_sd, "outcome": outcome,
            "n_fp_seeds": n_fp, "n_fn_seeds": n_fn,
            "glucose_now": g_now, "slope": sl, "accel": ac,
            "future_nadir": nadir, "n_below70": n_low,
            "frac_below70": frac, "max_run_below70": max_run,
            "t_persistent": t_pers, "valid": valid,
        })
        d = df[df.valid]

        L = ["", "=" * 78, f"HORIZON {h} min   n={len(d):,}  "
             f"prevalence any {y.mean():.3%}  consensus {yc.mean():.3%}",
             "=" * 78]

        # ---- seed stability ------------------------------------------
        L.append("\n-- how much of the error is structural? --")
        neg, pos = d[d.y_true == 0], d[d.y_true == 1]
        for nm, sub, col in (("FP", neg, "n_fp_seeds"), ("FN", pos, "n_fn_seeds")):
            vc = sub[col].value_counts().sort_index()
            tot = int((sub[col] > 0).sum())
            st = int((sub[col] >= 4).sum())
            L.append(f"  {nm}: {tot:,} windows wrong in >=1 seed, "
                     f"{st:,} in >=4/5 ({st/max(tot,1):.1%} structural)")
            L.append("     by seed count " +
                     " ".join(f"{i}:{vc.get(i,0):,}" for i in range(6)))

        sfp = d[(d.y_true == 0) & (d.n_fp_seeds >= 4)]
        sfn = d[(d.y_true == 1) & (d.n_fn_seeds >= 4)]

        # ---- the endpoint question -----------------------------------
        L.append("\n-- legacy vs consensus endpoint --")
        L.append(f"  stable FP (n={len(sfp):,}) that are consensus-POSITIVE: "
                 f"{sfp.y_consensus.mean() if len(sfp) else float('nan'):.1%}"
                 "   <- if high, these 'errors' are real episodes the legacy "
                 "label happened to agree with")
        L.append(f"  stable FN (n={len(sfn):,}) that are consensus-POSITIVE: "
                 f"{sfn.y_consensus.mean() if len(sfn) else float('nan'):.1%}"
                 "   <- if high, these are clinically real misses")
        if len(sfp):
            L.append(f"  stable FP that are consensus-NEGATIVE: "
                     f"{1 - sfp.y_consensus.mean():.1%}   <- transient dips; "
                     "not events under the primary endpoint")
        L.append(f"  stable FP max-run distribution: " +
                 ", ".join(f"{v}:{(sfp.max_run_below70==v).mean():.1%}"
                           for v in range(4)) +
                 f", >=4:{(sfp.max_run_below70>=4).mean():.1%}"
                 if len(sfp) else "  (none)")
        from sklearn.metrics import average_precision_score
        ap_any = average_precision_score(y, p_mean)
        ap_con = average_precision_score(yc, p_mean) if yc.sum() else np.nan
        L.append(f"  pooled AUPRC of the SAME predictions: "
                 f"any {ap_any:.4f} (prev {y.mean():.3%})  "
                 f"consensus {ap_con:.4f} (prev {yc.mean():.3%})")
        L.append("  NOT comparable to each other -- different prevalence.")

        # ---- requested tables ----------------------------------------
        d = d.copy()
        d["glu_bin"] = _bin(d.glucose_now, [0, 80, 100, 120, 150, 1e9],
                            ["<80", "80-99", "100-119", "120-149", ">=150"])
        d["slope_bin"] = _bin(d.slope, [-1e9, -3, -1, 0.5, 1e9],
                              ["rapid fall", "moderate fall", "stable", "rising"])
        d["nadir_bin"] = _bin(d.future_nadir, [0, 54, 70, 80, 100, 1e9],
                              ["<54", "54-69", "70-79", "80-99", ">=100"])
        d["run_bin"] = _bin(d.max_run_below70, [0, 1, 2, 3, 4, 7, 1e9],
                            ["0", "1", "2", "3", "4-6", ">6"])

        L.append(rate_table(d, "glu_bin", "A. by current glucose"))
        L.append(rate_table(d, "slope_bin", "B. by recent slope"))
        L.append(rate_table(d, "nadir_bin", "C. by future nadir"))
        L.append(rate_table(d, "run_bin",
                            "D. by max consecutive readings <70 "
                            "(consensus needs >=3)"))

        # the 2-vs-3 question, asked directly
        two = d[d.max_run_below70 == 2]
        three = d[d.max_run_below70 == 3]
        if len(two) and len(three):
            L.append(f"\n  2 vs 3 consecutive: mean prob {two.p_mean.mean():.4f} "
                     f"vs {three.p_mean.mean():.4f}  "
                     f"(separation {three.p_mean.mean()-two.p_mean.mean():+.4f})")
            L.append("  Under the LEGACY label both are positive, so the model "
                     "has no reason to separate them. Under CONSENSUS only 3+ "
                     "is positive. A small separation here means persistence "
                     "supervision has something to add; a large one means the "
                     "model already encodes it.")

        report.append("\n".join(L))
        d.drop(columns=["valid"]).to_csv(
            f"{a.outdir}/stable_h{h}.csv.gz", index=False, compression="gzip")

    txt = "\n".join(report)
    print(txt)
    with open(f"{a.outdir}/stable_errors.txt", "w") as f:
        f.write(txt)
    print(f"\nwrote -> {a.outdir}/stable_errors.txt")


if __name__ == "__main__":
    main()
