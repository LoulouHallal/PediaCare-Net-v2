"""
error_regime_diagnostics.py
=============================
Where do the two strongest models actually fail, and do they fail on the
same windows?

No training. Everything here is computed from predictions already saved
to disk plus the raw glucose timeline.

WHY THIS RATHER THAN ANOTHER ARCHITECTURE
-----------------------------------------
Six architectures and two output-structure diagnostics have now come back
negative. Rather than proposing a seventh cell, this asks which
trajectories the current models get wrong, and whether GRU and TCN get
different ones wrong.

The distinction matters, because the two outcomes point in opposite
directions:

    models fail on DIFFERENT windows
        the architectures capture different temporal regimes, and a model
        that adapts its computation to the regime has something real to
        exploit

    models fail on the SAME windows
        the failures are a property of the data, not the architecture. If
        both miss an event when glucose is still 120 mg/dL and flat, the
        information needed to predict it is not in the input, and no cell
        design recovers it. That would explain the ~0.86 ceiling far
        better than any remaining architectural hypothesis.

REGIMES EXAMINED
----------------
All derived causally from the window itself, never from the future:

    current glucose        70-90 / 90-120 / 120-150 / >150 mg/dL
    distance to threshold  (G_T - 70), how far above the clinical line
    fall velocity          (G_{T-3} - G_T)/15 min, recent slope
    acceleration           change in slope over the last 30 min
    variability            SD of glucose across the window
    time to event          for positives only: minutes until the first
                           reading below 70
    subject prevalence     per-patient hypoglycaemia rate

METRIC WITHIN A REGIME
----------------------
AUPRC is reported per regime, but it is NOT comparable across regimes:
each subgroup has a different base rate, and AUPRC scales with it. The
ratio AUPRC/base-rate is therefore reported alongside, and that is the
column to read when comparing one regime against another.

Usage:
    python error_regime_diagnostics.py
    python error_regime_diagnostics.py --horizon 30
"""

import json
import argparse
import itertools
import numpy as np
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "error_regimes"

# the two strongest conventional models, on identical windows
CANDIDATES = [
    ("GRU+TA", config.RESULTS / "RQ2_models" / "tap_gru" / "gru_ta__s42_probs.npz"),
    ("TCN+TA", config.RESULTS / "RQ2_models" / "dms_tcn" / "tcn_ta__s42_probs.npz"),
]


# ─── REGIME CONSTRUCTION ──────────────────────────────────────────────────────

def build_regimes(bw, idx, j):
    """
    Causal trajectory descriptors for each test window, plus the time
    until the next true hypoglycaemic reading (used only to characterise
    positives, never as a model input).
    """
    ends = bw.starts[idx] + bw.window_len - 1
    g_now = bw.raw_gluc[ends]

    # slope over the last 15 min and the 15 min before that
    g_3 = bw.raw_gluc[ends - 3]
    g_6 = bw.raw_gluc[ends - 6]
    vel = (g_3 - g_now) / 15.0                      # mg/dL per min, + = falling
    vel_prev = (g_6 - g_3) / 15.0
    accel = vel - vel_prev                          # + = fall accelerating

    # variability across the window
    var = np.empty(len(idx), dtype=np.float32)
    for k, s in enumerate(bw.starts[idx]):
        var[k] = bw.raw_gluc[s:s + bw.window_len].std()

    # minutes until the first reading below 70 inside the horizon
    h_steps = HORIZONS[j] // bw.sample_min
    ttl = np.full(len(idx), np.nan, dtype=np.float32)
    for k, e in enumerate(ends):
        fut = bw.raw_gluc[e + 1:e + 1 + h_steps]
        low = np.flatnonzero(fut < 70.0)
        if low.size:
            ttl[k] = (low[0] + 1) * bw.sample_min

    # per-subject hypoglycaemia prevalence
    meta = bw.meta[idx]
    Y = bw._labels_any[idx, j]
    prev_map = {s: float(Y[meta == s].mean()) for s in np.unique(meta)}
    prev = np.array([prev_map[s] for s in meta], dtype=np.float32)

    return {"glucose": g_now, "dist_to_70": g_now - 70.0, "velocity": vel,
            "acceleration": accel, "variability": var, "time_to_low": ttl,
            "subject_prevalence": prev, "meta": meta}


def bins_for(name, v):
    """Fixed clinically meaningful edges where they exist, else quartiles."""
    if name == "glucose":
        edges = [-np.inf, 90, 120, 150, np.inf]
        labels = ["<90", "90-120", "120-150", ">150"]
    elif name == "velocity":
        edges = [-np.inf, -0.5, 0.0, 0.5, 1.0, np.inf]
        labels = ["rising", "flat-", "flat+", "falling", "fast fall"]
    elif name == "acceleration":
        edges = [-np.inf, -0.2, 0.2, np.inf]
        labels = ["decelerating", "steady", "accelerating"]
    else:
        q = np.nanpercentile(v, [25, 50, 75])
        edges = [-np.inf, *q, np.inf]
        labels = ["Q1", "Q2", "Q3", "Q4"]
    return np.digitize(v, edges[1:-1]), labels


def regime_metrics(y, p, mask):
    """AUPRC within a subgroup, with its base rate and the ratio."""
    if mask.sum() < 200 or len(np.unique(y[mask])) < 2:
        return None
    yy, pp = y[mask], p[mask]
    base = float(yy.mean())
    ap = float(average_precision_score(yy, pp))
    return {"n": int(mask.sum()), "base_rate": base, "auprc": ap,
            "auprc_over_base": ap / base if base > 0 else float("nan"),
            "auroc": float(roc_auc_score(yy, pp))}


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=15, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--n_boot", type=int, default=2000)
    args = ap.parse_args()

    j = HORIZONS.index(args.horizon)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)

    models, idx_ref = {}, None
    for name, path in CANDIDATES:
        if not path.exists():
            print(f"  missing: {path}")
            continue
        z = np.load(path)
        if idx_ref is None:
            idx_ref = z["idx_test"]
        elif not np.array_equal(idx_ref, z["idx_test"]):
            print(f"  !! {name} was evaluated on different windows; skipping")
            continue
        models[name] = z["test"][:, j]
    if len(models) < 1:
        print("No saved predictions found.")
        return

    y = bw._labels_any[idx_ref, j].astype(int)
    R = build_regimes(bw, idx_ref, j)
    print(f"h={args.horizon} | {len(y):,} test windows | base rate {y.mean():.4f}")
    print(f"models: {', '.join(models)}\n")

    results = {"horizon": args.horizon, "n": int(len(y)),
               "base_rate": float(y.mean()), "regimes": {}}

    # ---- per-regime performance ------------------------------------------
    for var in ["glucose", "velocity", "acceleration", "variability",
                "subject_prevalence"]:
        b, labels = bins_for(var, R[var])
        print(f"\n{'='*94}\n{var}\n{'='*94}")
        print(f"  {'bin':>14} {'n':>9} {'base':>7} " +
              "  ".join(f"{m:>18}" for m in models))
        print(f"  {'':>14} {'':>9} {'':>7} " +
              "  ".join(f"{'AUPRC  ratio':>18}" for _ in models))
        entry = {}
        for bi, lab in enumerate(labels):
            mask = b == bi
            row, cells = {}, []
            for m, p in models.items():
                r = regime_metrics(y, p, mask)
                row[m] = r
                cells.append("      too few     " if r is None
                             else f"{r['auprc']:>8.4f} {r['auprc_over_base']:>8.1f}x")
            if any(v is not None for v in row.values()):
                n = int(mask.sum())
                base = float(y[mask].mean()) if n else 0.0
                print(f"  {lab:>14} {n:>9,} {base:>7.4f} " + "  ".join(cells))
                entry[lab] = row
        results["regimes"][var] = entry

    # ---- positives only: how far ahead is the event? ----------------------
    pos = y == 1
    if pos.any():
        print(f"\n{'='*94}\ntime until the event (positives only)\n{'='*94}")
        ttl = R["time_to_low"]
        print(f"  {'minutes':>14} {'n':>9} " +
              "  ".join(f"{m:>16}" for m in models))
        print(f"  {'':>14} {'':>9} " +
              "  ".join(f"{'mean p(hypo)':>16}" for _ in models))
        entry = {}
        for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 30), (30, 60), (60, 120)]:
            mask = pos & (ttl > lo) & (ttl <= hi)
            if mask.sum() < 50:
                continue
            cells, row = [], {}
            for m, p in models.items():
                mp = float(p[mask].mean())
                row[m] = {"n": int(mask.sum()), "mean_prob": mp}
                cells.append(f"{mp:>16.4f}")
            print(f"  {f'{lo}-{hi}':>14} {int(mask.sum()):>9,} " + "  ".join(cells))
            entry[f"{lo}-{hi}"] = row
        results["time_to_event"] = entry
        print("\n  A model that anticipates rather than reacts should keep a")
        print("  high probability even when the event is still 10-15 min away.")

    # ---- agreement between the two architectures --------------------------
    if len(models) == 2:
        (n1, p1), (n2, p2) = list(models.items())
        print(f"\n{'='*94}\nagreement between {n1} and {n2}\n{'='*94}")

        # threshold each at its own recall-matched operating point
        t1 = np.quantile(p1, 1 - y.mean() * 3)
        t2 = np.quantile(p2, 1 - y.mean() * 3)
        d1, d2 = p1 >= t1, p2 >= t2

        both_hit = (d1 & d2 & (y == 1)).sum()
        only1 = (d1 & ~d2 & (y == 1)).sum()
        only2 = (~d1 & d2 & (y == 1)).sum()
        both_miss = (~d1 & ~d2 & (y == 1)).sum()
        n_pos = int((y == 1).sum())

        print(f"\n  of {n_pos:,} true events, at matched alarm budgets:")
        print(f"    both detected      {both_hit:>8,}  ({100*both_hit/n_pos:>5.1f}%)")
        print(f"    only {n1:<12} {only1:>8,}  ({100*only1/n_pos:>5.1f}%)")
        print(f"    only {n2:<12} {only2:>8,}  ({100*only2/n_pos:>5.1f}%)")
        print(f"    BOTH MISSED        {both_miss:>8,}  ({100*both_miss/n_pos:>5.1f}%)")

        corr = float(np.corrcoef(p1, p2)[0, 1])
        print(f"\n  correlation of predicted probabilities: {corr:.4f}")

        results["agreement"] = {
            "n_positives": n_pos, "both_detected": int(both_hit),
            f"only_{n1}": int(only1), f"only_{n2}": int(only2),
            "both_missed": int(both_miss), "prob_correlation": corr}

        # what do the jointly-missed events look like?
        miss = (~d1) & (~d2) & (y == 1)
        if miss.sum() > 50:
            print(f"\n  the {int(miss.sum()):,} events BOTH models miss:")
            for var, unit in [("glucose", "mg/dL"), ("velocity", "mg/dL/min"),
                              ("acceleration", "mg/dL/min^2"),
                              ("variability", "mg/dL")]:
                a = float(np.nanmean(R[var][miss]))
                b = float(np.nanmean(R[var][y == 1]))
                print(f"    {var:>14}: {a:>8.2f} {unit:<12} "
                      f"vs {b:>8.2f} for all positives")
            results["joint_misses"] = {
                v: {"missed_mean": float(np.nanmean(R[v][miss])),
                    "all_positive_mean": float(np.nanmean(R[v][y == 1]))}
                for v in ["glucose", "velocity", "acceleration", "variability"]}

        print(f"\n{'='*94}\nREADING THIS\n{'='*94}")
        excl = only1 + only2
        if excl > 0.15 * n_pos:
            print(f"\n  {100*excl/n_pos:.1f}% of events are found by one model")
            print("  and missed by the other. The two architectures capture")
            print("  DIFFERENT temporal regimes, so a model that adapts its")
            print("  computation to the regime has something real to exploit.")
        else:
            print(f"\n  only {100*excl/n_pos:.1f}% of events are found by one")
            print("  model and missed by the other: the two architectures")
            print("  succeed and fail on largely the SAME windows.")
            print("\n  Combined with the joint-miss profile above, this points")
            print("  at the input rather than the architecture. If the missed")
            print("  events look unremarkable at prediction time -- glucose")
            print("  still high, no downward trend -- then the information")
            print("  required to predict them is not present in the window,")
            print("  and no cell design recovers it.")

    common.save_result(OUT_DIR, f"_error_regimes_h{args.horizon}", results)
    print(f"\n✓ Saved -> {OUT_DIR / f'_error_regimes_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
