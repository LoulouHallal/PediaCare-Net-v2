"""
gain_and_treatment_diagnostics.py
===================================
Two zero-training diagnostics that decide what, if anything, to build next.

DIAGNOSTIC 1 -- where did the absolute-state gain come from?
------------------------------------------------------------
Adding absolute glucose level and signed absolute rate improved AUPRC by
+0.0065 / +0.0055 / +0.0039 / +0.0036 across the four horizons, with every
paired CI excluding zero. Small, but credible in a way the architectural
ties were not.

The aggregate number does not say WHERE it helped. If the gain is
concentrated in the easy regime (glucose already below 90) it is
cosmetic. If it is concentrated where glucose is still high and the fall
has not started -- exactly where both GRU and TCN fail -- then absolute
state is doing something mechanistically interesting and is worth
building on.

DIAGNOSTIC 2 -- do the missed events carry recent treatment?
-------------------------------------------------------------
The jointly-missed events sit at ~120 mg/dL with velocity ~0.14
mg/dL/min, yet cross below 70 within 30 minutes. That requires an average
fall of about 1.66 mg/dL/min, roughly ten times the observed rate. So the
trajectory does not merely lack signal -- it points the wrong way.

Exogenous insulin is the obvious candidate: a bolus peaks 60-90 minutes
after delivery, so a dose given 45 minutes earlier would explain a fall
that has not yet begun. The bolus, basal and carbohydrate channels are
already inputs, but a generic hidden state may not preserve a delayed
effect across 60+ timesteps.

This compares causal treatment exposure over the preceding 30 / 60 / 120
minutes between events both models detected and events both models
missed.

    missed events carry MORE recent insulin
        -> a concrete, attackable failure: the model has the information
           but does not use it across the required delay. That justifies
           an architecture with explicit treatment-effect memory.

    the two groups look the same
        -> the trigger is not recorded in this dataset at all (an
           unlogged meal, exercise, a correction the family did not
           enter), and the information ceiling argument becomes much
           stronger.

A CAVEAT ON SCALE
-----------------
The treatment channels are causally per-patient normalised, so absolute
dose in units is not readable. Comparisons here are therefore RELATIVE
between the two groups of events, which is exactly what the question
needs: not "how many units" but "more or less than usual for this child".

Usage:
    python gain_and_treatment_diagnostics.py --gain --horizon 30
    python gain_and_treatment_diagnostics.py --treatment --horizon 30
    python gain_and_treatment_diagnostics.py --gain --treatment --horizon 15
"""

import json
import argparse
import numpy as np

from sklearn.metrics import average_precision_score

import config
import common
from ta_gru import attach_ta

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "gain_treatment"

ABS_DIR = config.RESULTS / "RQ2_models" / "absolute_state"
TAP_DIR = config.RESULTS / "RQ2_models" / "tap_gru"
DMS_DIR = config.RESULTS / "RQ2_models" / "dms_tcn"

CH_BASAL, CH_BOLUS, CH_CARBS = 1, 2, 3


# ─── SHARED ───────────────────────────────────────────────────────────────────

def regime_masks(bw, idx, j):
    """Causal descriptors of each window at prediction time."""
    ends = bw.starts[idx] + bw.window_len - 1
    g = bw.raw_gluc[ends]
    vel = (bw.raw_gluc[ends - 3] - g) / 15.0

    h_steps = HORIZONS[j] // bw.sample_min
    ttl = np.full(len(idx), np.nan, dtype=np.float32)
    for k, e in enumerate(ends):
        fut = bw.raw_gluc[e + 1:e + 1 + h_steps]
        low = np.flatnonzero(fut < 70.0)
        if low.size:
            ttl[k] = (low[0] + 1) * bw.sample_min

    return {"glucose": g, "velocity": vel, "time_to_low": ttl, "ends": ends}


def glucose_bins(g):
    edges = [-np.inf, 90, 120, 150, np.inf]
    return np.digitize(g, edges[1:-1]), ["<90", "90-120", "120-150", ">150"]


def velocity_bins(v):
    edges = [-np.inf, -0.5, 0.0, 0.5, 1.0, np.inf]
    return np.digitize(v, edges[1:-1]), ["rising", "flat-", "flat+",
                                         "falling", "fast fall"]


# ─── DIAGNOSTIC 1 ─────────────────────────────────────────────────────────────

def run_gain(args, bw):
    a_path = ABS_DIR / "gru_ta__s42_probs.npz"
    b_path = ABS_DIR / "gru_ta_abs__s42_probs.npz"
    for p in (a_path, b_path):
        if not p.exists():
            print(f"  missing {p}")
            return
    za, zb = np.load(a_path), np.load(b_path)
    if not np.array_equal(za["idx_test"], zb["idx_test"]):
        print("  !! the two arms were evaluated on different windows")
        return

    idx = za["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    pa, pb = za["test"][:, j], zb["test"][:, j]
    R = regime_masks(bw, idx, j)

    print(f"\n{'#'*94}")
    print(f"# DIAGNOSTIC 1 — where did the absolute-state gain appear? "
          f"(h={args.horizon})")
    print(f"#   7-channel GRU+TA  vs  9-channel GRU+TA+absolute")
    print(f"{'#'*94}")

    out = {}
    for var, (b, labels) in [("glucose", glucose_bins(R["glucose"])),
                             ("velocity", velocity_bins(R["velocity"]))]:
        print(f"\n{var}")
        print(f"  {'bin':>12} {'n':>9} {'base':>8} {'7ch':>9} {'9ch':>9} "
              f"{'delta':>9} {'ratio 7ch':>10} {'ratio 9ch':>10}")
        entry = {}
        for bi, lab in enumerate(labels):
            m = b == bi
            if m.sum() < 300 or len(np.unique(y[m])) < 2:
                continue
            base = float(y[m].mean())
            aa = float(average_precision_score(y[m], pa[m]))
            bb = float(average_precision_score(y[m], pb[m]))
            print(f"  {lab:>12} {int(m.sum()):>9,} {base:>8.4f} "
                  f"{aa:>9.4f} {bb:>9.4f} {bb-aa:>+9.4f} "
                  f"{aa/base:>9.1f}x {bb/base:>9.1f}x")
            entry[lab] = {"n": int(m.sum()), "base": base, "auprc_7ch": aa,
                          "auprc_9ch": bb, "delta": bb - aa}
        out[var] = entry

    # positives only, by how far ahead the event is
    pos = y == 1
    if pos.any():
        print(f"\ntime until the event (positives only, mean predicted prob)")
        print(f"  {'minutes':>12} {'n':>9} {'7ch':>9} {'9ch':>9} {'delta':>9}")
        entry = {}
        for lo, hi in [(0, 5), (5, 10), (10, 15), (15, 30), (30, 60), (60, 120)]:
            m = pos & (R["time_to_low"] > lo) & (R["time_to_low"] <= hi)
            if m.sum() < 50:
                continue
            aa, bb = float(pa[m].mean()), float(pb[m].mean())
            print(f"  {f'{lo}-{hi}':>12} {int(m.sum()):>9,} {aa:>9.4f} "
                  f"{bb:>9.4f} {bb-aa:>+9.4f}")
            entry[f"{lo}-{hi}"] = {"n": int(m.sum()), "prob_7ch": aa,
                                   "prob_9ch": bb, "delta": bb - aa}
        out["time_to_event"] = entry

    # did the previously jointly-missed events improve?
    tp = TAP_DIR / "gru_ta__s42_probs.npz"
    dp = DMS_DIR / "tcn_ta__s42_probs.npz"
    if tp.exists() and dp.exists():
        z1, z2 = np.load(tp), np.load(dp)
        if np.array_equal(z1["idx_test"], idx) and np.array_equal(z2["idx_test"], idx):
            p1, p2 = z1["test"][:, j], z2["test"][:, j]
            t1 = np.quantile(p1, 1 - y.mean() * 3)
            t2 = np.quantile(p2, 1 - y.mean() * 3)
            miss = (p1 < t1) & (p2 < t2) & (y == 1)
            if miss.sum() > 20:
                print(f"\nthe {int(miss.sum()):,} events GRU+TA and TCN+TA both missed")
                print(f"  mean predicted probability: 7ch {pa[miss].mean():.4f}  "
                      f"9ch {pb[miss].mean():.4f}  "
                      f"delta {pb[miss].mean()-pa[miss].mean():+.4f}")
                out["joint_misses"] = {
                    "n": int(miss.sum()), "prob_7ch": float(pa[miss].mean()),
                    "prob_9ch": float(pb[miss].mean())}

    print(f"\n{'#'*94}\nREADING THIS\n{'#'*94}")
    g = out.get("glucose", {})
    easy = g.get("<90", {}).get("delta", 0.0)
    hard = np.mean([v["delta"] for k, v in g.items() if k != "<90"]) if len(g) > 1 else 0.0
    print(f"\n  gain where glucose < 90 (already falling): {easy:+.4f}")
    print(f"  gain where glucose >= 90 (early prediction): {hard:+.4f}")
    if hard > 2 * abs(easy) and hard > 0.005:
        print("\n  -> The gain is concentrated in the hard, early regime.")
        print("     Absolute state is helping precisely where the models")
        print("     were failing, which is worth building on.")
    elif easy > hard:
        print("\n  -> The gain sits in the easy regime, where the fall has")
        print("     already begun. That is cosmetic rather than mechanistic.")
    else:
        print("\n  -> The gain is spread evenly across regimes: a general")
        print("     representation improvement rather than a targeted fix.")

    common.save_result(OUT_DIR, f"_gain_regimes_h{args.horizon}", out)


# ─── DIAGNOSTIC 2 ─────────────────────────────────────────────────────────────

def run_treatment(args, bw):
    tp = TAP_DIR / "gru_ta__s42_probs.npz"
    dp = DMS_DIR / "tcn_ta__s42_probs.npz"
    for p in (tp, dp):
        if not p.exists():
            print(f"  missing {p}")
            return
    z1, z2 = np.load(tp), np.load(dp)
    idx = z1["idx_test"]
    if not np.array_equal(z2["idx_test"], idx):
        print("  !! models evaluated on different windows")
        return

    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p1, p2 = z1["test"][:, j], z2["test"][:, j]
    t1 = np.quantile(p1, 1 - y.mean() * 3)
    t2 = np.quantile(p2, 1 - y.mean() * 3)
    det = (p1 >= t1) & (p2 >= t2) & (y == 1)
    miss = (p1 < t1) & (p2 < t2) & (y == 1)

    print(f"\n{'#'*94}")
    print(f"# DIAGNOSTIC 2 — treatment history of detected vs missed events "
          f"(h={args.horizon})")
    print(f"#   Missed events sit near 120 mg/dL and nearly flat, yet cross")
    print(f"#   below 70 within the horizon. Is insulin already on board?")
    print(f"{'#'*94}")
    print(f"\n  detected {int(det.sum()):,}   jointly missed {int(miss.sum()):,}")

    if miss.sum() < 20:
        print("  too few missed events for a meaningful comparison")
        return

    starts = bw.starts[idx]
    wl = bw.window_len
    out, flags = {}, []

    print(f"\n  {'channel':>10} {'window':>10} {'detected':>22} {'missed':>22} "
          f"{'ratio':>8}")
    for ch, cname in [(CH_BOLUS, "bolus"), (CH_BASAL, "basal"),
                      (CH_CARBS, "carbs")]:
        for mins, steps in [(30, 6), (60, 12), (120, 24)]:
            steps = min(steps, wl)
            # sum over the last `steps` readings of the window -- strictly
            # causal, it is part of the model's own input
            vals = np.empty(len(idx), dtype=np.float32)
            for k, s in enumerate(starts):
                vals[k] = bw.timeline[s + wl - steps:s + wl, ch].sum()
            a, b = vals[det], vals[miss]
            ratio = (b.mean() / a.mean()) if abs(a.mean()) > 1e-9 else float("nan")
            print(f"  {cname:>10} {f'{mins} min':>10} "
                  f"{a.mean():>10.4f} +/-{a.std():>9.4f} "
                  f"{b.mean():>10.4f} +/-{b.std():>9.4f} {ratio:>8.2f}")
            out[f"{cname}_{mins}min"] = {
                "detected_mean": float(a.mean()), "missed_mean": float(b.mean()),
                "ratio": float(ratio)}
            if cname == "bolus" and np.isfinite(ratio):
                flags.append(ratio)

    # fraction of windows with ANY bolus recorded
    print(f"\n  fraction of events with any recorded bolus in the window:")
    for mins, steps in [(30, 6), (60, 12), (120, 24)]:
        steps = min(steps, wl)
        anyb = np.empty(len(idx), dtype=bool)
        for k, s in enumerate(starts):
            anyb[k] = np.abs(bw.timeline[s + wl - steps:s + wl,
                                         CH_BOLUS]).max() > 1e-6
        print(f"    last {mins:>3} min: detected {100*anyb[det].mean():>6.2f}%   "
              f"missed {100*anyb[miss].mean():>6.2f}%")
        out[f"any_bolus_{mins}min"] = {
            "detected": float(anyb[det].mean()),
            "missed": float(anyb[miss].mean())}

    print(f"\n{'#'*94}\nREADING THIS\n{'#'*94}")
    mx = max(flags) if flags else 1.0
    if mx > 1.3:
        print(f"\n  Missed events carry up to {mx:.2f}x the recent bolus")
        print("  exposure of detected ones. The information is in the input")
        print("  but the model is not carrying its delayed effect across the")
        print("  window. That is a concrete, attackable failure and would")
        print("  justify an architecture with explicit treatment-effect")
        print("  memory.")
    elif mx < 0.77:
        print(f"\n  Missed events carry LESS recent insulin ({mx:.2f}x).")
        print("  Insulin does not explain them; a missed meal or exercise is")
        print("  more likely, and neither is recorded here.")
    else:
        print(f"\n  Treatment exposure is comparable between the two groups")
        print(f"  (bolus ratio {mx:.2f}x). Insulin timing does not explain the")
        print("  missed events, so the trigger is not recorded in this")
        print("  dataset. That strengthens the information-ceiling account")
        print("  and argues against another architecture.")

    common.save_result(OUT_DIR, f"_treatment_h{args.horizon}", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", action="store_true")
    ap.add_argument("--treatment", action="store_true")
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    args = ap.parse_args()

    if not (args.gain or args.treatment):
        print("Choose --gain and/or --treatment.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)

    if args.gain:
        run_gain(args, bw)
    if args.treatment:
        run_treatment(args, bw)
    print(f"\n✓ Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
