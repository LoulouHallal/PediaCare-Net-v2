"""
treatment_diagnostics.py
==========================
Do the events both models miss carry more recent insulin than the events
they detect?

WHY THIS REPLACES THE EARLIER ATTEMPT
-------------------------------------
A first version of this diagnostic read the bolus/basal/carbs channels
straight out of the windowed tensor. Those channels are causally
per-subject z-scored, so their values are deviations from each child's
running average rather than doses. That produced means straddling zero
and ratios like -2.59 and -24.41, which are arithmetically meaningless,
and an "any bolus recorded" statistic that read 88% in every lookback
because a z-scored value is essentially never exactly zero.

The two horizons also disagreed in sign, which is what a broken statistic
looks like.

This version re-streams the source parquet to recover absolute units --
the same reason the glucose timeline had to be rebuilt. Insulin in units
and carbohydrate in grams are comparable across children in a way that
per-child z-scores are not.

THE QUESTION
------------
The jointly-missed events sit near 120 mg/dL with velocity around
0.14 mg/dL/min, yet cross below 70 within 30 minutes -- a required fall
of roughly 1.66 mg/dL/min, about ten times what is visible. The
trajectory does not merely lack signal; it points the wrong way.

Rapid-acting insulin peaks 60-90 minutes after delivery, so a dose given
45 minutes earlier would explain a fall that has not yet started. If the
missed events carry more insulin-on-board, the model has the information
and is failing to carry its delayed effect -- a concrete, attackable
architectural weakness. If they do not, the trigger is not recorded in
this dataset and the information-ceiling account stands.

INSULIN ON BOARD
----------------
Raw dose sums are reported, but the more physiologically meaningful
quantity is insulin still active. A standard linear-decay IOB model with
a 180-minute duration is used:

    IOB(t) = sum over past boluses of  dose * max(0, 1 - dt/DIA)

This is a deliberate simplification -- real curves are biexponential --
but it captures the essential point that a bolus 45 minutes old is mostly
still active while one 170 minutes old is nearly spent.

Both raw sums and IOB are reported so the conclusion does not rest on the
decay model.

Usage:
    python treatment_diagnostics.py --horizon 30
    python treatment_diagnostics.py --horizon 30 --rebuild
"""

import gc
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import pyarrow.parquet as pq

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "treatment"
RAW_PARQUET = config.RAW_DATA["metabonet"]
COLS = ["id", "date", "CGM", "basal", "bolus", "carbs"]

DIA_MIN = 180.0            # duration of insulin action, minutes
LOOKBACKS = [30, 60, 120, 180]

TAP_DIR = config.RESULTS / "RQ2_models" / "tap_gru"
DMS_DIR = config.RESULTS / "RQ2_models" / "dms_tcn"


# ─── RAW TREATMENT TIMELINE ───────────────────────────────────────────────────

def build_raw_treatment(bw, verbose=True):
    """
    Recover absolute basal/bolus/carbs aligned to the windowed timeline.

    Alignment is by (subject, timestamp), which is exact: build_windows
    kept every CGM reading it used, so each timeline row has a unique
    matching parquet row.
    """
    cache = config.DATA_DERIVED / "raw_treatment_stride6.npz"
    if cache.exists():
        if verbose:
            print(f"Loading cached raw treatment -> {cache.name}")
        z = np.load(cache)
        return z["basal"], z["bolus"], z["carbs"]

    keep = set(bw.subjects.tolist())
    f = pq.ParquetFile(str(RAW_PARQUET))
    if verbose:
        print(f"Streaming {f.metadata.num_row_groups} row groups for raw "
              f"treatment values...")

    parts = []
    for g in range(f.metadata.num_row_groups):
        df = f.read_row_group(g, columns=COLS).to_pandas()
        df = df[df["CGM"].notna()]
        df["id"] = df["id"].astype(str)
        df = df[df["id"].isin(keep)]
        if len(df):
            parts.append(df[["id", "date", "basal", "bolus", "carbs"]])
        del df
        if verbose and (g + 1) % 40 == 0:
            print(f"  {g+1}/{f.metadata.num_row_groups}")
        gc.collect()

    raw = pd.concat(parts, ignore_index=True)
    del parts
    raw = raw.sort_values(["id", "date"]).drop_duplicates(["id", "date"])
    raw["t"] = raw["date"].values.astype("datetime64[m]").astype(np.int64)

    # align to the timeline by (subject, minute)
    n = len(bw.raw_gluc)
    basal = np.zeros(n, np.float32)
    bolus = np.zeros(n, np.float32)
    carbs = np.zeros(n, np.float32)
    matched = 0
    for si, pid in enumerate(bw.subjects):
        m = np.flatnonzero(bw.reading_subject == si)
        if len(m) == 0:
            continue
        sub = raw[raw["id"] == pid]
        if not len(sub):
            continue
        lut = dict(zip(sub["t"].to_numpy(),
                       zip(sub["basal"].fillna(0).to_numpy(),
                           sub["bolus"].fillna(0).to_numpy(),
                           sub["carbs"].fillna(0).to_numpy())))
        for k in m:
            v = lut.get(int(bw.reading_times[k]))
            if v is not None:
                basal[k], bolus[k], carbs[k] = v
                matched += 1
        if verbose and (si + 1) % 50 == 0:
            print(f"  aligned {si+1}/{len(bw.subjects)} subjects")

    if verbose:
        print(f"\nmatched {matched:,}/{n:,} readings "
              f"({100*matched/n:.1f}%)")
        print(f"  bolus:  nonzero {100*(bolus>0).mean():.2f}%  "
              f"max {bolus.max():.2f} U")
        print(f"  basal:  nonzero {100*(basal>0).mean():.2f}%  "
              f"max {basal.max():.2f} U")
        print(f"  carbs:  nonzero {100*(carbs>0).mean():.2f}%  "
              f"max {carbs.max():.1f} g")

    config.DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, basal=basal, bolus=bolus, carbs=carbs)
    if verbose:
        print(f"✓ Cached -> {cache}")
    return basal, bolus, carbs


def insulin_on_board(bolus, subject, times, dia=DIA_MIN):
    """
    Linear-decay IOB. A bolus contributes dose * (1 - dt/DIA) while
    dt < DIA, and nothing after.
    """
    iob = np.zeros(len(bolus), np.float32)
    for si in np.unique(subject):
        m = np.flatnonzero(subject == si)
        b, t = bolus[m], times[m]
        hits = np.flatnonzero(b > 0)
        if not len(hits):
            continue
        acc = np.zeros(len(m), np.float32)
        for hi in hits:
            dt = t - t[hi]
            live = (dt >= 0) & (dt < dia)
            acc[live] += b[hi] * (1.0 - dt[live] / dia)
        iob[m] = acc
    return iob


# ─── DIAGNOSTIC ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--rebuild", action="store_true",
                    help="ignore the cache and re-stream the parquet")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)

    if args.rebuild:
        c = config.DATA_DERIVED / "raw_treatment_stride6.npz"
        if c.exists():
            c.unlink()
    basal, bolus, carbs = build_raw_treatment(bw)
    iob = insulin_on_board(bolus, bw.reading_subject, bw.reading_times)
    print(f"  IOB: nonzero {100*(iob>0).mean():.2f}%  max {iob.max():.2f} U\n")

    # detected vs jointly missed, from the two saved prediction sets
    tp, dp = TAP_DIR / "gru_ta__s42_probs.npz", DMS_DIR / "tcn_ta__s42_probs.npz"
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

    ends = bw.starts[idx] + bw.window_len - 1
    print(f"{'#'*94}")
    print(f"# TREATMENT HISTORY IN ABSOLUTE UNITS (h={args.horizon})")
    print(f"#   detected {int(det.sum()):,}   jointly missed {int(miss.sum()):,}")
    print(f"{'#'*94}")

    if miss.sum() < 20:
        print("  too few missed events")
        return

    rng = np.random.default_rng(0)
    out = {}

    def compare(name, va, unit):
        """Difference in means with a bootstrap CI over events."""
        a, b = va[det], va[miss]
        d = b.mean() - a.mean()
        reps = np.array([rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean()
                         for _ in range(1000)])
        lo, hi = np.percentile(reps, [2.5, 97.5])
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {name:>22} {a.mean():>9.3f} {b.mean():>9.3f} {d:>+9.3f} "
              f"{f'[{lo:+.3f}, {hi:+.3f}]':>22}{sig} {unit}")
        out[name] = {"detected": float(a.mean()), "missed": float(b.mean()),
                     "delta": float(d), "lo": float(lo), "hi": float(hi),
                     "significant": bool(lo > 0 or hi < 0)}
        return d, (lo > 0 or hi < 0)

    print(f"\n  {'quantity':>22} {'detected':>9} {'missed':>9} {'delta':>9} "
          f"{'95% CI':>22}")
    print(f"  {'-'*78}")

    iob_res = compare("insulin on board", iob[ends], "U")

    for mins in LOOKBACKS:
        steps = mins // bw.sample_min
        tot = np.zeros(len(idx), np.float32)
        for k, e in enumerate(ends):
            tot[k] = bolus[max(0, e - steps):e + 1].sum()
        compare(f"bolus last {mins} min", tot, "U")

    for mins in [30, 60, 120]:
        steps = mins // bw.sample_min
        tot = np.zeros(len(idx), np.float32)
        for k, e in enumerate(ends):
            tot[k] = carbs[max(0, e - steps):e + 1].sum()
        compare(f"carbs last {mins} min", tot, "g")

    for mins in [60, 120]:
        steps = mins // bw.sample_min
        tot = np.zeros(len(idx), np.float32)
        for k, e in enumerate(ends):
            tot[k] = basal[max(0, e - steps):e + 1].sum()
        compare(f"basal last {mins} min", tot, "U")

    # how many events have ANY bolus, in absolute terms
    print(f"\n  fraction with a recorded bolus (dose > 0):")
    for mins in LOOKBACKS:
        steps = mins // bw.sample_min
        anyb = np.zeros(len(idx), bool)
        for k, e in enumerate(ends):
            anyb[k] = (bolus[max(0, e - steps):e + 1] > 0).any()
        print(f"    last {mins:>3} min: detected {100*anyb[det].mean():>6.2f}%   "
              f"missed {100*anyb[miss].mean():>6.2f}%")
        out[f"any_bolus_{mins}"] = {"detected": float(anyb[det].mean()),
                                    "missed": float(anyb[miss].mean())}

    print(f"\n{'#'*94}\nREADING THIS\n{'#'*94}")
    d_iob, sig_iob = iob_res
    if sig_iob and d_iob > 0:
        print(f"\n  Missed events carry {d_iob:+.3f} U more insulin on board,")
        print("  with a CI excluding zero. The model has the bolus in its")
        print("  input but is not carrying its delayed effect across the")
        print("  window -- a concrete failure that an architecture with")
        print("  explicit treatment-effect memory could target.")
    elif sig_iob and d_iob < 0:
        print(f"\n  Missed events carry {d_iob:+.3f} U LESS insulin on board.")
        print("  Insulin does not explain them; an unlogged meal or exercise")
        print("  is more likely, and neither is recorded here.")
    else:
        print("\n  Insulin on board does not differ between detected and")
        print("  missed events. The trigger is not in the recorded treatment")
        print("  data, which strengthens the information-ceiling account and")
        print("  argues against another architecture.")

    common.save_result(OUT_DIR, f"_treatment_raw_h{args.horizon}", out)
    print(f"\n✓ Saved -> {OUT_DIR / f'_treatment_raw_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
