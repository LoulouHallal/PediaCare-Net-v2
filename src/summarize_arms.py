"""
summarize_arms.py  --  a correct comparison table, read-only
=============================================================

cmr_pilot's collect() names every run in the folder "gru", so the CMR
verdict line has been subtracting CMR from whichever row it happened to
pick up:

    -0.00060   against the real baseline        <- correct
    +0.00021   against the nadir lam=0.1 run
    -0.00097   against the nadir lam=0.3 run
    -0.00055   against the nadir lam=1.0 run

This script derives the arm name from the filename instead, so each
experiment is reported as itself.

IT WRITES NOTHING AND CHANGES NOTHING. Every .json, .pt and _probs.npz
stays exactly where it is, and cmr_pilot.py is untouched, so the resume
and skip logic keeps working. This only prints.

    python summarize_arms.py
    python summarize_arms.py --dir results/RQ2_models/cmr_gru
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_name(fname: str):
    """gru__s42__lam0.3.json -> ('gru', 42, {'lam': '0.3'})"""
    stem = os.path.basename(fname)[:-5]
    parts = stem.split("__")
    arm, seed, opts = parts[0], None, {}
    for p in parts[1:]:
        if re.fullmatch(r"s\d+", p):
            seed = int(p[1:])
        elif p.startswith("lam"):
            opts["lam"] = p[3:]
        elif p in ("consensus", "any"):
            opts["endpoint"] = p
        else:
            opts.setdefault("other", []).append(p)
    # runs from before the seed suffix existed are the default seed
    return arm, (42 if seed is None else seed), opts


def label(arm: str, opts: dict) -> str:
    name = arm
    if "lam" in opts:
        name += f"+nadir(lam={opts['lam']})"
    if opts.get("endpoint") == "consensus":
        name += " [consensus]"
    for o in opts.get("other", []):
        name += f"+{o}"
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/RQ2_models/cmr_gru")
    ap.add_argument("--baseline", default="gru",
                    help="arm name treated as the reference")
    a = ap.parse_args()

    files = sorted(f for f in glob.glob(os.path.join(a.dir, "*.json"))
                   if not os.path.basename(f).startswith("_"))
    if not files:
        sys.exit(f"no run json files in {a.dir}")

    rows = []
    for f in files:
        try:
            r = json.load(open(f))
        except Exception as e:
            print(f"  skipping {os.path.basename(f)}: {e}")
            continue
        arm, seed, opts = parse_name(f)
        v = r.get("val_mean_auprc")
        if v is None:
            continue
        rows.append({
            "label": label(arm, opts),
            "arm": arm, "seed": seed,
            "endpoint": opts.get("endpoint", "any"),
            "lam": opts.get("lam"),
            "val_mean": float(v),
            "params": r.get("params") or r.get("n_params"),
            "best_epoch": r.get("best_epoch"),
            "h": r.get("val_horizons") or {},
            "file": os.path.basename(f),
        })

    print("=" * 86)
    print("ALL RUNS  (pooled validation mean AUPRC, as cmr_pilot selects on)")
    print("=" * 86)
    print(f"{'run':<34} {'seed':>5} {'endpoint':>10} {'params':>8} {'val mean':>10}")
    for r in sorted(rows, key=lambda x: (x["endpoint"], x["label"], x["seed"] or 0)):
        p = f"{r['params']:,}" if isinstance(r["params"], int) else "-"
        print(f"{r['label']:<34} {r['seed'] or '-':>5} {r['endpoint']:>10} "
              f"{p:>8} {r['val_mean']:>10.5f}")

    # ---- baseline, any endpoint only ----------------------------------
    base = [r for r in rows
            if r["arm"] == a.baseline and r["endpoint"] == "any"
            and r["lam"] is None]
    if not base:
        print(f"\nno clean '{a.baseline}' baseline runs found")
        return

    bs = {r["seed"]: r["val_mean"] for r in base}
    vals = np.array(sorted(bs.values()))
    print("\n" + "=" * 86)
    print(f"BASELINE  {a.baseline}, endpoint=any, no auxiliary loss")
    print("=" * 86)
    print(f"  seeds {sorted(bs)}")
    print(f"  mean {vals.mean():.5f}   sd {vals.std(ddof=1):.5f}   "
          f"range {vals.max()-vals.min():.5f}")
    print(f"  NOISE FLOOR: any effect smaller than {vals.max()-vals.min():.5f} "
          f"is indistinguishable from changing the seed.")

    # ---- paired comparisons -------------------------------------------
    print("\n" + "=" * 86)
    print("PAIRED vs BASELINE  (same seed; only comparable within an endpoint)")
    print("=" * 86)
    others = {}
    for r in rows:
        if r["arm"] == a.baseline and r["endpoint"] == "any" and r["lam"] is None:
            continue
        others.setdefault(r["label"], []).append(r)

    for lab in sorted(others):
        rs = others[lab]
        ep = rs[0]["endpoint"]
        if ep != "any":
            m = np.mean([r["val_mean"] for r in rs])
            print(f"  {lab:<34} mean {m:.5f}   "
                  f"NOT COMPARABLE -- different endpoint, different prevalence")
            continue
        ds = [(r["seed"], r["val_mean"] - bs[r["seed"]])
              for r in rs if r["seed"] in bs]
        if len(ds) < len(rs):
            print(f"    ({len(rs)-len(ds)} run(s) of {lab} have no matching "
                  f"baseline seed and are excluded from the pairing)")
        if not ds:
            continue
        d = np.array([x[1] for x in ds])
        verdict = ("ADOPT" if d.mean() >= 0.003 else
                   "below threshold" if abs(d.mean()) < 0.003 else "worse")
        seeds = " ".join(f"s{s}{v:+.5f}" for s, v in ds)
        print(f"  {lab:<34} {seeds}   mean {d.mean():+.5f}   -> {verdict}")

    print("\n  Decision threshold +0.003 as pre-registered. With a noise floor "
          f"of {vals.max()-vals.min():.5f},")
    print("  a single-seed difference below that is not evidence of anything.")

    # ---- per-horizon ---------------------------------------------------
    print("\n" + "=" * 86)
    print("PER-HORIZON  (pooled validation AUPRC)")
    print("=" * 86)
    hz = sorted({int(k) for r in rows for k in r["h"]},
                key=int) if any(r["h"] for r in rows) else []
    if hz:
        print(f"{'run':<34} {'seed':>5} " + "".join(f"{'h'+str(h):>9}" for h in hz))
        for r in sorted(rows, key=lambda x: (x["endpoint"], x["label"], x["seed"] or 0)):
            if not r["h"]:
                continue
            cells = "".join(f"{r['h'].get(str(h), float('nan')):>9.4f}" for h in hz)
            print(f"{r['label']:<34} {r['seed'] or '-':>5} {cells}")

    print("\nNote: these are POOLED values, which is what cmr_pilot selects on.")
    print("The project's primary metric is per-subject mean AUPRC, which is "
          "lower\n(0.8689 vs 0.9003 at h=15 for the seed-42 baseline). Do not "
          "mix the two.")


if __name__ == "__main__":
    main()
