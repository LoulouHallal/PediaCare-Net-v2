"""
operating_points.py
=====================
Sweep the decision threshold across its full range and report the
precision/recall trade-off, using the probabilities already saved by
stage2_deep.py. No retraining: this is pure post-processing.

WHY THIS TABLE MATTERS
----------------------
PPV and recall are not independent properties of a model. They are two
coordinates on one curve, and the threshold picks a point on it. Quoting
a single PPV therefore describes an operating-point choice as much as it
describes the model.

The project's default policy — maximise PPV subject to recall >= 0.85 —
deliberately buys recall with precision, because missing a hypoglycaemic
episode in a child is worse than a false alarm. That choice is why PPV
reads ~0.37 while recall reads ~0.94. At the default 0.5 threshold the
SAME model gives PPV ~0.69 at recall ~0.62.

Both are honest descriptions of the same classifier. Reporting the curve
rather than one point is the defensible presentation: it shows the
trade-off explicitly and cannot be accused of selecting whichever number
looked best.

THREE NAMED OPERATING POINTS
----------------------------
    high_sensitivity   max PPV subject to recall >= --recall_floor (0.85)
                       safety-first; the project default
    balanced           threshold 0.5, and separately the max-F1 point
    high_precision     max PPV subject to recall >= --precision_recall_floor
                       (0.40) -- for settings where alarm fatigue dominates

All three come from ONE model and ONE set of predictions.

IMPORTANT
---------
Thresholds here are read off the TEST set to draw the curve. That is
appropriate for characterising the trade-off, but a threshold intended
for deployment must be chosen on validation data (as stage1/stage2 do)
and then locked. This script reports the validation-selected point too,
so the difference is visible rather than hidden.

Usage:
    python operating_points.py                       # all stage-2 models
    python operating_points.py --model gru --loss weighted_bce
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import config
import common

HORIZONS = config.HORIZONS
# Both stages save probabilities in the same format, so the sweep is
# model-agnostic: classical and deep models get identical treatment.
PROB_DIRS = [config.RQ2_STAGES["stage1_ml"] / "v2_rebuilt",
             config.RQ2_STAGES["stage2_dl"]]
OUT = config.RESULTS / "tables"


def per_subject_at(y, p, meta, thr):
    """Per-subject mean metrics at one threshold."""
    res = common.evaluate(y, p, meta, thr)
    return res["per_subject_mean"], res["constraint"], res["per_subject"]


def sweep_one(name, probs_path, json_path, bw, args):
    z = np.load(probs_path)
    p_test, idx_test = z["test"], z["idx_test"]
    p_val, idx_val = z["val"], z["idx_val"]
    meta_te = bw.meta[idx_test]

    meta_json = json.load(open(json_path))
    label_set = meta_json.get("label_set", "any")
    Y = bw._labels if label_set == "consensus" else bw._labels_any
    Y_te, Y_va = Y[idx_test], Y[idx_val]

    rows, named = [], []
    for j, h in enumerate(HORIZONS):
        yt = Y_te[:, j].astype(int)
        yv = Y_va[:, j].astype(int)
        base = float(yt.mean())

        # full curve
        for thr in np.round(np.arange(0.05, 0.96, args.step), 3):
            m, c, _ = per_subject_at(yt, p_test[:, j], meta_te, thr)
            rows.append({"model": name, "horizon": h, "threshold": float(thr),
                         "base_rate": base, **{k: m[k] for k in common.METRICS},
                         "constraint_met": f"{c['n_meeting']}/38",
                         "worst_recall": c["worst_recall"]})

        d = pd.DataFrame([r for r in rows if r["model"] == name
                          and r["horizon"] == h])

        def pick(sub, label, how="ppv"):
            if len(sub) == 0:
                return None
            r = sub.loc[sub[how].idxmax()]
            return {"model": name, "horizon": h, "operating_point": label,
                    "threshold": float(r["threshold"]), "base_rate": base,
                    **{k: float(r[k]) for k in common.METRICS},
                    "constraint_met": r["constraint_met"],
                    "worst_recall": float(r["worst_recall"])}

        hs = pick(d[d.recall >= args.recall_floor], "high_sensitivity")
        hp = pick(d[d.recall >= args.precision_recall_floor], "high_precision")
        bf = pick(d, "balanced_maxF1", how="f1")

        # threshold 0.5 exactly, and the validation-selected threshold
        m05, c05, _ = per_subject_at(yt, p_test[:, j], meta_te, 0.5)
        default = {"model": name, "horizon": h, "operating_point": "default_0.5",
                   "threshold": 0.5, "base_rate": base,
                   **{k: m05[k] for k in common.METRICS},
                   "constraint_met": f"{c05['n_meeting']}/38",
                   "worst_recall": c05["worst_recall"]}

        thr_val, _ = common.find_threshold(yv, p_val[:, j])
        mv, cv, _ = per_subject_at(yt, p_test[:, j], meta_te, thr_val)
        valsel = {"model": name, "horizon": h,
                  "operating_point": "validation_selected",
                  "threshold": float(thr_val), "base_rate": base,
                  **{k: mv[k] for k in common.METRICS},
                  "constraint_met": f"{cv['n_meeting']}/38",
                  "worst_recall": cv["worst_recall"]}

        for x in [valsel, default, hs, bf, hp]:
            if x:
                named.append(x)
    return rows, named


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--model", default=None, help="filter, e.g. gru")
    ap.add_argument("--loss", default=None, help="filter, e.g. weighted_bce")
    ap.add_argument("--step", type=float, default=0.05)
    ap.add_argument("--recall_floor", type=float, default=0.85,
                    help="recall required at the high-sensitivity point")
    ap.add_argument("--precision_recall_floor", type=float, default=0.40,
                    help="recall required at the high-precision point")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)

    files = []
    for d in PROB_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("*_probs.npz")))
    if args.model:
        files = [f for f in files if f.name.startswith(args.model)]
    if args.loss:
        files = [f for f in files if f"__{args.loss}__" in f.name]
    if not files:
        print("No saved probabilities found in:")
        for d in PROB_DIRS:
            print(f"  {d}")
        print("Run stage1_v2.py / stage2_deep.py first (re-run with --force")
        print("if they completed before probability saving was added).")
        return
    print(f"Found {len(files)} models with saved predictions")

    all_rows, all_named = [], []
    for f in files:
        name = f.name.replace("_probs.npz", "")
        jp = f.with_name(name + ".json")
        if not jp.exists():
            print(f"  skip {name}: no results JSON")
            continue
        print(f"sweeping {name}...")
        r, n = sweep_one(name, f, jp, bw, args)
        all_rows.extend(r)
        all_named.extend(n)

    curve = pd.DataFrame(all_rows)
    named = pd.DataFrame(all_named)
    curve.to_csv(OUT / "T6_operating_curve.csv", index=False,
                 float_format="%.4f")
    named.to_csv(OUT / "T6_operating_points.csv", index=False,
                 float_format="%.4f")

    lines = ["# Operating points — same model, same predictions, "
             "different threshold", "",
             "PPV and recall are two coordinates on one curve. The threshold "
             "selects a point on it; it does not change the model. All rows "
             "below come from identical predictions.", ""]

    for name in named.model.unique():
        lines += [f"## {name}", ""]
        for h in HORIZONS:
            sub = named[(named.model == name) & (named.horizon == h)]
            if sub.empty:
                continue
            keep = ["operating_point", "threshold", "ppv", "recall", "f1",
                    "auroc", "auprc", "constraint_met", "worst_recall"]
            lines += [f"### h = {h} min  (base rate "
                      f"{sub.base_rate.iloc[0]:.4f})", "",
                      sub[keep].to_markdown(index=False, floatfmt=".4f"), ""]
    (OUT / "T6_operating_points.md").write_text("\n".join(lines))

    # console summary for the shortest horizon
    print(f"\n{'#'*96}")
    print("# OPERATING POINTS — h=15 min (same model, same predictions)")
    print(f"{'#'*96}")
    for name in named.model.unique():
        sub = named[(named.model == name) & (named.horizon == 15)]
        if sub.empty:
            continue
        print(f"\n{name}")
        print(f"  {'operating point':>22} {'thr':>6} {'PPV':>8} {'Recall':>8} "
              f"{'F1':>8} {'AUPRC':>8} {'constraint':>11}")
        for _, r in sub.iterrows():
            print(f"  {r['operating_point']:>22} {r['threshold']:>6.2f} "
                  f"{r['ppv']:>8.4f} {r['recall']:>8.4f} {r['f1']:>8.4f} "
                  f"{r['auprc']:>8.4f} {r['constraint_met']:>11}")

    print("\nAUPRC is identical across rows for a given model: it is "
          "threshold-free.")
    print("Only the threshold-dependent metrics move. That is the whole point.")
    print(f"\n✓ Saved -> {OUT / 'T6_operating_points.md'} (+ two CSVs)")


if __name__ == "__main__":
    main()
