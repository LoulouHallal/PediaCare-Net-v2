"""
make_improvement_table.py
===========================
Save a before/after comparison table for any two model arms, with every
metric at every horizon and paired subject-level bootstrap CIs.

WHY THIS EXISTS
---------------
The individual pilot scripts print their comparison to the console and
save only the raw per-run JSON. Nothing writes a table you can open,
cite, or hand to someone. This reads the JSONs any pilot produced and
writes CSV + Markdown.

It is deliberately generic: point it at any two arms in any results
folder and it produces the same table, so the absolute-state pilot, the
TSL-GRU ablation and the balancing runs all get comparable output.

WHAT IT REPORTS
---------------
Per horizon, for both arms: AUROC, AUPRC, PPV, recall, F1, specificity,
plus the delta with a 95% CI from a paired cluster bootstrap over test
subjects. Pairing matters here -- both arms score the identical subjects,
so pairing removes subject-sampling noise and resolves differences far
smaller than the marginal CIs would suggest.

A significance flag is printed from the CI, NOT from a fixed effect-size
threshold. The pilot scripts use thresholds like "> 0.005 is meaningful",
which is a decision rule about practical importance and says nothing
about whether an effect is real. A +0.0036 gain with a CI excluding zero
is a real effect that happens to be small; labelling it "adds nothing"
because it misses an arbitrary cutoff conflates the two questions.

Usage:
    # absolute-state pilot
    python make_improvement_table.py \\
        --dir results/RQ2_models/absolute_state \\
        --before gru_ta__s42 --after gru_ta_abs__s42 \\
        --name TA_vs_TA_absolute

    # any other pair
    python make_improvement_table.py \\
        --dir results/RQ2_models/tsl_gru \\
        --before tsl_static__weighted_bce__none__any \\
        --after  tsl_gru__weighted_bce__none__any \\
        --name TSL_static_vs_dynamic
"""

import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import config
import common

HORIZONS = config.HORIZONS
METRICS = ["auroc", "auprc", "ppv", "recall", "f1", "specificity"]
OUT = config.RESULTS / "tables"


def load(d, stem):
    p = Path(d) / f"{stem}.json"
    if not p.exists():
        # tolerate a stem given with or without the .json suffix
        p = Path(d) / stem
        if not p.exists():
            raise FileNotFoundError(f"No result at {p}")
    return json.load(open(p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True,
                    help="folder holding both result JSONs")
    ap.add_argument("--before", required=True, help="baseline arm stem")
    ap.add_argument("--after", required=True, help="proposed arm stem")
    ap.add_argument("--name", default=None, help="output filename stem")
    ap.add_argument("--label_before", default=None)
    ap.add_argument("--label_after", default=None)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    args = ap.parse_args()

    d = Path(args.dir)
    if not d.is_absolute():
        d = config.PROJECT_ROOT / d
    A = load(d, args.before)
    B = load(d, args.after)

    la = args.label_before or A.get("model", args.before)
    lb = args.label_after or B.get("model", args.after)
    name = args.name or f"{args.before}_vs_{args.after}"

    print(f"before: {la}  ({A.get('n_channels', '?')} channels, "
          f"{A.get('n_params', 0):,} params)")
    print(f"after : {lb}  ({B.get('n_channels', '?')} channels, "
          f"{B.get('n_params', 0):,} params)")

    rows, deltas = [], []
    for h in HORIZONS:
        ha, hb = A["horizons"].get(str(h)), B["horizons"].get(str(h))
        if ha is None or hb is None:
            continue
        ma, mb = ha["per_subject_mean"], hb["per_subject_mean"]

        for lab, m, ev in [(la, ma, ha), (lb, mb, hb)]:
            rows.append({
                "horizon": h, "model": lab,
                **{k: m[k] for k in METRICS},
                "base_rate": ev.get("test_prevalence", float("nan")),
                "auprc_over_base": (m["auprc"] / ev["test_prevalence"]
                                    if ev.get("test_prevalence") else float("nan")),
                "threshold": ev.get("threshold", float("nan")),
                "constraint_met": f"{ev['constraint']['n_meeting']}/{ev['n_subjects']}",
                "worst_recall": ev["constraint"]["worst_recall"],
            })

        # paired bootstrap: both arms scored the identical subjects
        pb = common.paired_delta(ha["per_subject"], hb["per_subject"],
                                 n_boot=args.n_boot)
        row = {"horizon": h}
        for k in METRICS:
            dd = pb[k]
            row[f"d_{k}"] = dd["delta"]
            row[f"d_{k}_lo"] = dd["lo"]
            row[f"d_{k}_hi"] = dd["hi"]
            row[f"d_{k}_sig"] = "yes" if (dd["lo"] > 0 or dd["hi"] < 0) else "no"
            row[f"d_{k}_pct"] = (100 * dd["delta"] / ma[k]
                                 if ma[k] else float("nan"))
        deltas.append(row)

    if not rows:
        print("No overlapping horizons between the two runs.")
        return

    vals = pd.DataFrame(rows)
    dels = pd.DataFrame(deltas)
    OUT.mkdir(parents=True, exist_ok=True)
    vals.to_csv(OUT / f"{name}_values.csv", index=False, float_format="%.4f")
    dels.to_csv(OUT / f"{name}_deltas.csv", index=False, float_format="%.4f")

    # ---- console + markdown ----------------------------------------------
    lines = [f"# {name.replace('_', ' ')}", "",
             f"**{la}** ({A.get('n_params', 0):,} params, "
             f"{A.get('n_channels', '?')} channels) versus "
             f"**{lb}** ({B.get('n_params', 0):,} params, "
             f"{B.get('n_channels', '?')} channels).", "",
             "Per-subject means. Deltas carry 95% CIs from a paired "
             "subject-level cluster bootstrap; `sig` marks intervals that "
             "exclude zero.", "",
             "Note that significance and practical size are separate "
             "questions: a small delta whose interval excludes zero is a "
             "real effect that happens to be small.", ""]

    print(f"\n{'#'*100}")
    print(f"# {la}  ->  {lb}")
    print(f"{'#'*100}")
    for h in HORIZONS:
        va = vals[(vals.horizon == h) & (vals.model == la)]
        vb = vals[(vals.horizon == h) & (vals.model == lb)]
        dd = dels[dels.horizon == h]
        if va.empty or vb.empty:
            continue
        print(f"\nh={h} min   (base rate {va.base_rate.iloc[0]:.4f})")
        print(f"  {'metric':>12} {'before':>10} {'after':>10} {'delta':>10} "
              f"{'95% CI':>22} {'sig':>5} {'change':>9}")
        block = []
        for k in METRICS:
            a, b = float(va[k].iloc[0]), float(vb[k].iloc[0])
            de = float(dd[f"d_{k}"].iloc[0])
            lo, hi = float(dd[f"d_{k}_lo"].iloc[0]), float(dd[f"d_{k}_hi"].iloc[0])
            sig = dd[f"d_{k}_sig"].iloc[0]
            pct = float(dd[f"d_{k}_pct"].iloc[0])
            print(f"  {k:>12} {a:>10.4f} {b:>10.4f} {de:>+10.4f} "
                  f"{f'[{lo:+.4f}, {hi:+.4f}]':>22} {sig:>5} {pct:>+8.2f}%")
            block.append({"metric": k, "before": a, "after": b, "delta": de,
                          "CI": f"[{lo:+.4f}, {hi:+.4f}]", "sig": sig,
                          "change_%": pct})
        lines += [f"## h = {h} min  (base rate "
                  f"{va.base_rate.iloc[0]:.4f})", "",
                  pd.DataFrame(block).to_markdown(index=False, floatfmt=".4f"),
                  "",
                  f"Constraint met: {va.constraint_met.iloc[0]} -> "
                  f"{vb.constraint_met.iloc[0]}   |   worst-subject recall "
                  f"{va.worst_recall.iloc[0]:.3f} -> "
                  f"{vb.worst_recall.iloc[0]:.3f}", ""]

    # summary of which metrics improved significantly
    n_sig = sum(1 for h in HORIZONS for k in METRICS
                if not dels[dels.horizon == h].empty
                and dels[dels.horizon == h][f"d_{k}_sig"].iloc[0] == "yes"
                and dels[dels.horizon == h][f"d_{k}"].iloc[0] > 0)
    n_tot = len(dels) * len(METRICS)
    lines += ["## Summary", "",
              f"{n_sig} of {n_tot} metric-horizon combinations improved with "
              f"a 95% CI excluding zero.", ""]
    print(f"\n  {n_sig} of {n_tot} metric-horizon combinations improved "
          f"significantly.")

    (OUT / f"{name}.md").write_text("\n".join(lines))
    print(f"\n✓ Saved -> {OUT / (name + '.md')}")
    print(f"           {OUT / (name + '_values.csv')}")
    print(f"           {OUT / (name + '_deltas.csv')}")


if __name__ == "__main__":
    main()
