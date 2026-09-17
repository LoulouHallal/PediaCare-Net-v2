"""
phase2_followup.py  --  two questions the main report raised
=============================================================

Q1  Is the error concentration real, or just window count?
    The report ranks subjects by RAW FP+FN count. A subject with 3x the
    recording produces 3x the errors while being no harder to predict.
    Rank instead by per-subject AUPRC and by error RATE, and see whether
    the same subjects survive. Phase 5 (patient-balanced batching) is
    justified only if they do.

Q2  How much of the residual error is the 70 mg/dL cut point slicing
    through CGM noise?
    At h=15, 91.6% of false positives have a future nadir of 70-80 with a
    median of 72. Sensor MARD is ~9-10%, so a true 68 routinely reads 72.
    Recompute AUPRC with an ambiguous band excluded to estimate the
    ceiling the labels permit.

    THIS IS A DIAGNOSTIC, NOT A RESULT. Dropping windows changes the
    evaluation set and the prevalence, so the number is NOT comparable to
    the headline AUPRC and must never be reported as an improvement. It
    answers one question only: how much of the gap to 1.0 is reachable.

Usage
-----
    python phase2_followup.py \
        --windows results/error_analysis/windows_gru_ta_abs__s42.csv.gz
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

HYPO = 70.0


def _ap(y, p):
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, p))


def concentration(df: pd.DataFrame) -> pd.DataFrame:
    """Q1: rank subjects by rate and by AUPRC, not by raw error count."""
    rows = []
    for (sid, h), g in df.groupby(["subject_id", "horizon"]):
        err = int((g.outcome.isin(["FP", "FN"])).sum())
        rows.append({
            "subject_id": sid, "horizon": h,
            "n_windows": len(g),
            "n_pos": int(g.y_true.sum()),
            "prevalence": float(g.y_true.mean()),
            "n_err": err,
            "err_rate": err / len(g),
            "auprc": _ap(g.y_true, g.y_prob),
        })
    return pd.DataFrame(rows)


def report_concentration(c: pd.DataFrame) -> str:
    L = ["=" * 78,
         "Q1  ERROR CONCENTRATION -- count artifact or genuinely hard subjects?",
         "=" * 78]
    for h in sorted(c.horizon.unique()):
        d = c[c.horizon == h]
        n = len(d)
        uniform = 5 / n
        by_count = d.nlargest(5, "n_err")
        by_rate = d.nlargest(5, "err_rate")
        by_auprc = d.nsmallest(5, "auprc")

        share = by_count.n_err.sum() / d.n_err.sum()
        L.append(f"\n-- h={h} ({n} subjects; 5 subjects would be "
                 f"{uniform:.1%} of errors if uniform) --")
        L.append(f"top-5 by raw count hold {share:.1%}  "
                 f"({share/uniform:.2f}x over-representation)")

        # is the raw count just window count?
        r_wins = d.n_err.corr(d.n_windows, method="spearman")
        r_pos = d.n_err.corr(d.n_pos, method="spearman")
        r_rate = d.err_rate.corr(d.auprc, method="spearman")
        L.append(f"spearman  n_err vs n_windows {r_wins:+.3f}   "
                 f"n_err vs n_pos {r_pos:+.3f}   err_rate vs auprc {r_rate:+.3f}")
        if r_wins > 0.8:
            L.append("  -> raw counts are largely a RECORDING-LENGTH artifact")

        overlap = len(set(by_count.subject_id) & set(by_auprc.subject_id))
        L.append(f"overlap between worst-5 by count and worst-5 by AUPRC: "
                 f"{overlap}/5")

        L.append("  by raw count : " + ", ".join(
            f"{r.subject_id}(n={r.n_windows:,}, rate={r.err_rate:.3f}, "
            f"auprc={r.auprc:.3f})" for r in by_count.itertuples()))
        L.append("  by error rate: " + ", ".join(
            f"{r.subject_id}({r.err_rate:.3f})" for r in by_rate.itertuples()))
        L.append("  by worst AUPRC: " + ", ".join(
            f"{r.subject_id}({r.auprc:.3f})" for r in by_auprc.itertuples()))

    L.append("\nVERDICT GUIDE")
    L.append("  overlap 4-5/5 and n_err vs n_windows weak  -> genuinely hard")
    L.append("     subjects exist; Phase 5 patient balancing is justified.")
    L.append("  overlap 0-2/5 or n_err vs n_windows > 0.8  -> concentration is")
    L.append("     a recording-length artifact; Phase 5 drops in priority.")
    return "\n".join(L)


def ceiling(df: pd.DataFrame, bands=((70, 75), (70, 80), (65, 80))) -> str:
    """
    Q2: AUPRC after removing windows whose future nadir sits inside a band
    around 70, i.e. windows whose label could flip under sensor error.
    Per-subject aggregated, same as the primary metric.
    """
    L = ["", "=" * 78,
         "Q2  LABEL-BOUNDARY CEILING  (diagnostic only -- never report as a result)",
         "=" * 78]
    for h in sorted(df.horizon.unique()):
        d = df[df.horizon == h]
        base = d.groupby("subject_id").apply(
            lambda g: _ap(g.y_true, g.y_prob), include_groups=False).mean()
        L.append(f"\n-- h={h} --")
        L.append(f"  as measured                         {base:.4f}")
        for lo, hi in bands:
            keep = d[~d.future_nadir.between(lo, hi)]
            drop_frac = 1 - len(keep) / len(d)
            v = keep.groupby("subject_id").apply(
                lambda g: _ap(g.y_true, g.y_prob), include_groups=False).mean()
            prev = keep.y_true.mean()
            L.append(f"  excl. nadir {lo}-{hi:<3}  {v:.4f}   "
                     f"(dropped {drop_frac:5.1%} of windows, "
                     f"prevalence {d.y_true.mean():.3%} -> {prev:.3%})")
    L.append("\nHOW TO READ THIS")
    L.append("  Prevalence shifts when windows are dropped, and AUPRC depends")
    L.append("  on prevalence, so part of any rise is mechanical. The signal is")
    L.append("  the SIZE of the jump relative to how few windows were removed.")
    L.append("  A large jump from a small exclusion at h=15 means the 70 mg/dL")
    L.append("  cut point, not the model, is what caps that horizon.")
    return "\n".join(L)


def recoverable(df: pd.DataFrame) -> str:
    """Where is error actually recoverable? Splits FN by whether the drop
    was visible in the recent trace."""
    L = ["", "=" * 78,
         "Q3  RECOVERABLE vs CEILING ERROR, by horizon",
         "=" * 78]
    for h in sorted(df.horizon.unique()):
        d = df[df.horizon == h]
        fp, fn = d[d.outcome == "FP"], d[d.outcome == "FN"]
        if not len(fp) or not len(fn):
            continue
        fp_near = fp.future_nadir.between(70, 80).mean()
        fp_wrong = (fp.future_nadir > 100).mean()
        fn_visible = (fn.slope_per5min < -1.0).mean()
        fn_blind = (fn.slope_per5min.abs() < 0.5).mean()
        L.append(f"\n-- h={h} --")
        L.append(f"  FP near-miss (nadir 70-80)      {fp_near:6.1%}  label noise")
        L.append(f"  FP genuinely wrong (nadir>100)  {fp_wrong:6.1%}  attackable")
        L.append(f"  FN falling at t=0 (slope<-1)    {fn_visible:6.1%}  attackable")
        L.append(f"  FN flat at t=0 (|slope|<0.5)    {fn_blind:6.1%}  likely ceiling")
        L.append(f"  FN median probability           {fn.y_prob.median():6.3f}  "
                 + ("threshold placement" if fn.y_prob.median() > 0.5
                    else "representation failure"))
    return "\n".join(L)



def cohort(df):
    """Q4: does per-subject AUPRC separate by source-study prefix?"""
    L = ["", "=" * 78, "Q4  SOURCE-COHORT EFFECT", "=" * 78]
    d = df.copy()
    sid = d.subject_id.astype(str)
    d["cohort"] = np.where(sid.str.contains("-"),
                           sid.str.split("-").str[0], "numeric")
    per = (d.groupby(["cohort", "subject_id", "horizon"])
             .apply(lambda g: _ap(g.y_true, g.y_prob), include_groups=False)
             .rename("auprc").reset_index())
    counts = per[per.horizon == per.horizon.min()].groupby("cohort").size()
    L.append("subjects per cohort: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    if len(counts) < 2:
        L.append("only one cohort -- nothing to compare")
        return "\n".join(L)
    for h in sorted(per.horizon.unique()):
        pp = per[per.horizon == h]
        L.append(f"\n-- h={h} --")
        for c, g in pp.groupby("cohort"):
            L.append(f"  {c:<10} n={len(g):<3} mean AUPRC {g.auprc.mean():.4f}"
                     f"   median {g.auprc.median():.4f}"
                     f"   range [{g.auprc.min():.3f}, {g.auprc.max():.3f}]")
        groups = [g.auprc.dropna().values for _, g in pp.groupby("cohort")]
        if len(groups) == 2 and all(len(x) >= 3 for x in groups):
            from scipy.stats import mannwhitneyu
            try:
                _, pv = mannwhitneyu(groups[0], groups[1], alternative="two-sided")
                L.append(f"  Mann-Whitney p={pv:.4f}"
                         + ("   <- separates" if pv < 0.05 else "   (n.s.)"))
            except Exception:
                pass
    L.append("\nConsistent separation across horizons = domain shift, not capacity.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    df = pd.read_csv(a.windows, dtype={'subject_id': str})
    df['subject_id'] = df['subject_id'].str.strip()
    n_sub = df.subject_id.nunique()
    print(f'{n_sub} unique validation subjects')
    if n_sub != 36:
        print(f'  WARNING: expected 36, got {n_sub}')
    c = concentration(df)
    txt = "\n".join([report_concentration(c), ceiling(df), recoverable(df), cohort(df)])
    print(txt)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
        c.to_csv(a.out.replace(".txt", "_subjects.csv"), index=False)
        print(f"\nwrote -> {a.out}")


if __name__ == "__main__":
    main()
