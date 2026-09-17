"""
error_analysis.py  --  Phase 2: what does GRU-Baseline actually get wrong?
==========================================================================

No training. No architecture. This loads the frozen GRU-Baseline checkpoint,
runs it over the VALIDATION subjects, and exports one row per window with the
prediction plus a set of trajectory descriptors computed for ANALYSIS ONLY.

Nothing computed here is fed back into the model. These columns exist to
answer the Phase 2 questions:

  FP: does glucose approach 70 and rebound, or hover just above it, or is a
      subgroup of patients producing most of the errors?
  FN: are misses fast drops from a high level (130->120->108->94), or slow
      persistent drifts (82->80->77->74)?

WHY VALIDATION AND NOT TEST
---------------------------
The test set is development evidence at this point (18 architectures were
selected after inspecting it) and is reserved for the single frozen final
model. Every Phase 2-7 decision is made on validation.

WHAT COMES OUT
--------------
  windows_<split>.csv      one row per (subject, window, horizon)
  subject_panel_<split>.csv per-subject metric panel
  report_<split>.txt       the FP/FN characterization, printed and saved

ADAPTER
-------
The only project-specific part is load_predictions() below. Point it at your
existing loader/checkpoint. Everything downstream is schema-driven and needs
no further edits.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

HYPO_MGDL = 70.0

# Trajectory descriptors are computed over the tail of the input window.
# 6 CGM points at 5 min = 30 min of recent context.
RECENT_N = 6


# ----------------------------------------------------------------------------
# ADAPTER -- the only part that touches project internals
# ----------------------------------------------------------------------------
@dataclass
class Batch:
    """What the adapter must return, all aligned on axis 0 = window index."""
    subject_id: np.ndarray      # (N,)   subject identifier, any dtype
    horizon: np.ndarray         # (N,)   15 / 30 / 60 / 120
    y_true: np.ndarray          # (N,)   0/1
    y_prob: np.ndarray          # (N,)   GRU-Baseline probability
    glucose_hist: np.ndarray    # (N, T) CGM history in mg/dL, UNNORMALIZED
    glucose_fut: np.ndarray     # (N, H) CGM future over the horizon, mg/dL
                                #        may be a masked/ragged array; NaN-pad


def load_predictions(split: str, ckpt: str, config: str) -> Batch:
    """
    Adapter. Replace the body with your project's loader.

    Sketch against the existing repo:

        from interaction_screen import seed_everything, make_loader
        seed_everything(42)
        loader = make_loader(split=split, config=config, shuffle=False)
        model  = load_gru_baseline(ckpt).eval().cuda()

        rows = []
        with torch.no_grad():
            for xb, yb, meta in loader:
                p = torch.sigmoid(model(xb.cuda())).cpu().numpy()
                rows.append((meta, yb.numpy(), p))

    The critical requirement is that glucose_hist and glucose_fut are in
    RAW mg/dL, not z-scored. If your loader only emits normalized tensors,
    invert with the stored train-split mean/std before returning. Every
    threshold in this file is in clinical units and will be silently wrong
    on normalized input -- assert_units() below will catch it.
    """
    raise NotImplementedError(
        "Wire load_predictions() to your loader. See docstring for a sketch."
    )


def assert_units(b: Batch) -> None:
    """Fail loudly if glucose looks normalized rather than mg/dL."""
    finite = b.glucose_hist[np.isfinite(b.glucose_hist)]
    if finite.size == 0:
        raise ValueError("glucose_hist is entirely non-finite")
    med = float(np.median(finite))
    if not (40.0 < med < 400.0):
        raise ValueError(
            f"median glucose_hist = {med:.3f}, which is not mg/dL. "
            "Invert normalization in the adapter before returning."
        )


# ----------------------------------------------------------------------------
# Trajectory descriptors -- analysis only, never inputs
# ----------------------------------------------------------------------------
def _slope_accel(hist: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Least-squares slope (mg/dL per 5 min) and acceleration over the last
    RECENT_N points. Vectorized: one polyfit over all windows at once, not
    a per-row loop.
    """
    tail = hist[:, -RECENT_N:]
    t = np.arange(RECENT_N, dtype=float)

    # slope via closed-form LS on a fixed design -- no per-row fit needed
    t_c = t - t.mean()
    denom = (t_c ** 2).sum()
    slope = ((tail - tail.mean(axis=1, keepdims=True)) * t_c).sum(axis=1) / denom

    # acceleration = slope(second half) - slope(first half)
    half = RECENT_N // 2
    a = tail[:, :half]
    c = tail[:, half:]
    ta = np.arange(a.shape[1], dtype=float)
    ta_c = ta - ta.mean()
    sa = ((a - a.mean(axis=1, keepdims=True)) * ta_c).sum(axis=1) / (ta_c ** 2).sum()
    tc = np.arange(c.shape[1], dtype=float)
    tc_c = tc - tc.mean()
    sc = ((c - c.mean(axis=1, keepdims=True)) * tc_c).sum(axis=1) / (tc_c ** 2).sum()
    accel = sc - sa

    return slope, accel


def _time_to_first_hypo(fut: np.ndarray, step_min: int = 5) -> np.ndarray:
    """Minutes until the first future reading below 70; NaN if none."""
    below = fut < HYPO_MGDL
    any_below = below.any(axis=1)
    first = np.where(any_below, below.argmax(axis=1), -1)
    out = np.full(fut.shape[0], np.nan)
    out[any_below] = (first[any_below] + 1) * step_min
    return out


def build_window_table(b: Batch, threshold: float) -> pd.DataFrame:
    assert_units(b)
    slope, accel = _slope_accel(b.glucose_hist)

    with np.errstate(invalid="ignore"):
        fut_min = np.nanmin(b.glucose_fut, axis=1)

    y_pred = (b.y_prob >= threshold).astype(int)
    outcome = np.where(
        (y_pred == 1) & (b.y_true == 1), "TP",
        np.where((y_pred == 1) & (b.y_true == 0), "FP",
                 np.where((y_pred == 0) & (b.y_true == 1), "FN", "TN")))

    df = pd.DataFrame({
        "subject_id": b.subject_id,
        "horizon": b.horizon,
        "y_true": b.y_true.astype(int),
        "y_prob": b.y_prob,
        "y_pred": y_pred,
        "outcome": outcome,
        "glucose_now": b.glucose_hist[:, -1],
        "recent_min": np.nanmin(b.glucose_hist[:, -RECENT_N:], axis=1),
        "recent_max": np.nanmax(b.glucose_hist[:, -RECENT_N:], axis=1),
        "recent_std": np.nanstd(b.glucose_hist[:, -RECENT_N:], axis=1),
        "slope_per5min": slope,
        "accel": accel,
        "dist_from_70": b.glucose_hist[:, -1] - HYPO_MGDL,
        "future_nadir": fut_min,
        "min_to_first_hypo": _time_to_first_hypo(b.glucose_fut),
    })

    # rebound: dips toward 70 during the horizon but the nadir stays above it
    df["near_miss_negative"] = (
        (df.y_true == 0) & (df.future_nadir < 85.0) & (df.future_nadir >= HYPO_MGDL)
    )
    return df


# ----------------------------------------------------------------------------
# Per-subject metric panel -- frozen set, reported for every experiment
# ----------------------------------------------------------------------------
def _safe(fn, y, p, default=np.nan):
    y = np.asarray(y)
    if y.size == 0 or len(np.unique(y)) < 2:
        return default
    try:
        return float(fn(y, p))
    except Exception:
        return default


def subject_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (sid, h), g in df.groupby(["subject_id", "horizon"], sort=True):
        tp = int(((g.y_pred == 1) & (g.y_true == 1)).sum())
        fp = int(((g.y_pred == 1) & (g.y_true == 0)).sum())
        fn = int(((g.y_pred == 0) & (g.y_true == 1)).sum())
        tn = int(((g.y_pred == 0) & (g.y_true == 0)).sum())
        out.append({
            "subject_id": sid,
            "horizon": h,
            "n_windows": len(g),
            "n_pos": int(g.y_true.sum()),
            "prevalence": float(g.y_true.mean()),
            "auprc": _safe(average_precision_score, g.y_true, g.y_prob),
            "auroc": _safe(roc_auc_score, g.y_true, g.y_prob),
            "brier": _safe(brier_score_loss, g.y_true, g.y_prob),
            "recall": tp / (tp + fn) if (tp + fn) else np.nan,
            "precision": tp / (tp + fp) if (tp + fp) else np.nan,
            "false_alarm_rate": fp / (fp + tn) if (fp + tn) else np.nan,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        })
    p = pd.DataFrame(out)
    p["f1"] = 2 * p.precision * p.recall / (p.precision + p.recall)
    return p


def subject_bootstrap_ci(panel: pd.DataFrame, col: str, horizon,
                         n_boot: int = 2000, seed: int = 42) -> tuple:
    """Subject-level bootstrap. Window-level inflates CIs ~143x here."""
    v = panel.loc[panel.horizon == horizon, col].dropna().to_numpy()
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return float(v.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------------------
# The Phase 2 questions
# ----------------------------------------------------------------------------
def characterize(df: pd.DataFrame, panel: pd.DataFrame) -> str:
    L: list[str] = []
    add = L.append

    add("=" * 78)
    add("PHASE 2 -- GRU-Baseline error analysis (validation)")
    add("=" * 78)

    for h in sorted(df.horizon.unique()):
        d = df[df.horizon == h]
        add(f"\n\n{'=' * 78}\nHORIZON {h} min   "
            f"({len(d):,} windows, prevalence {d.y_true.mean():.3%})\n{'=' * 78}")

        m, lo, hi = subject_bootstrap_ci(panel, "auprc", h)
        add(f"per-subject AUPRC  {m:.4f}  [{lo:.4f}, {hi:.4f}]")
        for c in ("auroc", "recall", "precision", "f1", "false_alarm_rate", "brier"):
            mm, ll, hh = subject_bootstrap_ci(panel, c, h)
            add(f"per-subject {c:<17} {mm:.4f}  [{ll:.4f}, {hh:.4f}]")

        # ---- concentration: is the error mass in a few subjects? ----
        add("\n-- error concentration across subjects --")
        ps = panel[panel.horizon == h].sort_values("auprc")
        q = ps.auprc.quantile([0.0, 0.25, 0.5, 0.75, 1.0])
        add(f"AUPRC quartiles  min {q[0.0]:.3f} | q1 {q[0.25]:.3f} | "
            f"med {q[0.5]:.3f} | q3 {q[0.75]:.3f} | max {q[1.0]:.3f}")
        add(f"worst-quartile mean AUPRC {ps.auprc.head(max(1, len(ps)//4)).mean():.4f} "
            f"(clinically the number that matters for an alarm)")

        err = d[d.outcome.isin(["FP", "FN"])]
        if len(err):
            share = err.groupby("subject_id").size().sort_values(ascending=False)
            top = share.head(5).sum() / share.sum()
            add(f"top-5 subjects hold {top:.1%} of all FP+FN "
                f"({len(share)} subjects contribute errors)")
            add("worst 5: " + ", ".join(
                f"{s}({n})" for s, n in share.head(5).items()))

        # ---- false positives ----
        fp = d[d.outcome == "FP"]
        add(f"\n-- FALSE POSITIVES  (n={len(fp):,}) --")
        if len(fp):
            nad = fp.future_nadir.dropna()
            add(f"future nadir     median {nad.median():.1f} mg/dL  "
                f"iqr [{nad.quantile(.25):.1f}, {nad.quantile(.75):.1f}]")
            add(f"  nadir 70-80     {(nad.between(70, 80)).mean():6.1%}   <- near-miss, arguably right")
            add(f"  nadir 80-100    {(nad.between(80, 100)).mean():6.1%}")
            add(f"  nadir >100      {(nad > 100).mean():6.1%}   <- genuinely wrong")
            add(f"glucose at t=0   median {fp.glucose_now.median():.1f} mg/dL")
            add(f"recent slope     median {fp.slope_per5min.median():+.2f} mg/dL per 5 min")
            add(f"  falling (<0)    {(fp.slope_per5min < 0).mean():6.1%}")
            add(f"  rebounding      {((fp.slope_per5min < 0) & (fp.accel > 0)).mean():6.1%}"
                "   <- dropping then decelerating")
            add(f"hovering 70-85 flat "
                f"{((fp.glucose_now.between(70, 85)) & (fp.slope_per5min.abs() < 0.5)).mean():6.1%}")
            hi_conf = fp[fp.y_prob > 0.9]
            add(f"high-confidence FP (p>0.9): {len(hi_conf):,} "
                f"({len(hi_conf)/max(1,len(fp)):.1%}), median nadir "
                f"{hi_conf.future_nadir.median() if len(hi_conf) else float('nan'):.1f}")

        # ---- false negatives ----
        fn = d[d.outcome == "FN"]
        add(f"\n-- FALSE NEGATIVES  (n={len(fn):,}) --")
        if len(fn):
            add(f"glucose at t=0   median {fn.glucose_now.median():.1f} mg/dL  "
                f"iqr [{fn.glucose_now.quantile(.25):.1f}, {fn.glucose_now.quantile(.75):.1f}]")
            add(f"future nadir     median {fn.future_nadir.median():.1f} mg/dL")
            add(f"  nadir <54 (severe) {(fn.future_nadir < 54).mean():6.1%}   <- the costly misses")
            add(f"recent slope     median {fn.slope_per5min.median():+.2f} mg/dL per 5 min")

            fast_high = (fn.glucose_now > 110) & (fn.slope_per5min < -2.0)
            slow_low = (fn.glucose_now.between(75, 95)) & (fn.slope_per5min.between(-1.5, 0))
            flat_far = (fn.glucose_now > 110) & (fn.slope_per5min.abs() < 0.5)
            add(f"\nmode A  fast drop from high   (>110, slope<-2)   {fast_high.mean():6.1%}"
                "   e.g. 130->120->108->94")
            add(f"mode B  slow persistent drift (75-95, slow neg)  {slow_low.mean():6.1%}"
                "   e.g. 82->80->77->74")
            add(f"mode C  flat and far from 70                     {flat_far.mean():6.1%}"
                "   -- unpredictable from CGM alone")
            add(f"unclassified                                     "
                f"{1 - fast_high.mean() - slow_low.mean() - flat_far.mean():6.1%}")
            add(f"\nmedian model probability on FN: {fn.y_prob.median():.3f} "
                "(near threshold => a threshold problem; near 0 => a representation problem)")

        # ---- the Phase 3 premise, measured ----
        add("\n-- supervision-collapse check (motivates the nadir head) --")
        pos = d[d.y_true == 1].future_nadir.dropna()
        neg = d[d.y_true == 0].future_nadir.dropna()
        if len(pos) and len(neg):
            add(f"positives: nadir range {pos.min():.0f}-{pos.max():.0f}, "
                f"iqr [{pos.quantile(.25):.0f}, {pos.quantile(.75):.0f}]  "
                f"-- all collapsed to y=1")
            add(f"negatives: nadir range {neg.min():.0f}-{neg.max():.0f}, "
                f"iqr [{neg.quantile(.25):.0f}, {neg.quantile(.75):.0f}]  "
                f"-- all collapsed to y=0")
            add(f"hard negatives (nadir 70-85): {(neg.between(70, 85)).mean():.1%} of negatives")
            add(f"severe positives (nadir <54): {(pos < 54).mean():.1%} of positives")

    add("\n" + "=" * 78)
    add("READ THIS BEFORE ACTING")
    add("=" * 78)
    add("Descriptors above are analysis-only and are NOT model inputs.")
    add("If FPs are dominated by nadir 70-80, many 'errors' are near-misses and")
    add("  the label boundary, not the model, is the limiting factor.")
    add("If FNs are mode C (flat, far from 70), no architecture recovers them:")
    add("  that is the information ceiling the screens already suggested.")
    add("If errors concentrate in a few subjects, Phase 5 (patient balancing)")
    add("  moves ahead of Phase 4 in priority.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val", choices=["val", "train"],
                    help="test is reserved for the single frozen final model")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default="gru_ta_absolute")
    ap.add_argument("--threshold", type=float, required=True,
                    help="the frozen operating threshold from GRU-Baseline")
    ap.add_argument("--outdir", default="results/error_analysis")
    args = ap.parse_args()

    if args.split == "test":
        raise SystemExit("test split is withheld until the architecture is frozen")

    os.makedirs(args.outdir, exist_ok=True)
    b = load_predictions(args.split, args.ckpt, args.config)
    df = build_window_table(b, args.threshold)
    panel = subject_panel(df)
    report = characterize(df, panel)

    df.to_csv(f"{args.outdir}/windows_{args.split}.csv", index=False)
    panel.to_csv(f"{args.outdir}/subject_panel_{args.split}.csv", index=False)
    with open(f"{args.outdir}/report_{args.split}.txt", "w") as f:
        f.write(report)

    summary = {}
    for h in sorted(df.horizon.unique()):
        summary[str(h)] = {
            c: subject_bootstrap_ci(panel, c, h)[0]
            for c in ("auprc", "auroc", "recall", "precision", "f1",
                      "false_alarm_rate", "brier")
        }
    with open(f"{args.outdir}/baseline_panel.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print(f"\nwrote -> {args.outdir}/")
    print("baseline_panel.json is the frozen reference every later phase is "
          "compared against.")


if __name__ == "__main__":
    main()
