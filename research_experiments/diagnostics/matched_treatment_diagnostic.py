"""
matched_treatment_diagnostic.py
=================================
Does the treatment-history difference between detected and jointly-missed
events survive matching on current glucose and velocity?

THE CONFOUND THIS TESTS
-----------------------
The unmatched comparison found missed events carrying +0.330 U more
insulin on board, with 22.1% having a bolus in the previous 30 minutes
against 8.0% of detected events.

But the two groups differ in current state: missed events sit near
119.8 mg/dL, detected ones near 77.3. A child at 120 mg/dL is naturally
more likely to have just received a correction bolus. So part of that
+0.330 U may simply reflect

    higher glucose  ->  more insulin

rather than insulin being independent information the models failed to
use. If the entire difference is explained that way, an architecture
built around treatment memory would be built on a confound.

THE CONTROL
-----------
Each missed event is matched to detected events with nearly the same
observable state at prediction time:

    |G_detected - G_missed|       <= 5 mg/dL
    |vel_detected - vel_missed|   <= 0.2 mg/dL/min

so both groups look the same to a model reading only glucose. Any
remaining treatment difference is information the CGM trajectory does not
carry.

Two matching modes are reported:

    pooled     matches drawn from any subject
    subject    matches drawn from the SAME child only

The subject-stratified version is stricter and also removes per-patient
differences in insulin regimen, body weight and carbohydrate ratio. It
retains fewer events, so both are shown.

WHAT WOULD SETTLE IT
--------------------
If matched missed events still carry significantly more active insulin,
the treatment history contains predictive information that survives
conditioning on everything the glucose channel shows -- a genuine
architectural target.

If the difference vanishes after matching, the association was driven by
current glucose, and no treatment-memory architecture is justified.

Usage:
    python matched_treatment_diagnostic.py --horizon 30
    python matched_treatment_diagnostic.py --horizon 30 --g_tol 3 --v_tol 0.15
"""

import json
import argparse
import numpy as np

import config
import common
from treatment_diagnostics import (build_raw_treatment, insulin_on_board,
                                   TAP_DIR, DMS_DIR)

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "treatment"


def match_events(g, v, det_idx, miss_idx, g_tol, v_tol, subj=None,
                 max_per_case=5, seed=0):
    """
    For each missed event, collect detected events with similar current
    glucose and velocity. Returns (missed kept, matched detected indices,
    weights) where each missed event contributes equally regardless of how
    many matches it found -- otherwise events in dense regions would
    dominate the comparison.
    """
    rng = np.random.default_rng(seed)
    kept_miss, matched_det, weights = [], [], []
    order = np.argsort(g[det_idx])
    d_sorted = det_idx[order]
    g_sorted = g[d_sorted]

    for m in miss_idx:
        lo = np.searchsorted(g_sorted, g[m] - g_tol, "left")
        hi = np.searchsorted(g_sorted, g[m] + g_tol, "right")
        cand = d_sorted[lo:hi]
        if len(cand) == 0:
            continue
        cand = cand[np.abs(v[cand] - v[m]) <= v_tol]
        if subj is not None and len(cand):
            cand = cand[subj[cand] == subj[m]]
        if len(cand) == 0:
            continue
        if len(cand) > max_per_case:
            cand = rng.choice(cand, max_per_case, replace=False)
        kept_miss.append(m)
        matched_det.append(cand)
        weights.append(1.0 / len(cand))       # equal weight per missed case
    return np.array(kept_miss), matched_det, np.array(weights)


def weighted_stats(vals, matched_det, weights):
    """Mean over matched controls, weighting each missed case equally."""
    per_case = np.array([vals[c].mean() for c in matched_det])
    return float(np.average(per_case, weights=np.ones_like(weights)))


def boot_delta(vals, kept_miss, matched_det, n_boot=2000, seed=0,
               g=None, v=None):
    """
    Paired bootstrap over MATCHED CASES: resample the missed events and
    carry their controls with them, so the pairing is preserved.

    Residual confounding is removed first when g and v are supplied.
    Matching to a tolerance leaves a small imbalance INSIDE each window --
    a validation run injecting a purely glucose-driven quantity still
    produced a significant +0.046 difference at +/-5 mg/dL. Regressing the
    quantity on glucose and velocity and taking residuals removes that
    leakage, so a difference that survives is not explainable by current
    state.
    """
    rng = np.random.default_rng(seed)
    x = vals.astype(np.float64)

    if g is not None and v is not None:
        # least squares on [1, g, v, g^2, g*v] over every event used here
        used = np.concatenate([kept_miss] + [np.asarray(c) for c in matched_det])
        used = np.unique(used)
        A = np.column_stack([np.ones(len(used)), g[used], v[used],
                             g[used] ** 2, g[used] * v[used]])
        beta, *_ = np.linalg.lstsq(A, x[used], rcond=None)
        full = np.column_stack([np.ones(len(x)), g, v, g ** 2, g * v])
        x = x - full @ beta                      # residual after state

    m_vals = x[kept_miss]
    c_vals = np.array([x[c].mean() for c in matched_det])
    d = m_vals - c_vals
    reps = np.array([rng.choice(d, len(d), replace=True).mean()
                     for _ in range(n_boot)])
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)



def empirical_null_floor(g, v, kept, matched, n_funcs=40, n_boot=400, seed=0):
    """
    Estimate how large a residualised delta can be when the quantity is a
    pure function of current state plus noise -- i.e. when there is no
    independent treatment effect at all.

    Tolerance matching cannot remove confounding entirely: a first
    calibration using ONE synthetic quantity produced +0.050 where the
    truth was zero. Relying on a single realisation is fragile, so this
    draws many null quantities with different dependencies on glucose and
    velocity (linear, quadratic, interaction, saturating, threshold) and
    different noise levels, and reports the 95th percentile of the
    absolute deltas.

    A real effect must clear that percentile to be distinguishable from
    residual confounding, however tight its confidence interval.
    """
    rng = np.random.default_rng(seed)
    gz = (g - g.mean()) / max(g.std(), 1e-9)
    vz = (v - v.mean()) / max(v.std(), 1e-9)
    deltas = []
    for k in range(n_funcs):
        form = k % 5
        if form == 0:
            base = gz
        elif form == 1:
            base = gz ** 2
        elif form == 2:
            base = gz * vz
        elif form == 3:
            base = np.tanh(gz)
        else:
            base = (gz > rng.normal(0, 0.5)).astype(float)
        scale = rng.uniform(0.2, 1.5)
        noise = rng.uniform(0.2, 1.0)
        q = (scale * base + rng.normal(0, noise, len(g))).astype(np.float32)
        d, _, _ = boot_delta(q, kept, matched, n_boot=n_boot,
                             seed=int(rng.integers(1e6)), g=g, v=v)
        deltas.append(abs(d))
    deltas = np.array(deltas)
    return {"p50": float(np.percentile(deltas, 50)),
            "p95": float(np.percentile(deltas, 95)),
            "max": float(deltas.max()), "n_funcs": n_funcs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--g_tol", type=float, default=5.0,
                    help="glucose matching tolerance, mg/dL")
    ap.add_argument("--v_tol", type=float, default=0.2,
                    help="velocity matching tolerance, mg/dL/min")
    ap.add_argument("--max_per_case", type=int, default=5)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--null_floor", type=float, default=0.06,
                    help="deltas below this are treated as indistinguishable "
                         "from residual confounding; see the note below")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    basal, bolus, carbs = build_raw_treatment(bw, verbose=False)
    iob = insulin_on_board(bolus, bw.reading_subject, bw.reading_times)

    z1 = np.load(TAP_DIR / "gru_ta__s42_probs.npz")
    z2 = np.load(DMS_DIR / "tcn_ta__s42_probs.npz")
    idx = z1["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p1, p2 = z1["test"][:, j], z2["test"][:, j]
    t1 = np.quantile(p1, 1 - y.mean() * 3)
    t2 = np.quantile(p2, 1 - y.mean() * 3)
    det_mask = (p1 >= t1) & (p2 >= t2) & (y == 1)
    miss_mask = (p1 < t1) & (p2 < t2) & (y == 1)

    ends = bw.starts[idx] + bw.window_len - 1
    g = bw.raw_gluc[ends]
    v = (bw.raw_gluc[ends - 3] - g) / 15.0
    subj = bw.reading_subject[ends]

    det_idx = np.flatnonzero(det_mask)
    miss_idx = np.flatnonzero(miss_mask)

    print(f"{'#'*96}")
    print(f"# MATCHED-STATE TREATMENT CONTROL (h={args.horizon})")
    print(f"#   matching on glucose +/-{args.g_tol} mg/dL and velocity "
          f"+/-{args.v_tol} mg/dL/min")
    print(f"{'#'*96}")
    print(f"\n  before matching: detected {len(det_idx):,}  missed {len(miss_idx):,}")
    print(f"    glucose  {g[det_idx].mean():>7.1f} vs {g[miss_idx].mean():>7.1f} mg/dL")
    print(f"    velocity {v[det_idx].mean():>7.3f} vs {v[miss_idx].mean():>7.3f} mg/dL/min")

    quantities = [("insulin on board", iob[ends], "U")]
    for mins in [30, 60, 120]:
        steps = mins // bw.sample_min
        tot = np.array([bolus[max(0, e - steps):e + 1].sum() for e in ends],
                       dtype=np.float32)
        quantities.append((f"bolus last {mins} min", tot, "U"))
    for mins in [30, 60]:
        steps = mins // bw.sample_min
        tot = np.array([carbs[max(0, e - steps):e + 1].sum() for e in ends],
                       dtype=np.float32)
        quantities.append((f"carbs last {mins} min", tot, "g"))
    steps = 12
    basal_tot = np.array([basal[max(0, e - steps):e + 1].sum() for e in ends],
                         dtype=np.float32)
    quantities.append(("basal last 60 min", basal_tot, "U"))

    results = {}
    for mode, use_subj in [("pooled", None), ("subject-stratified", subj)]:
        kept, matched, w = match_events(g, v, det_idx, miss_idx,
                                        args.g_tol, args.v_tol, use_subj,
                                        args.max_per_case)
        if len(kept) < 30:
            print(f"\n  [{mode}] only {len(kept)} matched cases -- too few")
            continue

        gm = g[kept].mean()
        gc_ = np.mean([g[c].mean() for c in matched])
        vm = v[kept].mean()
        vc = np.mean([v[c].mean() for c in matched])
        print(f"\n{'='*96}")
        print(f"{mode}: {len(kept):,} of {len(miss_idx):,} missed events matched "
              f"({np.mean([len(c) for c in matched]):.1f} controls each)")
        print(f"  balance check   glucose {gc_:>7.1f} vs {gm:>7.1f} mg/dL   "
              f"velocity {vc:>6.3f} vs {vm:>6.3f}")
        print(f"{'='*96}")
        print(f"  {'quantity':>20} {'matched det':>12} {'missed':>10} "
              f"{'delta':>9} {'95% CI':>22} {'floor':>7} {'clears':>6}")
        print(f"  (delta is residual after regressing out glucose and velocity;")
        print(f"   'floor' is the 95th percentile of null deltas in these units)")

        # the floor is estimated in STANDARDISED units, then rescaled to
        # each quantity's own spread, since a 0.06 U floor for insulin is
        # not the same as a 0.06 g floor for carbohydrate
        nf = empirical_null_floor(g, v, kept, matched, seed=1)
        print(f"  empirical null floor (standardised): p50 {nf['p50']:.4f}  "
              f"p95 {nf['p95']:.4f}  max {nf['max']:.4f}")
        results.setdefault("null_floor", {})[mode] = nf

        entry = {}
        for name, vals, unit in quantities:
            # residualise on current state so any surviving difference is
            # not explainable by glucose or velocity
            d, lo, hi = boot_delta(vals, kept, matched, args.n_boot,
                                   g=g, v=v)
            c_mean = np.mean([vals[c].mean() for c in matched])
            floor = nf["p95"] * float(np.std(vals))      # rescale to units
            sig = "*" if (lo > 0 or hi < 0) else " "
            clears = "yes" if abs(d) > floor else "no "
            print(f"  {name:>20} {c_mean:>12.3f} {vals[kept].mean():>10.3f} "
                  f"{d:>+9.3f} {f'[{lo:+.3f}, {hi:+.3f}]':>22}{sig} "
                  f"{floor:>7.3f} {clears:>6} {unit}")
            entry[name] = {"matched_detected": float(c_mean),
                           "missed": float(vals[kept].mean()),
                           "delta": d, "lo": lo, "hi": hi,
                           "significant": bool(lo > 0 or hi < 0),
                           "null_floor": float(floor),
                           "clears_floor": bool(abs(d) > floor)}

        # any-bolus fraction, matched
        for mins in [30, 60]:
            steps = mins // bw.sample_min
            anyb = np.array([(bolus[max(0, e - steps):e + 1] > 0).any()
                             for e in ends])
            cm = np.mean([anyb[c].mean() for c in matched])
            mm = anyb[kept].mean()
            print(f"  {f'any bolus {mins} min':>20} {100*cm:>11.2f}% "
                  f"{100*mm:>9.2f}% {100*(mm-cm):>+8.2f}%")
            entry[f"any_bolus_{mins}"] = {"matched_detected": float(cm),
                                          "missed": float(mm)}
        results[mode] = {"n_matched": int(len(kept)), "metrics": entry}

    print(f"\n{'#'*96}\nREADING THIS\n{'#'*96}")
    print("""
  The 'floor' column is the 95th percentile of deltas obtained from many
  synthetic quantities generated from current glucose and velocity ALONE,
  with no independent effect. Tolerance matching cannot remove confounding
  entirely, so a delta below its floor is not distinguishable from that
  residual leakage however tight its interval.

  Interpretation scale, applied to insulin on board:
      below floor        probably residual confounding
      floor .. 0.10 U    weak evidence
      > 0.10 U           meaningful, if the sign agrees across both
                         matching modes
      > 0.15 U with recent-bolus measures also clearing their floors
                         strong architectural target
""")

    # the subject-stratified result carries the most weight: it also removes
    # per-patient differences in regimen, body weight and carbohydrate ratio
    primary = "subject-stratified" if "subject-stratified" in results else "pooled"
    if primary not in results:
        print("  Too few matched cases to conclude.")
    else:
        pm = results[primary]["metrics"]
        iobr = pm["insulin on board"]
        d, floor = iobr["delta"], iobr["null_floor"]
        agree = True
        if "pooled" in results and "subject-stratified" in results:
            dp = results["pooled"]["metrics"]["insulin on board"]["delta"]
            agree = (np.sign(dp) == np.sign(d))
            print(f"  pooled IOB delta {dp:+.3f} U | subject-stratified "
                  f"{d:+.3f} U | signs {'agree' if agree else 'DISAGREE'}")
            if not agree:
                print("    -> a large pooled effect with a collapsing "
                      "same-subject effect points at regimen or body-size\n"
                      "       differences between children, not at the "
                      "mechanism.")
        bolus_clears = sum(1 for k, vv in pm.items()
                           if k.startswith("bolus") and vv.get("clears_floor"))

        print(f"\n  primary result ({primary}): IOB delta {d:+.3f} U, "
              f"floor {floor:.3f} U")
        if not iobr["significant"] or abs(d) <= floor:
            print(f"\n  -> Below the calibrated floor. After conditioning on")
            print(f"     current glucose and velocity, the treatment")
            print(f"     difference is not distinguishable from residual")
            print(f"     confounding: children at 120 mg/dL simply receive")
            print(f"     more correction insulin than children at 77.")
            print(f"     A treatment-memory architecture is NOT justified.")
        elif d > 0.15 and bolus_clears >= 2 and agree:
            print(f"\n  -> STRONG. At the same glucose, velocity and in the")
            print(f"     same child, missed events carry {d:+.3f} U more active")
            print(f"     insulin, and {bolus_clears} recent-bolus measures also")
            print(f"     clear their floors. The treatment history holds")
            print(f"     predictive information the glucose trajectory does")
            print(f"     not. This is a defensible architectural target.")
        elif d > 0.10 and agree:
            print(f"\n  -> MEANINGFUL. The effect survives matching and")
            print(f"     residualisation with consistent sign across both")
            print(f"     matching modes. Worth building on, though less")
            print(f"     decisive than the strong case.")
        else:
            print(f"\n  -> WEAK. The effect clears the floor but is small")
            print(f"     ({d:+.3f} U). I would not freeze an architecture on")
            print(f"     this alone.")

        print(f"\n  NOTE: even a surviving effect shows only that treatment")
        print(f"  history carries signal beyond current glucose. It does not")
        print(f"  show that the GRU specifically discards it. That needs an")
        print(f"  intervention on the trained model -- zeroing recent bolus")
        print(f"  while holding glucose fixed and measuring how much the")
        print(f"  prediction moves.")

    common.save_result(OUT_DIR, f"_matched_treatment_h{args.horizon}",
                       {"config": vars(args), "results": results})
    print(f"\n✓ Saved -> {OUT_DIR / f'_matched_treatment_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
