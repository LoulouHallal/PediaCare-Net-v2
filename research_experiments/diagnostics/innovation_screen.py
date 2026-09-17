"""
innovation_screen.py
======================
Two cheap checks before any more GPU time.

1. REPRODUCIBILITY
   Two nominally identical GRU runs (seed 42, same code) differed by
   +0.00173 mean AUPRC — larger than the entire effect of the last four
   proposed architectures (DRS +0.0004, PAC +0.0011, KEW -0.0020,
   ADEW -0.0013). Until that spread is known and controlled, a claimed
   +0.002 architectural gain cannot be distinguished from run noise.

   This reports the spread between whatever GRU checkpoints exist on disk,
   and prints the determinism settings needed to shrink it.

2. THE INNOVATION HYPOTHESIS, WITHOUT BUILDING THE ARCHITECTURE
   IGRU / IRNN feed recent prediction error back into the recurrence. The
   claim for this task would be: when the recent glucose trajectory
   surprises a simple causal predictor, that surprise carries information
   about impending hypoglycaemia which the trained model has not already
   captured.

   That is testable directly. Form a causal one-step prediction from the
   preceding readings, take the residual

       eps_t = g_t - ghat_t

   summarise it over each window, then fit logistic regression on the
   trained model's own logit and ask whether adding the innovation
   features improves AUPRC. Nothing is trained; one forward pass is
   already cached in the saved probabilities.

       lift ~ 0        the innovation carries nothing the model lacks, so
                       an innovation-corrected cell has nothing to exploit

       lift clear      a genuine source of unused signal, and the first
                       one found in this project since TA features

   Calibration matters here: two REDUNDANT predictors still yield about
   +0.005 from logistic regression alone, purely from extra free
   parameters (measured in the AR-RHU fusion diagnostic). The threshold is
   set above that floor rather than at it.

Usage:
    python innovation_screen.py --horizon 30
"""

import json
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "innovation"
REDUNDANT_FLOOR = 0.005          # measured, see docstring


def gru_run_spread():
    """How far apart are nominally identical GRU runs on disk?"""
    rows = []
    for sub in ["adew_gru", "kew_gru", "drs_gru", "pac_gru", "ar_rhu",
                "absolute_state"]:
        d = config.RESULTS / "RQ2_models" / sub
        if not d.exists():
            continue
        for f in sorted(d.glob("*.json")):
            if f.name.startswith("_"):
                continue
            try:
                r = json.load(open(f))
            except Exception:
                continue
            if r.get("model") not in ("gru", "gru_abs", "gru_ta_abs"):
                continue
            hs = r.get("horizons", {})
            if not all(str(h) in hs for h in HORIZONS):
                continue
            ap = [hs[str(h)]["per_subject_mean"]["auprc"] for h in HORIZONS]
            rows.append((f"{sub}/{r['model']}", r.get("n_params"),
                         ap, float(np.mean(ap))))
    return rows


def causal_innovation(g, sub_ids, order=3):
    """
    One-step prediction from the preceding `order` readings, per subject,
    never across a subject boundary. Returns eps_t = g_t - ghat_t.

    A least-squares line through `order` points on a fixed integer grid,
    extrapolated one step, is a FIXED LINEAR FILTER -- the weights do not
    depend on the data:

        xbar = (k-1)/2,  Sxx = sum (x - xbar)^2
        w_i  = 1/k + (i - xbar)(k - xbar)/Sxx
        ghat = sum_i w_i * g_{t-k+i}

    Fitting per point with polyfit instead costs ~60 us each, which is
    18 minutes over 18.1M readings. The filter form is 2.5 seconds and
    agrees to 2.3e-13. The predictor is deliberately simple: the question
    is whether the RESIDUAL carries signal, not how good the predictor is.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    k = int(order)
    x = np.arange(k, dtype=np.float64)
    xb = (k - 1) / 2.0
    w = 1.0 / k + (x - xb) * (k - xb) / ((x - xb) @ (x - xb))

    g = g.astype(np.float64)
    pred = np.full(len(g), np.nan)
    for si in np.unique(sub_ids):
        m = np.flatnonzero(sub_ids == si)
        if len(m) <= k:
            continue
        gg = g[m]
        out = np.full(len(gg), np.nan)
        out[k:] = sliding_window_view(gg, k)[:-1] @ w
        pred[m] = out
    return g - pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--probs", default=None,
                    help="path to a saved *_probs.npz; defaults to the "
                         "adew_gru or kew_gru reference run")
    ap.add_argument("--order", type=int, default=3)
    ap.add_argument("--n_boot", type=int, default=400)
    args = ap.parse_args()

    print(f"{'#'*88}\n# 1. REPRODUCIBILITY OF THE GRU BASELINE\n{'#'*88}\n")
    rows = gru_run_spread()
    if len(rows) < 2:
        print("  fewer than two GRU runs on disk; skipping")
    else:
        print(f"  {'run':>28} {'params':>8} " +
              "".join(f"{f'h{h}':>9}" for h in HORIZONS) + f"{'mean':>10}")
        for nm, npar, aps, mu in rows:
            print(f"  {nm:>28} {npar or 0:>8,} " +
                  "".join(f"{a:>9.4f}" for a in aps) + f"{mu:>10.5f}")
        mus = np.array([r[3] for r in rows])
        print(f"\n  spread across nominally comparable GRU runs: "
              f"{mus.max()-mus.min():+.5f} mean AUPRC")
        print(f"  for reference, the last four proposed architectures gave")
        print(f"    DRS +0.0004   PAC +0.0011   KEW -0.0020   ADEW -0.0013")
        if mus.max() - mus.min() > 0.001:
            print(f"\n  -> the baseline moves by more than the architectures do.")
            print(f"     Any future gain below that spread is uninterpretable")
            print(f"     from a single seed.")

    print(f"""
  To tighten this before the next comparison:

      random.seed(s); np.random.seed(s)
      torch.manual_seed(s); torch.cuda.manual_seed_all(s)
      torch.backends.cudnn.benchmark = False
      torch.backends.cudnn.deterministic = True
      torch.use_deterministic_algorithms(True)
      DataLoader(..., generator=g, worker_init_fn=seed_worker)

  Note the two runs above also differed because one RESUMED after a
  disconnect: resuming restores optimiser state but not the DataLoader
  shuffle order, so the trajectory diverges from that epoch on.""")

    # ---- 2. innovation ----------------------------------------------------
    from pathlib import Path
    zb = Path(args.probs) if args.probs else None
    if zb is None:
        for cand in [config.RESULTS / "RQ2_models" / "adew_gru" / "gru__s42_probs.npz",
                     config.RESULTS / "RQ2_models" / "kew_gru" / "gru__s42_probs.npz",
                     config.RESULTS / "RQ2_models" / "drs_gru" / "gru_abs__s42_probs.npz"]:
            if cand.exists():
                zb = cand
                break
    if zb is None or not zb.exists():
        print("\n  no saved GRU probabilities found; run a gru arm first")
        return

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    z = np.load(zb)
    idx = z["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p = z["test"][:, j]

    print(f"\n\n{'#'*88}")
    print(f"# 2. DOES PREDICTION ERROR CARRY SIGNAL THE MODEL LACKS?  "
          f"(h={args.horizon})")
    print(f"{'#'*88}\n")
    print(f"  using {zb.parent.name}/{zb.name}, {len(idx):,} test windows, "
          f"base rate {y.mean():.4f}")

    eps = causal_innovation(bw.raw_gluc, bw.reading_subject, args.order)
    ends = bw.starts[idx] + bw.window_len - 1
    wl = bw.window_len

    # summarise the residual over each window: recent, mean and extreme
    feats, names = [], []
    e_last = eps[ends]
    e_prev = eps[np.maximum(ends - 1, 0)]
    seg = np.stack([eps[np.maximum(ends - k, 0)] for k in range(12)], 1)
    feats += [e_last, np.abs(e_last), e_prev,
              np.nanmean(seg, 1), np.nanmax(np.abs(seg), 1),
              np.nanstd(seg, 1)]
    names += ["eps_last", "|eps_last|", "eps_prev",
              "eps_mean_1h", "max|eps|_1h", "sd_eps_1h"]
    X = np.column_stack(feats)
    ok = np.isfinite(X).all(1) & np.isfinite(p)
    X, yy, pp = X[ok], y[ok], p[ok]
    print(f"  usable windows: {ok.sum():,}\n")

    print(f"  {'feature':>14} {'negatives':>11} {'positives':>11} "
          f"{'AUROC alone':>12}")
    for k, nm in enumerate(names):
        v = X[:, k]
        try:
            a = roc_auc_score(yy, v)
        except ValueError:
            a = float("nan")
        print(f"  {nm:>14} {v[yy == 0].mean():>11.4f} "
              f"{v[yy == 1].mean():>11.4f} {a:>12.4f}")

    lp = np.log(np.clip(pp, 1e-6, 1 - 1e-6) / (1 - np.clip(pp, 1e-6, 1 - 1e-6)))
    base = LogisticRegression(max_iter=1000).fit(lp.reshape(-1, 1), yy)
    both = LogisticRegression(max_iter=1000).fit(
        np.column_stack([lp, X]), yy)
    pb = base.predict_proba(lp.reshape(-1, 1))[:, 1]
    pj = both.predict_proba(np.column_stack([lp, X]))[:, 1]
    ap_b, ap_j = average_precision_score(yy, pb), average_precision_score(yy, pj)

    print(f"\n  {'model':>26} {'AUPRC':>9} {'AUROC':>9}")
    print(f"  {'GRU score alone':>26} {ap_b:>9.4f} {roc_auc_score(yy, pb):>9.4f}")
    print(f"  {'+ innovation features':>26} {ap_j:>9.4f} "
          f"{roc_auc_score(yy, pj):>9.4f}")
    print(f"  {'lift':>26} {ap_j-ap_b:>+9.4f}")

    rng = np.random.default_rng(0)
    reps = []
    for _ in range(args.n_boot):
        b = rng.integers(0, len(yy), len(yy))
        if len(np.unique(yy[b])) < 2:
            continue
        reps.append(average_precision_score(yy[b], pj[b]) -
                    average_precision_score(yy[b], pb[b]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    print(f"  {'95% CI':>26} [{lo:+.4f}, {hi:+.4f}]"
          f"{'*' if (lo > 0 or hi < 0) else ''}")

    d = ap_j - ap_b
    print(f"\n{'#'*88}\nREADING THIS\n{'#'*88}\n")
    print(f"  Calibration: two fully REDUNDANT predictors still give about")
    print(f"  +{REDUNDANT_FLOOR:.3f} from logistic regression alone, purely from")
    print(f"  the extra free parameters. The bar is therefore above that.\n")
    if d >= 0.010 and lo > 0:
        print(f"  -> {d:+.4f} lift, clearly above the floor. Prediction error")
        print(f"     carries information the model does not already have.")
        print(f"     An innovation-corrected cell has something to exploit,")
        print(f"     and this is the first unused signal found since TA.")
    elif d >= REDUNDANT_FLOOR:
        print(f"  -> {d:+.4f} lift, at roughly the size two redundant")
        print(f"     predictors would produce. Not evidence of new signal.")
    else:
        print(f"  -> {d:+.4f} lift. The residual of a causal one-step")
        print(f"     predictor tells us nothing the trained model has not")
        print(f"     already captured, so feeding it back into the recurrence")
        print(f"     would have nothing to correct with.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, f"_innovation_h{args.horizon}", {
        "horizon": args.horizon, "probs": str(zb), "n": int(ok.sum()),
        "auprc_base": float(ap_b), "auprc_joint": float(ap_j),
        "lift": float(d), "lo": float(lo), "hi": float(hi),
        "gru_runs": [{"run": r[0], "mean_auprc": r[3]} for r in rows]})
    print(f"\n✓ Saved -> {OUT_DIR / f'_innovation_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
