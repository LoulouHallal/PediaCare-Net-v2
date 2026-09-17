"""
interaction_screen.py
=======================
Two preconditions before any more architecture GPU time.

1. DETERMINISM
   The nn.GRU arms in this project are bit-reproducible (drs_gru, pac_gru
   and ar_rhu all report 0.8640 / 0.7528 / 0.6313 / 0.5365). The
   hand-written-loop arms are not: two nominally identical runs differed by
   0.00168 mean AUPRC. Every proposed cell runs through that loop, so the
   non-determinism sits exactly where the architecture comparisons are
   made.

   `seed_everything()` and `make_loader()` below fix it. Import them into
   any pilot that compares custom cells.

2. THE MULTIPLICATIVE-INTERACTION HYPOTHESIS
   SIF-GRU's premise is that the GRU candidate is additive before its
   nonlinearity,

       h~ = tanh(W x_t + U (r * h_{t-1}) + b)

   and therefore does not explicitly encode products like
   glucose-state x insulin-context. The proposed fix gives the cell
   f(x_t) * g(h_{t-1}).

   Part of that is testable without training: if explicit pairwise
   products of the input channels carry information the trained GRU has
   not already captured, an interaction-capable candidate has something to
   exploit. If they add nothing, the additive candidate plus tanh plus
   64 units across 60 timesteps has already represented whatever
   interactions matter.

   This does NOT test the x*h state interaction, only x*x. It is a
   necessary-condition screen, not a sufficient one: a null here does not
   strictly prove SIF-GRU would fail, but it removes the main motivating
   argument.

   Two independent probes:
     (a) logistic regression on the GRU logit, with and without the 45
         pairwise products of the 9 channels at the final timestep
     (b) gradient boosting -- an interaction machine by construction --
         on the raw channels, tested for lift over the same GRU logit

   Calibration: two fully redundant predictors still yield about +0.005
   from logistic regression alone, purely from the extra degrees of
   freedom. The bar is above that floor, not at zero.

Usage:
    python interaction_screen.py --horizon 30
    python interaction_screen.py --horizon 30 --no_gbm    # skip (b), faster
"""

import os
import json
import random
import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "interaction"
REDUNDANT_FLOOR = 0.005


# ─── 1. determinism ───────────────────────────────────────────────────────────

def seed_everything(seed: int = 42, strict: bool = True):
    """
    Call once at the top of any pilot that compares hand-written cells.

    cuDNN picks algorithms by benchmarking, and several reductions are
    non-deterministic by default, so two identical runs can diverge. That
    divergence measured 0.00168 mean AUPRC here -- larger than every
    architecture effect since AR-RHU.
    """
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if strict:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError:
            torch.use_deterministic_algorithms(True)


def _worker_init(worker_id):
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(dataset, batch_size, shuffle, seed=42, **kw):
    """DataLoader with a seeded generator and seeded workers."""
    import torch
    from torch.utils.data import DataLoader
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=g, worker_init_fn=_worker_init, **kw)


def determinism_note():
    print(f"{'#'*88}\n# 1. DETERMINISM\n{'#'*88}\n")
    print("  Measured in this project:")
    print("    nn.GRU arms (drs/pac/ar_rhu)  identical to 4 decimals")
    print("    hand-written loop arms        differ by 0.00168 mean AUPRC")
    print("\n  The custom loop is where every proposed cell lives, so that")
    print("  spread sits directly on top of the comparisons being made.\n")
    print("  Import into every pilot that compares custom cells:\n")
    print("      from interaction_screen import seed_everything, make_loader")
    print("      seed_everything(args.seed)")
    print("      dl = make_loader(ds, args.batch, shuffle=True,")
    print("                       seed=args.seed, num_workers=args.workers,")
    print("                       drop_last=True)\n")
    print("  Then run the SAME arm twice and confirm |delta| < 1e-4 before")
    print("  trusting any architecture result. Note that resuming after a")
    print("  disconnect breaks this regardless: optimiser state is restored")
    print("  but the sampler position is not.")


# ─── 2. interaction screen ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--probs", default=None)
    ap.add_argument("--no_gbm", action="store_true")
    ap.add_argument("--n_boot", type=int, default=400)
    ap.add_argument("--subsample", type=int, default=200_000)
    args = ap.parse_args()

    determinism_note()

    from pathlib import Path
    zb = Path(args.probs) if args.probs else None
    if zb is None:
        for c in [config.RESULTS / "RQ2_models" / "adew_gru" / "gru__s42_probs.npz",
                  config.RESULTS / "RQ2_models" / "drs_gru" / "gru_abs__s42_probs.npz"]:
            if c.exists():
                zb = c; break
    if zb is None or not zb.exists():
        print("\n  no saved GRU probabilities found")
        return

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    from ta_gru import attach_ta
    from absolute_state import attach_absolute
    attach_ta(bw, verbose=False)
    attach_absolute(bw, verbose=False)

    z = np.load(zb)
    idx = z["idx_test"]
    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    p = z["test"][:, j]

    print(f"\n\n{'#'*88}")
    print(f"# 2. DO CHANNEL INTERACTIONS CARRY SIGNAL THE GRU LACKS?  "
          f"(h={args.horizon})")
    print(f"{'#'*88}\n")
    print(f"  {zb.parent.name}/{zb.name}, {len(idx):,} windows, "
          f"base rate {y.mean():.4f}")

    ends = bw.starts[idx] + bw.window_len - 1
    X1 = bw.timeline[ends].astype(np.float64)          # 9 channels, final step
    nch = X1.shape[1]
    names = list(bw.features)

    pairs, pnames = [], []
    for a in range(nch):
        for b in range(a, nch):
            pairs.append(X1[:, a] * X1[:, b])
            pnames.append(f"{names[a][:6]}*{names[b][:6]}")
    X2 = np.column_stack(pairs)
    print(f"  {nch} channels -> {X2.shape[1]} pairwise products "
          f"(including squares)")

    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    ok = np.isfinite(lp) & np.isfinite(X2).all(1) & np.isfinite(X1).all(1)
    lp, yy, X1, X2 = lp[ok], y[ok], X1[ok], X2[ok]

    rng = np.random.default_rng(0)
    if args.subsample and len(yy) > args.subsample:
        s = rng.choice(len(yy), args.subsample, replace=False)
        lp, yy, X1, X2 = lp[s], yy[s], X1[s], X2[s]
    print(f"  fitting on {len(yy):,} windows\n")

    def lift(extra, label):
        base = LogisticRegression(max_iter=2000).fit(lp.reshape(-1, 1), yy)
        Xj = np.column_stack([lp, extra])
        mu, sd = Xj.mean(0), Xj.std(0) + 1e-9
        both = LogisticRegression(max_iter=2000).fit((Xj - mu) / sd, yy)
        pb = base.predict_proba(lp.reshape(-1, 1))[:, 1]
        pj = both.predict_proba((Xj - mu) / sd)[:, 1]
        ab, aj = (average_precision_score(yy, pb),
                  average_precision_score(yy, pj))
        reps = []
        for _ in range(args.n_boot):
            b = rng.integers(0, len(yy), len(yy))
            if len(np.unique(yy[b])) < 2:
                continue
            reps.append(average_precision_score(yy[b], pj[b]) -
                        average_precision_score(yy[b], pb[b]))
        lo, hi = np.percentile(reps, [2.5, 97.5])
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {label:>34} {aj:>8.4f}  lift {aj-ab:>+8.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]{sig}")
        return aj - ab, ab

    print(f"  {'model':>34} {'AUPRC':>8}")
    base_only = LogisticRegression(max_iter=2000).fit(lp.reshape(-1, 1), yy)
    ab = average_precision_score(
        yy, base_only.predict_proba(lp.reshape(-1, 1))[:, 1])
    print(f"  {'GRU score alone':>34} {ab:>8.4f}")
    d_lin, _ = lift(X1, "+ 9 raw channels (linear)")
    d_int, _ = lift(X2, "+ 45 pairwise products")
    d_all, _ = lift(np.column_stack([X1, X2]), "+ channels and products")

    d_gbm = None
    if not args.no_gbm:
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            print("\n  gradient boosting on the raw channels -- an interaction")
            print("  machine by construction, so it should find any that exist")
            n = len(yy)
            tr = rng.random(n) < 0.7
            gb = HistGradientBoostingClassifier(
                max_iter=200, max_depth=6, learning_rate=0.1,
                random_state=0).fit(X1[tr], yy[tr])
            gscore = gb.predict_proba(X1[~tr])[:, 1]
            lp_te, y_te = lp[~tr], yy[~tr]
            b2 = LogisticRegression(max_iter=2000).fit(
                lp_te.reshape(-1, 1), y_te)
            j2 = LogisticRegression(max_iter=2000).fit(
                np.column_stack([lp_te, gscore]), y_te)
            a_b = average_precision_score(
                y_te, b2.predict_proba(lp_te.reshape(-1, 1))[:, 1])
            a_j = average_precision_score(
                y_te, j2.predict_proba(np.column_stack([lp_te, gscore]))[:, 1])
            d_gbm = a_j - a_b
            print(f"  {'GBM alone':>34} "
                  f"{average_precision_score(y_te, gscore):>8.4f}")
            print(f"  {'GRU + GBM':>34} {a_j:>8.4f}  lift {d_gbm:>+8.4f}")
        except Exception as e:
            print(f"  (GBM skipped: {type(e).__name__})")

    print(f"\n{'#'*88}\nREADING THIS\n{'#'*88}\n")
    print(f"  Floor: two REDUNDANT predictors give about "
          f"+{REDUNDANT_FLOOR:.3f} from the")
    print(f"  extra degrees of freedom alone. The bar is above that.\n")
    best = max(d_int, d_all, d_gbm if d_gbm is not None else -1)
    if best >= 0.010:
        print(f"  -> {best:+.4f}. Explicit interactions carry information the")
        print(f"     GRU has not captured. SIF-GRU's multiplicative candidate")
        print(f"     has something real to exploit -- the first unused signal")
        print(f"     found since the TA features.")
    elif best >= REDUNDANT_FLOOR:
        print(f"  -> {best:+.4f}, about what redundant predictors produce.")
        print(f"     Weak; not a strong basis for the architecture.")
    else:
        print(f"  -> {best:+.4f}. Neither explicit products nor a gradient-")
        print(f"     boosted interaction model adds anything over the GRU")
        print(f"     score. The additive candidate, composed over 64 units")
        print(f"     and 60 timesteps, has already represented whatever")
        print(f"     interactions matter here.")
        print(f"\n     Note this screens x*x, not the x*h state interaction")
        print(f"     SIF-GRU actually proposes -- so it is a necessary")
        print(f"     condition, not a sufficient one. But it removes the")
        print(f"     motivating argument.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, f"_interaction_h{args.horizon}", {
        "horizon": args.horizon, "n": int(len(yy)), "auprc_base": float(ab),
        "lift_linear": float(d_lin), "lift_pairwise": float(d_int),
        "lift_all": float(d_all),
        "lift_gbm": float(d_gbm) if d_gbm is not None else None})
    print(f"\n✓ Saved -> {OUT_DIR / f'_interaction_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
