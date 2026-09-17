"""
generalization_gap.py  --  does TSL-GRU overfit less than GRU+TA?
=================================================================

Inference only. Loads the saved .pt checkpoints and scores each on a
fixed subsample of TRAINING windows and on the full validation set,
reporting per-subject mean AUPRC on both and the gap between them.

WHAT A "GAP" MEANS HERE
-----------------------
    gap = train AUPRC - validation AUPRC

Both computed with the SAME metric on the SAME model. The per-epoch
training LOSS printed during training cannot be used for this: it is a
different quantity on a different scale from validation AUPRC, and
subtracting them would be meaningless.

THE CONFOUND, STATED UP FRONT
-----------------------------
gru_ta uses StdGRUCell; the other three use LightCell, which differs in
FOUR ways at once: fewer parameters, BatchNorm on the input projection,
ReLU instead of tanh, and no reset gate. A smaller gap for tsl_gru vs
gru_ta therefore cannot be attributed to parameter count alone.

The clean comparison is WITHIN the LightCell family, where normalisation
and activation are held fixed and only size and retention differ:

    ligru_ta     28,324   update gate
    tsl_gru      15,974   trajectory-conditioned retention (kappa free)
    tsl_static   15,460   constant retention (kappa = 0 by construction)

gru_ta is reported alongside as context, with the confound noted.

TRAIN SUBSAMPLE
---------------
The training index is capped and stratified by run_one, and scoring all
of it is unnecessary. A fixed stratified subsample with a fixed seed is
used so every arm sees exactly the same windows -- otherwise arms would
be compared on different data.

INTERPRETATION THRESHOLD, FIXED BEFORE LOOKING
----------------------------------------------
Validation AUPRC in this project has a measured noise floor of 0.00207.
A DIFFERENCE IN GAPS between two arms should exceed roughly 0.01 across
all three seeds before it is called an effect. Smaller than that and it
is not distinguishable from seed variation.

    python generalization_gap.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
from stage2_deep import WindowDataset  # noqa: E402
from ta_gru import attach_ta  # noqa: E402
from tsl_gru import TSLGRU  # noqa: E402

LIGHT = ("ligru_ta", "tsl_gru", "tsl_static")


def per_subject_auprc(y, p, subj):
    """Mean over subjects of per-subject AUPRC, per horizon, then averaged."""
    out = []
    for k in range(y.shape[1]):
        vals = []
        for s in np.unique(subj):
            m = subj == s
            yy = y[m, k].astype(int)
            if len(np.unique(yy)) < 2:
                continue
            vals.append(average_precision_score(yy, p[m, k]))
        out.append(float(np.mean(vals)) if vals else float("nan"))
    return out


@torch.no_grad()
def score(model, bw, idx, Y, device, batch=2048, workers=2):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P = [torch.sigmoid(model(xb.to(device))).cpu().numpy() for xb, _ in dl]
    return np.concatenate(P).astype(np.float32)


def parse_ckpt(path):
    stem = os.path.basename(path)[:-3]
    variant = stem.split("__")[0]
    m = re.search(r"__s(\d+)$", stem)
    return variant, (int(m.group(1)) if m else 42)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="../results/RQ2_models/tsl_gru")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--n_train", type=int, default=120_000,
                    help="fixed stratified subsample of training windows")
    ap.add_argument("--sub_seed", type=int, default=12345)
    ap.add_argument("--out", default="../results/tables/T13_generalization_gap.csv")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    bw = common.load_built(a.tag)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_tr_all = np.flatnonzero(tr)
    idx_va = np.flatnonzero(va)

    # one fixed subsample, shared by every arm
    rng = np.random.default_rng(a.sub_seed)
    n = min(a.n_train, idx_tr_all.size)
    idx_tr = np.sort(rng.choice(idx_tr_all, size=n, replace=False))

    subj_tr = np.asarray(bw.meta)[idx_tr].astype(str)
    subj_va = np.asarray(bw.meta)[idx_va].astype(str)
    print(f"device {device}")
    print(f"train subsample {len(idx_tr):,} of {idx_tr_all.size:,} "
          f"({len(np.unique(subj_tr))} subjects)")
    print(f"validation      {len(idx_va):,} ({len(np.unique(subj_va))} subjects)")
    print(f"train prevalence {Y[idx_tr].mean(0).round(4).tolist()}")
    print(f"val   prevalence {Y[idx_va].mean(0).round(4).tolist()}\n")

    ckpts = sorted(glob.glob(os.path.join(a.dir, "*.pt")))
    if not ckpts:
        sys.exit(f"no checkpoints in {a.dir}")

    rows = []
    for c in ckpts:
        variant, seed = parse_ckpt(c)
        blob = torch.load(c, map_location=device, weights_only=False)
        cfg = blob.get("config", {})
        model = TSLGRU(variant, in_ch=bw.timeline.shape[1],
                       hidden=cfg.get("hidden", 64),
                       dropout=cfg.get("dropout", 0.2),
                       rho=cfg.get("rho", 0.9)).to(device)
        model.load_state_dict(blob["model_state"])
        n_par = sum(p.numel() for p in model.parameters())

        ptr = score(model, bw, idx_tr, Y, device)
        pva = score(model, bw, idx_va, Y, device)
        atr = per_subject_auprc(Y[idx_tr], ptr, subj_tr)
        ava = per_subject_auprc(Y[idx_va], pva, subj_va)

        r = {"variant": variant, "seed": seed, "params": n_par,
             "family": "LightCell" if variant in LIGHT else "StdGRUCell"}
        for j, h in enumerate(bw.horizons):
            r[f"train_h{h}"] = atr[j]
            r[f"val_h{h}"] = ava[j]
            r[f"gap_h{h}"] = atr[j] - ava[j]
        r["train_mean"] = float(np.nanmean(atr))
        r["val_mean"] = float(np.nanmean(ava))
        r["gap_mean"] = r["train_mean"] - r["val_mean"]
        rows.append(r)
        print(f"  {variant:<12} s{seed}  {n_par:>7,}  "
              f"train {r['train_mean']:.4f}  val {r['val_mean']:.4f}  "
              f"gap {r['gap_mean']:+.4f}")

    df = pd.DataFrame(rows)

    print("\n" + "=" * 78)
    print("GENERALISATION GAP  (train - validation, per-subject mean AUPRC)")
    print("=" * 78)
    agg = (df.groupby(["variant", "family", "params"])
             .agg(n=("seed", "size"), train=("train_mean", "mean"),
                  val=("val_mean", "mean"), gap=("gap_mean", "mean"),
                  gap_sd=("gap_mean", "std"))
             .reset_index().sort_values("params", ascending=False))
    print(agg.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n-- within LightCell (normalisation and activation held fixed) --")
    lc = agg[agg.family == "LightCell"]
    if len(lc) >= 2:
        base = lc.iloc[0]
        for _, r in lc.iloc[1:].iterrows():
            d = r.gap - base.gap
            print(f"  {r.variant} vs {base.variant}: "
                  f"gap {r.gap:+.4f} vs {base.gap:+.4f}  "
                  f"difference {d:+.4f}"
                  + ("   <- exceeds 0.01, worth reporting" if abs(d) > 0.01
                     else "   (below the 0.01 threshold, not an effect)"))

    print("\n-- vs gru_ta (CONFOUNDED: differs in params, BatchNorm, "
          "activation, gating) --")
    g = agg[agg.variant == "gru_ta"]
    if len(g):
        gb = g.iloc[0]
        for _, r in agg[agg.family == "LightCell"].iterrows():
            d = r.gap - gb.gap
            print(f"  {r.variant} vs gru_ta: gap {r.gap:+.4f} vs "
                  f"{gb.gap:+.4f}  difference {d:+.4f}")
        print("  Any difference here has four possible causes at once and "
              "cannot be attributed to parameter count.")

    print("\n-- per horizon (mean over seeds) --")
    cols = ["variant"] + [f"gap_h{h}" for h in bw.horizons]
    print(df.groupby("variant")[[c for c in cols if c != "variant"]]
            .mean().to_string(float_format=lambda x: f"{x:+.4f}"))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote -> {a.out}")
    print("\nNote: these are FINAL-EPOCH gaps from early-stopped checkpoints, "
          "not curves.\nA per-epoch overfitting figure would need retraining "
          "with train AUPRC logged.")


if __name__ == "__main__":
    main()
