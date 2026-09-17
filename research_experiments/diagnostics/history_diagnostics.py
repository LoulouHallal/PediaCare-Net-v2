"""
history_diagnostics.py
========================
Does the trained model use the early part of its 5-hour window, and do
gradients reach it?

These decide whether delayed-state feedback (the tau-GRU mechanism, M3)
is solving a problem this task actually has. Both run on the existing
GRU+TA+Absolute checkpoint. No retraining.

DIAGNOSTIC 1 — is the early history used?
-----------------------------------------
Naive truncation is confounded: a model trained on 60 steps and fed 24
starts its hidden state from zero much closer to the prediction point, so
a drop could reflect that distribution shift rather than lost
information. Three corruptions are therefore compared, all keeping the
window at 60 steps:

    shuffle   permute the first (60-K) steps in time. Same values, same
              length, temporal order destroyed. This is the cleanest
              test: if it costs nothing, long-range temporal STRUCTURE
              is unused.
    freeze    replace the first (60-K) steps with the value at step
              (60-K). Same length, but the early trajectory is flattened
              to its endpoint.
    truncate  feed only the last K steps. Included for completeness, and
              the one to read most cautiously.

DIAGNOSTIC 2 — do gradients reach the early history?
----------------------------------------------------
The norm of dL/dx_t as a function of t. A recurrent model that cannot
propagate gradient to early timesteps cannot learn to use them, which is
the specific failure delayed feedback is designed to repair.

Reading the two together:

    early history matters AND gradients decay
        the delayed pathway has something to fix -> M3 justified

    early history matters, gradients healthy
        the model already reaches it; a delayed path may still help as
        an inductive bias, but the case is weaker

    early history does not matter
        drop M3. Any gain from a delayed pathway would not be coming
        from the mechanism it claims.

Usage:
    python history_diagnostics.py
    python history_diagnostics.py --horizon 30
"""

import json
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import average_precision_score

import config
import common
from stage2_deep import WindowDataset, Heads
from ta_gru import attach_ta
from absolute_state import attach_absolute

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "history_diag"
DEFAULT_CKPT = config.RESULTS / "RQ2_models" / "drs_gru" / "gru_abs__s42.pt"
KEEPS = [60, 48, 36, 24, 12, 6]


class GRUBaseline(nn.Module):
    def __init__(self, in_ch=9, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


def corrupt(x, keep, mode, gen):
    """
    Degrade the first (T - keep) steps. x is [B,T,F]; the window length is
    preserved except in 'truncate'.
    """
    T = x.shape[1]
    if keep >= T:
        return x
    cut = T - keep
    if mode == "truncate":
        return x[:, cut:, :]
    y = x.clone()
    if mode == "shuffle":
        perm = torch.randperm(cut, generator=gen).to(x.device)
        y[:, :cut, :] = x[:, perm, :]
    elif mode == "freeze":
        y[:, :cut, :] = x[:, cut:cut + 1, :].expand(-1, cut, -1)
    else:
        raise ValueError(mode)
    return y


@torch.no_grad()
def score(model, bw, idx, Y, device, keep, mode, batch=2048, workers=2,
          seed=0):
    model.eval()
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    out = []
    for xb, _ in dl:
        xb = corrupt(xb, keep, mode, gen).to(device)
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def gradient_profile(model, bw, idx, Y, device, j, n=4096, batch=512):
    """
    Mean |dL/dx_t| per timestep, over a sample of windows.

    cuDNN refuses RNN backward in eval mode, so the fused kernel is
    disabled for this pass. Keeping the model in eval() rather than
    switching to train() matters: train() would enable dropout and the
    gradient profile would then be measured on a different network.
    """
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx[:n], Y), batch_size=batch,
                    shuffle=False, num_workers=0)
    acc, cnt = None, 0
    with torch.backends.cudnn.flags(enabled=False):
        for xb, yb in dl:
            xb = xb.to(device).requires_grad_(True)
            p = model(xb)[:, j].clamp(1e-6, 1 - 1e-6)
            y = yb[:, j].to(device)
            loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p)).sum()
            model.zero_grad()
            loss.backward()
            g = xb.grad.abs().mean(dim=(0, 2)).detach().cpu().numpy()   # [T]
            acc = g if acc is None else acc + g
            cnt += 1
    return acc / max(cnt, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    from pathlib import Path
    ckpt = Path(args.ckpt)
    if not ckpt.exists():
        print(f"  missing checkpoint {ckpt}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    j = HORIZONS.index(args.horizon)

    model = GRUBaseline(bw.timeline.shape[1], args.hidden).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device)["model_state"])
    print(f"Loaded {ckpt.name}\n")

    yt = Y[idx_te, j].astype(int)
    meta_te = bw.meta[idx_te]
    res = {"horizon": args.horizon, "checkpoint": ckpt.name, "history": {}}

    print(f"{'#'*88}")
    print(f"# 1. IS THE EARLY HISTORY USED?  (h={args.horizon})")
    print("#    all rows keep the 60-step window except 'truncate'")
    print(f"{'#'*88}")
    print(f"\n  {'kept':>10} {'minutes':>8}   " +
          "  ".join(f"{m:>10}" for m in ["shuffle", "freeze", "truncate"]))

    full = None
    for keep in KEEPS:
        row, cells = {}, []
        for mode in ["shuffle", "freeze", "truncate"]:
            pv = score(model, bw, idx_va, Y, device, keep, mode,
                       workers=args.workers)
            pt = score(model, bw, idx_te, Y, device, keep, mode,
                       workers=args.workers)
            thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
            ev = common.evaluate(yt, pt[:, j], meta_te, thr)
            a = ev["per_subject_mean"]["auprc"]
            row[mode] = float(a)
            cells.append(f"{a:>10.4f}")
        if keep == 60:
            full = row["shuffle"]
        print(f"  {keep:>10} {keep*5:>8}   " + "  ".join(cells))
        res["history"][str(keep)] = row

    print(f"\n  change from the full window ({full:.4f}):")
    print(f"  {'kept':>10}   " +
          "  ".join(f"{m:>10}" for m in ["shuffle", "freeze", "truncate"]))
    for keep in KEEPS:
        r = res["history"][str(keep)]
        print(f"  {keep:>10}   " +
              "  ".join(f"{r[m]-full:>+10.4f}"
                        for m in ["shuffle", "freeze", "truncate"]))

    # ---- diagnostic 2 -----------------------------------------------------
    print(f"\n{'#'*88}")
    print("# 2. DO GRADIENTS REACH THE EARLY HISTORY?")
    print(f"{'#'*88}")
    g = gradient_profile(model, bw, idx_te, Y, device, j)
    gn = g / max(g.max(), 1e-12)
    print(f"\n  mean |dL/dx_t|, normalised to the maximum")
    print(f"  {'step':>6} {'minutes before':>15} {'gradient':>10}  ")
    T = len(g)
    for t in [0, 6, 12, 24, 36, 48, 54, 59]:
        if t < T:
            bar = "#" * int(40 * gn[t])
            print(f"  {t:>6} {(T-1-t)*5:>15} {gn[t]:>10.5f}  {bar}")
    half = float(gn[:T // 2].mean())
    late = float(gn[T // 2:].mean())
    print(f"\n  mean over the first half {half:.5f}   second half {late:.5f}   "
          f"ratio {half/max(late,1e-12):.4f}")
    res["gradient_profile"] = [float(v) for v in g]
    res["grad_first_half"], res["grad_second_half"] = half, late

    # ---- verdict ----------------------------------------------------------
    d24 = res["history"]["24"]["shuffle"] - full
    d12 = res["history"]["12"]["shuffle"] - full
    print(f"\n{'#'*88}\nREADING THIS\n{'#'*88}")
    print(f"\n  shuffling everything before the last 2 hours: {d24:+.4f}")
    print(f"  shuffling everything before the last 1 hour:  {d12:+.4f}")

    uses_history = abs(d24) > 0.005
    grad_decays = half < 0.25 * late
    if uses_history and grad_decays:
        print("\n  -> Early history matters AND gradients decay toward it.")
        print("     A delayed-state pathway has a real problem to fix, so")
        print("     M3 is justified.")
    elif uses_history:
        print("\n  -> Early history matters, but gradients still reach it.")
        print("     The model can already learn from that range; a delayed")
        print("     path might help as an inductive bias, but the case is")
        print("     weaker than the gradient argument would suggest.")
    else:
        print("\n  -> Destroying the temporal order of everything before the")
        print("     last two hours costs almost nothing. The model is not")
        print("     using long-range structure, so delayed-state feedback")
        print("     would be solving a problem this task does not have.")
        print("     Drop M3 and consider the M1 + M2 + M4 variant instead.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, f"_history_h{args.horizon}", res)
    print(f"\n✓ Saved -> {OUT_DIR / f'_history_h{args.horizon}.json'}")


if __name__ == "__main__":
    main()
