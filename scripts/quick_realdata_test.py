"""Fast end-to-end verification on a real preprocessed dataset.

This is not a thesis-performance reproduction. It performs one optimizer step
on a small deterministic training subset and a forward pass on held-out
validation windows to verify data loading, the frozen subject split,
threshold-aware channels, TSL-GRU, loss, gradients, and finite predictions.

Example:
    python scripts/quick_realdata_test.py --tag stride6 --train 128 --val 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import common  # noqa: E402
from stage2_deep import MultiHorizonLoss, WindowDataset  # noqa: E402
from ta_gru import attach_ta  # noqa: E402
from tsl_gru import TSLGRU  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--train", type=int, default=128)
    ap.add_argument("--val", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bw = common.load_built(args.tag, label_set="any")
    common.verify_split(bw.meta)
    attach_ta(bw, verbose=False)
    if bw.n_channels != 7:
        raise RuntimeError(f"expected 7 channels after TA attachment, got {bw.n_channels}")

    tr, va, _ = common.get_split(bw.meta)
    idx_tr = np.flatnonzero(tr)[: args.train]
    idx_va = np.flatnonzero(va)[: args.val]
    if len(idx_tr) < 2 or len(idx_va) < 1:
        raise RuntimeError("not enough windows for the requested quick test")

    Y = bw._labels_any
    prevalence = Y[np.flatnonzero(tr)].mean(axis=0)
    pos_weight = np.clip((1 - prevalence) / np.maximum(prevalence, 1e-6), 1.0, 50.0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TSLGRU("tsl_gru", in_ch=7, hidden=64, layers=2, dropout=0.2, rho=0.9).to(device)
    criterion = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    ds = WindowDataset(bw, idx_tr, Y)
    xb = torch.stack([ds[i][0] for i in range(len(ds))]).to(device)
    yb = torch.stack([ds[i][1] for i in range(len(ds))]).to(device)

    model.train()
    optimizer.zero_grad()
    pred = model(xb)
    loss = criterion(pred, yb)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite training loss: {loss.item()}")
    loss.backward()
    optimizer.step()

    model.eval()
    xv = torch.from_numpy(bw.windows(idx_va)).to(device)
    with torch.no_grad():
        pv = model(xv)
    if not bool(torch.isfinite(pv).all()):
        raise RuntimeError("non-finite validation predictions")

    print("PediaCare-Net real-data quick test: PASSED")
    print(f"  device: {device}")
    print(f"  windows: train={len(idx_tr)}, validation={len(idx_va)}")
    print(f"  channels: {bw.n_channels}; sequence length: {bw.window_len}")
    print(f"  one-step weighted-BCE loss: {loss.item():.6f}")
    print(f"  validation output shape: {tuple(pv.shape)}")
    print("  NOTE: this is an execution check, not a thesis metric reproduction.")


if __name__ == "__main__":
    main()
