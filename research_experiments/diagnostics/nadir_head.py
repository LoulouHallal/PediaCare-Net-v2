"""
nadir_head.py  --  Phase 3 (E1): future-nadir auxiliary supervision
====================================================================

Adds a second training head that regresses min(future glucose) alongside
the existing hypoglycemia classifier. Classification at inference is
UNCHANGED and still comes from the classification head -- the regression
is never thresholded, never used to decide anything. It exists only to
put trajectory severity into the shared representation.

    L = L_bce  +  lambda * L_huber(nadir)

WHAT PHASE 2 SAID, AND WHAT THIS PREDICTS
------------------------------------------
h=15  90.1% of FPs are near-misses (nadir 70-80, median 72) and FN median
      probability is 0.751 against a 0.899 threshold. There is essentially
      no model error left. PREDICTION: null here.
h=120 FN median probability 0.261 with 22.9% "fast drop from high", and
      28.5% of FPs genuinely wrong. PREDICTION: gain here if anywhere.

Committed before running. A gain at h=15 but not h=120 refutes the stated
mechanism and does NOT get relabelled a success.

NO CHANGES TO THE MODEL CLASS
------------------------------
The head needs the hidden state, but the model returns logits. Rather than
edit the architecture, a forward hook on the final Linear captures its
INPUT, which is the representation feeding the classifier. This keeps E0
and E1 running identical architecture code -- the only difference is the
extra head and loss term, which is what the ablation is supposed to
isolate.

INVARIANCE CHECK
----------------
At lambda=0 this must reproduce the baseline bit-for-bit. assert_lambda0
verifies that. If it fails, the wrapper is perturbing the base model and
no comparison it produces is trustworthy.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
class NadirDataset(torch.utils.data.Dataset):
    """
    Wraps the project's WindowDataset and appends the auxiliary target.

    base[i] -> (x, y)          becomes      (x, y, g, m)

    where g is the z-scored nadir for window idx[i] and m is its validity
    mask. idx must be the SAME index array passed to WindowDataset, in the
    same order, or targets are silently misaligned with inputs -- the
    single most dangerous failure mode here, because it trains fine and
    just quietly learns nothing.
    """

    def __init__(self, base, idx, nadir_z, valid):
        self.base = base
        self.g = np.asarray(nadir_z)[np.asarray(idx)].astype(np.float32)
        self.m = np.asarray(valid)[np.asarray(idx)].astype(np.float32)
        if len(self.base) != len(self.g):
            raise ValueError(
                f"base dataset has {len(self.base)} items but idx selects "
                f"{len(self.g)} targets -- these must match exactly")
        # NaN in a masked-out slot is fine, but NaN where valid is True is a
        # build error and would poison the loss.
        bad = np.isnan(self.g) & (self.m > 0)
        if bad.any():
            raise ValueError(f"{bad.sum()} valid slots hold NaN targets")
        self.g = np.nan_to_num(self.g, nan=0.0)

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        x, y = self.base[i]
        return x, y, torch.from_numpy(self.g[i]), torch.from_numpy(self.m[i])


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------
def _find_output_linear(model: nn.Module, n_out: int) -> tuple[str, nn.Linear]:
    """Last nn.Linear whose out_features == n_out. That is the classifier."""
    hit = None
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Linear) and mod.out_features == n_out:
            hit = (name, mod)
    if hit is None:
        raise RuntimeError(
            f"no nn.Linear with out_features={n_out} found; pass head_name "
            f"explicitly. Modules: "
            f"{[n for n, m in model.named_modules() if isinstance(m, nn.Linear)]}")
    return hit


class WithNadirHead(nn.Module):
    """
    Base model plus a parallel linear head on the same representation.

    forward(x)                  -> logits                (identical to base)
    forward(x, want_nadir=True) -> (logits, nadir_pred)
    """

    def __init__(self, base: nn.Module, n_out: int = 4, head_name: str | None = None):
        super().__init__()
        self.base = base
        if head_name is None:
            head_name, head = _find_output_linear(base, n_out)
        else:
            head = dict(base.named_modules())[head_name]
        self.head_name = head_name
        self.in_features = head.in_features
        self.nadir = nn.Linear(head.in_features, n_out)
        nn.init.zeros_(self.nadir.bias)
        nn.init.normal_(self.nadir.weight, std=0.01)

        self._h = None
        head.register_forward_hook(self._capture)

    def _capture(self, module, inp, out):
        self._h = inp[0]

    def forward(self, x, want_nadir: bool = False, **kw):
        self._h = None
        out = self.base(x, **kw)
        if not want_nadir:
            return out
        if self._h is None:
            raise RuntimeError("hook did not fire; head_name is wrong")
        return out, self.nadir(self._h)


# ---------------------------------------------------------------------------
# loss
# ---------------------------------------------------------------------------
def masked_huber(pred, target, mask, delta: float = 1.0):
    """
    Huber over valid slots only. Windows with no future readings (539 in
    this build) carry a classification label but no nadir, and must not
    contribute. Returns a 0-D tensor; 0.0 if nothing is valid.
    """
    n = mask.sum()
    if n.item() == 0:
        return pred.sum() * 0.0
    per = nn.functional.huber_loss(pred, target, reduction="none", delta=delta)
    return (per * mask).sum() / n


class NadirLoss(nn.Module):
    """L = base(logits, y) + lambda * huber(nadir)."""

    def __init__(self, base_criterion, lam: float = 0.3, delta: float = 1.0):
        super().__init__()
        self.base = base_criterion
        self.lam = float(lam)
        self.delta = float(delta)
        self.last = {}

    def forward(self, logits, y, nadir_pred=None, g=None, m=None):
        lc = self.base(logits, y)
        if self.lam == 0.0 or nadir_pred is None:
            self.last = {"bce": float(lc.detach()), "nadir": 0.0}
            return lc
        ln = masked_huber(nadir_pred, g, m, self.delta)
        self.last = {"bce": float(lc.detach()), "nadir": float(ln.detach())}
        return lc + self.lam * ln


# ---------------------------------------------------------------------------
# invariance check
# ---------------------------------------------------------------------------
def assert_lambda0(base, x, n_out: int = 4, tol: float = 0.0):
    """
    The wrapper must not change the base model's outputs. Run this before
    trusting any E1 vs E0 comparison.
    """
    base.eval()
    with torch.no_grad():
        a = base(x)
        w = WithNadirHead(base, n_out=n_out).eval()
        b = w(x)
        c, _ = w(x, want_nadir=True)
    d1 = (a - b).abs().max().item()
    d2 = (a - c).abs().max().item()
    print(f"  wrapper invariance: max|base - wrapped| = {d1:.3e}, "
          f"with head = {d2:.3e}")
    if d1 > tol or d2 > tol:
        raise AssertionError("wrapper perturbs the base model")
    return True
