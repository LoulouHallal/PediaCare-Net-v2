"""
drs_gru.py
============
DRS-GRU — signed recurrent history with a learned admission gate.

THE MECHANISM
-------------
A standard GRU computes its candidate as

    h~ = tanh(W_h x + r * (U_h h)),     r = sigmoid(...) in [0, 1]

For a fixed recurrent projection m = U_h h, the SIGN of dimension j's
historical contribution is fixed by U_h. The reset gate can only scale it
between 0 and its full value. History can be admitted or forgotten; it
cannot be opposed.

DRS-GRU replaces that with a signed reset,

    r_t = tanh(W_r x_t + U_r h_{t-1} + b_r)      in [-1, 1]

so the same hidden dimension can reinforce history in one context and
oppose it in another. Verified: with identical weights, DRS flips a
dimension from +0.995 to -0.995 across two inputs where a GRU moves only
from 0.953 to 0.047 -- scaling, never inverting.

FROZEN EQUATIONS
----------------
    z_t   = sigmoid(W_z x_t + U_z h_{t-1} + b_z)      standard GRU gate
    r_t   = tanh(W_r x_t + U_r h_{t-1} + b_r)         SIGNED
    a_t   = W_h x_t + b_h
    m_t   = U_h h_{t-1}

    h~S_t = tanh(a_t + r_t * m_t)                     signed-history
    h~0_t = tanh(a_t)                                 history-free
    lam_t = sigmoid(W_l x_t + b_l)
    h~_t  = lam_t * h~S_t + (1 - lam_t) * h~0_t

    h_t   = (1 - z_t) * h_{t-1} + z_t * h~_t

WHY THIS FORM AND NOT THE ORIGINAL DUAL-CANDIDATE ONE
-----------------------------------------------------
The design began as two candidates built from r+ = relu(r) and
r- = relu(-r), blended by a context gate c. That reduces exactly, because
-r- = r whenever r < 0:

    r_j > 0:   h~+ = tanh(a + r m),  h~- = tanh(a)
    r_j < 0:   h~+ = tanh(a),        h~- = tanh(a + r m)

So one candidate is always signed-history and the other always
history-free -- and c therefore means the OPPOSITE thing depending on
sign(r). High c admits history when r > 0 and suppresses it when r < 0.
The optimiser would have had to learn that coupling to a sign it does not
control.

The lambda form above is algebraically the same family with one
consistent meaning: how much signed recurrent history to admit. Same
parameter count, no sign-dependent reinterpretation.

The update gate keeps its standard GRU form. An input-free variant
(z from h alone) was considered and rejected for the primary model: it is
strictly less expressive, it is not where the hypothesis lives, and a
weaker gate could mask whatever the signed pathway contributes. It
remains available as an ablation via --input_free_z.

DIAGNOSTICS
-----------
Every earlier architecture in this project had its mechanism suppressed
by the optimiser, so a tie is only informative if we can see whether the
mechanism was used. Logged per epoch:

    P(r < 0)                is the signed pathway used at all?
    E[|r|]                  or has the reset collapsed toward zero?
    E[lambda]               is signed history admitted?
    E[lambda | r < 0]       THE key one: negative r is inert if lambda
                            suppresses history exactly when r is negative
    ||h~S - h~0||           if this is ~0 the two candidates coincide
    E[z], sd(z)             the update gate should not be pinned

and, on the test set, P(r<0) and lambda split by outcome and within the
glucose >= 120 regime where the error-regime audit found the models
weakest.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


class DRSGRUCell(nn.Module):
    def __init__(self, input_size: int, hidden_size: int,
                 input_free_z: bool = False):
        super().__init__()
        self.input_size, self.hidden_size = int(input_size), int(hidden_size)
        self.input_free_z = bool(input_free_z)

        D, H = self.input_size, self.hidden_size
        self.W_z = None if input_free_z else nn.Linear(D, H, bias=False)
        self.U_z = nn.Linear(H, H, bias=True)
        self.W_r = nn.Linear(D, H, bias=False)
        self.U_r = nn.Linear(H, H, bias=True)
        self.W_h = nn.Linear(D, H, bias=True)
        self.U_h = nn.Linear(H, H, bias=False)
        self.W_l = nn.Linear(D, H, bias=True)

        for mod in self.modules():
            if isinstance(mod, nn.Linear):
                nn.init.xavier_uniform_(mod.weight)
                if mod.bias is not None:
                    nn.init.zeros_(mod.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor,
                want_diag: bool = False):
        zin = self.U_z(h) if self.input_free_z else self.W_z(x) + self.U_z(h)
        z = torch.sigmoid(zin)
        r = torch.tanh(self.W_r(x) + self.U_r(h))          # SIGNED, in [-1,1]

        a = self.W_h(x)
        m = self.U_h(h)
        h_signed = torch.tanh(a + r * m)
        h_free = torch.tanh(a)
        lam = torch.sigmoid(self.W_l(x))

        cand = lam * h_signed + (1 - lam) * h_free
        h_new = (1 - z) * h + z * cand

        if not want_diag:
            return h_new, None
        return h_new, {
            "r": r.detach(), "lam": lam.detach(), "z": z.detach(),
            "cand_sep": (h_signed - h_free).abs().mean(-1).detach(),
        }


class DRSGRU(nn.Module):
    """Stacked DRS-GRU + the shared 4-horizon head."""

    def __init__(self, input_size: int, hidden_size: int = 60, layers: int = 2,
                 n_horizons: int = 4, dropout: float = 0.2,
                 input_free_z: bool = False):
        super().__init__()
        self.variant = "drs_gru"
        self.hidden_size, self.layers = int(hidden_size), int(layers)
        self.cells = nn.ModuleList([
            DRSGRUCell(input_size if i == 0 else hidden_size, hidden_size,
                       input_free_z)
            for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        h2 = hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(hidden_size, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, n_horizons))

    def forward(self, x: torch.Tensor, return_diagnostics: bool = False):
        B, T, _ = x.shape
        hs = [x.new_zeros(B, self.hidden_size) for _ in range(self.layers)]
        acc: Dict[str, list] = {k: [] for k in ["r", "lam", "z", "cand_sep"]}

        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                # diagnostics from layer 0 only: it reads the raw features,
                # so its gates are the interpretable ones
                hs[li], d = cell(inp, hs[li],
                                 want_diag=return_diagnostics and li == 0)
                if d is not None:
                    for k in acc:
                        acc[k].append(d[k])
                inp = self.drop(hs[li]) if li < self.layers - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        R = torch.stack(acc["r"], 1)                  # [B,T,H]
        L = torch.stack(acc["lam"], 1)
        Z = torch.stack(acc["z"], 1)
        S = torch.stack(acc["cand_sep"], 1)           # [B,T]
        neg = (R < 0).float()

        diag = {
            "frac_r_negative": neg.mean(),
            "abs_r_mean": R.abs().mean(),
            "lambda_mean": L.mean(),
            # negative r is inert if lambda suppresses history exactly when
            # r is negative, so these two must be compared
            "lambda_where_r_neg": (L * neg).sum() / neg.sum().clamp_min(1),
            "lambda_where_r_pos": (L * (1 - neg)).sum() / (1 - neg).sum().clamp_min(1),
            "candidate_separation": S.mean(),
            "z_mean": Z.mean(), "z_sd": Z.std(),
        }
        # per-window traces so the split by outcome needs no second pass
        diag["_frac_r_neg_per_window"] = neg.mean(dim=(1, 2))
        diag["_lambda_per_window"] = L.mean(dim=(1, 2))
        diag["_cand_sep_per_window"] = S.mean(dim=1)
        return logits, diag


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    import torch.nn as nn2

    g = nn2.GRU(9, 64, num_layers=2, batch_first=True)
    hd = nn2.Sequential(nn2.Linear(64, 32), nn2.ReLU(), nn2.Dropout(0.1),
                        nn2.Linear(32, 4))
    base = sum(p.numel() for p in g.parameters()) + \
        sum(p.numel() for p in hd.parameters())
    print(f"gru_abs baseline: {base:,}\n")

    for h in [58, 60, 61, 62]:
        m = DRSGRU(9, h)
        n = count_trainable_parameters(m)
        print(f"  hidden {h}: {n:,}  ({100*(n-base)/base:+.2f}%)"
              f"{'   <-- capacity matched' if h == 60 else ''}")

    m = DRSGRU(9, 60)
    x = torch.randn(8, 60, 9)
    logits, d = m(x, return_diagnostics=True)
    print(f"\nlogits {tuple(logits.shape)}")
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
