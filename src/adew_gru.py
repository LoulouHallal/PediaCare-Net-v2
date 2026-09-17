"""
adew_gru.py
=============
ADEW-GRU — Anchored Delay–Erase–Write GRU.

THE DEFINING PROPERTY
---------------------
The standard GRU is an EXACT special case. At alpha_E = alpha_D = 0:

    h_t = (1 - z_t) h_{t-1} + z_t u_t

which is the GRU verbatim (verified to 0.00e+00). Every earlier proposed
cell in this project replaced the GRU recurrence outright and had to
re-derive whatever the baseline already did well. This one starts there
and must learn to depart.

WHAT WENT WRONG IN KEW-GRU, AND HOW THIS FIXES IT
-------------------------------------------------
KEW put Keep, Erase and Write on one simplex:

    k + e + w = 1     so     k + w = 1 - e

The coefficients multiplying live information therefore shrink whenever
erase carries mass. At the measured e ~ 0.19, a step where the candidate
agrees with the state still contracts it by 0.81. Sustained over ten
steps that is 0.12, over twenty 0.015. Erase was not a rare discrete
operation either: 19% average mass, dominant on 15% of coordinates.

ADEW factorises instead, so erase draws only from the RETAINED path:

    K_t = (1 - z_t)(1 - eps_t)
    E_t = (1 - z_t) eps_t
    W_t = z_t
    K + E + W = 1     exactly, for any eps

Writing no longer competes with erasing. That is also a more faithful
transfer of the Gated DeltaNet-2 principle: its contribution is
DECOUPLED erase and write control, which a shared categorical
normalisation quietly re-couples.

THE CELL
--------
    r_t   = sigmoid(W_r x + U_r h + b_r)
    z_t   = sigmoid(W_z x + U_z h + b_z)
    u_t   = tanh(W_u x + U_u (r_t * h) + b_u)
    e_t   = sigmoid(W_e x + U_e h + b_e)
    eps_t = alpha_E * e_t

    d_t   = tanh(W_d x + U_d h_{t-L} + b_d)          [delay arm only]
    a_t   = sigmoid(W_a x + U_a h + b_a)

    h_t = (1 - z_t) * (1 - eps_t) * h_{t-1}
        + z_t * (u_t + alpha_D * a_t * d_t)

THE ANCHOR COLD START
---------------------
At alpha_E EXACTLY 0, W_e/U_e/b_e receive zero gradient -- the controller
only enters through the product alpha_E * e_t. The anchor moves first,
driven by a randomly initialised e_t, and only then can the controller
learn. Measured: |grad W_e| is 0.000000 at alpha=0, 0.107 at 0.01, 0.529
at 0.05.

The default is therefore alpha_init = 0.01, not 0. That is at most 1%
erasure at initialisation -- against KEW's 16% -- so the model still
starts essentially at the GRU, but the controller can learn from step one.
Use --alpha_init 0.0 for strict nesting at initialisation, accepting the
cold start.

The reset gate is RESTORED. Li-GRU showed reset removal can work in ASR,
not that it is universally safe, and the KEW experiment gave no evidence
it helped here. It is no longer part of the proposed novelty.

Sparsity is NOT used in the core cell. Fixed entmax-1.5 was part of KEW
and KEW lost; making it mandatory again would be assuming the answer.
Adaptive-alpha entmax can return later as an ablation, once the cell
beats the baseline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class ADEWConfig:
    input_size: int
    hidden_size: int = 64
    layers: int = 2
    n_horizons: int = 4
    dropout: float = 0.2
    use_erase: bool = True
    use_delay: bool = False
    delay_steps: int = 12          # 12 * 5 min = 60 min
    alpha_init: float = 0.01       # see the cold-start note above


class ADEWGRUCell(nn.Module):
    def __init__(self, cfg: ADEWConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        I, H = int(input_size), cfg.hidden_size
        self.H = H

        # the standard GRU, untouched
        self.x_r = nn.Linear(I, H); self.h_r = nn.Linear(H, H, bias=False)
        self.x_z = nn.Linear(I, H); self.h_z = nn.Linear(H, H, bias=False)
        self.x_u = nn.Linear(I, H); self.h_u = nn.Linear(H, H, bias=False)

        if cfg.use_erase:
            self.x_e = nn.Linear(I, H); self.h_e = nn.Linear(H, H, bias=False)
            self.alpha_E = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
        if cfg.use_delay:
            self.x_d = nn.Linear(I, H); self.h_d = nn.Linear(H, H, bias=False)
            self.x_a = nn.Linear(I, H); self.h_a = nn.Linear(H, H, bias=False)
            self.alpha_D = nn.Parameter(torch.tensor(float(cfg.alpha_init)))
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for nm in ["h_r", "h_z", "h_u", "h_e", "h_d", "h_a"]:
            if hasattr(self, nm):
                nn.init.orthogonal_(getattr(self, nm).weight)

    def forward(self, x, h, h_delay=None, delay_ok=None, want_diag=False):
        r = torch.sigmoid(self.x_r(x) + self.h_r(h))
        z = torch.sigmoid(self.x_z(x) + self.h_z(h))
        u = torch.tanh(self.x_u(x) + self.h_u(r * h))

        keep_path = 1.0 - z
        eps = None
        if self.cfg.use_erase:
            e = torch.sigmoid(self.x_e(x) + self.h_e(h))
            eps = self.alpha_E * e
            keep_path = keep_path * (1.0 - eps)

        write_val = u
        agd = None
        if self.cfg.use_delay and h_delay is not None:
            a = torch.sigmoid(self.x_a(x) + self.h_a(h))
            d = torch.tanh(self.x_d(x) + self.h_d(h_delay))
            # delay_ok masks the first L steps, where h_{t-L} does not exist
            agd = self.alpha_D * a * d * delay_ok
            write_val = u + agd

        h_new = keep_path * h + z * write_val

        if not want_diag:
            return h_new, None
        diag = {"z": z.detach(), "r": r.detach()}
        if eps is not None:
            diag["eps"] = eps.detach()
            diag["e_raw"] = e.detach()
            diag["alpha_E"] = self.alpha_E.detach()
            # the factorised K/E/W, which sum to 1 by construction
            diag["K"] = ((1 - z) * (1 - eps)).detach()
            diag["E"] = ((1 - z) * eps).detach()
            diag["W"] = z.detach()
        if agd is not None:
            diag["delay_contrib"] = agd.abs().detach()
            diag["alpha_D"] = self.alpha_D.detach()
        return h_new, diag


class ADEWGRU(nn.Module):
    def __init__(self, cfg: ADEWConfig):
        super().__init__()
        self.cfg = cfg
        self.variant = "adew"
        self.cells = nn.ModuleList([
            ADEWGRUCell(cfg, cfg.input_size if i == 0 else cfg.hidden_size)
            for i in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        h2 = cfg.hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, cfg.n_horizons))

    def forward(self, x, return_diagnostics: bool = False):
        B, T, _ = x.shape
        H, L = self.cfg.hidden_size, self.cfg.layers
        hs = [x.new_zeros(B, H) for _ in range(L)]
        hist: List[List[torch.Tensor]] = [[] for _ in range(L)]
        acc: Dict[str, list] = {}
        lag = self.cfg.delay_steps

        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hd, ok = None, None
                if self.cfg.use_delay:
                    if len(hist[li]) >= lag:
                        hd = hist[li][-lag]
                        ok = x.new_ones(B, 1)
                    else:
                        hd = hs[li].detach() * 0
                        ok = x.new_zeros(B, 1)
                hs[li], d = cell(inp, hs[li], hd, ok,
                                 want_diag=return_diagnostics and li == 0)
                if self.cfg.use_delay:
                    hist[li].append(hs[li])
                if d is not None:
                    for k, v in d.items():
                        acc.setdefault(k, []).append(v)
                inp = self.drop(hs[li]) if li < L - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        diag: Dict[str, torch.Tensor] = {}
        for k in ["z", "r", "eps", "e_raw", "K", "E", "W", "delay_contrib"]:
            if k in acc:
                st = torch.stack(acc[k], 1)
                diag[f"{k}_mean"] = st.mean()
                if k == "E":
                    diag["E_max"] = st.max()
                    diag["_E_per_window"] = st.mean(dim=(1, 2))
                if k == "eps":
                    diag["_eps_per_window"] = st.mean(dim=(1, 2))
        for k in ["alpha_E", "alpha_D"]:
            if k in acc:
                diag[k] = acc[k][0]
        if "K" in acc:
            s = (torch.stack(acc["K"], 1) + torch.stack(acc["E"], 1)
                 + torch.stack(acc["W"], 1))
            diag["kew_sum_err"] = (s - 1.0).abs().max()
        return logits, diag


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


ARMS = {
    "gru":        dict(use_erase=False, use_delay=False),
    "adew":       dict(use_erase=True,  use_delay=False),
    "adew_delay": dict(use_erase=True,  use_delay=True),
}


if __name__ == "__main__":
    torch.manual_seed(42)
    print(f"{'arm':>12} {'H':>4} {'params':>9}")
    for nm, kw in ARMS.items():
        for H in [64, 60, 56]:
            n = count_parameters(ADEWGRU(ADEWConfig(9, hidden_size=H, **kw)))
            print(f"{nm:>12} {H:>4} {n:>9,}")
        print()

    print("NESTING at alpha = 0")
    x = torch.randn(4, 20, 9)
    g = ADEWGRU(ADEWConfig(9, 32, **ARMS["gru"]))
    a = ADEWGRU(ADEWConfig(9, 32, alpha_init=0.0, **ARMS["adew"]))
    a.load_state_dict(g.state_dict(), strict=False)
    g.eval(); a.eval()
    with torch.no_grad():
        print(f"  max |ADEW(alpha=0) - GRU| = {(g(x)-a(x)).abs().max():.2e}")

    m = ADEWGRU(ADEWConfig(9, 64, **ARMS["adew_delay"]))
    lo, d = m(torch.randn(8, 60, 9), return_diagnostics=True)
    print(f"\nlogits {tuple(lo.shape)}")
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
