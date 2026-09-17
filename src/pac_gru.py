"""
pac_gru.py
============
PAC-GRU — Prediction-Anchored, Adaptive-order, Coupled-gate GRU.

THREE MECHANISMS, EACH MOTIVATED BY A DIAGNOSTIC
------------------------------------------------
M1' prediction-anchored scaling
    T-GRU scales the state update by t^(alpha-1), where t is position in
    the sequence. In its battery setting t is elapsed time along a real
    degradation trajectory. Here every 60-step sliding window restarts at
    t=1 regardless of where it sits in the child's record, and the
    prediction is always made at t=60 -- so t^(alpha-1) would suppress
    the MOST predictive step by up to 71% (0.29 at alpha=0.7) while
    giving full weight to the step five hours earlier.

    Anchoring on the prediction origin instead:

        d_t   = (T - 1 - t) / (T - 1)      1 at the oldest step, 0 at the newest
        rho_t = 1 + d_t                    in [1, 2]
        s_t   = rho_t ^ (alpha_t - 1)      = 1 exactly at the prediction origin

    This matches what the history diagnostic found: shuffling everything
    before the last 3 hours costs -0.0005, before the last 2 hours
    -0.0054, before the last hour -0.0149. Relevance is
    prediction-relative, not window-relative.

    Note the range is narrow. With alpha_min = 0.80, s spans [0.871, 1.0]
    -- at most 13% suppression at the oldest step. alpha_min is exposed as
    a flag because that may be too little room for the mechanism to show
    an effect either way.

M2 adaptive order
        alpha_t = alpha_min + (1 - alpha_min) * sigmoid(W_a [x_t; h_{t-1}])
    initialised near 0.95, so the cell starts close to an ordinary GRU
    and must learn to deviate. alpha_t -> 1 switches M1' off entirely.

M4 reset-conditioned update gate
        z_t = sigmoid(W_z x_t + U_z h_{t-1} + V_z r_t + b_z)
    V_z is initialised to ZERO, so at epoch 0 this is exactly a standard
    GRU update gate. If coupling helps, V_z moves away from zero; if not,
    it stays. The mechanism cannot destabilise the baseline.

WHAT WAS DROPPED AND WHY
------------------------
M3, tau-GRU's delayed-state feedback, was dropped on evidence. The
gradient profile of the trained GRU spans 46x from the newest to the
oldest step (1.7 orders of magnitude) and is NOT monotonic -- the
gradient at t=0 exceeds that at t=6. Genuine vanishing gradient decays
monotonically over many orders; a deliberately under-trained control
spanned 7 orders. So gradients reach the early history perfectly well.
The early steps contribute little because they carry little information,
which the shuffle test showed independently. A delayed shortcut would
have been repairing a failure this task does not have.

FINAL CELL
----------
    r_t   = sigmoid(W_r x_t + U_r h_{t-1} + b_r)
    z_t   = sigmoid(W_z x_t + U_z h_{t-1} + V_z r_t + b_z)      [M4]
    h~_t  = tanh(W_h x_t + U_h (r_t * h_{t-1}) + b_h)
    alpha_t = alpha_min + (1 - alpha_min) * sigmoid(W_a [x_t; h_{t-1}])  [M2]
    s_t   = (1 + d_t) ^ (alpha_t - 1)                            [M1']
    h_t   = h_{t-1} + s_t * z_t * (h~_t - h_{t-1})

With s_t = 1 this is exactly h_t = (1-z_t) h_{t-1} + z_t h~_t.

ABLATION ARMS
-------------
    gru        none                          baseline
    pa_gru     M1' with fixed alpha
    paa_gru    M1' + M2
    pac_gru    M1' + M2 + M4                 proposed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class PACConfig:
    input_size: int
    hidden_size: int = 64
    layers: int = 2
    n_horizons: int = 4
    dropout: float = 0.2
    alpha_min: float = 0.80
    alpha_init: float = 0.95
    fixed_alpha: float = 0.90
    anchored_scaling: bool = True      # M1'
    adaptive_alpha: bool = True        # M2
    coupled_gates: bool = True         # M4


class PACGRUCell(nn.Module):
    def __init__(self, cfg: PACConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        I, H = int(input_size), cfg.hidden_size
        if not 0.0 < cfg.alpha_min <= 1.0:
            raise ValueError("alpha_min must be in (0, 1].")
        if not cfg.alpha_min <= cfg.alpha_init <= 1.0:
            raise ValueError("need alpha_min <= alpha_init <= 1.")

        self.x_r = nn.Linear(I, H); self.h_r = nn.Linear(H, H, bias=False)
        self.x_z = nn.Linear(I, H); self.h_z = nn.Linear(H, H, bias=False)
        self.x_h = nn.Linear(I, H); self.h_h = nn.Linear(H, H, bias=False)

        if cfg.coupled_gates:
            self.V_z = nn.Linear(H, H, bias=False)
        if cfg.anchored_scaling and cfg.adaptive_alpha:
            self.alpha_net = nn.Linear(I + H, 1)
        self.reset_parameters()

    def reset_parameters(self):
        for m in [self.x_r, self.x_z, self.x_h]:
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        for m in [self.h_r, self.h_z, self.h_h]:
            nn.init.orthogonal_(m.weight)
        if self.cfg.coupled_gates:
            # ZERO init: at epoch 0 z_t is exactly a standard GRU gate, so
            # the coupling cannot destabilise the baseline before it has
            # shown it helps
            nn.init.zeros_(self.V_z.weight)
        if self.cfg.anchored_scaling and self.cfg.adaptive_alpha:
            nn.init.xavier_uniform_(self.alpha_net.weight, gain=0.1)
            am, ai = self.cfg.alpha_min, self.cfg.alpha_init
            p = min(max((ai - am) / max(1e-8, 1.0 - am), 1e-4), 1.0 - 1e-4)
            nn.init.constant_(self.alpha_net.bias, math.log(p / (1 - p)))

    def alpha(self, x, h):
        if not self.cfg.anchored_scaling:
            return x.new_ones(x.shape[0], 1)
        if not self.cfg.adaptive_alpha:
            return x.new_full((x.shape[0], 1), self.cfg.fixed_alpha)
        raw = torch.sigmoid(self.alpha_net(torch.cat([x, h], dim=-1)))
        return self.cfg.alpha_min + (1.0 - self.cfg.alpha_min) * raw

    def forward(self, x, h, d_t: float, want_diag: bool = False):
        r = torch.sigmoid(self.x_r(x) + self.h_r(h))
        zin = self.x_z(x) + self.h_z(h)
        if self.cfg.coupled_gates:
            zin = zin + self.V_z(r)
        z = torch.sigmoid(zin)
        cand = torch.tanh(self.x_h(x) + self.h_h(r * h))

        a = self.alpha(x, h)
        if self.cfg.anchored_scaling:
            rho = torch.as_tensor(1.0 + d_t, dtype=x.dtype, device=x.device)
            s = torch.pow(rho, a - 1.0)          # = 1 at the prediction origin
        else:
            s = x.new_ones(x.shape[0], 1)

        h_new = h + s * z * (cand - h)
        if not want_diag:
            return h_new, None
        return h_new, {"alpha": a.detach(), "scale": s.detach(),
                       "r": r.detach(), "z": z.detach()}


class PACGRU(nn.Module):
    def __init__(self, cfg: PACConfig):
        super().__init__()
        self.cfg = cfg
        self.variant = "pac_gru"
        self.cells = nn.ModuleList([
            PACGRUCell(cfg, cfg.input_size if i == 0 else cfg.hidden_size)
            for i in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        h2 = cfg.hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, cfg.n_horizons))

    def forward(self, x, return_diagnostics: bool = False):
        B, T, _ = x.shape
        hs = [x.new_zeros(B, self.cfg.hidden_size)
              for _ in range(self.cfg.layers)]
        acc = {k: [] for k in ["alpha", "scale", "r", "z"]}

        for t in range(T):
            # distance from the PREDICTION ORIGIN, identical for every
            # window regardless of where it falls in the record
            d_t = (T - 1 - t) / max(T - 1, 1)
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hs[li], dg = cell(inp, hs[li], d_t,
                                  want_diag=return_diagnostics and li == 0)
                if dg is not None:
                    for k in acc:
                        acc[k].append(dg[k])
                inp = self.drop(hs[li]) if li < self.cfg.layers - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        A = torch.stack(acc["alpha"], 1).squeeze(-1)      # [B,T]
        S = torch.stack(acc["scale"], 1)
        S = S.squeeze(-1) if S.dim() == 3 else S
        R = torch.stack(acc["r"], 1)                      # [B,T,H]
        Z = torch.stack(acc["z"], 1)
        diag = {
            "alpha_mean": A.mean(), "alpha_sd": A.std(),
            "alpha_min_seen": A.min(), "alpha_max_seen": A.max(),
            "scale_mean": S.mean(), "scale_at_oldest": S[:, 0].mean(),
            "scale_at_newest": S[:, -1].mean(),
            "r_mean": R.mean(), "z_mean": Z.mean(), "z_sd": Z.std(),
        }
        if self.cfg.coupled_gates:
            v = self.cells[0].V_z.weight
            diag["Vz_norm"] = v.detach().norm()
            diag["Vz_vs_Uz"] = (v.detach().norm() /
                                self.cells[0].h_z.weight.detach().norm())
        diag["_alpha_per_window"] = A.mean(dim=1)
        diag["_z_per_window"] = Z.mean(dim=(1, 2))
        return logits, diag


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


ARMS = {
    "pa_gru":  dict(anchored_scaling=True,  adaptive_alpha=False, coupled_gates=False),
    "paa_gru": dict(anchored_scaling=True,  adaptive_alpha=True,  coupled_gates=False),
    "pac_gru": dict(anchored_scaling=True,  adaptive_alpha=True,  coupled_gates=True),
}


if __name__ == "__main__":
    import torch.nn as nn2
    torch.manual_seed(42)

    g = nn2.GRU(9, 64, num_layers=2, batch_first=True)
    hd = nn2.Sequential(nn2.Linear(64, 32), nn2.ReLU(), nn2.Dropout(0.1),
                        nn2.Linear(32, 4))
    base = sum(p.numel() for p in g.parameters()) + \
        sum(p.numel() for p in hd.parameters())
    print(f"gru baseline (2 layers, hidden 64): {base:,}\n")

    for name, kw in ARMS.items():
        for h in [64, 62, 60]:
            n = count_parameters(PACGRU(PACConfig(9, hidden_size=h, **kw)))
            if h == 64 or abs(n - base) < 400:
                print(f"  {name:>8} hidden {h}: {n:>8,}  "
                      f"({100*(n-base)/base:+6.2f}%)")
        print()

    m = PACGRU(PACConfig(9, hidden_size=64))
    x = torch.randn(8, 60, 9)
    lo, d = m(x, return_diagnostics=True)
    print(f"logits {tuple(lo.shape)}")
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
