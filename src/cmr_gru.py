"""
cmr_gru.py
============
CMR-GRU — Context-Modulated Routed GRU.

WHY THESE THREE MECHANISMS AND NOT THE OTHERS
---------------------------------------------
Ten candidate mechanisms were triaged against experiments already run in
this project. Seven are already refuted here:

  attention gate in the cell     TA-GRU: q fell 0.056 -> 0.041, suppressed
  self-attention over history    TRM-GRU: shuffle control indistinguishable
  dilated / multi-scale memory   DMS-TCN +0.001; shuffling >3h costs -0.0005
  gated residual / highway       this is the GRU update gate; ADEW's residual
                                 branch drove its anchor NEGATIVE
  learned time decay             TSL-GRU (kappa 0.10->0.60, causally
                                 verified), PAC-GRU, ADEW -- all tie
  cross-module attention         AR-RHU coupled two states; active, redundant
  continuous-time / ODE          dt is constant at 5 min, so the problem
                                 ODEs solve is absent here

Three survive, and this cell is built from exactly those:

  M1 FiLM context modulation
     h_hat = gamma(c_t) * h~ + beta(c_t)
     This is an x*h PRODUCT. The corrected interaction screen tested x*x
     only (-0.0008 at h=30) and explicitly could not speak to x*h: two
     windows with identical current inputs but different histories are
     indistinguishable to every x*x feature and different under x*h.
     Physiologically: glucose falling after a bolus is not the same event
     as glucose falling with no insulin on board, and an additive
     candidate must represent that through composition rather than
     directly.

  M2 Modular recurrent state
     h splits into M blocks with block-diagonal recurrence, so each block
     can specialise. AR-RHU is the nearest prior attempt but used two
     states with an ORTHOGONALITY constraint; the states ended up
     information-redundant. Block-diagonal specialisation with routing is
     a different hypothesis.

  M3 Sparse module routing
     Context decides which modules update at each step, so an insulin
     module can stay quiet during fasting and engage after a bolus. Never
     tested here. KEW used sparse ACTIONS on a single state, not sparse
     module UPDATES.

NESTING, AND WHAT IT DOES NOT BUY
---------------------------------
At M=1, lambda=0, mu=0 the cell is EXACTLY a standard GRU (verified to
0.00e+00). That is a good initialisation and a clean fallback. It is NOT
a guarantee of better held-out performance -- ADEW nested the GRU exactly
and still lost. Nesting bounds the training optimum, not generalisation.

Anchors follow the ADEW lesson: lambda and mu are initialised small but
NONZERO, because at exactly zero their controllers receive no gradient
(measured: |grad| 0.000000 at 0, 0.107 at 0.01). Both are bounded in
[0, 1) via 1 - exp(-rho^2), so neither can go negative -- the failure mode
where ADEW's unconstrained anchor became an anti-erase term.

THE CELL
--------
    c_t = treatment context channels of x_t

    per module m of M:
      z_m  = sigmoid(W_zm x + U_zm h_m + b_zm)
      r_m  = sigmoid(W_rm x + U_rm h_m + b_rm)
      h~_m = tanh(W_m x + U_m (r_m * h_m) + b_m)

      gamma_m = 1 + lambda * tanh(A_m c_t)              [M1]
      beta_m  =     lambda * tanh(B_m c_t)
      hhat_m  = gamma_m * h~_m + beta_m

      s       = softmax_or_entmax over modules of (W_s [c_t; h])   [M3]
      rho_m   = 1 + mu * (M * s_m - 1)          mean 1, so mu=0 is uniform

      h_m = (1 - rho_m z_m) h_m + rho_m z_m hhat_m

ABLATION LADDER (what the pilot runs)
-------------------------------------
    gru          M=1, no FiLM, no routing      standard GRU
    modular      M=4, no FiLM, no routing      block-diagonal alone
    film         M=1, FiLM on                  context modulation alone
    mod_film     M=4, FiLM on                  the pair
    cmr          M=4, FiLM + routing           proposed
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn


@dataclass
class CMRConfig:
    input_size: int
    hidden_size: int = 64
    n_modules: int = 4
    layers: int = 2
    n_horizons: int = 4
    dropout: float = 0.2
    context_idx: tuple = (1, 2, 3, 4)      # basal, bolus, carbs, carbs_obs
    use_film: bool = False
    use_routing: bool = False
    anchor_init: float = 0.05


def _anchor(v: float) -> float:
    return math.sqrt(-math.log(1.0 - v))


class CMRCell(nn.Module):
    def __init__(self, cfg: CMRConfig, input_size: int, first_layer: bool):
        super().__init__()
        self.cfg = cfg
        I, H, M = int(input_size), cfg.hidden_size, cfg.n_modules
        if H % M:
            raise ValueError("hidden_size must divide by n_modules")
        self.I, self.H, self.M, self.D = I, H, M, H // M
        # only layer 0 sees the raw channels, so context is read there
        self.first_layer = first_layer
        C = len(cfg.context_idx) if first_layer else 0
        self.C = C

        # block-diagonal recurrence: per-module weights, no cross-module mixing
        D = self.D
        self.W_z = nn.Linear(I, H); self.U_z = nn.Parameter(torch.empty(M, D, D))
        self.W_r = nn.Linear(I, H); self.U_r = nn.Parameter(torch.empty(M, D, D))
        self.W_h = nn.Linear(I, H); self.U_h = nn.Parameter(torch.empty(M, D, D))

        if cfg.use_film and C:
            self.A = nn.Linear(C, H)          # gamma
            self.B = nn.Linear(C, H)          # beta
            self.rho_lam = nn.Parameter(torch.tensor(_anchor(cfg.anchor_init)))
        if cfg.use_routing:
            self.W_s = nn.Linear((C if C else I) + H, M)
            self.rho_mu = nn.Parameter(torch.tensor(_anchor(cfg.anchor_init)))
        self.reset_parameters()

    def lam(self):
        return 1.0 - torch.exp(-self.rho_lam ** 2)

    def mu(self):
        return 1.0 - torch.exp(-self.rho_mu ** 2)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for P in [self.U_z, self.U_r, self.U_h]:
            for k in range(self.M):
                nn.init.orthogonal_(P.data[k])

    def _blk(self, P, h):
        """Block-diagonal matmul: h [B,H] -> [B,M,D] -> per-module -> [B,H]."""
        B = h.shape[0]
        hm = h.view(B, self.M, self.D)
        return torch.einsum("bmd,mde->bme", hm, P).reshape(B, self.H)

    def forward(self, x, h, want_diag=False):
        B = x.shape[0]
        z = torch.sigmoid(self.W_z(x) + self._blk(self.U_z, h))
        r = torch.sigmoid(self.W_r(x) + self._blk(self.U_r, h))
        cand = torch.tanh(self.W_h(x) + self._blk(self.U_h, r * h))

        diag: Dict[str, torch.Tensor] = {}
        c = x[:, list(self.cfg.context_idx)] if (self.C and self.first_layer) \
            else None

        if self.cfg.use_film and c is not None:
            lam = self.lam()
            gamma = 1.0 + lam * torch.tanh(self.A(c))
            beta = lam * torch.tanh(self.B(c))
            cand = gamma * cand + beta
            if want_diag:
                diag["lam"] = lam.detach()
                diag["gamma_dev"] = (gamma - 1).abs().mean().detach()
                diag["beta_abs"] = beta.abs().mean().detach()

        rho = None
        if self.cfg.use_routing:
            mu = self.mu()
            s = torch.softmax(self.W_s(torch.cat(
                [c if c is not None else x, h], -1)), dim=-1)     # [B,M]
            # mean 1 by construction, so mu = 0 is exactly uniform routing
            rho_m = 1.0 + mu * (self.M * s - 1.0)
            rho = rho_m.repeat_interleave(self.D, dim=1).clamp(0.0, 2.0)
            if want_diag:
                diag["mu"] = mu.detach()
                diag["route_entropy"] = (
                    -(s * torch.log(s.clamp_min(1e-8))).sum(-1).mean().detach())
                diag["route_max"] = s.max(-1).values.mean().detach()
                diag["_route"] = s.detach()

        eff = z if rho is None else (rho * z).clamp(0.0, 1.0)
        h_new = (1 - eff) * h + eff * cand

        if not want_diag:
            return h_new, None
        diag["z"] = z.detach()
        return h_new, diag


class CMRGRU(nn.Module):
    def __init__(self, cfg: CMRConfig):
        super().__init__()
        self.cfg = cfg
        self.variant = "cmr"
        self.cells = nn.ModuleList([
            CMRCell(cfg, cfg.input_size if i == 0 else cfg.hidden_size, i == 0)
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
        acc: Dict[str, list] = {}

        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hs[li], d = cell(inp, hs[li],
                                 want_diag=return_diagnostics and li == 0)
                if d:
                    for k, v in d.items():
                        acc.setdefault(k, []).append(v)
                inp = self.drop(hs[li]) if li < self.cfg.layers - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        diag = {}
        for k in ["gamma_dev", "beta_abs", "route_entropy", "route_max", "z"]:
            if k in acc:
                diag[f"{k}_mean"] = torch.stack(acc[k]).mean()
        for k in ["lam", "mu"]:
            if k in acc:
                diag[k] = acc[k][0]
        if "_route" in acc:
            R = torch.stack(acc["_route"], 1)              # [B,T,M]
            diag["_route_per_window"] = R.mean(dim=1)      # [B,M]
            diag["uniform_entropy"] = torch.tensor(
                math.log(self.cfg.n_modules))
        if "gamma_dev" in acc:
            diag["_gamma_per_window"] = torch.stack(
                acc["gamma_dev"]).mean().expand(B).clone()
        return logits, diag


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


ARMS = {
    "gru":      dict(n_modules=1, use_film=False, use_routing=False),
    "modular":  dict(n_modules=4, use_film=False, use_routing=False),
    "film":     dict(n_modules=1, use_film=True,  use_routing=False),
    "mod_film": dict(n_modules=4, use_film=True,  use_routing=False),
    "cmr":      dict(n_modules=4, use_film=True,  use_routing=True),
}


if __name__ == "__main__":
    torch.manual_seed(42)
    base = count_parameters(CMRGRU(CMRConfig(9, 64, **ARMS["gru"])))
    print(f"in-loop GRU reference (M=1, H=64): {base:,}\n")
    print(f"{'arm':>10} {'H':>4} {'M':>3} {'params':>8} {'vs base':>9}")
    for nm, kw in ARMS.items():
        for H in [64, 72, 80, 88, 96]:
            if H % kw["n_modules"]:
                continue
            n = count_parameters(CMRGRU(CMRConfig(9, H, **kw)))
            if abs(n - base) / base < 0.06 or H == 64:
                print(f"{nm:>10} {H:>4} {kw['n_modules']:>3} {n:>8,} "
                      f"{100*(n-base)/base:>+8.1f}%")
        print()

    m = CMRGRU(CMRConfig(9, 64, **ARMS["cmr"]))
    lo, d = m(torch.randn(8, 60, 9), return_diagnostics=True)
    print(f"logits {tuple(lo.shape)}")
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
