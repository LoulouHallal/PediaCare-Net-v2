"""
xgru.py
=========
xGRU — exponential similarity gating with low-rank matrix memory.

WHY THIS AXIS
-------------
Every mechanism tested in this project so far manipulates a single hidden
VECTOR h_t: which coordinates are kept, erased, written, delayed,
corrected, or scaled. Six corrected diagnostics say the information those
mechanisms reach for is already available to the baseline.

Matrix memory changes something none of them touched: the recurrent state
is no longer a vector but an associative matrix M_t, holding relationships
BETWEEN latent features rather than a compressed summary of them. That is
the one recurrent axis the diagnostics cannot speak to, and unlike KEW or
ADEW it comes with external evidence of a material gain elsewhere
(xLSTM's exponential gating and matrix memory; subsequent xGRU transfers
in traffic forecasting and biometrics).

This is a TRANSFER experiment, not the proposed contribution. The question
is narrow: does this memory structure help on this data at all? Deriving a
new cell from it is only worth doing if the answer is yes.

THE CELL
--------
Base GRU, unchanged:

    r_t = sigmoid(W_r x + U_r h + b_r)
    z_t = sigmoid(W_z x + U_z h + b_z)
    u_t = tanh(W_u x + U_u (r_t * h) + b_u)
    hG  = (1 - z_t) h + z_t u_t

Low-rank associative memory, rank R << H:

    k_t, v_t, q_t = R-dim projections of [x_t, h_{t-1}]
    d_t = mean_j (P_x x - P_h h)^2                  input/state discrepancy
    s_t = exp(-softplus(theta) * d_t)               in (0, 1]
    M_t = s_t M_{t-1} + (1 - s_t) v_t k_t^T
    m_t = M_t q_t / (||M_t q_t|| + eps)

    h_t = hG + beta_t * tanh(P m_t)

s_t is the exponential gate: when the current input matches what the state
expects, discrepancy is low, s_t -> 1, and the memory is retained. When
they disagree, s_t -> 0 and the memory is overwritten. This is gating by
input/state AGREEMENT rather than by an affine sigmoid of both.

The read is L2-normalised because the outer-product accumulation is
otherwise unbounded in scale; xLSTM stabilises exponential gating for the
same reason.

NESTING, STATED HONESTLY
------------------------
beta_t = alpha * sigmoid(...) with alpha = 1 - exp(-rho^2) in [0, 1), so
at alpha = 0 this is exactly the GRU. That is a good initialisation and a
clean fallback — it is NOT a guarantee of better held-out performance.
ADEW nested the GRU exactly and still lost: nesting bounds the training
optimum, not generalisation. alpha is initialised at 0.01 rather than 0
because at exactly 0 the memory projections receive no gradient (measured:
|grad| 0.000000 at 0, 0.107 at 0.01), and alpha cannot go negative, which
is the failure mode ADEW's unconstrained signed anchor hit.

DETERMINISM
-----------
The pilot imports seed_everything and make_loader. Two nominally identical
hand-written-loop runs previously differed by 0.00168 mean AUPRC, larger
than any architecture effect measured since AR-RHU.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn


@dataclass
class XGRUConfig:
    input_size: int
    hidden_size: int = 64
    rank: int = 8
    layers: int = 2
    n_horizons: int = 4
    dropout: float = 0.2
    use_memory: bool = True
    alpha_init: float = 0.01


class XGRUCell(nn.Module):
    def __init__(self, cfg: XGRUConfig, input_size: int):
        super().__init__()
        self.cfg = cfg
        I, H, R = int(input_size), cfg.hidden_size, cfg.rank
        self.H, self.R = H, R

        self.x_r = nn.Linear(I, H); self.h_r = nn.Linear(H, H, bias=False)
        self.x_z = nn.Linear(I, H); self.h_z = nn.Linear(H, H, bias=False)
        self.x_u = nn.Linear(I, H); self.h_u = nn.Linear(H, H, bias=False)

        if cfg.use_memory:
            self.x_k = nn.Linear(I, R); self.h_k = nn.Linear(H, R, bias=False)
            self.x_v = nn.Linear(I, R); self.h_v = nn.Linear(H, R, bias=False)
            self.x_q = nn.Linear(I, R); self.h_q = nn.Linear(H, R, bias=False)
            # matched projections for the similarity gate
            self.P_x = nn.Linear(I, R); self.P_h = nn.Linear(H, R, bias=False)
            self.theta = nn.Parameter(torch.tensor(0.0))     # softplus -> ~0.69
            self.read = nn.Linear(R, H, bias=False)
            self.x_b = nn.Linear(I, H); self.h_b = nn.Linear(H, H, bias=False)
            # alpha = 1 - exp(-rho^2), bounded in [0,1), cannot go negative
            a0 = float(cfg.alpha_init)
            self.rho = nn.Parameter(torch.tensor(math.sqrt(-math.log(1 - a0))))
        self.reset_parameters()

    def alpha(self):
        return 1.0 - torch.exp(-self.rho ** 2)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for nm in ["h_r", "h_z", "h_u"]:
            nn.init.orthogonal_(getattr(self, nm).weight)

    def forward(self, x, h, M=None, want_diag=False):
        r = torch.sigmoid(self.x_r(x) + self.h_r(h))
        z = torch.sigmoid(self.x_z(x) + self.h_z(h))
        u = torch.tanh(self.x_u(x) + self.h_u(r * h))
        hG = (1 - z) * h + z * u

        if not self.cfg.use_memory:
            return hG, None, ({"z": z.detach()} if want_diag else None)

        k = self.x_k(x) + self.h_k(h)
        v = self.x_v(x) + self.h_v(h)
        q = self.x_q(x) + self.h_q(h)

        # exponential similarity gate: agreement between input and state
        d = ((self.P_x(x) - self.P_h(h)) ** 2).mean(-1, keepdim=True)
        s = torch.exp(-torch.nn.functional.softplus(self.theta) * d)  # (0,1]

        M_new = s.unsqueeze(-1) * M + (1 - s).unsqueeze(-1) * \
            torch.einsum("bi,bj->bij", v, k)
        m = torch.einsum("bij,bj->bi", M_new, q)
        m = m / (m.norm(dim=-1, keepdim=True) + 1e-6)

        beta = self.alpha() * torch.sigmoid(self.x_b(x) + self.h_b(h))
        h_new = hG + beta * torch.tanh(self.read(m))

        if not want_diag:
            return h_new, M_new, None
        return h_new, M_new, {
            "z": z.detach(), "s": s.detach(), "beta": beta.detach(),
            "alpha": self.alpha().detach(),
            "mem_norm": M_new.norm(dim=(1, 2)).detach(),
            "read_norm": m.norm(dim=-1).detach(),
        }


class XGRU(nn.Module):
    def __init__(self, cfg: XGRUConfig):
        super().__init__()
        self.cfg = cfg
        self.variant = "xgru" if cfg.use_memory else "gru"
        self.cells = nn.ModuleList([
            XGRUCell(cfg, cfg.input_size if i == 0 else cfg.hidden_size)
            for i in range(cfg.layers)])
        self.drop = nn.Dropout(cfg.dropout)
        h2 = cfg.hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(cfg.hidden_size, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, cfg.n_horizons))

    def forward(self, x, return_diagnostics: bool = False):
        B, T, _ = x.shape
        H, R, L = self.cfg.hidden_size, self.cfg.rank, self.cfg.layers
        hs = [x.new_zeros(B, H) for _ in range(L)]
        Ms = [x.new_zeros(B, R, R) for _ in range(L)] \
            if self.cfg.use_memory else [None] * L
        acc: Dict[str, list] = {}

        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                hs[li], Ms[li], d = cell(
                    inp, hs[li], Ms[li],
                    want_diag=return_diagnostics and li == 0)
                if d is not None:
                    for kk, vv in d.items():
                        acc.setdefault(kk, []).append(vv)
                inp = self.drop(hs[li]) if li < L - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        diag = {}
        for kk in ["z", "s", "beta", "mem_norm", "read_norm"]:
            if kk in acc:
                st = torch.stack(acc[kk], 1)
                diag[f"{kk}_mean"] = st.mean()
                if kk == "s":
                    diag["s_min"] = st.min()
                    diag["s_sd"] = st.std()
                if kk == "beta":
                    diag["beta_max"] = st.max()
                    diag["_beta_per_window"] = st.mean(dim=tuple(
                        range(1, st.dim())))
        if "alpha" in acc:
            diag["alpha"] = acc["alpha"][0]
        if "s" in acc:
            diag["_s_per_window"] = torch.stack(acc["s"], 1).mean(
                dim=tuple(range(1, torch.stack(acc["s"], 1).dim())))
        return logits, diag


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    base = count_parameters(XGRU(XGRUConfig(9, 64, use_memory=False)))
    print(f"in-loop GRU reference (H=64): {base:,}\n")
    print(f"{'rank':>5} {'H':>4} {'params':>8} {'vs base':>9}")
    for R in [4, 8, 16]:
        for H in [64, 56, 52, 48]:
            n = count_parameters(XGRU(XGRUConfig(9, H, rank=R)))
            if abs(n - base) / base < 0.25:
                print(f"{R:>5} {H:>4} {n:>8,} {100*(n-base)/base:>+8.1f}%")

    m = XGRU(XGRUConfig(9, 64, rank=8))
    lo, d = m(torch.randn(8, 60, 9), return_diagnostics=True)
    print(f"\nlogits {tuple(lo.shape)}")
    for k, v in d.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
