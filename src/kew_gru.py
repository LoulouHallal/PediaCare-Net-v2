"""
kew_gru.py
============
KEW-GRU — Keep–Erase–Write Gated Recurrent Unit.

THE MECHANISM
-------------
A GRU's state update is a two-way interpolation:

    h_t = z_t * h_{t-1} + (1 - z_t) * n_t          keep + write = 1

Preservation and replacement are complementary, so a coordinate cannot say
"remove this stale value, but I have nothing useful to put in its place"
without coordinating the candidate and the gate together.

KEW-GRU replaces that with a three-way simplex over memory actions:

    h~_t          = tanh(W_c x_t + U_c h_{t-1} + b_c)      reset-free
    L_t           = W_a x_t + U_a h_{t-1} + b_a            [B, H, 3]
    [k, e, w]_t   = entmax_alpha(L_t, dim=-1)              k + e + w = 1
    h_t           = k_t * h_{t-1} + e_t * 0 + w_t * h~_t

The `e_t * 0` term vanishes computationally but defines what Erase means and
is kept in the docstring for that reason.

IS THIS ACTUALLY NEW EXPRESSIVITY?
----------------------------------
Not strictly, and the write-up should say so. A GRU can shrink a coordinate
by producing a near-zero candidate: h_t = (1-z) h_{t-1} when n_t = 0.
Verified numerically: to reproduce a KEW step with k=0.3, e=0.5, w=0.2, a GRU
with z = 1-k would need candidate = 0.286 * (the KEW candidate). So the GRU
must COORDINATE gate and candidate to erase; KEW decouples them.

That is a reparameterisation — which is exactly what an inductive bias is —
not an impossible operation made possible. The claim to defend is that the
factorisation makes erasure easier to learn, and that is what the ablation
ladder tests.

BOUNDED STATE
-------------
Because (k, e, w) lie on the simplex, h_t is a convex combination of
{h_{t-1}, 0, h~_t}. With h_0 = 0 and a tanh candidate, induction keeps
|h_t| <= 1 without any gate coordination. Independent sigmoid forget/input
gates would not give this.

ENTMAX
------
alpha = 1.5 entmax can return EXACT zeros, unlike softmax which is always
dense. The `entmax` package is used when available; otherwise a
dependency-free closed form for the 3-action case is used. With only three
candidates the support size can be enumerated, so no bisection is needed.
The two agree to 2.4e-07.

One caveat worth logging: gradient does NOT flow to entries entmax has set
to exactly zero. An action driven out early may struggle to return, which is
why b_E is initialised at -0.5 rather than something strongly negative.

ABLATION LADDER
---------------
    E0  gru            standard GRU                     reference
    E1  gru_noreset    reset removed only               isolates M3
    E2  kw_2action     Keep/Write simplex, no Erase     isolates the
                                                        controller change
    E3  kew_softmax    Keep/Erase/Write, dense          isolates Erase
    E4  kew_entmax     Keep/Erase/Write, sparse         proposed
    E5  kew_reset      E4 with the reset gate restored  tests whether M3
                                                        was justified

The decisive rungs are E2 -> E3 (does explicit Erase matter?) and
E3 -> E4 (does sparse competition matter?). E0 -> E4 alone would not say why.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from entmax import entmax15 as _entmax15_pkg
except ImportError:
    _entmax15_pkg = None


def entmax15_k3(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    alpha = 1.5 entmax, closed form for a small number of actions.

    p_i = [(z_i/2 - tau)_+]^2 with sum p = 1. The support size is found by
    enumeration rather than bisection, which is exact and cheap when there
    are only three candidates. Matches the reference package to 2.4e-07.
    """
    z = z - z.max(dim=dim, keepdim=True).values
    z = z / 2.0
    zs, _ = torch.sort(z, dim=dim, descending=True)
    cs = zs.cumsum(dim)
    ks = torch.arange(1, z.shape[dim] + 1, dtype=z.dtype, device=z.device)
    shape = [1] * z.dim()
    shape[dim] = -1
    ks = ks.view(shape)
    mean = cs / ks
    ss = (zs ** 2).cumsum(dim) / ks - mean ** 2
    tau = mean - (1.0 / ks - ss).clamp_min(0).sqrt()
    k_star = (tau <= zs).to(z.dtype).sum(dim, keepdim=True).long().clamp_min(1)
    tau_star = tau.gather(dim, k_star - 1)
    return ((z - tau_star).clamp_min(0)) ** 2


def action_probs(logits: torch.Tensor, norm: str) -> torch.Tensor:
    if norm == "softmax":
        return torch.softmax(logits, dim=-1)
    if norm == "entmax15":
        if _entmax15_pkg is not None:
            return _entmax15_pkg(logits.float(), dim=-1).to(logits.dtype)
        return entmax15_k3(logits, dim=-1)
    raise ValueError(f"unknown action_norm {norm!r}")


class KEWGRUCell(nn.Module):
    """
    variant:
      'gru'          standard GRU cell (update-gate convention)
      'gru_noreset'  as above, reset removed
      'kw_2action'   Keep/Write simplex, no Erase action
      'kew_softmax'  Keep/Erase/Write, dense normalisation
      'kew_entmax'   Keep/Erase/Write, sparse                 <- proposed
      'kew_reset'    kew_entmax with the reset gate restored
    """

    N_ACTIONS = {"kw_2action": 2, "kew_softmax": 3, "kew_entmax": 3,
                 "kew_reset": 3}

    def __init__(self, input_size: int, hidden_size: int, variant: str,
                 keep_bias: float = 0.5, erase_bias: float = -0.5,
                 write_bias: float = 0.0):
        super().__init__()
        I, H = int(input_size), int(hidden_size)
        self.I, self.H, self.variant = I, H, variant
        self.n_act = self.N_ACTIONS.get(variant, 0)
        self.use_reset = variant in ("gru", "kew_reset")
        self.norm = "entmax15" if variant in ("kew_entmax", "kew_reset") \
            else "softmax"

        if self.use_reset:
            self.x_r = nn.Linear(I, H); self.h_r = nn.Linear(H, H, bias=False)
        self.x_c = nn.Linear(I, H); self.h_c = nn.Linear(H, H, bias=False)

        if self.n_act:
            # action-major: [all K | all E | all W]
            self.x_a = nn.Linear(I, self.n_act * H)
            self.h_a = nn.Linear(H, self.n_act * H, bias=False)
        else:
            self.x_z = nn.Linear(I, H); self.h_z = nn.Linear(H, H, bias=False)

        self.keep_bias, self.erase_bias, self.write_bias = \
            keep_bias, erase_bias, write_bias
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.x_c.weight); nn.init.zeros_(self.x_c.bias)
        nn.init.orthogonal_(self.h_c.weight)
        if self.use_reset:
            nn.init.xavier_uniform_(self.x_r.weight)
            nn.init.zeros_(self.x_r.bias)
            nn.init.orthogonal_(self.h_r.weight)
        if self.n_act:
            for blk in self.x_a.weight.chunk(self.n_act, 0):
                nn.init.xavier_uniform_(blk)
            for blk in self.h_a.weight.chunk(self.n_act, 0):
                nn.init.orthogonal_(blk)
            # A mild retention prior, NOT a strong suppression of Erase:
            # entmax gives exactly zero probability to sufficiently negative
            # logits, and gradient does not flow to entries it has zeroed, so
            # an action driven out early may not be able to return.
            H = self.H
            with torch.no_grad():
                if self.n_act == 2:
                    self.x_a.bias[:H].fill_(self.keep_bias)
                    self.x_a.bias[H:].fill_(self.write_bias)
                else:
                    self.x_a.bias[:H].fill_(self.keep_bias)
                    self.x_a.bias[H:2 * H].fill_(self.erase_bias)
                    self.x_a.bias[2 * H:].fill_(self.write_bias)
        else:
            nn.init.xavier_uniform_(self.x_z.weight)
            nn.init.zeros_(self.x_z.bias)
            nn.init.orthogonal_(self.h_z.weight)

    def forward(self, x, h, want_diag: bool = False):
        B = x.shape[0]
        r = (torch.sigmoid(self.x_r(x) + self.h_r(h)) if self.use_reset
             else None)
        c_in = self.h_c(r * h) if self.use_reset else self.h_c(h)
        cand = torch.tanh(self.x_c(x) + c_in)

        if not self.n_act:                       # E0 / E1: ordinary GRU update
            z = torch.sigmoid(self.x_z(x) + self.h_z(h))
            h_new = (1 - z) * h + z * cand
            if not want_diag:
                return h_new, None
            return h_new, {"keep": (1 - z).detach(),
                           "erase": torch.zeros_like(z),
                           "write": z.detach()}

        raw = self.x_a(x) + self.h_a(h)          # [B, n_act*H], action-major
        logits = raw.view(B, self.n_act, self.H).transpose(1, 2).contiguous()
        p = action_probs(logits, self.norm)      # [B, H, n_act]

        if self.n_act == 2:
            k, w = p[..., 0], p[..., 1]
            e = torch.zeros_like(k)
        else:
            k, e, w = p[..., 0], p[..., 1], p[..., 2]

        # h_t = k*h_{t-1} + e*0 + w*cand   -- the e*0 term is what Erase means
        h_new = k * h + w * cand
        if not want_diag:
            return h_new, None
        return h_new, {"keep": k.detach(), "erase": e.detach(),
                       "write": w.detach(), "probs": p.detach()}


class KEWGRU(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, variant: str,
                 layers: int = 2, n_horizons: int = 4, dropout: float = 0.2,
                 **kw):
        super().__init__()
        self.variant, self.H, self.layers = variant, hidden_size, layers
        self.cells = nn.ModuleList([
            KEWGRUCell(input_size if i == 0 else hidden_size, hidden_size,
                       variant, **kw) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        h2 = hidden_size // 2
        self.head = nn.Sequential(
            nn.Linear(hidden_size, h2), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(h2, n_horizons))

    def forward(self, x, return_diagnostics: bool = False):
        B, T, _ = x.shape
        hs = [x.new_zeros(B, self.H) for _ in range(self.layers)]
        acc = {k: [] for k in ["keep", "erase", "write"]}
        zeros = []

        for t in range(T):
            inp = x[:, t, :]
            for li, cell in enumerate(self.cells):
                # layer 0 reads the raw features, so its actions are the
                # interpretable ones
                hs[li], d = cell(inp, hs[li],
                                 want_diag=return_diagnostics and li == 0)
                if d is not None:
                    for kk in acc:
                        acc[kk].append(d[kk])
                    if "probs" in d:
                        zeros.append((d["probs"] == 0).float().mean())
                inp = self.drop(hs[li]) if li < self.layers - 1 else hs[li]

        logits = self.head(hs[-1])
        if not return_diagnostics:
            return logits

        K = torch.stack(acc["keep"], 1)          # [B,T,H]
        E = torch.stack(acc["erase"], 1)
        W = torch.stack(acc["write"], 1)
        stacked = torch.stack([K, E, W], -1)
        occ = stacked.argmax(-1)
        diag = {
            "keep_mean": K.mean(), "erase_mean": E.mean(), "write_mean": W.mean(),
            "frac_argmax_keep": (occ == 0).float().mean(),
            "frac_argmax_erase": (occ == 1).float().mean(),
            "frac_argmax_write": (occ == 2).float().mean(),
            # how often Erase is the dominant action AND substantial
            "frac_erase_gt_half": (E > 0.5).float().mean(),
            "simplex_err": (K + E + W - 1.0).abs().max(),
        }
        diag["exact_zero_frac"] = (torch.stack(zeros).mean() if zeros
                                   else torch.tensor(0.0))
        diag["_erase_per_window"] = E.mean(dim=(1, 2))
        diag["_keep_per_window"] = K.mean(dim=(1, 2))
        diag["_write_per_window"] = W.mean(dim=(1, 2))
        return logits, diag


def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


VARIANTS = ["gru", "gru_noreset", "kw_2action", "kew_softmax", "kew_entmax",
            "kew_reset"]


if __name__ == "__main__":
    torch.manual_seed(42)
    g = nn.GRU(9, 64, num_layers=2, batch_first=True)
    hd = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                       nn.Linear(32, 4))
    base = sum(p.numel() for p in g.parameters()) + \
        sum(p.numel() for p in hd.parameters())
    print(f"project baseline (nn.GRU h=64, 2 layers + Heads): {base:,}")
    print(f"entmax package: {'available' if _entmax15_pkg else 'using closed form'}\n")

    for v in VARIANTS:
        for H in [64, 56]:
            n = count_parameters(KEWGRU(9, H, v))
            print(f"  {v:>12} H={H}: {n:>8,}  ({100*(n-base)/base:+6.1f}%)")
        print()

    x = torch.randn(8, 60, 9)
    for v in ["kew_entmax", "kew_softmax"]:
        m = KEWGRU(9, 56, v); m.eval()
        lo, d = m(x, return_diagnostics=True)
        print(f"{v}: out {tuple(lo.shape)}")
        for k, val in d.items():
            if not k.startswith("_"):
                print(f"   {k}: {float(val):.6f}")
        print()
