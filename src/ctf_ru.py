"""
ctf_ru.py — CTF-RU v2
=======================
Clinical Threshold-Flux Recurrent Unit.

The hidden state is not an unstructured vector. It is distributed over an
evenly spaced grid of clinical glucose states, and latent evidence is
physically transported through that grid according to the observed
glucose velocity. The quantity crossing the 70 mg/dL boundary is
accumulated as explicit "approach to hypoglycaemia" memory.

WHAT CHANGED FROM v1, AND WHY
-----------------------------
v1 moved a fraction tanh(v / 1.0) of the state one bin per step. Three
measured problems:

* At a sustained fall of 1 mg/dL/min, 99.8% of latent mass reached the
  bottom bin within 12 of 60 steps. The state pinned to the grid edge
  during exactly the trajectories the architecture exists to model.
* The grid (40/60/70/80/100/130/190) had gaps from 10 to 60 mg/dL, but
  one transport step moved one bin regardless, so the same physical rate
  implied a 6x different latent speed depending on position.
* rate_scale = 1.0 saturated tanh at typical CGM rates (0.5-2.0
  mg/dL/min), which is what drove the first problem.

v2 replaces the heuristic with physical advection on a uniform grid.

MEASURED CONSTRAINT
-------------------
A rate audit over 18.1M readings found |v| exceeding 4 mg/dL/min in
1.65% of readings -- and 27.8% of 60-step windows contain at least one
such step, with the rate HIGHER before events (1.93%) than before
non-events (1.38%). Single-neighbour transport with clipping would
therefore have systematically understated the fastest falls, biased
precisely against the cases that matter. Hence multi-cell transport.

THE TRANSPORT BLOCK
-------------------
1. Physical displacement, in bins. No clipping, no tanh, no free scale:

       delta_t = v_t * dt / dx        (dt = 5 min, dx = 20 mg/dL)

2. Conservative multi-cell remap. Each SOURCE bin deposits its content at
   destination k + delta, split linearly between the two neighbouring
   bins; overflow past an edge stays in the edge bin:

       H_move = Remap(H_{t-1}, delta_t),   sum_k H_move = sum_k H_{t-1}

   Verified exact to float precision for displacements from 0 to 18 bins.
   Semi-Lagrangian interpolation was rejected: it loses 2-4% of the state
   per step through edge clamping, and H is a signed learned
   representation, so post-hoc renormalisation would alter its amplitude
   for purely numerical reasons.

3. Learned participation. The network decides how much of the latent
   evidence follows the physical movement -- it does NOT get to rescale
   the movement itself:

       alpha_t = sigmoid(W_a e_t + b_a)
       Hbar_t  = (1 - alpha_t) H_{t-1} + alpha_t H_move

   A convex combination of two states with equal total content preserves
   that total, so the whole stage remains conservative.

4. Boundary flux, using the same alpha for consistency. Treating each
   bin's content as uniform over [k-1/2, k+1/2], with the 70 mg/dL
   boundary at grid coordinate 2.5:

       f_before(k) = clip(3 - k, 0, 1)
       f_after(k)  = clip(3 - k - delta_t, 0, 1)
       F_t         = alpha_t * sum_k relu(f_after - f_before) H_{t-1,k}

   Computing flux from the full physical displacement while transporting
   only a fraction of the state would be inconsistent.

5. Flux memory:  M_t = rho M_{t-1} + (1 - rho) phi(F_t),  rho init 0.90,
   which at 5-minute cadence is roughly a 33-minute half-life.

THE GRID
--------
Centres 20, 40, ..., 220 mg/dL with dx = 20 throughout. The clinical
threshold 70 falls BETWEEN bin 2 (60) and bin 3 (80) rather than on a bin
centre, so downward flux 80 -> 60 crosses exactly 70, and the readout
uses the two states flanking the boundary. Membership width is dx/2 = 10,
so at 70 mg/dL the 60 and 80 states receive near-equal weight.

Note that the number of bins does not affect the parameter count --
update_gate and candidate are shared across bins -- only compute.

INPUTS
------
x_context     [B,T,F]  normalised contextual channels
glucose_mgdl  [B,T]    physical glucose
rate_mgdl_min [B,T]    physical rate

The TA channels are deliberately NOT passed in the primary arm. The point
is to test whether the architecture constructs threshold awareness
internally rather than being handed the engineered representation.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ClinicalThresholdFluxRU(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_per_bin: int = 55,
        head_hidden: int = 114,
        n_horizons: int = 4,
        threshold_mgdl: float = 70.0,
        grid_min_mgdl: float = 20.0,
        grid_max_mgdl: float = 220.0,
        dx_mgdl: float = 20.0,
        dt_min: float = 5.0,
        membership_sigma_mgdl: float | None = None,
        flux_decay_init: float = 0.90,
        dropout: float = 0.10,
    ):
        super().__init__()
        if not (0.0 < flux_decay_init < 1.0):
            raise ValueError("flux_decay_init must be in (0, 1).")

        centers = torch.arange(grid_min_mgdl, grid_max_mgdl + 1e-6, dx_mgdl,
                               dtype=torch.float32)
        if centers.numel() < 4:
            raise ValueError("grid needs at least 4 states.")
        if not (centers[0] < threshold_mgdl < centers[-1]):
            raise ValueError("threshold must lie inside the grid range.")

        # The threshold sits BETWEEN two states, not on one. below_idx is the
        # last state under it, above_idx the first over it, and the boundary
        # coordinate is the midpoint between them.
        below = int((centers < threshold_mgdl).sum().item()) - 1
        above = below + 1
        self.below_idx, self.above_idx = below, above
        self.boundary_coord = below + (threshold_mgdl - float(centers[below])) / dx_mgdl

        # the pilot harness dispatches on this
        self.variant = "ctf_ru"
        self.register_buffer("bin_centers_mgdl", centers)
        self.n_bins = int(centers.numel())
        self.input_size = int(input_size)
        self.hidden_per_bin = int(hidden_per_bin)
        self.n_horizons = int(n_horizons)
        self.threshold_mgdl = float(threshold_mgdl)
        self.dx = float(dx_mgdl)
        self.dt = float(dt_min)
        self.sigma_b = float(membership_sigma_mgdl if membership_sigma_mgdl
                             is not None else dx_mgdl / 2.0)

        D = self.hidden_per_bin
        self.input_proj = nn.Sequential(
            nn.Linear(self.input_size, D), nn.LayerNorm(D), nn.Tanh())

        # decides PARTICIPATION in the physical movement, never its magnitude
        self.alpha_proj = nn.Linear(D, 1)

        cell_in = 2 * D + 1                     # [evidence, transported, membership]
        self.update_gate = nn.Linear(cell_in, D)
        self.candidate = nn.Linear(cell_in, D)

        self.flux_proj = nn.Sequential(nn.Linear(D, D), nn.Tanh())
        self.logit_flux_decay = nn.Parameter(
            torch.logit(torch.tensor(float(flux_decay_init))).clone())

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Sequential(
            nn.Linear(4 * D, head_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(head_hidden, self.n_horizons))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.alpha_proj.bias, 0.0)      # alpha starts at 0.5

    @staticmethod
    def _squeeze_last(x: torch.Tensor) -> torch.Tensor:
        return x[..., 0] if (x.ndim == 3 and x.shape[-1] == 1) else x

    # ─── grid operations ──────────────────────────────────────────────────

    def membership(self, glucose_mgdl: torch.Tensor) -> torch.Tensor:
        """Soft position over clinical states. [B] -> [B,K]."""
        g = glucose_mgdl.unsqueeze(-1)
        c = self.bin_centers_mgdl.to(dtype=g.dtype, device=g.device)
        z = (g - c) / self.sigma_b
        return torch.softmax(-0.5 * z.square(), dim=-1)

    def displacement(self, rate_mgdl_min: torch.Tensor) -> torch.Tensor:
        """delta = v * dt / dx, in bins. Purely physical."""
        return rate_mgdl_min * (self.dt / self.dx)

    def _remap(self, H: torch.Tensor, delta: torch.Tensor
               ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Conservative forward remap. Returns (H_move, lower_edge_hit,
        upper_edge_hit), the last two being the fraction of source states
        whose destination fell outside the grid.
        """
        B, K, D = H.shape
        k = torch.arange(K, dtype=H.dtype, device=H.device)[None, :]
        raw = k + delta[:, None]
        lower_hit = (raw < 0).to(H.dtype).mean(dim=1)
        upper_hit = (raw > K - 1).to(H.dtype).mean(dim=1)

        d = torch.clamp(raw, 0, K - 1)
        j = d.floor().long()
        w = (d - j.to(H.dtype)).unsqueeze(-1)
        j1 = torch.clamp(j + 1, max=K - 1)

        out = torch.zeros_like(H)
        idx = j.unsqueeze(-1).expand(-1, -1, D)
        idx1 = j1.unsqueeze(-1).expand(-1, -1, D)
        out.scatter_add_(1, idx, (1 - w) * H)
        out.scatter_add_(1, idx1, w * H)
        return out, lower_hit, upper_hit

    def _boundary_flux(self, H: torch.Tensor, delta: torch.Tensor,
                       alpha: torch.Tensor) -> torch.Tensor:
        """
        Latent content crossing the threshold downward this step, scaled by
        the same alpha that governs the transport.
        """
        K = H.shape[1]
        k = torch.arange(K, dtype=H.dtype, device=H.device)[None, :]
        edge = self.boundary_coord + 0.5
        f_before = torch.clamp(edge - k, 0, 1)
        f_after = torch.clamp(edge - k - delta[:, None], 0, 1)
        cross = F.relu(f_after - f_before)                    # [B,K]
        return alpha[:, None] * (cross.unsqueeze(-1) * H).sum(dim=1)

    # ─── forward ──────────────────────────────────────────────────────────

    def forward(self, x_context: torch.Tensor, glucose_mgdl: torch.Tensor,
                rate_mgdl_min: torch.Tensor, return_diagnostics: bool = False):
        if x_context.ndim != 3:
            raise ValueError(f"x_context must be [B,T,F], got "
                             f"{tuple(x_context.shape)}")
        g_all = self._squeeze_last(glucose_mgdl)
        r_all = self._squeeze_last(rate_mgdl_min)
        if g_all.shape != x_context.shape[:2] or r_all.shape != x_context.shape[:2]:
            raise ValueError("glucose_mgdl and rate_mgdl_min must be [B,T] "
                             "matching x_context.")

        B, T, _ = x_context.shape
        D, K = self.hidden_per_bin, self.n_bins
        H = x_context.new_zeros(B, K, D)
        flux_mem = x_context.new_zeros(B, D)
        rho = torch.sigmoid(self.logit_flux_decay)

        (d_alpha, d_delta, d_flux, d_ent, d_lo, d_hi,
         d_cons_move, d_cons_bar) = ([] for _ in range(8))
        b_t = None

        for t in range(T):
            e_t = self.input_proj(x_context[:, t, :])          # [B,D]
            b_t = self.membership(g_all[:, t])                 # [B,K]
            alpha = torch.sigmoid(self.alpha_proj(e_t).squeeze(-1))   # [B]
            delta = self.displacement(r_all[:, t])             # [B]

            H_move, lo_hit, hi_hit = self._remap(H, delta)
            a3 = alpha[:, None, None]
            H_bar = (1 - a3) * H + a3 * H_move                 # conservative
            if return_diagnostics:
                # Conservation holds for H -> H_move -> H_bar ONLY. The
                # recurrent update below is free to change the magnitude of
                # the hidden representation, so measuring "mass" after it
                # would not test the transport at all.
                before = H.sum(dim=(1, 2))
                d_cons_move.append((H_move.sum(dim=(1, 2)) - before).abs().detach())
                d_cons_bar.append((H_bar.sum(dim=(1, 2)) - before).abs().detach())
            flux = self._boundary_flux(H, delta, alpha)        # [B,D]

            e_rep = e_t[:, None, :].expand(-1, K, -1)
            cell_in = torch.cat([e_rep, H_bar, b_t.unsqueeze(-1)], dim=-1)
            z = torch.sigmoid(self.update_gate(cell_in))
            cand = torch.tanh(self.candidate(cell_in))
            # evidence enters in proportion to where glucose actually is;
            # no unexplained floor constant
            z_eff = z * b_t.unsqueeze(-1)
            H = (1 - z_eff) * H_bar + z_eff * cand

            flux_mem = rho * flux_mem + (1 - rho) * self.flux_proj(flux)

            if return_diagnostics:
                d_alpha.append(alpha.detach())
                d_delta.append(delta.detach())
                d_flux.append(flux.norm(dim=-1).detach())
                d_ent.append(-(b_t * torch.log(b_t.clamp_min(1e-8))).sum(-1).detach())
                d_lo.append(lo_hit.detach())
                d_hi.append(hi_hit.detach())

        current = (b_t.unsqueeze(-1) * H).sum(dim=1)
        below = H[:, self.below_idx, :]
        above = H[:, self.above_idx, :]
        logits = self.head(self.dropout(
            torch.cat([current, below, above, flux_mem], dim=-1)))

        if not return_diagnostics:
            return logits

        diag: Dict[str, torch.Tensor] = {
            "alpha_mean": torch.stack(d_alpha, 1).mean(),
            "alpha_sd": torch.stack(d_alpha, 1).std(),
            "delta_abs_mean": torch.stack(d_delta, 1).abs().mean(),
            "delta_abs_max": torch.stack(d_delta, 1).abs().max(),
            "boundary_flux_norm_mean": torch.stack(d_flux, 1).mean(),
            "membership_entropy_mean": torch.stack(d_ent, 1).mean(),
            "edge_hit_rate_lower": torch.stack(d_lo, 1).mean(),
            "edge_hit_rate_upper": torch.stack(d_hi, 1).mean(),
            "flux_decay": rho.detach(),
            # numerical conservation error of the transport stage
            "conservation_err_move": torch.stack(d_cons_move, 1).max(),
            "conservation_err_bar": torch.stack(d_cons_bar, 1).max(),
            "boundary_membership_mean": (
                b_t[:, self.below_idx] + b_t[:, self.above_idx]).mean().detach(),
        }
        # Per-window traces, so alpha and flux can be split by outcome
        # outside the model without a second forward pass. Named _norm
        # because what is stored is ||F_t||, not the signed latent flux.
        diag["_flux_norm_per_window"] = torch.stack(d_flux, 1).mean(dim=1)
        diag["_alpha_per_window"] = torch.stack(d_alpha, 1).mean(dim=1)
        return logits, diag


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, Fdim = 8, 60, 5

    model = ClinicalThresholdFluxRU(input_size=Fdim, hidden_per_bin=55,
                                    head_hidden=114, n_horizons=4)
    print(f"grid: {model.bin_centers_mgdl.tolist()}")
    print(f"threshold {model.threshold_mgdl} between index "
          f"{model.below_idx} ({model.bin_centers_mgdl[model.below_idx]:.0f}) "
          f"and {model.above_idx} "
          f"({model.bin_centers_mgdl[model.above_idx]:.0f}), "
          f"boundary coord {model.boundary_coord:.2f}")
    print(f"params: {count_trainable_parameters(model):,}  "
          f"(GRU+TA+Absolute baseline 41,572)\n")

    x = torch.randn(B, T, Fdim)
    g = 140.0 + torch.cumsum(torch.randn(B, T) * 2.0, dim=1)
    rate = torch.zeros_like(g)
    rate[:, 1:] = (g[:, 1:] - g[:, :-1]) / 5.0

    logits, diag = model(x, g, rate, return_diagnostics=True)
    print("logits:", tuple(logits.shape))
    for k, v in diag.items():
        if not k.startswith("_"):
            print(f"  {k}: {float(v):.6f}")
