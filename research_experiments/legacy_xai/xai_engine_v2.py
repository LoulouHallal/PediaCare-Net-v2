"""
xai_engine_v2.py  --  Temporal IG + DiCE counterfactuals for PediaCare-Net v2
==============================================================================

Ported from the Phase-3 engine written for the earlier cohort. The method
is the same; four things had to change for this dataset and these models,
and each one is a correctness fix rather than a preference.

1. MODEL INTERFACE
   The old model returned a dict keyed by horizon. TSLGRU returns a
   (B, 4) tensor of LOGITS, so a sigmoid is applied and horizons are
   selected by index.

2. BASELINE
   The old engine used a zero baseline described as "absence of any
   signal". bw.timeline is z-scored with a causal expanding mean, so zero
   is the SUBJECT'S RUNNING MEAN, not absence. The attribution is
   therefore relative to an average trajectory, and is documented as
   such. A true "absence" baseline does not exist in this representation.

3. TA CONSISTENCY  (the important one)
   ta_proximity_c and ta_downslope_vplus are deterministic functions of
   raw glucose:

       c_t  = sigmoid(-(g_t - 70) / sigma_t)
       v+_t = softplus((g_{t-1} - g_t) / sigma_t)

   A counterfactual that changes glucose without recomputing them
   produces an input that cannot physically occur -- glucose says one
   thing, the threshold channels say another. Models are free to exploit
   that inconsistency, which is how "counterfactuals" that move the
   prediction without corresponding to any real intervention arise. Here
   the TA channels are RECOMPUTED from perturbed glucose inside the
   optimisation loop.

4. ACTIONABILITY
   carbs_observed is a SUBJECT-LEVEL constant (1 if that subject records
   carbs at all) and is locked: whether something was written down is not
   an intervention. carbs are absent for 73 of 244 subjects, so carb
   counterfactuals are only generated where carbs_observed = 1.

   Glucose itself is NOT actionable. "If the child's glucose were higher"
   is not an action anyone can take.

VALIDITY REPORTING
------------------
Every counterfactual carries a validity record: how far each changed
channel moved in RAW units (grams, U, U/hr) and whether that value falls
inside the observed distribution for that channel. A counterfactual
requiring a carb quantity never seen in the cohort is an artifact, not an
intervention, and is reported as such rather than silently averaged into
a success rate.

The engine reports what the MODEL would predict under a changed input.
That is an associational statement, not a causal one: it does not
establish that performing the action would prevent hypoglycaemia.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

HYPO_MGDL = 70.0
TA_CLAMP = 10.0
FEAT_CLAMP = 10.0
HORIZONS = [15, 30, 60, 120]

FEATURE_NAMES = ["glucose", "basal", "bolus", "carbs", "carbs_observed",
                 "ta_proximity_c", "ta_downslope_vplus"]

CLINICAL = {
    "glucose":            "Blood glucose (CGM)",
    "basal":              "Basal insulin rate",
    "bolus":              "Insulin bolus",
    "carbs":              "Carbohydrate intake",
    "carbs_observed":     "Carb recording available (dataset flag)",
    "ta_proximity_c":     "Proximity to 70 mg/dL",
    "ta_downslope_vplus": "Rate of glucose fall",
}

# Short labels for figures and JSON. The raw column names are internal
# identifiers and mean nothing to a reader; carbs_observed is flagged as a
# dataset artifact in its own label because it is a subject-level constant
# (1 if that subject records carbs at all), not a physiological quantity.
DISPLAY = {
    "glucose":            "Glucose (CGM)",
    "basal":              "Basal rate",
    "bolus":              "Bolus",
    "carbs":              "Carbs",
    "carbs_observed":     "Carb-recording flag*",
    "ta_proximity_c":     "Proximity to 70",
    "ta_downslope_vplus": "Rate of fall",
}
DISPLAY_NAMES = [DISPLAY[n] for n in FEATURE_NAMES]

# Smallest change worth reporting, in RAW clinical units. Below this a
# "counterfactual" is a rounding artifact: it prints as "increase by 0.0
# U/hr" and moves the prediction not at all.
MIN_RAW_CHANGE = {"basal": 0.05, "bolus": 0.1, "carbs": 1.0}

# A counterfactual must also move the prediction by at least this much to
# be worth showing at all.
MIN_RISK_DELTA = 0.005

RAW_UNITS = {"basal": "U/hr", "bolus": "U", "carbs": "g"}

# basal is a RATE (U/hr): summing 60 per-reading changes is meaningless, the
# clinically sensible summary is the mean rate change. bolus and carbs are
# discrete EVENTS: their total over the window is what a person would give.
AGG = {"basal": "mean", "bolus": "sum", "carbs": "sum"}

IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}

# Only these can be intervened on. glucose is an OUTCOME, not an action;
# carbs_observed is a recording flag; TA channels are derived from glucose
# and cannot be set independently.
ACTIONABLE = ["basal", "bolus", "carbs"]

TRAFFIC_GREEN, TRAFFIC_AMBER = 0.30, 0.60


# ─── data structures ────────────────────────────────────────────────────────
@dataclass
class Counterfactual:
    cf_id: int
    predicted_risk: float
    risk_reduction: float
    changes_norm: Dict[str, float] = field(default_factory=dict)
    changes_raw: Dict[str, float] = field(default_factory=dict)
    validity: Dict[str, Dict] = field(default_factory=dict)
    plausible: bool = True
    action_en: str = ""


@dataclass
class XAIResult:
    original_risk: float
    traffic_light: str
    attribution: np.ndarray            # (T, F) absolute attribution
    top_features: List[Dict]
    top_timesteps: List[Dict]
    counterfactuals: List[Counterfactual]
    ig_summary_en: str = ""
    cf_note: str = ""


def traffic_light(r: float) -> str:
    return "GREEN" if r < TRAFFIC_GREEN else ("AMBER" if r < TRAFFIC_AMBER
                                              else "RED")


# ─── model adapter ──────────────────────────────────────────────────────────
class ModelAdapter:
    """
    Wraps TSLGRU so the XAI code sees a single probability per horizon.

    TSLGRU.forward(x) -> (B, 4) logits. The old engine indexed a dict by
    horizon value; here horizons are positional.
    """

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model, self.device = model, device

    def prob(self, x: torch.Tensor, h_idx: int) -> torch.Tensor:
        return torch.sigmoid(self.model(x))[:, h_idx]


# ─── TA recomputation ───────────────────────────────────────────────────────
def recompute_ta_torch(g_norm: torch.Tensor, g_mean: torch.Tensor,
                       g_std: torch.Tensor, sigma: torch.Tensor):
    """
    Rebuild the two TA channels from a (possibly perturbed) NORMALISED
    glucose sequence, differentiably.

        g_raw = g_norm * g_std + g_mean
        c     = sigmoid(-(g_raw - 70) / sigma)
        v+    = softplus((g_raw[t-1] - g_raw[t]) / sigma)

    sigma is the causal expanding std of RAW glucose that compute_ta_channels
    uses. It is treated as fixed: it is a property of the subject's history
    up to t, and a counterfactual at time t cannot rewrite that history.

    g_norm : (B, T)      g_mean, g_std, sigma : (B, T)
    """
    g_raw = g_norm * g_std + g_mean
    delta = (g_raw - HYPO_MGDL) / sigma
    c = torch.sigmoid(-torch.clamp(delta, -60, 60))

    prev = torch.cat([g_raw[:, :1], g_raw[:, :-1]], dim=1)
    v = (prev - g_raw) / sigma
    vp = torch.nn.functional.softplus(torch.clamp(v, -30, 30))

    return (torch.clamp(c, -TA_CLAMP, TA_CLAMP),
            torch.clamp(vp, -TA_CLAMP, TA_CLAMP))


# ─── temporal integrated gradients ──────────────────────────────────────────
class TemporalIG:
    """
    Standard IG along a straight path from baseline to input.

    BASELINE: zeros in normalised space = the subject's causal running
    mean. Attribution therefore answers "what pushed the risk away from
    what this subject typically looks like", which is the meaningful
    question for a per-subject normalised representation.
    """

    def __init__(self, adapter: ModelAdapter, steps: int = 50):
        self.a, self.M = adapter, steps

    def compute(self, x_window: np.ndarray, h_idx: int) -> np.ndarray:
        self.a.model.eval()
        T, F = x_window.shape
        x0 = np.zeros_like(x_window)
        grads = np.zeros((T, F), dtype=np.float64)

        for m in range(1, self.M + 1):
            alpha = m / self.M
            xi = torch.tensor(
                (x0 + alpha * (x_window - x0))[None],
                dtype=torch.float32, device=self.a.device, requires_grad=True)
            p = self.a.prob(xi, h_idx)
            self.a.model.zero_grad()
            p.backward()
            if xi.grad is not None:
                grads += xi.grad.detach().cpu().numpy()[0].astype(np.float64)

        attr = (x_window - x0) * (grads / self.M)
        return np.abs(attr).astype(np.float32)

    @staticmethod
    def top_features(attr: np.ndarray, k: int = 3) -> List[Dict]:
        imp = attr.sum(0)
        tot = imp.sum() + 1e-8
        order = np.argsort(imp)[::-1]
        return [{"rank": i + 1, "index": int(j), "name": FEATURE_NAMES[j],
                 "clinical": CLINICAL[FEATURE_NAMES[j]],
                 "importance": float(imp[j]),
                 "pct": float(100 * imp[j] / tot)}
                for i, j in enumerate(order[:k])]

    @staticmethod
    def top_timesteps(attr: np.ndarray, k: int = 3) -> List[Dict]:
        imp = attr.sum(1)
        T = len(imp)
        order = np.argsort(imp)[::-1]
        return [{"rank": i + 1, "step": int(t),
                 "minutes_ago": int((T - 1 - t) * 5),
                 "importance": float(imp[t])}
                for i, t in enumerate(order[:k])]


# ─── DiCE counterfactuals ───────────────────────────────────────────────────
class DiCE:
    """
    Gradient counterfactuals restricted to actionable treatment channels,
    with the TA channels rebuilt from glucose at every step so the input
    stays internally consistent.

    Perturbations are bounded per channel by `bound_sd` standard
    deviations AND the resulting raw value is required to be
    non-negative (a negative bolus or a negative carb quantity is not an
    intervention). Both constraints are applied inside the loop, not
    checked afterwards.
    """

    def __init__(self, adapter: ModelAdapter, n_cf: int = 3,
                 lam_dist: float = 1.0, lam_div: float = 0.3,
                 n_iter: int = 200, lr: float = 0.05,
                 bound_sd: float = 1.5, act_last: int = 12,
                 lam_sparse: float = 0.5):
        """
        act_last : only the final `act_last` readings may be changed.
                   A counterfactual spread over all 60 steps ("add 0.2 g
                   at every reading for five hours") is not an action
                   anyone can take. 12 readings = the last hour.
        lam_sparse : L1 penalty concentrating the change onto few steps,
                   so the result reads as one snack or one dose rather
                   than a smear.
        """
        self.a = adapter
        self.n_cf, self.lam_dist, self.lam_div = n_cf, lam_dist, lam_div
        self.n_iter, self.lr, self.bound_sd = n_iter, lr, bound_sd
        self.act_last, self.lam_sparse = act_last, lam_sparse

    def generate(self, x: np.ndarray, h_idx: int, ctx: Dict,
                 orig_risk: float) -> List[Counterfactual]:
        """
        x   : (T, F) normalised window
        ctx : per-window raw context, all length T --
              g_mean, g_std, sigma, and for each actionable channel
              {name}_mean, {name}_std, {name}_raw
        """
        dev = self.a.device
        T, F = x.shape
        x_t = torch.tensor(x[None], dtype=torch.float32, device=dev)

        act = [n for n in ACTIONABLE if ctx.get(f"{n}_ok", True)]
        if not act:
            return []
        act_idx = [IDX[n] for n in act]

        g_mean = torch.tensor(ctx["g_mean"][None], dtype=torch.float32, device=dev)
        g_std = torch.tensor(ctx["g_std"][None], dtype=torch.float32, device=dev)
        sigma = torch.tensor(ctx["sigma"][None], dtype=torch.float32, device=dev)

        # only recent steps are editable
        tmask = torch.zeros((1, T, 1), device=dev)
        tmask[:, max(0, T - self.act_last):, :] = 1.0
        # small random init, otherwise all K counterfactuals start identical
        # and the diversity term has no gradient to separate them
        delta = (torch.randn((self.n_cf, T, len(act)), device=dev) * 0.05
                 ).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=self.lr)

        for _ in range(self.n_iter):
            opt.zero_grad()
            xb = x_t.repeat(self.n_cf, 1, 1)

            # bounded additive change on actionable channels only
            d = torch.tanh(delta) * self.bound_sd * tmask
            for j, ci in enumerate(act_idx):
                name = act[j]
                mu = torch.tensor(ctx[f"{name}_mean"][None], dtype=torch.float32,
                                  device=dev)
                sd = torch.tensor(ctx[f"{name}_std"][None], dtype=torch.float32,
                                  device=dev)
                new_norm = xb[:, :, ci] + d[:, :, j]
                # raw value must stay >= 0
                raw = new_norm * sd + mu
                raw = torch.clamp(raw, min=0.0)
                new_norm = (raw - mu) / torch.clamp(sd, min=1e-6)
                xb = xb.clone()
                xb[:, :, ci] = torch.clamp(new_norm, -FEAT_CLAMP, FEAT_CLAMP)

            # glucose is untouched, so TA is unchanged here; recomputed
            # anyway so the same code path serves glucose-perturbing
            # variants and the consistency guarantee is explicit
            c, vp = recompute_ta_torch(xb[:, :, IDX["glucose"]],
                                       g_mean, g_std, sigma)
            xb = xb.clone()
            xb[:, :, IDX["ta_proximity_c"]] = c
            xb[:, :, IDX["ta_downslope_vplus"]] = vp

            p = self.a.prob(xb, h_idx)
            loss_pred = nn.functional.binary_cross_entropy(
                torch.clamp(p, 1e-7, 1 - 1e-7), torch.zeros_like(p))
            loss_dist = d.abs().mean()
            loss_div = torch.tensor(0.0, device=dev)
            if self.n_cf > 1:
                fl = d.reshape(self.n_cf, -1)
                loss_div = -torch.cdist(fl, fl).mean()
            loss_sparse = d.abs().sum(dim=1).mean()
            (loss_pred + self.lam_dist * loss_dist
             + self.lam_div * loss_div
             + self.lam_sparse * loss_sparse).backward()
            opt.step()

        # ---- materialise ---------------------------------------------------
        out = []
        with torch.no_grad():
            d = torch.tanh(delta) * self.bound_sd * tmask
            xb = x_t.repeat(self.n_cf, 1, 1).clone()
            for j, ci in enumerate(act_idx):
                name = act[j]
                mu = torch.tensor(ctx[f"{name}_mean"][None], dtype=torch.float32,
                                  device=dev)
                sd = torch.tensor(ctx[f"{name}_std"][None], dtype=torch.float32,
                                  device=dev)
                raw = torch.clamp((xb[:, :, ci] + d[:, :, j]) * sd + mu, min=0.0)
                xb[:, :, ci] = torch.clamp((raw - mu) / torch.clamp(sd, min=1e-6),
                                           -FEAT_CLAMP, FEAT_CLAMP)
            c, vp = recompute_ta_torch(xb[:, :, IDX["glucose"]],
                                       g_mean, g_std, sigma)
            xb[:, :, IDX["ta_proximity_c"]] = c
            xb[:, :, IDX["ta_downslope_vplus"]] = vp
            risks = self.a.prob(xb, h_idx).cpu().numpy()

            for k in range(self.n_cf):
                ch_n, ch_r, val = {}, {}, {}
                plausible = True
                for j, ci in enumerate(act_idx):
                    name = act[j]
                    dn = float((xb[k, :, ci] - x_t[0, :, ci]).abs().mean())
                    sd = ctx[f"{name}_std"]
                    raw_new = (xb[k, :, ci].cpu().numpy() * sd
                               + ctx[f"{name}_mean"])
                    raw_old = ctx[f"{name}_raw"]
                    diff = raw_new - raw_old
                    draw = float(diff.mean() if AGG[name] == "mean"
                                 else diff.sum())
                    # reject changes too small to state in clinical units
                    if abs(draw) < MIN_RAW_CHANGE.get(name, 0.0):
                        continue
                    ch_n[name] = round(dn, 4)
                    ch_r[name] = round(draw, 3)
                    hi = ctx.get(f"{name}_p99", np.inf)
                    ok = bool(np.all(raw_new >= -1e-6) and np.max(raw_new) <= hi)
                    val[name] = {"unit": RAW_UNITS.get(name, ""),
                                 "total_change": round(draw, 3),
                                 "max_value": float(np.max(raw_new)),
                                 "cohort_p99": float(hi),
                                 "within_observed": ok}
                    plausible &= ok

                red = float(max(0.0, orig_risk - risks[k]))
                # a counterfactual with no statable change, or one that does
                # not move the prediction, is not a counterfactual. Emitting
                # it produces lines like "increase basal by 0.0 U/hr, risk
                # falls from 73% to 73%", which are worse than no output.
                if not ch_r or red < MIN_RISK_DELTA:
                    continue
                out.append(Counterfactual(
                    cf_id=len(out) + 1,
                    predicted_risk=float(risks[k]),
                    risk_reduction=red,
                    changes_norm=ch_n, changes_raw=ch_r,
                    validity=val, plausible=plausible,
                    action_en=_action_text(ch_r, orig_risk, float(risks[k]),
                                           plausible)))
        return out


def _action_text(changes_raw: Dict[str, float], r0: float, r1: float,
                 plausible: bool) -> str:
    if not changes_raw:
        return ("No actionable change to insulin or carbohydrate altered the "
                "model's prediction. The risk is driven by the glucose "
                "trajectory itself.")
    parts = []
    for n, d in changes_raw.items():
        u = RAW_UNITS.get(n, "")
        how = "on average" if AGG.get(n) == "mean" else "in total"
        parts.append(f"{'increase' if d > 0 else 'reduce'} {CLINICAL[n].lower()} "
                     f"by {abs(d):.1f} {u} {how} over the last hour")
    s = ("Model-predicted risk falls from "
         f"{r0*100:.0f}% to {r1*100:.0f}% if you " + " and ".join(parts) + ".")
    if not plausible:
        s += (" NOTE: this requires values outside the range observed in the "
              "cohort and should not be treated as an actionable "
              "recommendation.")
    return s


# ─── engine ─────────────────────────────────────────────────────────────────
class XAIEngineV2:
    def __init__(self, model, device="cpu", ig_steps=50, n_cf=3,
                 risk_thresh=0.35, bound_sd=1.5, n_iter=200):
        self.a = ModelAdapter(model, device)
        self.ig = TemporalIG(self.a, ig_steps)
        self.cf = DiCE(self.a, n_cf=n_cf, bound_sd=bound_sd, n_iter=n_iter)
        self.risk_thresh = risk_thresh

    def explain(self, x: np.ndarray, h_idx: int, ctx: Dict,
                with_cf: bool = True, force: bool = False
                ) -> Optional[XAIResult]:
        with torch.no_grad():
            r = float(self.a.prob(
                torch.tensor(x[None], dtype=torch.float32,
                             device=self.a.device), h_idx)[0])
        if r < self.risk_thresh and not force:
            return None

        attr = self.ig.compute(x, h_idx)
        tf = self.ig.top_features(attr)
        cfs = self.cf.generate(x, h_idx, ctx, r) if with_cf else []

        note = ""
        if with_cf and not cfs:
            note = ("No change to basal, bolus or carbohydrate within "
                    "clinically plausible bounds altered the predicted risk "
                    "by more than 0.5 percentage points. The prediction is "
                    "carried by the glucose trajectory channels.")
        elif cfs and not any(c.plausible for c in cfs):
            note = ("All counterfactuals required values outside the observed "
                    "cohort range and are reported as implausible.")

        return XAIResult(
            original_risk=r, traffic_light=traffic_light(r),
            attribution=attr, top_features=tf,
            top_timesteps=self.ig.top_timesteps(attr),
            counterfactuals=cfs,
            ig_summary_en=_ig_summary(r, tf, HORIZONS[h_idx]),
            cf_note=note)


def _ig_summary(risk: float, top: List[Dict], horizon: int) -> str:
    if not top:
        return ""
    lead = ", ".join(f"{t['clinical']} ({t['pct']:.0f}%)" for t in top)
    return (f"The model's {risk*100:.0f}% predicted risk at {horizon} minutes "
            f"is attributed mainly to: {lead}. Attribution is measured "
            f"relative to this subject's running average trajectory.")
