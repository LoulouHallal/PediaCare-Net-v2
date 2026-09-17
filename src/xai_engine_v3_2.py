"""
xai_engine_v3_2.py -- Temporal Integrated Gradients + decision-flipping constrained counterfactuals
for PediaCare-Net v2.

Correctness changes relative to v2
----------------------------------
1) TSLGRU/GRU+TA already return probabilities through stage2_deep.Heads.
   ModelAdapter therefore NEVER applies a second sigmoid.
2) Human-readable feature labels are carried into JSON/figures.
3) Counterfactual generation is deterministic per case when a seed is supplied.
4) Counterfactual diagnostics retain the best optimisation attempt even when no
   candidate passes the reporting threshold, so a failed search is interpretable.
5) Plausibility checks are applied to actually changed recent values rather than
   rejecting a candidate because an unchanged historical value happened to exceed
   the cohort p99.

Important interpretation note
-----------------------------
Counterfactuals are model-based sensitivity statements. They change recorded
historical treatment inputs while holding the observed glucose trajectory fixed.
They are NOT causal treatment recommendations and do not establish what would
happen clinically if a treatment were administered.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

HYPO_MGDL = 70.0
TA_CLAMP = 10.0
FEAT_CLAMP = 10.0
HORIZONS = [15, 30, 60, 120]

FEATURE_NAMES = [
    "glucose", "basal", "bolus", "carbs", "carbs_observed",
    "ta_proximity_c", "ta_downslope_vplus",
]

CLINICAL = {
    "glucose": "Blood glucose (CGM)",
    "basal": "Basal insulin rate",
    "bolus": "Insulin bolus",
    "carbs": "Carbohydrate intake",
    "carbs_observed": "Carbohydrate-recording availability",
    "ta_proximity_c": "Proximity to 70 mg/dL",
    "ta_downslope_vplus": "Rate of glucose fall",
}

DISPLAY = {
    "glucose": "Glucose (CGM)",
    "basal": "Basal insulin rate",
    "bolus": "Insulin bolus",
    "carbs": "Carbohydrate intake",
    "carbs_observed": "Carb-recording flag",
    "ta_proximity_c": "Proximity to 70 mg/dL",
    "ta_downslope_vplus": "Rate of glucose fall",
}
DISPLAY_NAMES = [DISPLAY[n] for n in FEATURE_NAMES]

RAW_UNITS = {"basal": "U/hr", "bolus": "U", "carbs": "g"}
AGG = {"basal": "mean", "bolus": "sum", "carbs": "sum"}
IDX = {n: i for i, n in enumerate(FEATURE_NAMES)}
ACTIONABLE = ["basal", "bolus", "carbs"]

# Reporting thresholds. These remove numerical/rounding artifacts, not valid
# optimisation candidates with clinically visible changes.
MIN_RAW_CHANGE = {"basal": 0.05, "bolus": 0.1, "carbs": 1.0}
CF_MARGIN = 0.01  # target one percentage point below the decision threshold

TRAFFIC_GREEN, TRAFFIC_AMBER = 0.30, 0.60


@dataclass
class Counterfactual:
    cf_id: int
    predicted_risk: float
    risk_reduction: float
    decision_threshold: float
    decision_flipped: bool
    changes_norm: Dict[str, float] = field(default_factory=dict)
    changes_raw: Dict[str, float] = field(default_factory=dict)
    validity: Dict[str, Dict] = field(default_factory=dict)
    plausible: bool = True
    action_en: str = ""


@dataclass
class CounterfactualDiagnostics:
    n_candidates: int = 0
    n_reported: int = 0
    decision_threshold: Optional[float] = None
    n_plausible: int = 0
    n_flipped: int = 0
    best_attempt_risk: Optional[float] = None
    best_attempt_reduction: float = 0.0
    best_attempt_changes_raw: Dict[str, float] = field(default_factory=dict)
    best_partial_risk: Optional[float] = None
    best_partial_reduction: float = 0.0
    best_partial_changes_raw: Dict[str, float] = field(default_factory=dict)
    note: str = ""


@dataclass
class XAIResult:
    original_risk: float
    traffic_light: str
    attribution: np.ndarray
    top_features: List[Dict]
    top_timesteps: List[Dict]
    counterfactuals: List[Counterfactual]
    cf_diagnostics: CounterfactualDiagnostics
    ig_summary_en: str = ""
    cf_note: str = ""


def traffic_light(r: float) -> str:
    return "GREEN" if r < TRAFFIC_GREEN else ("AMBER" if r < TRAFFIC_AMBER else "RED")


class ModelAdapter:
    """Expose one horizon probability from a model that already outputs probabilities."""

    def __init__(self, model: nn.Module, device: str = "cpu"):
        self.model = model
        self.device = device

    def prob(self, x: torch.Tensor, h_idx: int) -> torch.Tensor:
        # stage2_deep.Heads.forward() already applies torch.sigmoid.
        # Applying another sigmoid here is a correctness bug.
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.ndim != 2 or out.shape[1] != len(HORIZONS):
            raise RuntimeError(
                f"Expected model output shape (B,{len(HORIZONS)}), got {tuple(out.shape)}"
            )
        return out[:, h_idx]

    @torch.no_grad()
    def validate_probability_output(self, x: torch.Tensor) -> Dict[str, float]:
        out = self.model(x)
        if isinstance(out, tuple):
            out = out[0]
        if not torch.isfinite(out).all():
            raise RuntimeError("Model produced non-finite values during XAI preflight.")
        lo, hi = float(out.min()), float(out.max())
        if lo < -1e-6 or hi > 1.0 + 1e-6:
            raise RuntimeError(
                "XAI expects probabilities because stage2_deep.Heads already applies "
                f"sigmoid, but observed model output range [{lo:.6f}, {hi:.6f}]."
            )
        return {
            "min": lo,
            "median": float(out.median()),
            "max": hi,
            "range": hi - lo,
        }


def recompute_ta_torch(
    g_norm: torch.Tensor,
    g_mean: torch.Tensor,
    g_std: torch.Tensor,
    sigma: torch.Tensor,
):
    """Rebuild TA channels from normalised glucose using fixed causal history stats."""
    g_raw = g_norm * g_std + g_mean
    delta = (g_raw - HYPO_MGDL) / torch.clamp(sigma, min=1e-6)
    c = torch.sigmoid(-torch.clamp(delta, -60, 60))

    prev = torch.cat([g_raw[:, :1], g_raw[:, :-1]], dim=1)
    v = (prev - g_raw) / torch.clamp(sigma, min=1e-6)
    vp = torch.nn.functional.softplus(torch.clamp(v, -30, 30))
    return torch.clamp(c, -TA_CLAMP, TA_CLAMP), torch.clamp(vp, -TA_CLAMP, TA_CLAMP)


class TemporalIG:
    """
    Temporal Integrated Gradients with a zero reference in normalised feature space.

    For the subject-level recording flag, the baseline is kept equal to the observed
    flag so this metadata indicator cannot receive attribution merely because the
    subject records carbohydrates. TA channels remain explicit model inputs and are
    therefore attributed as explicit channels, consistent with the thesis comparison.
    """

    def __init__(self, adapter: ModelAdapter, steps: int = 50):
        if steps < 2:
            raise ValueError("Integrated Gradients requires at least 2 steps.")
        self.a = adapter
        self.M = int(steps)

    def compute(self, x_window: np.ndarray, h_idx: int) -> np.ndarray:
        self.a.model.eval()
        x_window = np.asarray(x_window, dtype=np.float32)
        if x_window.ndim != 2 or x_window.shape[1] != len(FEATURE_NAMES):
            raise ValueError(
                f"Expected window shape (T,{len(FEATURE_NAMES)}), got {x_window.shape}"
            )
        if not np.isfinite(x_window).all():
            raise ValueError("Non-finite value found in XAI input window.")

        T, F = x_window.shape
        x0 = np.zeros_like(x_window, dtype=np.float32)
        # Metadata is not a physiological intervention; hold it fixed.
        x0[:, IDX["carbs_observed"]] = x_window[:, IDX["carbs_observed"]]
        grads = np.zeros((T, F), dtype=np.float64)

        for m in range(1, self.M + 1):
            alpha = m / self.M
            interp = x0 + alpha * (x_window - x0)
            xi = torch.tensor(
                interp[None], dtype=torch.float32, device=self.a.device,
                requires_grad=True,
            )
            p = self.a.prob(xi, h_idx).sum()
            self.a.model.zero_grad(set_to_none=True)
            if xi.grad is not None:
                xi.grad.zero_()
            p.backward()
            if xi.grad is None:
                raise RuntimeError("Integrated Gradients could not obtain input gradients.")
            grads += xi.grad.detach().cpu().numpy()[0].astype(np.float64)

        attr = (x_window - x0) * (grads / self.M)
        return np.abs(attr).astype(np.float32)

    @staticmethod
    def top_features(attr: np.ndarray, k: int = 3) -> List[Dict]:
        imp = np.asarray(attr).sum(0)
        tot = float(imp.sum()) + 1e-12
        order = np.argsort(imp)[::-1]
        out = []
        for rank, j in enumerate(order[:k], start=1):
            internal = FEATURE_NAMES[int(j)]
            out.append({
                "rank": rank,
                "index": int(j),
                "internal_name": internal,
                "feature": CLINICAL[internal],
                "display": DISPLAY[internal],
                "importance": float(imp[j]),
                "pct": float(100.0 * imp[j] / tot),
            })
        return out

    @staticmethod
    def top_timesteps(attr: np.ndarray, k: int = 3) -> List[Dict]:
        imp = np.asarray(attr).sum(1)
        T = len(imp)
        order = np.argsort(imp)[::-1]
        return [
            {
                "rank": rank,
                "step": int(t),
                "minutes_ago": int((T - 1 - int(t)) * 5),
                "importance": float(imp[t]),
            }
            for rank, t in enumerate(order[:k], start=1)
        ]


class DiCE:
    """Constrained gradient search over recorded basal, bolus and carbohydrate inputs."""

    def __init__(
        self,
        adapter: ModelAdapter,
        n_cf: int = 6,
        lam_dist: float = 1.0,
        lam_div: float = 0.3,
        n_iter: int = 200,
        lr: float = 0.05,
        bound_sd: float = 1.5,
        act_last: int = 12,
        lam_sparse: float = 0.5,
    ):
        self.a = adapter
        self.n_cf = int(n_cf)
        self.lam_dist = float(lam_dist)
        self.lam_div = float(lam_div)
        self.n_iter = int(n_iter)
        self.lr = float(lr)
        self.bound_sd = float(bound_sd)
        self.act_last = int(act_last)
        self.lam_sparse = float(lam_sparse)
        self.last_diagnostics = CounterfactualDiagnostics()

    def generate(
        self,
        x: np.ndarray,
        h_idx: int,
        ctx: Dict,
        orig_risk: float,
        decision_threshold: float,
        seed: int = 0,
    ) -> List[Counterfactual]:
        dev = self.a.device
        x = np.asarray(x, dtype=np.float32)
        T, F = x.shape
        if not (0.0 < float(decision_threshold) < 1.0):
            raise ValueError(f"decision_threshold must be in (0,1), got {decision_threshold}")
        decision_threshold = float(decision_threshold)
        x_t = torch.tensor(x[None], dtype=torch.float32, device=dev)

        # A risk-reducing decision-flip counterfactual is only defined for a
        # currently positive model decision.  For an already-negative case we
        # still provide IG, but do not manufacture an unnecessary CF.
        if orig_risk < decision_threshold:
            self.last_diagnostics = CounterfactualDiagnostics(
                decision_threshold=decision_threshold,
                note=(
                    f"Original predicted risk ({orig_risk*100:.1f}%) is already below "
                    f"the decision threshold ({decision_threshold*100:.1f}%); no "
                    "risk-reducing decision-flip counterfactual was sought."
                ),
            )
            return []

        act = [n for n in ACTIONABLE if ctx.get(f"{n}_ok", True)]
        if not act:
            self.last_diagnostics = CounterfactualDiagnostics(note="No actionable channels available.")
            return []
        act_idx = [IDX[n] for n in act]

        g_mean = torch.tensor(ctx["g_mean"][None], dtype=torch.float32, device=dev)
        g_std = torch.tensor(ctx["g_std"][None], dtype=torch.float32, device=dev)
        sigma = torch.tensor(ctx["sigma"][None], dtype=torch.float32, device=dev)

        tmask = torch.zeros((1, T, 1), device=dev)
        first_edit = max(0, T - self.act_last)
        tmask[:, first_edit:, :] = 1.0

        gen = torch.Generator(device=dev)
        gen.manual_seed(int(seed))
        delta = (
            torch.randn((self.n_cf, T, len(act)), generator=gen, device=dev) * 0.05
        ).requires_grad_(True)
        opt = torch.optim.Adam([delta], lr=self.lr)

        for _ in range(self.n_iter):
            opt.zero_grad()
            xb = x_t.repeat(self.n_cf, 1, 1).clone()
            d = torch.tanh(delta) * self.bound_sd * tmask

            for j, ci in enumerate(act_idx):
                name = act[j]
                mu = torch.tensor(ctx[f"{name}_mean"][None], dtype=torch.float32, device=dev)
                sd = torch.tensor(ctx[f"{name}_std"][None], dtype=torch.float32, device=dev)
                new_norm = xb[:, :, ci] + d[:, :, j]
                raw = torch.clamp(new_norm * sd + mu, min=0.0)
                xb[:, :, ci] = torch.clamp(
                    (raw - mu) / torch.clamp(sd, min=1e-6),
                    -FEAT_CLAMP,
                    FEAT_CLAMP,
                )

            # TA features are deterministic functions of glucose. Glucose itself is
            # locked here, but recomputing keeps the model input internally consistent.
            c, vp = recompute_ta_torch(
                xb[:, :, IDX["glucose"]], g_mean, g_std, sigma
            )
            xb[:, :, IDX["ta_proximity_c"]] = c
            xb[:, :, IDX["ta_downslope_vplus"]] = vp

            p = self.a.prob(xb, h_idx)
            # Optimize specifically for a decision flip, not merely a tiny risk
            # reduction.  Once the prediction is one percentage point below the
            # validation-selected threshold, the prediction term becomes zero and
            # the distance/sparsity terms favor a smaller explanation.
            target = max(0.0, decision_threshold - CF_MARGIN)
            # Give boundary crossing clear priority.  The hinge becomes zero only
            # after the target side of the decision boundary is reached; distance
            # and sparsity then favor a smaller change.
            loss_pred = 100.0 * torch.relu(p - target).mean()
            loss_dist = d.abs().mean()
            loss_div = torch.tensor(0.0, device=dev)
            if self.n_cf > 1:
                flat = d.reshape(self.n_cf, -1)
                loss_div = -torch.cdist(flat, flat).mean()
            loss_sparse = d.abs().sum(dim=1).mean()
            loss = (
                loss_pred
                + self.lam_dist * loss_dist
                + self.lam_div * loss_div
                + self.lam_sparse * loss_sparse
            )
            loss.backward()
            opt.step()

        reported: List[Counterfactual] = []
        attempts = []
        with torch.no_grad():
            d = torch.tanh(delta) * self.bound_sd * tmask
            xb = x_t.repeat(self.n_cf, 1, 1).clone()
            for j, ci in enumerate(act_idx):
                name = act[j]
                mu = torch.tensor(ctx[f"{name}_mean"][None], dtype=torch.float32, device=dev)
                sd = torch.tensor(ctx[f"{name}_std"][None], dtype=torch.float32, device=dev)
                raw = torch.clamp((xb[:, :, ci] + d[:, :, j]) * sd + mu, min=0.0)
                xb[:, :, ci] = torch.clamp(
                    (raw - mu) / torch.clamp(sd, min=1e-6),
                    -FEAT_CLAMP,
                    FEAT_CLAMP,
                )

            c, vp = recompute_ta_torch(
                xb[:, :, IDX["glucose"]], g_mean, g_std, sigma
            )
            xb[:, :, IDX["ta_proximity_c"]] = c
            xb[:, :, IDX["ta_downslope_vplus"]] = vp
            risks = self.a.prob(xb, h_idx).detach().cpu().numpy()

            for k in range(self.n_cf):
                changes_norm: Dict[str, float] = {}
                changes_raw: Dict[str, float] = {}
                validity: Dict[str, Dict] = {}
                plausible = True

                for j, ci in enumerate(act_idx):
                    name = act[j]
                    arr_new = xb[k, :, ci].detach().cpu().numpy()
                    arr_old = x_t[0, :, ci].detach().cpu().numpy()
                    dn = float(np.mean(np.abs(arr_new - arr_old)))
                    sd = np.asarray(ctx[f"{name}_std"], dtype=np.float32)
                    mu = np.asarray(ctx[f"{name}_mean"], dtype=np.float32)
                    raw_new = arr_new * sd + mu
                    raw_old = np.asarray(ctx[f"{name}_raw"], dtype=np.float32)
                    diff = raw_new - raw_old
                    draw = float(diff.mean() if AGG[name] == "mean" else diff.sum())

                    if abs(draw) < MIN_RAW_CHANGE.get(name, 0.0):
                        continue
                    changes_norm[name] = round(dn, 4)
                    changes_raw[name] = round(draw, 3)

                    # Check only readings that the optimisation actually changed.
                    changed = np.abs(diff) > 1e-6
                    if not np.any(changed):
                        changed = np.arange(T) >= first_edit
                    p99 = float(ctx.get(f"{name}_p99", np.inf))
                    nonnegative = bool(np.all(raw_new[changed] >= -1e-6))
                    under_value_p99 = bool(np.all(raw_new[changed] <= p99 + 1e-6))
                    # For summed event variables (bolus/carbohydrate), also keep
                    # the total one-hour modification within the cohort p99 event
                    # magnitude.  This prevents a sequence of individually plausible
                    # edits from adding up to an implausibly large intervention.
                    under_total_p99 = bool(abs(draw) <= p99 + 1e-6)
                    ok = nonnegative and under_value_p99 and under_total_p99
                    validity[name] = {
                        "unit": RAW_UNITS.get(name, ""),
                        "total_change": round(draw, 3),
                        "max_changed_value": float(np.max(raw_new[changed])),
                        "cohort_p99": p99,
                        "within_value_p99_bound": under_value_p99,
                        "within_total_change_p99_bound": under_total_p99,
                        "within_p99_bound": ok,
                    }
                    plausible = plausible and ok

                cf_risk = float(risks[k])
                reduction = float(max(0.0, orig_risk - cf_risk))
                flipped = bool(cf_risk < decision_threshold)
                attempts.append(
                    {
                        "risk": cf_risk,
                        "reduction": reduction,
                        "changes_raw": dict(changes_raw),
                        "plausible": plausible,
                        "flipped": flipped,
                    }
                )

                # A thesis-level successful counterfactual must satisfy all three:
                # (1) a clinically reportable input change, (2) cohort-bound
                # plausibility, and (3) crossing the actual model decision boundary.
                if not changes_raw or not plausible or not flipped:
                    continue

                reported.append(
                    Counterfactual(
                        cf_id=len(reported) + 1,
                        predicted_risk=cf_risk,
                        risk_reduction=reduction,
                        decision_threshold=decision_threshold,
                        decision_flipped=True,
                        changes_norm=changes_norm,
                        changes_raw=changes_raw,
                        validity=validity,
                        plausible=True,
                        action_en=_action_text(
                            changes_raw, orig_risk, cf_risk, decision_threshold
                        ),
                    )
                )

        attempts.sort(key=lambda d: d["reduction"], reverse=True)
        best = attempts[0] if attempts else None
        plausible_attempts = [a for a in attempts if a["plausible"] and a["changes_raw"]]
        flipped_attempts = [a for a in plausible_attempts if a["flipped"]]
        partial_attempts = [a for a in plausible_attempts if not a["flipped"]]
        best_partial = partial_attempts[0] if partial_attempts else None

        note = ""
        if not reported:
            if best is None:
                note = "Counterfactual optimisation produced no candidates."
            elif not plausible_attempts:
                note = (
                    "No candidate produced a clinically reportable, cohort-plausible "
                    "input change."
                )
            elif best_partial is not None:
                note = (
                    f"No plausible candidate crossed the {decision_threshold*100:.1f}% "
                    f"decision threshold. The best partial sensitivity scenario reduced "
                    f"risk from {orig_risk*100:.1f}% to {best_partial['risk']*100:.1f}% "
                    "but the model decision remained positive."
                )
            else:
                note = (
                    f"No plausible candidate crossed the {decision_threshold*100:.1f}% "
                    "decision threshold."
                )

        self.last_diagnostics = CounterfactualDiagnostics(
            n_candidates=len(attempts),
            n_reported=len(reported),
            decision_threshold=decision_threshold,
            n_plausible=len(plausible_attempts),
            n_flipped=len(flipped_attempts),
            best_attempt_risk=None if best is None else float(best["risk"]),
            best_attempt_reduction=0.0 if best is None else float(best["reduction"]),
            best_attempt_changes_raw={} if best is None else dict(best["changes_raw"]),
            best_partial_risk=None if best_partial is None else float(best_partial["risk"]),
            best_partial_reduction=0.0 if best_partial is None else float(best_partial["reduction"]),
            best_partial_changes_raw={} if best_partial is None else dict(best_partial["changes_raw"]),
            note=note,
        )

        # Best model-based risk reduction first.
        reported.sort(key=lambda c: c.risk_reduction, reverse=True)
        for i, c in enumerate(reported, start=1):
            c.cf_id = i
        return reported


def _action_text(
    changes_raw: Dict[str, float],
    r0: float,
    r1: float,
    decision_threshold: float,
) -> str:
    parts = []
    for name, change in changes_raw.items():
        unit = RAW_UNITS.get(name, "")
        how = "on average" if AGG.get(name) == "mean" else "in total"
        direction = "higher" if change > 0 else "lower"
        parts.append(
            f"the recorded {CLINICAL[name].lower()} were {direction} by "
            f"{abs(change):.1f} {unit} {how} over the last hour"
        )
    return (
        f"Under this model-based what-if, predicted risk changes from "
        f"{r0*100:.1f}% to {r1*100:.1f}%, crossing the "
        f"{decision_threshold*100:.1f}% decision threshold, if "
        + " and ".join(parts) + "."
    )


class XAIEngineV32:
    def __init__(
        self,
        model,
        device="cpu",
        ig_steps=50,
        n_cf=6,
        risk_thresh=0.35,
        bound_sd=1.5,
        n_iter=200,
        decision_threshold=0.5,
    ):
        self.a = ModelAdapter(model, device)
        self.ig = TemporalIG(self.a, ig_steps)
        self.cf = DiCE(
            self.a, n_cf=n_cf, bound_sd=bound_sd, n_iter=n_iter
        )
        self.risk_thresh = float(risk_thresh)
        self.decision_threshold = float(decision_threshold)
        if not (0.0 < self.decision_threshold < 1.0):
            raise ValueError("decision_threshold must be in (0,1)")

    def explain(
        self,
        x: np.ndarray,
        h_idx: int,
        ctx: Dict,
        with_cf: bool = True,
        force: bool = False,
        cf_seed: int = 0,
    ) -> Optional[XAIResult]:
        with torch.no_grad():
            xt = torch.tensor(x[None], dtype=torch.float32, device=self.a.device)
            risk = float(self.a.prob(xt, h_idx)[0])
        if risk < self.risk_thresh and not force:
            return None

        attr = self.ig.compute(x, h_idx)
        top_features = self.ig.top_features(attr)
        cfs = (
            self.cf.generate(
                x, h_idx, ctx, risk, self.decision_threshold, seed=cf_seed
            )
            if with_cf else []
        )
        diag = self.cf.last_diagnostics if with_cf else CounterfactualDiagnostics()

        if with_cf and not cfs:
            cf_note = diag.note or "No reportable counterfactual was found."
        elif cfs and not any(c.plausible for c in cfs):
            cf_note = "Counterfactuals were found, but all exceeded at least one cohort p99 bound."
        else:
            cf_note = ""

        return XAIResult(
            original_risk=risk,
            traffic_light=traffic_light(risk),
            attribution=attr,
            top_features=top_features,
            top_timesteps=self.ig.top_timesteps(attr),
            counterfactuals=cfs,
            cf_diagnostics=diag,
            ig_summary_en=_ig_summary(risk, top_features, HORIZONS[h_idx]),
            cf_note=cf_note,
        )


def _ig_summary(risk: float, top: List[Dict], horizon: int) -> str:
    if not top:
        return ""
    lead = ", ".join(f"{t['feature']} ({t['pct']:.0f}%)" for t in top)
    return (
        f"The model's {risk*100:.1f}% predicted risk at {horizon} minutes is "
        f"attributed mainly to {lead}. Attribution is relative to the model's "
        "normalised reference trajectory."
    )


def dataclass_to_dict(obj):
    """JSON-safe helper used by the runner."""
    return asdict(obj)
