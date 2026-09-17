"""
run_xai_v3.py -- Patient-diverse XAI for PediaCare-Net v2
==========================================================

This runner fixes the v2 issues discovered during thesis validation:
  * no double sigmoid: TSLGRU/GRU+TA already output probabilities;
  * case selection is patient-diverse (one highest-risk window per subject);
  * plots use reader-friendly clinical feature names;
  * heatmaps are generated for multiple different subjects;
  * counterfactual failures retain diagnostics instead of producing an empty plot;
  * outputs go to a new v3 directory, so stale v2 heatmaps are not mixed in.

Recommended workflow
--------------------
1) Preflight only (fast; verifies probability range and patient selection):
   python run_xai_v3.py --models tsl_gru --horizon 30 --check_only

2) Small smoke run:
   python run_xai_v3.py --models tsl_gru --horizon 30 --n_pop 20 --n_case 3 \\
       --n_heatmaps 3 --ig_steps 8 --n_iter 30 --n_cf 4

3) Full thesis run:
   python run_xai_v3.py --models tsl_gru --horizon 30 --n_pop 2000 \\
       --n_case 12 --n_heatmaps 6 --ig_steps 50 --n_iter 200 --n_cf 6

Counterfactuals are model-based what-if sensitivity analyses, not causal treatment
recommendations.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import common  # noqa: E402
import config  # noqa: E402
from ta_gru import attach_ta  # noqa: E402
from tsl_gru import TSLGRU, OUT_DIR as TSL_DIR  # noqa: E402
from xai_engine_v3 import (  # noqa: E402
    ACTIONABLE,
    CLINICAL,
    DISPLAY_NAMES,
    FEATURE_NAMES,
    HORIZONS,
    IDX,
    RAW_UNITS,
    XAIEngineV3,
    dataclass_to_dict,
)


MODEL_DISPLAY = {
    "gru_ta": "GRU+TA",
    "tsl_gru": "TSL-GRU",
    "tsl_static": "TSL-static",
    "ligru_ta": "LiGRU-TA",
}


def model_label(name: str) -> str:
    return MODEL_DISPLAY.get(name, name.replace("_", " ").upper())


# ---------------------------------------------------------------------------
# Causal statistics in raw clinical units
# ---------------------------------------------------------------------------
def expanding_stats(x: np.ndarray, subject: np.ndarray):
    """Reproduce the causal expanding mean/std convention used by preprocessing."""
    x = np.asarray(x)
    subject = np.asarray(subject)
    if len(x) != len(subject):
        raise ValueError("x and subject must have the same length")
    mean = np.zeros_like(x, dtype=np.float32)
    std = np.zeros_like(x, dtype=np.float32)
    for sid in np.unique(subject):
        m = np.flatnonzero(subject == sid)
        v = x[m].astype(np.float64)
        cnt = np.arange(1, len(v) + 1, dtype=np.float64)
        c1, c2 = np.cumsum(v), np.cumsum(v ** 2)
        mu = c1 / cnt
        sd = np.sqrt(np.maximum(c2 / cnt - mu ** 2, 0.0))
        # strictly causal: statistic at t uses history before t
        mu = np.concatenate([mu[:1], mu[:-1]])
        sd = np.concatenate([sd[:1], sd[:-1]])
        mean[m], std[m] = mu.astype(np.float32), sd.astype(np.float32)
    return mean, std


def ta_sigma(raw_gluc: np.ndarray, subject: np.ndarray):
    _, sd = expanding_stats(raw_gluc, subject)
    return np.maximum(sd, 1e-6)


def build_context(bw, raw_tr: Dict[str, np.ndarray], stats: Dict, end_idx: int, window_len: int):
    start = int(end_idx) - int(window_len) + 1
    if start < 0:
        raise IndexError(f"Invalid window start {start} for end_idx={end_idx}")
    sl = slice(start, int(end_idx) + 1)
    ctx = {
        "g_mean": stats["glucose"][0][sl].astype(np.float32),
        "g_std": np.maximum(stats["glucose"][1][sl], 1e-6).astype(np.float32),
        "sigma": np.maximum(stats["sigma"][sl], 1e-6).astype(np.float32),
    }
    for name in ACTIONABLE:
        mu, sd = stats[name]
        ctx[f"{name}_mean"] = mu[sl].astype(np.float32)
        ctx[f"{name}_std"] = np.maximum(sd[sl], 1e-6).astype(np.float32)
        ctx[f"{name}_raw"] = raw_tr[name][sl].astype(np.float32)
        ctx[f"{name}_p99"] = float(stats[f"{name}_p99"])
        ctx[f"{name}_ok"] = True
    return ctx


# ---------------------------------------------------------------------------
# Model loading / prediction / preflight
# ---------------------------------------------------------------------------
def load_model(variant: str, in_ch: int, device: str, suffix: str = ""):
    name = f"{variant}__weighted_bce__none__any" + (f"__{suffix}" if suffix else "")
    path = TSL_DIR / f"{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    model = TSLGRU(
        variant,
        in_ch=in_ch,
        hidden=cfg.get("hidden", 64),
        dropout=cfg.get("dropout", 0.2),
        rho=cfg.get("rho", 0.9),
    )
    model.load_state_dict(blob["model_state"])
    model = model.to(device).eval()
    n_params = sum(p.numel() for p in model.parameters())
    return model, n_params, path


def _window_batch(bw, end: np.ndarray, window_ids: Sequence[int]) -> np.ndarray:
    arr = [
        bw.timeline[end[w] - bw.window_len + 1 : end[w] + 1]
        for w in window_ids
    ]
    return np.ascontiguousarray(np.stack(arr).astype(np.float32))


@torch.no_grad()
def predict_windows(model, bw, end: np.ndarray, idx: np.ndarray, h_idx: int, device: str, batch: int = 4096):
    out: List[np.ndarray] = []
    for i in range(0, len(idx), batch):
        chunk = idx[i : i + batch]
        xb = _window_batch(bw, end, chunk)
        pred = model(torch.from_numpy(xb).to(device))
        if isinstance(pred, tuple):
            pred = pred[0]
        if pred.ndim != 2 or pred.shape[1] != len(HORIZONS):
            raise RuntimeError(f"Unexpected model output shape {tuple(pred.shape)}")
        out.append(pred[:, h_idx].detach().cpu().numpy())
    p = np.concatenate(out).astype(np.float32)
    if not np.isfinite(p).all():
        raise RuntimeError("Non-finite validation probabilities detected.")
    if p.min() < -1e-6 or p.max() > 1.0 + 1e-6:
        raise RuntimeError(
            "Model output is outside [0,1]. This runner expects probabilities because "
            "stage2_deep.Heads already applies sigmoid."
        )
    return p


def print_probability_preflight(name: str, probs: np.ndarray):
    q = np.quantile(probs, [0.0, 0.01, 0.50, 0.99, 1.0])
    rounded_unique = len(np.unique(np.round(probs, 6)))
    print(
        f"  probability preflight {model_label(name)}: "
        f"min={q[0]:.4f} p01={q[1]:.4f} median={q[2]:.4f} "
        f"p99={q[3]:.4f} max={q[4]:.4f} | unique(6dp)={rounded_unique:,}"
    )
    if np.std(probs) < 1e-6 or rounded_unique <= 2:
        raise RuntimeError(
            "Validation risks are effectively constant. Stop here: this indicates a "
            "model/output integration problem and should not be used for XAI."
        )


# ---------------------------------------------------------------------------
# Patient-diverse case selection
# ---------------------------------------------------------------------------
def subject_value(meta: np.ndarray, w: int) -> str:
    value = np.asarray(meta)[int(w)]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    return str(value)


def select_unique_patient_cases(
    idx_va: np.ndarray,
    probs: np.ndarray,
    meta: np.ndarray,
    n_case: int,
) -> List[Dict]:
    """Take at most one window per subject: that subject's highest-risk window."""
    if len(idx_va) != len(probs):
        raise ValueError("idx_va and probs must align")
    order = np.argsort(-probs, kind="mergesort")
    selected: List[Dict] = []
    seen = set()
    for pos in order:
        w = int(idx_va[int(pos)])
        sid = subject_value(meta, w)
        if sid in seen:
            continue
        seen.add(sid)
        selected.append(
            {
                "window": w,
                "subject": sid,
                "risk": float(probs[int(pos)]),
                "val_position": int(pos),
            }
        )
        if len(selected) >= n_case:
            break
    return selected


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def plot_heatmap(attr, risk, horizon, subject, variant, path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(
        f"Temporal Integrated Gradients — {model_label(variant)} — Subject {subject}\n"
        f"{horizon}-min predicted risk: {risk*100:.1f}%",
        fontsize=12,
    )
    im = ax[0].imshow(attr.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax[0], label="Absolute attribution")
    ax[0].set_yticks(range(len(DISPLAY_NAMES)))
    ax[0].set_yticklabels(DISPLAY_NAMES, fontsize=8)
    ax[0].set_xlabel("Time step (0 = 5 h ago, 59 = now)")
    ax[0].set_title("Attribution across time and features")

    imp = np.asarray(attr).sum(0)
    order = np.argsort(imp)
    ax[1].barh(range(len(imp)), imp[order], alpha=0.85)
    ax[1].set_yticks(range(len(imp)))
    ax[1].set_yticklabels([DISPLAY_NAMES[i] for i in order], fontsize=9)
    ax[1].set_xlabel("Total absolute attribution")
    ax[1].set_title("Feature importance for this patient window")
    ax[1].grid(axis="x", alpha=0.3)
    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_population(pop: Dict, horizon: int, path):
    models = list(pop)
    ncols = 3 if len(models) == 2 else 2
    fig, ax = plt.subplots(1, ncols, figsize=(19 if ncols == 3 else 13, 5))
    ax = np.atleast_1d(ax)
    y = np.arange(len(DISPLAY_NAMES))

    if len(models) == 1:
        m = models[0]
        ax[0].barh(y, pop[m]["channel_share"], alpha=0.85, label=model_label(m))
    else:
        w = 0.36
        for i, m in enumerate(models):
            ax[0].barh(
                y + (i - 0.5) * w,
                pop[m]["channel_share"],
                w,
                label=model_label(m),
                alpha=0.85,
            )
    ax[0].set_yticks(y)
    ax[0].set_yticklabels(DISPLAY_NAMES, fontsize=9)
    ax[0].set_xlabel("Share of total attribution")
    ax[0].set_title(f"Population feature attribution ({horizon} min)")
    ax[0].legend()
    ax[0].grid(axis="x", alpha=0.3)

    for m in models:
        prof = np.asarray(pop[m]["time_profile"])
        minutes_ago = (len(prof) - 1 - np.arange(len(prof))) * 5
        ax[1].plot(minutes_ago, prof, label=model_label(m), linewidth=2)
    ax[1].invert_xaxis()
    ax[1].set_xlabel("Minutes before prediction")
    ax[1].set_ylabel("Mean absolute attribution")
    ax[1].set_title("Population temporal attribution profile")
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    if len(models) == 2:
        a = np.asarray(pop[models[0]]["channel_share"])
        b = np.asarray(pop[models[1]]["channel_share"])
        ax[2].scatter(a, b, s=70, alpha=0.8)
        for i, label in enumerate(DISPLAY_NAMES):
            ax[2].annotate(label, (a[i], b[i]), fontsize=7, xytext=(3, 3), textcoords="offset points")
        lim = max(float(a.max()), float(b.max())) * 1.15
        ax[2].plot([0, lim], [0, lim], "--", alpha=0.4)
        r = float(np.corrcoef(a, b)[0, 1]) if np.std(a) > 0 and np.std(b) > 0 else np.nan
        ax[2].set_xlabel(model_label(models[0]))
        ax[2].set_ylabel(model_label(models[1]))
        ax[2].set_title(f"Attribution agreement: r = {r:.3f}")
        ax[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def plot_cf_quality(results: Dict[str, List[Dict]], path):
    models = list(results)
    x = np.arange(len(models))
    success = []
    best_attempt_pp = []
    valid_mean_pp = []

    for m in models:
        cases = results[m]
        success.append(100.0 * np.mean([len(c["counterfactuals"]) > 0 for c in cases]) if cases else 0.0)
        best_attempt_pp.append(
            100.0 * np.mean([c["cf_diagnostics"]["best_attempt_reduction"] for c in cases])
            if cases else 0.0
        )
        valid = [cf["risk_reduction"] for c in cases for cf in c["counterfactuals"]]
        valid_mean_pp.append(100.0 * np.mean(valid) if valid else 0.0)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    labels = [model_label(m) for m in models]
    ax[0].bar(x, success, width=0.55, alpha=0.85)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("Cases with ≥1 reportable counterfactual (%)")
    ax[0].set_title("Counterfactual success across unique patients")
    ax[0].set_ylim(0, 105)
    ax[0].grid(axis="y", alpha=0.3)

    width = 0.34
    ax[1].bar(x - width / 2, best_attempt_pp, width, label="Best attempt (all cases)", alpha=0.85)
    ax[1].bar(x + width / 2, valid_mean_pp, width, label="Reported counterfactuals", alpha=0.85)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_ylabel("Risk reduction (percentage points)")
    ax[1].set_title("Model-predicted counterfactual effect")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_case_csv(results: Dict[str, List[Dict]], path: Path):
    fields = [
        "model", "case_rank", "subject", "window", "true_label", "predicted_risk",
        "top_feature_1", "top_feature_1_pct", "top_feature_2", "top_feature_2_pct",
        "top_feature_3", "top_feature_3_pct", "top_time_1_min_ago",
        "top_time_2_min_ago", "top_time_3_min_ago", "n_counterfactuals",
        "best_cf_risk", "best_cf_reduction", "best_cf_plausible", "best_cf_action",
        "best_attempt_reduction", "cf_note",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for model, cases in results.items():
            for case in cases:
                tf = case["top_features"] + [{}] * 3
                tt = case["top_timesteps"] + [{}] * 3
                best = case["counterfactuals"][0] if case["counterfactuals"] else {}
                writer.writerow({
                    "model": model_label(model),
                    "case_rank": case["case_rank"],
                    "subject": case["subject"],
                    "window": case["window"],
                    "true_label": case["true_label"],
                    "predicted_risk": case["risk"],
                    "top_feature_1": tf[0].get("feature", ""),
                    "top_feature_1_pct": tf[0].get("pct", ""),
                    "top_feature_2": tf[1].get("feature", ""),
                    "top_feature_2_pct": tf[1].get("pct", ""),
                    "top_feature_3": tf[2].get("feature", ""),
                    "top_feature_3_pct": tf[2].get("pct", ""),
                    "top_time_1_min_ago": tt[0].get("minutes_ago", ""),
                    "top_time_2_min_ago": tt[1].get("minutes_ago", ""),
                    "top_time_3_min_ago": tt[2].get("minutes_ago", ""),
                    "n_counterfactuals": len(case["counterfactuals"]),
                    "best_cf_risk": best.get("predicted_risk", ""),
                    "best_cf_reduction": best.get("risk_reduction", ""),
                    "best_cf_plausible": best.get("plausible", ""),
                    "best_cf_action": best.get("action_en", ""),
                    "best_attempt_reduction": case["cf_diagnostics"].get("best_attempt_reduction", 0.0),
                    "cf_note": case.get("cf_note", ""),
                })


def prepare_output(base_out: Path):
    base_out.mkdir(parents=True, exist_ok=True)
    plots = base_out / "plots"
    if plots.exists():
        shutil.rmtree(plots)
    plots.mkdir(parents=True, exist_ok=True)
    for name in ["xai_results_v3.json", "case_explanations.csv"]:
        p = base_out / name
        if p.exists():
            p.unlink()
    return plots


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="tsl_gru")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--n_pop", type=int, default=2000)
    ap.add_argument("--n_case", type=int, default=12,
                    help="number of UNIQUE patients for case explanations")
    ap.add_argument("--n_heatmaps", type=int, default=6,
                    help="heatmaps to save per model; cases are already unique by patient")
    ap.add_argument("--risk_thresh", type=float, default=0.35)
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--n_cf", type=int, default=6)
    ap.add_argument("--bound_sd", type=float, default=1.5)
    ap.add_argument("--act_last", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check_only", action="store_true",
                    help="load data/checkpoint, verify real probability distribution and patient selection, then exit")
    a = ap.parse_args()

    variants = [v.strip() for v in a.models.split(",") if v.strip()]
    if not variants:
        raise SystemExit("--models did not contain any model name")
    if a.n_case < 1 or a.n_pop < 1 or a.n_heatmaps < 0:
        raise SystemExit("n_case and n_pop must be >=1; n_heatmaps must be >=0")

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    h_idx = HORIZONS.index(a.horizon)

    run_id = f"h{a.horizon}_" + "_".join(variants)
    base_out = Path(a.out) if a.out else Path(config.RESULTS) / "RQ3_xai_v3" / run_id
    plots = prepare_output(base_out)
    print(f"device {device} | horizon {a.horizon} (index {h_idx}) | out {base_out}")

    bw = common.load_built(a.tag)
    attach_ta(bw)
    build_features = list(bw.features)
    if build_features != FEATURE_NAMES:
        raise RuntimeError(
            "XAI channel order does not match the trained 7-channel representation.\n"
            f"build : {build_features}\nengine: {FEATURE_NAMES}"
        )

    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_va = np.flatnonzero(va)
    meta = np.asarray(bw.meta)
    end = bw.starts + bw.window_len - 1
    if not np.array_equal(bw.raw_gluc[end], bw.raw_gluc_at_pred):
        raise RuntimeError("Window-end convention mismatch between raw_gluc and raw_gluc_at_pred")

    rng = np.random.default_rng(a.seed)
    sample_n = min(256, len(idx_va))
    sample_idx = rng.choice(idx_va, size=sample_n, replace=False)

    # Load all models and perform probability preflight before expensive XAI.
    loaded = {}
    val_probs = {}
    selected = {}
    for variant in variants:
        model, n_params, ckpt = load_model(variant, bw.timeline.shape[1], device, a.suffix)
        eng = XAIEngineV3(model, device=device, ig_steps=max(2, min(a.ig_steps, 8)), n_cf=a.n_cf)
        xb = torch.from_numpy(_window_batch(bw, end, sample_idx)).to(device)
        stats = eng.a.validate_probability_output(xb)
        print(f"\n{model_label(variant)} ({n_params:,} params) | checkpoint {ckpt.name}")
        print(
            f"  direct output check: min={stats['min']:.4f} median={stats['median']:.4f} "
            f"max={stats['max']:.4f}"
        )
        p = predict_windows(model, bw, end, idx_va, h_idx, device)
        print_probability_preflight(variant, p)
        cases = select_unique_patient_cases(idx_va, p, meta, a.n_case)
        if not cases:
            raise RuntimeError("No validation cases could be selected")
        print(
            f"  selected {len(cases)} unique patients from "
            f"{len(np.unique([subject_value(meta, w) for w in idx_va]))} validation subjects"
        )
        print("  top selected patients: " + ", ".join(
            f"{c['subject']} ({c['risk']*100:.1f}%)" for c in cases[:min(8, len(cases))]
        ))
        loaded[variant] = (model, n_params)
        val_probs[variant] = p
        selected[variant] = cases

    if a.check_only:
        print("\nCHECK-ONLY PASSED: probability outputs are not double-sigmoided and case selection is patient-diverse.")
        print("No IG/counterfactual computation was run.")
        return

    # Reconstruct raw-unit statistics only after preflight passes.
    print("\nreconstructing causal expanding statistics...")
    t0 = time.time()
    rt = np.load(config.DATA_DERIVED / f"raw_treatment_{a.tag}.npz")
    subj = bw.reading_subject
    stats = {"glucose": expanding_stats(bw.raw_gluc, subj), "sigma": ta_sigma(bw.raw_gluc, subj)}
    raw_tr = {}
    for name in ACTIONABLE:
        if name not in rt:
            raise KeyError(f"{name!r} missing from raw_treatment_{a.tag}.npz")
        raw_tr[name] = rt[name]
        if len(raw_tr[name]) != len(subj):
            raise RuntimeError(f"raw treatment length mismatch for {name}")
        stats[name] = expanding_stats(rt[name], subj)
        nz = rt[name][rt[name] > 0]
        stats[f"{name}_p99"] = float(np.percentile(nz, 99)) if len(nz) else 0.0
        print(
            f"  {name:<7} p99(nonzero) {stats[f'{name}_p99']:.2f} {RAW_UNITS[name]} "
            f"| nonzero {float((rt[name] > 0).mean()):.3%}"
        )
    print(f"  done in {time.time()-t0:.0f}s")

    pop_idx = rng.choice(idx_va, size=min(a.n_pop, len(idx_va)), replace=False)
    results: Dict[str, List[Dict]] = {}
    pop: Dict[str, Dict] = {}

    for variant in variants:
        model, n_params = loaded[variant]
        eng = XAIEngineV3(
            model,
            device=device,
            ig_steps=a.ig_steps,
            n_cf=a.n_cf,
            risk_thresh=a.risk_thresh,
            bound_sd=a.bound_sd,
            n_iter=a.n_iter,
        )
        eng.cf.act_last = a.act_last
        print(f"\n{'='*72}\n{model_label(variant)} ({n_params:,} params)\n{'='*72}")

        # Population IG
        print(f"  population IG over {len(pop_idx):,} validation windows...")
        acc = np.zeros((bw.window_len, len(FEATURE_NAMES)), dtype=np.float64)
        t0 = time.time()
        report_every = max(1, min(250, len(pop_idx)))
        for i, w in enumerate(pop_idx):
            x = np.ascontiguousarray(
                bw.timeline[end[w] - bw.window_len + 1 : end[w] + 1]
            )
            acc += eng.ig.compute(x, h_idx)
            if (i + 1) % report_every == 0 or i + 1 == len(pop_idx):
                print(
                    f"    {i+1:,}/{len(pop_idx):,} "
                    f"({(time.time()-t0)/(i+1)*1000:.0f} ms/window)"
                )
        share = acc.sum(0) / (acc.sum() + 1e-12)
        pop[variant] = {
            "channel_internal_names": FEATURE_NAMES,
            "channel_names": DISPLAY_NAMES,
            "channel_share": share.tolist(),
            "time_profile": acc.mean(1).tolist(),
            "n_windows": int(len(pop_idx)),
        }
        print("  channel share: " + ", ".join(
            f"{label} {s:.1%}" for label, s in
            sorted(zip(DISPLAY_NAMES, share), key=lambda z: -z[1])
        ))

        # One case per patient
        cases = selected[variant]
        print(
            f"  case explanations on {len(cases)} unique patients "
            f"(risk {cases[0]['risk']:.3f} to {cases[-1]['risk']:.3f})"
        )
        recs: List[Dict] = []
        t0 = time.time()
        for i, case in enumerate(cases):
            w = int(case["window"])
            sid = case["subject"]
            x = np.ascontiguousarray(
                bw.timeline[end[w] - bw.window_len + 1 : end[w] + 1]
            )
            ctx = build_context(bw, raw_tr, stats, int(end[w]), bw.window_len)
            if bw.timeline[end[w], IDX["carbs_observed"]] <= 0:
                ctx["carbs_ok"] = False

            result = eng.explain(
                x,
                h_idx,
                ctx,
                with_cf=True,
                force=True,
                cf_seed=a.seed + w,
            )
            if result is None:
                raise RuntimeError(f"Unexpected empty explanation for window {w}")

            cfs = [dataclass_to_dict(c) for c in result.counterfactuals]
            rec = {
                "case_rank": i + 1,
                "window": w,
                "subject": sid,
                "true_label": int(Y[w, h_idx]),
                "risk": float(result.original_risk),
                "traffic_light": result.traffic_light,
                "top_features": result.top_features,
                "top_timesteps": result.top_timesteps,
                "ig_summary_en": result.ig_summary_en,
                "cf_note": result.cf_note,
                "cf_diagnostics": dataclass_to_dict(result.cf_diagnostics),
                "counterfactuals": cfs,
            }
            recs.append(rec)

            if i < a.n_heatmaps:
                plot_heatmap(
                    result.attribution,
                    result.original_risk,
                    a.horizon,
                    sid,
                    variant,
                    plots / f"heatmap_{variant}_rank{i+1:02d}_subject_{sid}_window_{w}.png",
                )

            top1 = result.top_features[0]
            cf_text = (
                f"CF Δ={result.counterfactuals[0].risk_reduction*100:.2f}pp"
                if result.counterfactuals else
                f"no reportable CF (best attempt Δ={result.cf_diagnostics.best_attempt_reduction*100:.2f}pp)"
            )
            print(
                f"    {i+1:>2}/{len(cases)} subject {sid:<8} risk={result.original_risk*100:5.1f}% "
                f"top={top1['feature']} ({top1['pct']:.1f}%) | {cf_text}"
            )

        results[variant] = recs
        del loaded[variant]
        if device == "cuda":
            torch.cuda.empty_cache()

    plot_population(pop, a.horizon, plots / "population_attribution.png")
    plot_cf_quality(results, plots / "cf_quality.png")
    write_case_csv(results, base_out / "case_explanations.csv")

    agree = None
    if len(pop) == 2:
        m = list(pop)
        a0 = np.asarray(pop[m[0]]["channel_share"])
        a1 = np.asarray(pop[m[1]]["channel_share"])
        if np.std(a0) > 0 and np.std(a1) > 0:
            agree = float(np.corrcoef(a0, a1)[0, 1])
        print(f"\nattribution agreement {model_label(m[0])} vs {model_label(m[1])}: {agree}")

    doc = {
        "version": 3,
        "horizon": a.horizon,
        "n_pop": int(len(pop_idx)),
        "n_case_requested": a.n_case,
        "case_selection": "one highest-risk validation window per unique subject",
        "n_heatmaps_per_model": a.n_heatmaps,
        "ig_steps": a.ig_steps,
        "n_cf": a.n_cf,
        "n_iter": a.n_iter,
        "risk_thresh": a.risk_thresh,
        "bound_sd": a.bound_sd,
        "act_last": a.act_last,
        "population": pop,
        "attribution_agreement_r": agree,
        "explanations": results,
        "caveat": (
            "Counterfactuals are model-based sensitivity analyses under modified historical "
            "input records. They are associational, not causal, and are not treatment recommendations."
        ),
    }
    with open(base_out / "xai_results_v3.json", "w", encoding="utf-8") as f:
        json.dump(doc, f, indent=2, default=float)

    print(f"\nWROTE:\n  {base_out / 'xai_results_v3.json'}")
    print(f"  {base_out / 'case_explanations.csv'}")
    print(f"  {plots / 'population_attribution.png'}")
    print(f"  {plots / 'cf_quality.png'}")
    print(f"  {plots}/heatmap_*.png")


if __name__ == "__main__":
    main()
