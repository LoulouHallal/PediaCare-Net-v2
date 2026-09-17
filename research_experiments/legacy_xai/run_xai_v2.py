"""
run_xai_v2.py  --  XAI for GRU+TA and TSL-GRU on MetaboNet
===========================================================

Two passes, because they answer different questions and cost very
different amounts:

  POPULATION  Temporal IG over a large sample (default 2000 windows per
              model). Gives the attribution statistics: which channels
              carry the prediction, and at which time offsets. Cheap
              enough to be representative.

  CASE        IG + DiCE counterfactuals on high-risk windows (default 50
              per model). Gives the per-patient explanations and the
              counterfactual quality metrics. Expensive: 200 optimisation
              steps each.

WHY BOTH MODELS
---------------
TSL-GRU matches GRU+TA within the noise floor at 61% fewer parameters.
If the two also attribute to the same channels at the same time offsets,
that is independent evidence they learned the same function rather than
two different functions that happen to score alike. The attribution
correlation between the arms is reported for exactly this reason.

CLINICAL UNITS
--------------
build_windows.py normalises causally per subject:

    mean_t, std_t = expanding stats over readings < t   (shift by one)
    norm_t        = (x_t - mean_t) / max(std_t, 1e-6)

Those statistics were never saved, so they are recomputed here from
raw_gluc and raw_treatment_stride6.npz with the identical formula. This
makes the inversion exact rather than approximate, which is what allows
a counterfactual to be stated in grams and units instead of standard
deviations. Verified invertible to ~5e-07.

sigma for the TA channels is a SEPARATE quantity -- the expanding std of
RAW glucose used by compute_ta_channels -- and is recomputed with that
function's own formula, not the normalisation one.

    python run_xai_v2.py --models gru_ta,tsl_gru --horizon 30
    python run_xai_v2.py --models tsl_gru --n_case 10 --n_pop 200   # quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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
from xai_engine_v2 import (XAIEngineV2, FEATURE_NAMES, IDX, HORIZONS,  # noqa
                           ACTIONABLE, RAW_UNITS)

HYPO = 70.0


# ─── causal expanding statistics, per subject ───────────────────────────────
def expanding_stats(x: np.ndarray, subject: np.ndarray):
    """
    Reproduces build_windows.py's causal normalisation statistics.

        c1/c2 cumulative -> mean, std -> shift by one

    Returns (mean, std) aligned to x, computed within each subject.
    """
    mean = np.zeros_like(x, dtype=np.float32)
    std = np.zeros_like(x, dtype=np.float32)
    for si in np.unique(subject):
        m = np.flatnonzero(subject == si)
        v = x[m].astype(np.float64)
        cnt = np.arange(1, len(v) + 1)
        c1, c2 = np.cumsum(v), np.cumsum(v ** 2)
        mu = c1 / cnt
        sd = np.sqrt(np.maximum(c2 / cnt - mu ** 2, 0.0))
        mu = np.concatenate([mu[:1], mu[:-1]])
        sd = np.concatenate([sd[:1], sd[:-1]])
        mean[m], std[m] = mu.astype(np.float32), sd.astype(np.float32)
    return mean, std


def ta_sigma(raw_gluc: np.ndarray, subject: np.ndarray):
    """The sigma compute_ta_channels uses: expanding std of RAW glucose,
    shifted by one, floored at 1e-6."""
    _, sd = expanding_stats(raw_gluc, subject)
    return np.maximum(sd, 1e-6)


# ─── context per window ─────────────────────────────────────────────────────
def build_context(bw, raw_tr, stats, end_idx, window_len):
    """All per-reading quantities the counterfactual needs, sliced to the
    window ending at end_idx."""
    sl = slice(end_idx - window_len + 1, end_idx + 1)
    ctx = {
        "g_mean": stats["glucose"][0][sl].astype(np.float32),
        "g_std": np.maximum(stats["glucose"][1][sl], 1e-6).astype(np.float32),
        "sigma": stats["sigma"][sl].astype(np.float32),
    }
    for n in ACTIONABLE:
        mu, sd = stats[n]
        ctx[f"{n}_mean"] = mu[sl].astype(np.float32)
        ctx[f"{n}_std"] = np.maximum(sd[sl], 1e-6).astype(np.float32)
        ctx[f"{n}_raw"] = raw_tr[n][sl].astype(np.float32)
        ctx[f"{n}_p99"] = stats[f"{n}_p99"]
        ctx[f"{n}_ok"] = True
    return ctx


def load_model(variant, in_ch, device, suffix=""):
    name = f"{variant}__weighted_bce__none__any" + (f"__{suffix}" if suffix else "")
    p = TSL_DIR / f"{name}.pt"
    if not p.exists():
        raise SystemExit(f"checkpoint not found: {p}")
    blob = torch.load(p, map_location=device, weights_only=False)
    cfg = blob.get("config", {})
    m = TSLGRU(variant, in_ch=in_ch, hidden=cfg.get("hidden", 64),
               dropout=cfg.get("dropout", 0.2), rho=cfg.get("rho", 0.9))
    m.load_state_dict(blob["model_state"])
    return m.to(device).eval(), sum(p_.numel() for p_ in m.parameters())


# ─── plots ──────────────────────────────────────────────────────────────────
def plot_heatmap(attr, risk, horizon, subj, variant, path):
    fig, ax = plt.subplots(1, 2, figsize=(16, 5.5))
    fig.suptitle(f"Temporal Integrated Gradients — {variant} — subject {subj}\n"
                 f"h={horizon} min | predicted risk {risk*100:.0f}%", fontsize=12)
    im = ax[0].imshow(attr.T, aspect="auto", cmap="YlOrRd",
                      interpolation="nearest")
    plt.colorbar(im, ax=ax[0], label="|attribution|")
    ax[0].set_yticks(range(len(FEATURE_NAMES)))
    ax[0].set_yticklabels(FEATURE_NAMES, fontsize=8)
    ax[0].set_xlabel("timestep (0 = 5 h ago, 59 = now)")
    ax[0].set_title("attribution (time × channel)")

    imp = attr.sum(0)
    order = np.argsort(imp)
    ax[1].barh(range(len(imp)), imp[order], color="#2196F3", alpha=.85)
    ax[1].set_yticks(range(len(imp)))
    ax[1].set_yticklabels([FEATURE_NAMES[i] for i in order], fontsize=9)
    ax[1].set_xlabel("total |attribution|")
    ax[1].set_title("channel importance")
    ax[1].grid(axis="x", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_population(pop, horizon, path):
    """Channel share and temporal profile, both models side by side."""
    models = list(pop)
    fig, ax = plt.subplots(1, 3, figsize=(19, 5))
    w = 0.38
    y = np.arange(len(FEATURE_NAMES))
    for i, m in enumerate(models):
        share = pop[m]["channel_share"]
        ax[0].barh(y + (i - .5) * w, share, w, label=m, alpha=.85)
    ax[0].set_yticks(y); ax[0].set_yticklabels(FEATURE_NAMES, fontsize=9)
    ax[0].set_xlabel("share of total attribution")
    ax[0].set_title(f"channel attribution, h={horizon} min")
    ax[0].legend(); ax[0].grid(axis="x", alpha=.3)

    for m in models:
        prof = pop[m]["time_profile"]
        ax[1].plot(np.arange(len(prof)), prof, label=m, linewidth=2)
    ax[1].set_xlabel("timestep (59 = now)")
    ax[1].set_ylabel("mean |attribution|")
    ax[1].set_title("temporal profile")
    ax[1].legend(); ax[1].grid(alpha=.3)

    if len(models) == 2:
        a = np.array(pop[models[0]]["channel_share"])
        b = np.array(pop[models[1]]["channel_share"])
        ax[2].scatter(a, b, s=70, alpha=.8)
        for i, n in enumerate(FEATURE_NAMES):
            ax[2].annotate(n, (a[i], b[i]), fontsize=7,
                           xytext=(3, 3), textcoords="offset points")
        lim = max(a.max(), b.max()) * 1.15
        ax[2].plot([0, lim], [0, lim], "k--", alpha=.4)
        r = float(np.corrcoef(a, b)[0, 1])
        ax[2].set_xlabel(models[0]); ax[2].set_ylabel(models[1])
        ax[2].set_title(f"do the two models agree?  r = {r:.3f}")
        ax[2].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_cf_quality(res, path):
    models = list(res)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(len(models)); w = .35
    red = [np.mean([c["risk_reduction"] for e in res[m] for c in e["counterfactuals"]] or [0])
           for m in models]
    plaus = [100 * np.mean([c["plausible"] for e in res[m] for c in e["counterfactuals"]] or [0])
             for m in models]
    ax[0].bar(x, red, w * 1.6, color="#2196F3", alpha=.85)
    ax[0].set_xticks(x); ax[0].set_xticklabels(models)
    ax[0].set_ylabel("mean risk reduction"); ax[0].set_title("counterfactual effect")
    ax[0].grid(axis="y", alpha=.3)
    ax[1].bar(x, plaus, w * 1.6, color="#4CAF50", alpha=.85)
    ax[1].set_xticks(x); ax[1].set_xticklabels(models)
    ax[1].set_ylabel("% within observed cohort range")
    ax[1].set_title("counterfactual plausibility")
    ax[1].set_ylim(0, 105); ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout(); plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


# ─── main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="gru_ta,tsl_gru")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--n_pop", type=int, default=2000,
                    help="windows for population attribution (IG only)")
    ap.add_argument("--n_case", type=int, default=50,
                    help="high-risk windows for IG + counterfactuals")
    ap.add_argument("--risk_thresh", type=float, default=0.35)
    ap.add_argument("--ig_steps", type=int, default=50)
    ap.add_argument("--n_iter", type=int, default=200)
    ap.add_argument("--bound_sd", type=float, default=1.5)
    ap.add_argument("--act_last", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = a.out or str(config.RESULTS / "RQ3_xai")
    plots = os.path.join(out, "plots")
    os.makedirs(plots, exist_ok=True)
    h_idx = HORIZONS.index(a.horizon)
    print(f"device {dev} | horizon {a.horizon} (index {h_idx}) | out {out}")

    bw = common.load_built(a.tag)
    attach_ta(bw)
    if list(bw.features) != FEATURE_NAMES:
        print(f"  WARNING: channel order differs from the engine's list\n"
              f"    build : {list(bw.features)}\n    engine: {FEATURE_NAMES}")
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_va = np.flatnonzero(va)

    # ---- reconstruct causal statistics in clinical units ------------------
    print("reconstructing causal expanding statistics...")
    t0 = time.time()
    rt = np.load(config.DATA_DERIVED / f"raw_treatment_{a.tag}.npz")
    subj = bw.reading_subject
    stats = {"glucose": expanding_stats(bw.raw_gluc, subj),
             "sigma": ta_sigma(bw.raw_gluc, subj)}
    raw_tr = {}
    for n in ACTIONABLE:
        raw_tr[n] = rt[n]
        stats[n] = expanding_stats(rt[n], subj)
        stats[f"{n}_p99"] = float(np.percentile(rt[n][rt[n] > 0], 99)) \
            if (rt[n] > 0).any() else 0.0
        print(f"  {n:<7} p99(nonzero) {stats[f'{n}_p99']:.2f} {RAW_UNITS[n]}"
              f"   nonzero {float((rt[n] > 0).mean()):.3%}")
    print(f"  done in {time.time()-t0:.0f}s")

    end = bw.starts + bw.window_len - 1
    assert np.array_equal(bw.raw_gluc[end], bw.raw_gluc_at_pred), \
        "window end convention mismatch"

    rng = np.random.default_rng(a.seed)
    pop_idx = rng.choice(idx_va, size=min(a.n_pop, idx_va.size), replace=False)

    results, pop = {}, {}
    for variant in a.models.split(","):
        model, n_par = load_model(variant, bw.timeline.shape[1], dev, a.suffix)
        eng = XAIEngineV2(model, device=dev, ig_steps=a.ig_steps,
                          risk_thresh=a.risk_thresh, bound_sd=a.bound_sd,
                          n_iter=a.n_iter)
        eng.cf.act_last = a.act_last
        print(f"\n{'='*72}\n{variant}  ({n_par:,} params)\n{'='*72}")

        # ---- population attribution ---------------------------------------
        print(f"  population IG over {len(pop_idx):,} windows...")
        acc = np.zeros((bw.window_len, len(FEATURE_NAMES)), dtype=np.float64)
        t0 = time.time()
        for i, w in enumerate(pop_idx):
            x = bw.timeline[end[w] - bw.window_len + 1: end[w] + 1]
            acc += eng.ig.compute(np.ascontiguousarray(x), h_idx)
            if (i + 1) % 250 == 0:
                print(f"    {i+1:,}/{len(pop_idx):,}  "
                      f"({(time.time()-t0)/(i+1)*1000:.0f} ms/window)")
        share = acc.sum(0) / (acc.sum() + 1e-12)
        pop[variant] = {"channel_share": share.tolist(),
                        "time_profile": acc.mean(1).tolist(),
                        "n_windows": int(len(pop_idx))}
        print("  channel share: " + ", ".join(
            f"{n} {s:.1%}" for n, s in
            sorted(zip(FEATURE_NAMES, share), key=lambda z: -z[1])))

        # ---- case explanations with counterfactuals -----------------------
        with torch.no_grad():
            probs = []
            for i in range(0, len(idx_va), 4096):
                chunk = idx_va[i:i + 4096]
                xb = np.stack([bw.timeline[end[w] - bw.window_len + 1: end[w] + 1]
                               for w in chunk])
                probs.append(torch.sigmoid(model(
                    torch.tensor(xb, dtype=torch.float32, device=dev)
                ))[:, h_idx].cpu().numpy())
            probs = np.concatenate(probs)

        hi = idx_va[np.argsort(-probs)[:a.n_case]]
        print(f"  counterfactuals on {len(hi)} highest-risk windows "
              f"(risk {probs.max():.3f} to {probs[np.argsort(-probs)[len(hi)-1]]:.3f})")

        recs = []
        t0 = time.time()
        for i, w in enumerate(hi):
            x = np.ascontiguousarray(
                bw.timeline[end[w] - bw.window_len + 1: end[w] + 1])
            ctx = build_context(bw, raw_tr, stats, int(end[w]), bw.window_len)
            if bw.timeline[end[w], IDX["carbs_observed"]] <= 0:
                ctx["carbs_ok"] = False        # subject never records carbs
            r = eng.explain(x, h_idx, ctx, with_cf=True, force=True)
            if r is None:
                continue
            sid = str(np.asarray(bw.meta)[w])
            recs.append({
                "window": int(w), "subject": sid,
                "risk": r.original_risk, "traffic_light": r.traffic_light,
                "top_features": r.top_features,
                "top_timesteps": r.top_timesteps,
                "ig_summary_en": r.ig_summary_en,
                "cf_note": r.cf_note,
                "counterfactuals": [
                    {"id": c.cf_id, "predicted_risk": c.predicted_risk,
                     "risk_reduction": c.risk_reduction,
                     "changes_raw": c.changes_raw, "validity": c.validity,
                     "plausible": c.plausible, "action_en": c.action_en}
                    for c in r.counterfactuals],
            })
            if i < 3:
                plot_heatmap(r.attribution, r.original_risk, a.horizon, sid,
                             variant,
                             os.path.join(plots,
                                          f"heatmap_{variant}_{sid}_{w}.png"))
            if (i + 1) % 10 == 0:
                print(f"    {i+1}/{len(hi)}  "
                      f"({(time.time()-t0)/(i+1):.1f} s/window)")
        results[variant] = recs

        # ---- per-model counterfactual summary -----------------------------
        allcf = [c for e in recs for c in e["counterfactuals"]]
        if allcf:
            red = np.array([c["risk_reduction"] for c in allcf])
            pl = np.array([c["plausible"] for c in allcf])
            none = np.mean([not c["changes_raw"] for c in allcf])
            print(f"  mean risk reduction   {red.mean():.4f}")
            print(f"  >20% reduction        {(red > 0.20).mean():.1%}")
            print(f"  within cohort range   {pl.mean():.1%}")
            print(f"  no actionable change  {none:.1%}")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ---- outputs ----------------------------------------------------------
    plot_population(pop, a.horizon, os.path.join(plots, "population_attribution.png"))
    plot_cf_quality(results, os.path.join(plots, "cf_quality.png"))

    agree = None
    if len(pop) == 2:
        m = list(pop)
        agree = float(np.corrcoef(pop[m[0]]["channel_share"],
                                  pop[m[1]]["channel_share"])[0, 1])
        print(f"\nattribution agreement between {m[0]} and {m[1]}: r = {agree:.4f}")
        print("  high r means the smaller model reads the same inputs the same")
        print("  way -- evidence it learned the same function, not merely a")
        print("  different one that scores alike.")

    doc = {"horizon": a.horizon, "n_pop": int(len(pop_idx)),
           "n_case": a.n_case, "risk_thresh": a.risk_thresh,
           "bound_sd": a.bound_sd, "act_last": a.act_last,
           "population": pop, "attribution_agreement_r": agree,
           "explanations": results,
           "caveat": ("Counterfactuals report what the MODEL would predict "
                      "under a changed input. This is associational, not "
                      "causal: it does not establish that performing the "
                      "action would prevent hypoglycaemia.")}
    with open(os.path.join(out, "xai_results_v2.json"), "w") as f:
        json.dump(doc, f, indent=2, default=float)
    print(f"\nwrote -> {out}/xai_results_v2.json and {plots}/")


if __name__ == "__main__":
    main()
