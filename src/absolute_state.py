"""
absolute_state.py
===================
Does causal per-patient normalisation still hide absolute physiological
scale, beyond the threshold location that TA already restored?

THE ARGUMENT
------------
Channel 0 is (G_t - mu_t)/sigma_t. The TA channels add
c_t = sigmoid(-(G_t - 70)/sigma_t) and a scaled downward velocity.

From those the network can recover (G_t - 70)/sigma_t by inverting c_t,
and it already has (G_t - mu_t)/sigma_t. That is two equations in three
unknowns (G, mu, sigma), so **absolute glucose in mg/dL is not
determined**. Everything the model sees is expressed in units of that
child's own variability.

c_t also saturates. At sigma = 35 mg/dL:

        G = 120  ->  c = 0.193
        G = 150  ->  c = 0.092
        G = 180  ->  c = 0.041
        G = 250  ->  c = 0.006

so the entire hyperglycaemic range is compressed into a narrow band near
zero -- exactly the region where the error-regime audit found performance
collapsing (AUPRC 0.879 below 90 mg/dL, 0.105 above 150 at h=30) and
where the jointly-missed events sit (100.9 mg/dL at h=15, 119.8 at h=30).

TA restored the threshold LOCATION and produced roughly +0.09 on both GRU
and TCN. This asks whether the absolute SCALE is a second, separate
omission.

WHAT IS ADDED
-------------
Two channels, both scaled by FIXED clinical constants rather than by
per-patient statistics, so they carry absolute information that no
per-patient normalisation can express:

    g_abs   = (G_t - 100) / 50            absolute level, mg/dL
    gdot    = (G_{t-1} - G_t) / 2         absolute rate, mg/dL/min

A third channel is available but held back for a later step, because
mixing two hypotheses in one experiment would make the result
uninterpretable:

    tau_t   = projected minutes to reach 70 at the current rate,
              clipped to [0, 120] and scaled

THE PILOT
---------
    A  gru_ta        7 channels                    (current baseline)
    B  gru_ta_abs    9 channels, plus absolute level and rate

Same training indices, same loss, same seed, same early stopping. Only
the input differs.

    B - A  <  0.005   absolute scale adds nothing; stop
    B - A  >= 0.010   a second representation bottleneck exists
    B - A  >= 0.020   comparable to the original TA finding

DIAGNOSTIC FIRST
----------------
--diagnose runs with no training. It compares c_t, absolute glucose and
velocity between events both models detected and events both models
missed. If the missed events sit where c_t is saturated, that is direct
evidence for the mechanism rather than an inference from the aggregate
numbers.

Usage:
    python absolute_state.py --diagnose
    python absolute_state.py --model all --seed 42
"""

import gc
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import common
from stage2_deep import WindowDataset, MultiHorizonLoss, Heads
from ta_gru import attach_ta

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "absolute_state"
MODELS = ["gru_ta", "gru_ta_abs"]

# fixed clinical scaling -- deliberately NOT per-patient
G_CENTER, G_SCALE = 100.0, 50.0
RATE_SCALE = 2.0
ABS_CLAMP = 10.0


def attach_absolute(bw, verbose=True):
    """
    Append absolute level and absolute rate as channels 7 and 8.

    Both use fixed constants, so the same mg/dL maps to the same value for
    every child. That is the whole point: it is information no
    per-patient normalisation can represent.
    """
    g = bw.raw_gluc.astype(np.float32)
    g_abs = (g - G_CENTER) / G_SCALE

    gdot = np.zeros_like(g)
    for si in range(len(bw.subjects)):
        m = np.flatnonzero(bw.reading_subject == si)
        if len(m) < 2:
            continue
        gg = g[m]
        d = np.diff(gg, prepend=gg[0]) / bw.sample_min      # mg/dL per min
        gdot[m] = -d                                        # positive = falling
    gdot = gdot / RATE_SCALE

    extra = np.stack([np.clip(g_abs, -ABS_CLAMP, ABS_CLAMP),
                      np.clip(gdot, -ABS_CLAMP, ABS_CLAMP)], axis=1)
    bw.timeline = np.concatenate([bw.timeline, extra], axis=1)
    bw.features = list(bw.features) + ["glucose_absolute", "rate_absolute"]
    if verbose:
        print(f"  absolute channels: g_abs in "
              f"[{extra[:,0].min():.2f},{extra[:,0].max():.2f}]  "
              f"rate in [{extra[:,1].min():.2f},{extra[:,1].max():.2f}]")
    return bw


# ─── DIAGNOSTIC ───────────────────────────────────────────────────────────────

def diagnose(args):
    """
    Compare detected against jointly-missed positives on c_t, absolute
    glucose and velocity. No training.
    """
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)

    paths = [("GRU+TA", config.RESULTS / "RQ2_models" / "tap_gru" / "gru_ta__s42_probs.npz"),
             ("TCN+TA", config.RESULTS / "RQ2_models" / "dms_tcn" / "tcn_ta__s42_probs.npz")]
    P, idx = {}, None
    for name, p in paths:
        if not p.exists():
            print(f"  missing {p}")
            return
        z = np.load(p)
        idx = z["idx_test"] if idx is None else idx
        P[name] = z["test"]

    j = HORIZONS.index(args.horizon)
    y = bw._labels_any[idx, j].astype(int)
    ends = bw.starts[idx] + bw.window_len - 1
    c_t = bw.timeline[ends, 5]                       # threshold proximity
    g = bw.raw_gluc[ends]
    vel = (bw.raw_gluc[ends - 3] - g) / 15.0

    p1, p2 = P["GRU+TA"][:, j], P["TCN+TA"][:, j]
    t1 = np.quantile(p1, 1 - y.mean() * 3)
    t2 = np.quantile(p2, 1 - y.mean() * 3)
    det = (p1 >= t1) & (p2 >= t2) & (y == 1)
    miss = (p1 < t1) & (p2 < t2) & (y == 1)

    print(f"\n{'#'*90}")
    print(f"# TA SATURATION on detected vs jointly-missed events (h={args.horizon})")
    print(f"{'#'*90}")
    print(f"\n  detected: {det.sum():,}   jointly missed: {miss.sum():,}")
    print(f"\n  {'quantity':>22} {'detected':>22} {'missed':>22}")
    out = {}
    for nm, v in [("threshold proximity c_t", c_t),
                  ("absolute glucose mg/dL", g),
                  ("velocity mg/dL/min", vel)]:
        a, b = v[det], v[miss]
        print(f"  {nm:>22} {a.mean():>10.4f} +/- {a.std():>7.4f} "
              f"{b.mean():>10.4f} +/- {b.std():>7.4f}")
        out[nm] = {"detected_mean": float(a.mean()),
                   "missed_mean": float(b.mean())}

    print(f"\n  c_t percentiles           p50        p90        p99      max")
    for lab, m in [("detected", det), ("missed", miss)]:
        q = np.percentile(c_t[m], [50, 90, 99])
        print(f"    {lab:>20} {q[0]:>10.4f} {q[1]:>10.4f} {q[2]:>10.4f} "
              f"{c_t[m].max():>8.4f}")

    sat = float((c_t[miss] < 0.05).mean())
    sat_d = float((c_t[det] < 0.05).mean())
    print(f"\n  fraction with c_t < 0.05 (saturated region):")
    print(f"    detected {100*sat_d:>6.2f}%     missed {100*sat:>6.2f}%")
    out["frac_saturated"] = {"detected": sat_d, "missed": sat}

    print(f"\n{'#'*90}")
    if sat > 0.5 and sat > 2 * sat_d:
        print("  -> Most missed events sit where c_t is saturated. In that")
        print("     region c_t cannot distinguish 120 from 180 mg/dL, and")
        print("     absolute glucose is not recoverable from the other")
        print("     channels, so adding it is worth one run.")
    else:
        print("  -> Missed events are not concentrated in the saturated")
        print("     region, so c_t compression is unlikely to be the")
        print("     limiting factor.")
    print(f"{'#'*90}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, f"_saturation_h{args.horizon}", out)


# ─── MODEL ────────────────────────────────────────────────────────────────────

class GRUModel(nn.Module):
    def __init__(self, in_ch, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


@torch.no_grad()
def predict(model, bw, idx, labels, device, ch, batch=4096, workers=2):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    return np.concatenate([model(xb[:, :, :ch].to(device)).cpu().numpy()
                           for xb, _ in dl]).astype(np.float32)


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    ch = 7 if variant == "gru_ta" else 9
    print(f"\n{'='*78}\n{variant} | {ch} channels | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    model = GRUModel(ch, hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, f"{variant}__s{args.seed}",
                                resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state = None
    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0
        for xb, yb in dl:
            xb, yb = xb[:, :, :ch].to(device, non_blocking=True), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item(); nb += 1

        pv = predict(model, bw, idx_va, Y, device, ch, workers=args.workers)
        v = float(np.mean([average_precision_score(Y[idx_va, k].astype(int),
                                                   pv[:, k])
                           for k in range(len(HORIZONS))]))
        sched.step(v)
        improved = v > best
        if improved:
            best, best_ep, bad = v, ep, 0
            best_state = {k: t.detach().cpu().clone()
                          for k, t in model.state_dict().items()}
        else:
            bad += 1
        ck.save(model, opt, sched, ep, best, best_ep, bad, improved)
        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"val mean-AUPRC {v:.4f}{' *' if improved else ''}")
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, ch, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, ch, workers=args.workers)

    res = {"model": variant, "seed": args.seed, "n_channels": ch,
           "n_params": int(n_par), "val_mean_auprc": best,
           "best_epoch": int(best_ep), "horizons": {}}
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, tinfo = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["test_prevalence"] = float(Y[idx_te, j].mean())
        res["horizons"][str(h)] = ev
        m, c = ev["per_subject_mean"], ev["constraint"]
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")

    res["minutes"] = (time.time() - t0) / 60
    name = f"{variant}__s{args.seed}"
    np.savez_compressed(OUT_DIR / f"{name}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    common.save_result(OUT_DIR, name, res)
    ck.cleanup()
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*90}")
    print("# ABSOLUTE-STATE PILOT — does absolute glucose scale add information?")
    print(f"{'#'*90}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>14} {'ch':>4} {'AUPRC':>9} {'PPV':>9} {'Recall':>9}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>14} {lst[0]['n_channels']:>4} {m['auprc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*90}\n# DECISION\n{'#'*90}")
    for h in HORIZONS:
        A = {r["seed"]: r for r in runs.get("gru_ta", []) if str(h) in r["horizons"]}
        B = {r["seed"]: r for r in runs.get("gru_ta_abs", []) if str(h) in r["horizons"]}
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        a = np.mean([A[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        b = np.mean([B[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        d = b - a
        verdict = ("comparable to the original TA finding" if d >= 0.020 else
                   "a second representation bottleneck exists" if d >= 0.010 else
                   "marginal" if d >= 0.005 else
                   "STOP: absolute scale adds nothing")
        print(f"\nh={h}:  A {a:.4f}   B {b:.4f}   B-A = {d:+.4f}   -> {verdict}")
        if len(shared) == 1:
            s = shared[0]
            dd = common.paired_delta(A[s]["horizons"][str(h)]["per_subject"],
                                     B[s]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"        paired bootstrap: {dd['delta']:+.4f} "
                  f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    common.save_result(OUT_DIR, "_absolute_state_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_absolute_state_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--horizon", type=int, default=30, choices=HORIZONS)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.diagnose:
        diagnose(args)
        return
    if args.collect:
        collect(args)
        return
    if args.model is None:
        print("Choose --diagnose (free), --model all, or --collect.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    print(f"  channels: {bw.features}")
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"train {len(idx_tr):,} (shared) val {len(idx_va):,} test {len(idx_te):,}")

    for v in (MODELS if args.model == "all" else [args.model]):
        if (OUT_DIR / f"{v}__s{args.seed}.json").exists() and not args.force:
            print(f"\n[skip] {v}__s{args.seed} already done")
            continue
        try:
            run_one(v, bw, Y, idx_tr, idx_va, idx_te, device, args)
        except Exception as e:
            print(f"  !! {v} FAILED: {type(e).__name__}: {e}")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    collect(args)


if __name__ == "__main__":
    main()
