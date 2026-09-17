"""
adew_pilot.py
===============
ADEW-GRU pilot.

    gru           reference, same Python loop        H=64   41,188
    adew          + anchored factorised erase        H=64   54,182  (+31%)
    adew_delay    + gated delayed feedback           H=64   80,168  (+95%)

    adew_eq       equal-parameter control            H=56   41,810  (+1.5%)
    adew_delay_eq equal-parameter control            H=45   40,528  (-1.6%)

TWO COMPARISONS, RUN AND REPORTED SEPARATELY
--------------------------------------------
Equal width asks: does the mechanism help at the same recurrent state
dimension? Equal parameters asks: does it survive once the extra capacity
is controlled? Collapsing them into one arm — which is what the KEW run
did, at H=56 against the GRU's 64 — leaves both questions unanswered.

WHY THIS ARCHITECTURE CAN'T LOSE BY CONSTRUCTION
------------------------------------------------
At alpha_E = alpha_D = 0 the cell IS the GRU, verified to 0.00e+00. So a
tie means the anchors stayed near zero and the mechanism was not adopted,
which is itself a clean finding. Every earlier proposed cell in this
project replaced the GRU recurrence and had to re-derive what the baseline
already did.

THE ANCHOR IS THE PRIMARY DIAGNOSTIC
------------------------------------
Logged every epoch:

    alpha_E, alpha_D    did the anchors move off their initialisation?
    E_mean, E_max       how much erasure is actually applied
    kew_sum_err         K + E + W = 1 (must stay ~0)
    delay_contrib       magnitude of the delayed correction
    E by outcome        does erasure differ before events?

alpha staying at ~0.01 means the optimiser declined the mechanism. That is
a different result from "the mechanism was used and did not help", and the
distinction is exactly what thirteen previous ties in this project turned on.

STOPPING RULE, fixed before running
-----------------------------------
Primary outcome is the mean AUPRC across the four horizons.

    delta < +0.003    stop
    delta >= +0.003   interesting, run seeds 43/44
    delta >= +0.005   run the equal-parameter control and full ablation

Usage:
    python adew_pilot.py --model gru --seed 42
    python adew_pilot.py --model adew --seed 42
    python adew_pilot.py --collect
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
from stage2_deep import WindowDataset, MultiHorizonLoss
from ta_gru import attach_ta
from absolute_state import attach_absolute
from adew_gru import ADEWGRU, ADEWConfig, ARMS, count_parameters

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "adew_gru"
MODELS = ["gru", "adew", "adew_delay", "adew_eq", "adew_delay_eq"]
SPEC = {
    "gru":           ("gru", 64),
    "adew":          ("adew", 64),
    "adew_delay":    ("adew_delay", 64),
    "adew_eq":       ("adew", 56),
    "adew_delay_eq": ("adew_delay", 45),
}


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, E = [], []
    for xb, _ in dl:
        if want_diag:
            o, d = model(xb.to(device), return_diagnostics=True)
            if "_E_per_window" in d:
                E.append(d["_E_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        P.append(torch.sigmoid(o).cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    return (out, np.concatenate(E)) if (want_diag and E) else (out, None)


def run_one(name, bw, Y, idx_tr, idx_va, idx_te, device, args):
    arm, H = SPEC[name]
    H = args.hidden if args.hidden else H
    print(f"\n{'='*78}\n{name} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cfg = ADEWConfig(bw.timeline.shape[1], hidden_size=H,
                     n_horizons=len(HORIZONS), delay_steps=args.delay_steps,
                     alpha_init=args.alpha_init, **ARMS[arm])
    model = ADEWGRU(cfg).to(device)
    n_par = count_parameters(model)
    print(f"  arm {arm}  hidden {H}  params {n_par:,}  train n={len(idx_tr):,}")
    if arm != "gru":
        print(f"  alpha init {args.alpha_init}  "
              f"(0 gives exact GRU nesting but zero gradient to the "
              f"controller)")
        if cfg.use_delay:
            print(f"  delay {args.delay_steps} steps = "
                  f"{args.delay_steps * bw.sample_min} min")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)

    def crit(out, y):
        return crit_ce(torch.sigmoid(out), y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, f"{name}__s{args.seed}",
                                resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state = None

    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
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

        pv, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(f"{name}: non-finite predictions, epoch {ep}")
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

        if arm != "gru":
            with torch.no_grad():
                xb, _ = next(iter(DataLoader(
                    WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                    num_workers=0)))
                _, d = model(xb.to(device), return_diagnostics=True)
            line = f"      alpha_E {float(d['alpha_E']):+.5f}"
            if "alpha_D" in d:
                line += f"  alpha_D {float(d['alpha_D']):+.5f}"
            line += (f"   E {float(d['E_mean']):.5f} (max "
                     f"{float(d['E_max']):.4f})   z {float(d['z_mean']):.3f}")
            print(line)
            print(f"      K+E+W err {float(d['kew_sum_err']):.2e}"
                  + (f"   delay contrib "
                     f"{float(d['delay_contrib_mean']):.5f}"
                     if "delay_contrib_mean" in d else ""))
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt, e_w = predict(model, bw, idx_te, Y, device, workers=args.workers,
                      want_diag=(arm != "gru"))

    res = {"model": name, "arm": arm, "seed": args.seed, "hidden": H,
           "n_params": int(n_par), "alpha_init": args.alpha_init,
           "val_mean_auprc": best, "best_epoch": int(best_ep), "horizons": {}}
    meta_te = bw.meta[idx_te]
    aps = []
    for j, h in enumerate(HORIZONS):
        thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["test_prevalence"] = float(Y[idx_te, j].mean())
        res["horizons"][str(h)] = ev
        m, c = ev["per_subject_mean"], ev["constraint"]
        aps.append(m["auprc"])
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")
    res["mean_auprc"] = float(np.mean(aps))
    print(f"  mean AUPRC across horizons: {res['mean_auprc']:.5f}")

    if e_w is not None:
        yj = Y[idx_te, 1].astype(bool)
        ends = bw.starts[idx_te] + bw.window_len - 1
        high = bw.raw_gluc[ends] >= 120
        print(f"\n  erase on test windows: positives {e_w[yj].mean():.5f}  "
              f"negatives {e_w[~yj].mean():.5f}")
        if (yj & high).any():
            print(f"    within glucose >= 120: {e_w[yj & high].mean():.5f} "
                  f"vs {e_w[~yj & high].mean():.5f}")
        res["mechanism"] = {"E_pos": float(e_w[yj].mean()),
                            "E_neg": float(e_w[~yj].mean())}
        with torch.no_grad():
            res["alpha_E_final"] = float(model.cells[0].alpha_E)
            if hasattr(model.cells[0], "alpha_D"):
                res["alpha_D_final"] = float(model.cells[0].alpha_D)

    res["minutes"] = (time.time() - t0) / 60
    fn = f"{name}__s{args.seed}"
    np.savez_compressed(OUT_DIR / f"{fn}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state, "config": vars(args)},
               OUT_DIR / f"{fn}.pt")
    common.save_result(OUT_DIR, fn, res)
    ck.cleanup()
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No ADEW results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*92}\n# ADEW-GRU PILOT\n{'#'*92}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>14} {'H':>4} {'params':>8} {'AUPRC':>9} "
              f"{'PPV':>9} {'Recall':>9}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>14} {lst[0]['hidden']:>4} {lst[0]['n_params']:>8,} "
                  f"{m['auprc']:>9.4f} {m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*92}\n# DECISION — primary outcome is mean AUPRC "
          f"across horizons\n{'#'*92}\n")
    gl = runs.get("gru", [])
    if not gl:
        print("  run the gru arm first")
        return
    g_mean = gl[0]["mean_auprc"]
    print(f"  {'variant':>14} {'mean AUPRC':>11} {'delta':>9}   verdict")
    print(f"  {'gru':>14} {g_mean:>11.5f} {'':>9}   reference")
    for v in MODELS[1:]:
        lst = runs.get(v, [])
        if not lst:
            continue
        d = lst[0]["mean_auprc"] - g_mean
        verdict = ("run the equal-parameter control and ablations"
                   if d >= 0.005 else
                   "interesting, run seeds 43/44" if d >= 0.003 else "stop")
        print(f"  {v:>14} {lst[0]['mean_auprc']:>11.5f} {d:>+9.5f}   {verdict}")

    for h in HORIZONS:
        A = [r for r in gl if str(h) in r["horizons"]]
        if not A:
            continue
        for v in MODELS[1:]:
            B = [r for r in runs.get(v, [])
                 if str(h) in r["horizons"] and r["seed"] == A[0]["seed"]]
            if not B:
                continue
            dd = common.paired_delta(A[0]["horizons"][str(h)]["per_subject"],
                                     B[0]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"\n  h={h:>3} {v:>14}: {dd['delta']:+.4f} "
                  f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}  "
                  f"(subject-level bootstrap)")

    print(f"\n\n{'#'*92}\n# WERE THE ANCHORS ADOPTED?\n{'#'*92}")
    for v in MODELS[1:]:
        for r in runs.get(v, []):
            if "alpha_E_final" not in r:
                continue
            aE = r["alpha_E_final"]
            init = r.get("alpha_init", 0.01)
            line = f"\n{v} (seed {r['seed']}): alpha_E {init:.3f} -> {aE:+.5f}"
            if "alpha_D_final" in r:
                line += f"   alpha_D {init:.3f} -> {r['alpha_D_final']:+.5f}"
            print(line)
            mech = r.get("mechanism")
            if mech:
                print(f"  erase  positives {mech['E_pos']:.5f}  "
                      f"negatives {mech['E_neg']:.5f}")
            if abs(aE) < 2 * init:
                print("  -> the anchor barely moved: the optimiser declined the")
                print("     mechanism and the cell stayed at the GRU.")
            elif mech and abs(mech["E_pos"] - mech["E_neg"]) < 1e-3:
                print("  -> erasure is applied but identically before events and")
                print("     non-events: a fixed policy, not an adaptive one.")
            else:
                print("  -> the anchor grew and erasure differs before events.")

    common.save_result(OUT_DIR, "_adew_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items()
                        for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_adew_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--alpha_init", type=float, default=0.01,
                    help="0.0 gives exact GRU nesting but zero gradient to "
                         "the erase/delay controllers until the anchor moves")
    ap.add_argument("--delay_steps", type=int, default=12)
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
    if args.collect:
        collect(args)
        return
    if args.model is None:
        print(f"Choose --model {{{','.join(MODELS)}}} or --collect.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    print(f"  channels ({bw.timeline.shape[1]}): {list(bw.features)}")
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"  train {len(idx_tr):,} (shared) val {len(idx_va):,} "
          f"test {len(idx_te):,}")

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
