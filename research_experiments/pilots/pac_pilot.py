"""
pac_pilot.py
==============
PAC-GRU seed-42 pilot and ablation.

    gru       baseline, 9 channels                    41,572 params
    pa_gru    M1' anchored scaling, fixed alpha       41,188  (-0.92%)
    paa_gru   M1' + M2 adaptive alpha                 41,391  (-0.44%)
    pac_gru   M1' + M2 + M4 coupled gates             40,934  (-1.53%)

Hidden size is set PER ARM so every arm lands within ~1.5% of the
baseline. M4 adds an H x H matrix per layer, so pac_gru needs hidden 58
where the others use 64. Without that, "pac beats paa" would partly be a
capacity result -- the mistake we avoided in AR-RHU and DRS-GRU.

RUN ORDER
---------
Run gru and pac_gru first. If the full model does not clear +0.005 at
h=15 there is no reason to spend GPU on the intermediate arms; the
ablation only earns its cost once there is something to decompose.

    python pac_pilot.py --model gru --seed 42
    python pac_pilot.py --model pac_gru --seed 42
    python pac_pilot.py --collect
    # only if it clears the bar:
    python pac_pilot.py --model pa_gru --seed 42
    python pac_pilot.py --model paa_gru --seed 42

DIAGNOSTICS
-----------
Eleven earlier mechanisms in this project were either suppressed by the
optimiser or active-but-redundant, so a tie is only informative with
these:

    alpha_mean / sd     alpha -> 1 means M1' switched itself off
    scale_at_oldest     how much suppression the mechanism actually applies
    scale_at_newest     must stay exactly 1.0
    Vz_norm / Vz_vs_Uz  starts at 0; growth means M4 is being used
    alpha by outcome    does the order differ before events?

A note on headroom: with alpha_min = 0.80 the scale spans [0.871, 1.0],
so M1' can suppress the oldest step by at most 13%. If alpha settles near
1 AND scale_at_oldest stays near 1, the mechanism was not used. If alpha
saturates at alpha_min, the range was the binding constraint and
--alpha_min 0.5 is worth one more run.
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
from absolute_state import attach_absolute
from pac_gru import PACGRU, PACConfig, ARMS, count_parameters

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "pac_gru"
MODELS = ["gru", "pa_gru", "paa_gru", "pac_gru"]
HIDDEN = {"gru": 64, "pa_gru": 64, "paa_gru": 64, "pac_gru": 58}


class GRUBaseline(nn.Module):
    def __init__(self, in_ch=9, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.variant = "gru"
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x, return_diagnostics=False):
        h, _ = self.gru(x)
        out = self.head(h[:, -1, :])
        return (out, None) if return_diagnostics else out


def build(variant, in_ch, args):
    if variant == "gru":
        return GRUBaseline(in_ch, HIDDEN["gru"])
    return PACGRU(PACConfig(in_ch, hidden_size=HIDDEN[variant],
                            n_horizons=len(HORIZONS),
                            alpha_min=args.alpha_min,
                            alpha_init=args.alpha_init,
                            fixed_alpha=args.fixed_alpha, **ARMS[variant]))


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, A, Z = [], [], []
    for xb, _ in dl:
        if want_diag and model.variant != "gru":
            o, d = model(xb.to(device), return_diagnostics=True)
            A.append(d["_alpha_per_window"].cpu().numpy())
            Z.append(d["_z_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        P.append((torch.sigmoid(o) if model.variant != "gru"
                  else o).cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    if want_diag and A:
        return out, np.concatenate(A), np.concatenate(Z)
    return out


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    print(f"\n{'='*78}\n{variant} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build(variant, bw.timeline.shape[1], args).to(device)
    n_par = count_parameters(model)
    print(f"  hidden {HIDDEN[variant]}  params {n_par:,}  "
          f"train n={len(idx_tr):,}")
    if variant != "gru":
        print(f"  alpha_min {args.alpha_min}  "
              f"scale range [{2**(args.alpha_min-1):.3f}, 1.000]")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)

    def crit(out, y):
        return crit_ce(torch.sigmoid(out) if variant != "gru" else out, y)

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

        pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(f"{variant}: non-finite predictions, epoch {ep}")
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

        if variant != "gru":
            with torch.no_grad():
                xb, _ = next(iter(DataLoader(
                    WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                    num_workers=0)))
                _, d = model(xb.to(device), return_diagnostics=True)
            line = (f"      alpha {float(d['alpha_mean']):.4f} "
                    f"(sd {float(d['alpha_sd']):.4f}, "
                    f"[{float(d['alpha_min_seen']):.3f}, "
                    f"{float(d['alpha_max_seen']):.3f}])   "
                    f"scale oldest {float(d['scale_at_oldest']):.4f} "
                    f"newest {float(d['scale_at_newest']):.4f}")
            if "Vz_norm" in d:
                line += (f"\n      Vz {float(d['Vz_norm']):.4f}  "
                         f"Vz/Uz {float(d['Vz_vs_Uz']):.4f}  "
                         f"z {float(d['z_mean']):.3f}")
            print(line)
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    got = predict(model, bw, idx_te, Y, device, workers=args.workers,
                  want_diag=(variant != "gru"))
    if variant != "gru":
        pt, alpha_w, z_w = got
    else:
        pt, alpha_w, z_w = got, None, None

    res = {"model": variant, "seed": args.seed, "n_params": int(n_par),
           "hidden": HIDDEN[variant], "alpha_min": args.alpha_min,
           "val_mean_auprc": best, "best_epoch": int(best_ep), "horizons": {}}
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["test_prevalence"] = float(Y[idx_te, j].mean())
        res["horizons"][str(h)] = ev
        m, c = ev["per_subject_mean"], ev["constraint"]
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")

    if alpha_w is not None:
        yj = Y[idx_te, 1].astype(bool)
        ends = bw.starts[idx_te] + bw.window_len - 1
        high = bw.raw_gluc[ends] >= 120
        print(f"\n  mechanism on test windows:")
        print(f"    {'':>16} {'positives':>11} {'negatives':>11}")
        print(f"    {'alpha':>16} {alpha_w[yj].mean():>11.5f} "
              f"{alpha_w[~yj].mean():>11.5f}")
        print(f"    {'update gate z':>16} {z_w[yj].mean():>11.5f} "
              f"{z_w[~yj].mean():>11.5f}")
        if (yj & high).any():
            print(f"    within glucose >= 120: alpha "
                  f"{alpha_w[yj & high].mean():.5f} vs "
                  f"{alpha_w[~yj & high].mean():.5f}")
        res["mechanism"] = {"alpha_pos": float(alpha_w[yj].mean()),
                            "alpha_neg": float(alpha_w[~yj].mean()),
                            "z_pos": float(z_w[yj].mean()),
                            "z_neg": float(z_w[~yj].mean())}

    res["minutes"] = (time.time() - t0) / 60
    name = f"{variant}__s{args.seed}"
    np.savez_compressed(OUT_DIR / f"{name}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state, "config": vars(args)},
               OUT_DIR / f"{name}.pt")
    common.save_result(OUT_DIR, name, res)
    ck.cleanup()
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No PAC-GRU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*92}")
    print("# PAC-GRU ABLATION")
    print("#   pa  = M1' anchored scaling      paa = + M2 adaptive alpha")
    print("#   pac = + M4 coupled gates  (proposed)")
    print(f"{'#'*92}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>10} {'params':>9} {'AUPRC':>9} {'PPV':>9} "
              f"{'Recall':>9}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>10} {lst[0]['n_params']:>9,} {m['auprc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*92}\n# DECISION\n{'#'*92}")
    for h in HORIZONS:
        A = {r["seed"]: r for r in runs.get("gru", [])
             if str(h) in r["horizons"]}
        if not A:
            continue
        prev_key, prev_val = "gru", None
        for v in ["pa_gru", "paa_gru", "pac_gru"]:
            B = {r["seed"]: r for r in runs.get(v, [])
                 if str(h) in r["horizons"]}
            shared = sorted(set(A) & set(B))
            if not shared:
                continue
            a = np.mean([A[s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            b = np.mean([B[s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            d = b - a
            verdict = ("run seeds 43/44" if d >= 0.010 else
                       "interesting" if d >= 0.005 else "below threshold")
            print(f"\nh={h}:  gru {a:.4f}   {v} {b:.4f}   delta {d:+.4f}"
                  f"   -> {verdict}")
            if prev_val is not None:
                print(f"        incremental over {prev_key}: {b-prev_val:+.4f}")
            prev_key, prev_val = v, b
            if len(shared) == 1:
                s = shared[0]
                dd = common.paired_delta(
                    A[s]["horizons"][str(h)]["per_subject"],
                    B[s]["horizons"][str(h)]["per_subject"],
                    n_boot=args.n_boot)["auprc"]
                sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
                print(f"        paired bootstrap: {dd['delta']:+.4f} "
                      f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    for v in ["pa_gru", "paa_gru", "pac_gru"]:
        for r in runs.get(v, []):
            mech = r.get("mechanism")
            if not mech:
                continue
            print(f"\n{'#'*92}\n# MECHANISM — {v} (seed {r['seed']})\n{'#'*92}")
            print(f"  alpha  positives {mech['alpha_pos']:.5f}  "
                  f"negatives {mech['alpha_neg']:.5f}")
            print(f"  z      positives {mech['z_pos']:.5f}  "
                  f"negatives {mech['z_neg']:.5f}")
            if mech["alpha_pos"] > 0.99:
                print("  -> alpha went to 1: M1' switched itself off, so the")
                print("     anchored scaling contributed nothing.")
            elif abs(mech["alpha_pos"] - mech["alpha_neg"]) < 1e-3:
                print("  -> alpha is active but identical before events and")
                print("     non-events: a fixed recency profile, not an")
                print("     adaptive one.")
            else:
                print("  -> alpha differs before events: the adaptive order")
                print("     is responding to the trajectory.")

    common.save_result(OUT_DIR, "_pac_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items()
                        for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_pac_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--alpha_min", type=float, default=0.80)
    ap.add_argument("--alpha_init", type=float, default=0.95)
    ap.add_argument("--fixed_alpha", type=float, default=0.90)
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
        print("Choose --model {gru,pa_gru,paa_gru,pac_gru,all} or --collect.")
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
