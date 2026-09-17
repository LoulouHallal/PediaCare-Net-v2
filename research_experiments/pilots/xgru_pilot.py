"""
xgru_pilot.py
===============
xGRU transfer pilot — validation-first, deterministic, multi-seed.

    gru       in-loop GRU reference           H=64  rank -   41,188
    xgru      + low-rank matrix memory        H=51  rank 8   40,928  (-0.63%)
    xgru_r16  wider memory, same budget       H=47  rank 16  41,012  (-0.43%)

This is a TRANSFER experiment. xGRU is not the proposed contribution; the
question is only whether exponential similarity gating with matrix memory
helps on this data. Deriving a new cell from it is worth doing only if it
does.

THREE PROTOCOL CHANGES FROM THE PREVIOUS SIXTEEN PILOTS
-------------------------------------------------------
1. VALIDATION-FIRST. The decision block reports VALIDATION mean AUPRC and
   nothing else. Test metrics are computed and saved but printed only
   under --unlock_test. Sixteen architectures were chosen after looking at
   test results, which makes those 38 subjects part of the development
   loop; from here they are development evidence, not a clean confirmatory
   estimate. That limitation should be disclosed in the write-up rather
   than papered over.

2. DETERMINISM. seed_everything and make_loader are imported from
   interaction_screen. Run --model gru twice with --force and confirm the
   predictions match to ~1e-6 BEFORE trusting any comparison: two
   nominally identical hand-written-loop runs previously differed by
   0.00168 mean AUPRC. Comparing AUPRC alone is not enough — different
   predictions can give nearly identical AUPRC — so the check compares
   saved probabilities directly.

3. MULTI-SEED, PAIRED. --seeds 42,43,44 runs matched pairs. A candidate
   winning on one seed is not evidence; per-seed deltas are reported
   individually so a lucky trajectory is visible as such.

PROMOTION RULE, fixed before running
------------------------------------
Primary quantity is mean VALIDATION AUPRC across the four horizons.

    delta < +0.003        stop, the mechanism does not transfer
    +0.003 to +0.005      borderline, add seeds before deciding
    delta >= +0.005       promote: derive a cell from this mechanism

+0.003 is a project decision threshold chosen because the baseline itself
has moved by 0.00305 across nominally comparable runs — not a statistical
law.

Usage:
    python xgru_pilot.py --model gru --seed 42
    python xgru_pilot.py --model gru --seed 42 --force --suffix rep
    python xgru_pilot.py --check_determinism
    python xgru_pilot.py --model xgru --seeds 42,43,44
    python xgru_pilot.py --collect
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
from xgru import XGRU, XGRUConfig, count_parameters
from interaction_screen import seed_everything, make_loader

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "xgru"
MODELS = ["gru", "xgru", "xgru_r16"]
SPEC = {"gru": (64, 8, False), "xgru": (51, 8, True), "xgru_r16": (47, 16, True)}


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, S, B = [], [], []
    for xb, _ in dl:
        if want_diag and model.variant == "xgru":
            o, d = model(xb.to(device), return_diagnostics=True)
            S.append(d["_s_per_window"].cpu().numpy())
            B.append(d["_beta_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        P.append(torch.sigmoid(o).cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    if want_diag and S:
        return out, np.concatenate(S), np.concatenate(B)
    return out, None, None


def val_mean_auprc(Y, idx, pv):
    from sklearn.metrics import average_precision_score
    return float(np.mean([average_precision_score(Y[idx, k].astype(int),
                                                  pv[:, k])
                          for k in range(len(HORIZONS))]))


def run_one(name, bw, Y, idx_tr, idx_va, idx_te, device, args, seed):
    H, R, mem = SPEC[name]
    tag = f"{name}__s{seed}" + (f"__{args.suffix}" if args.suffix else "")
    print(f"\n{'='*78}\n{tag}\n{'='*78}")
    t0 = time.time()
    seed_everything(seed)

    cfg = XGRUConfig(bw.timeline.shape[1], hidden_size=H, rank=R,
                     n_horizons=len(HORIZONS), use_memory=mem,
                     alpha_init=args.alpha_init)
    model = XGRU(cfg).to(device)
    n_par = count_parameters(model)
    print(f"  H {H}  rank {R if mem else '-'}  params {n_par:,}  "
          f"train n={len(idx_tr):,}")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)

    def crit(o, y):
        return crit_ce(torch.sigmoid(o), y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = make_loader(WindowDataset(bw, idx_tr, Y), args.batch, shuffle=True,
                     seed=seed, num_workers=args.workers, drop_last=True)

    ck = common.TrainCheckpoint(OUT_DIR, tag, resume=not args.no_resume)
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

        pv, _, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(f"{tag}: non-finite predictions, epoch {ep}")
        v = val_mean_auprc(Y, idx_va, pv)
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
              f"VAL mean-AUPRC {v:.4f}{' *' if improved else ''}")

        if mem:
            with torch.no_grad():
                xb, _ = next(iter(DataLoader(
                    WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                    num_workers=0)))
                _, d = model(xb.to(device), return_diagnostics=True)
            print(f"      alpha {float(d['alpha']):.5f}  beta "
                  f"{float(d['beta_mean']):.5f} (max "
                  f"{float(d['beta_max']):.4f})   s {float(d['s_mean']):.4f} "
                  f"(min {float(d['s_min']):.4f}, sd {float(d['s_sd']):.4f})")
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best VAL mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv, _, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt, s_w, b_w = predict(model, bw, idx_te, Y, device, workers=args.workers,
                           want_diag=mem)

    res = {"model": name, "seed": seed, "hidden": H, "rank": R if mem else None,
           "n_params": int(n_par), "val_mean_auprc": best,
           "best_epoch": int(best_ep), "val_horizons": {}, "horizons": {}}
    from sklearn.metrics import average_precision_score
    for j, h in enumerate(HORIZONS):
        res["val_horizons"][str(h)] = float(average_precision_score(
            Y[idx_va, j].astype(int), pv[:, j]))
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        res["horizons"][str(h)] = ev
    res["mean_auprc"] = float(np.mean(
        [res["horizons"][str(h)]["per_subject_mean"]["auprc"]
         for h in HORIZONS]))

    print(f"  validation by horizon: " +
          "  ".join(f"h{h} {res['val_horizons'][str(h)]:.4f}"
                    for h in HORIZONS))
    if args.unlock_test:
        for h in HORIZONS:
            m = res["horizons"][str(h)]["per_subject_mean"]
            print(f"  [test] h={h:>3}  AUPRC {m['auprc']:.4f}  "
                  f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}")
    else:
        print("  (test metrics computed and saved but withheld; "
              "--unlock_test to print)")

    if s_w is not None:
        yj = Y[idx_te, 1].astype(bool)
        res["mechanism"] = {"s_pos": float(s_w[yj].mean()),
                            "s_neg": float(s_w[~yj].mean()),
                            "beta_pos": float(b_w[yj].mean()),
                            "beta_neg": float(b_w[~yj].mean())}
        with torch.no_grad():
            res["alpha_final"] = float(model.cells[0].alpha())

    res["minutes"] = (time.time() - t0) / 60
    np.savez_compressed(OUT_DIR / f"{tag}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state}, OUT_DIR / f"{tag}.pt")
    common.save_result(OUT_DIR, tag, res)
    ck.cleanup()
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def check_determinism(args):
    """Compare saved probabilities from two runs of the same arm."""
    import itertools
    files = sorted(OUT_DIR.glob("gru__s*_probs.npz"))
    if len(files) < 2:
        print("Need two gru runs. Do:\n"
              "  python xgru_pilot.py --model gru --seed 42\n"
              "  python xgru_pilot.py --model gru --seed 42 --force "
              "--suffix rep --no_resume")
        return
    print(f"{'#'*88}\n# DETERMINISM CHECK\n{'#'*88}\n")
    for a, b in itertools.combinations(files, 2):
        za, zb = np.load(a), np.load(b)
        if za["test"].shape != zb["test"].shape:
            continue
        dmax = float(np.abs(za["test"] - zb["test"]).max())
        print(f"  {a.name}\n  {b.name}")
        print(f"    max |p_A - p_B| over test predictions: {dmax:.3e}")
        if dmax < 1e-6:
            print("    -> reproducible. Architecture comparisons are "
                  "trustworthy.\n")
        else:
            print("    -> NOT reproducible. Find the first divergence before")
            print("       interpreting any architecture delta; the spread "
                  "here\n       has previously exceeded every measured "
                  "architecture effect.\n")


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No xGRU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*88}\n# xGRU TRANSFER PILOT — VALIDATION\n{'#'*88}\n")
    print(f"  {'variant':>10} {'seed':>5} {'H':>4} {'rank':>5} {'params':>8} "
          f"{'val mean':>10}")
    for v in MODELS:
        for r in sorted(runs.get(v, []), key=lambda x: x["seed"]):
            print(f"  {v:>10} {r['seed']:>5} {r['hidden']:>4} "
                  f"{str(r['rank'] or '-'):>5} {r['n_params']:>8,} "
                  f"{r['val_mean_auprc']:>10.5f}")

    g = {r["seed"]: r["val_mean_auprc"] for r in runs.get("gru", [])}
    if not g:
        print("\n  run the gru arm first")
        return
    print(f"\n\n{'#'*88}\n# DECISION — mean VALIDATION AUPRC, paired by seed"
          f"\n{'#'*88}\n")
    for v in MODELS[1:]:
        rs = runs.get(v, [])
        pairs = [(r["seed"], r["val_mean_auprc"] - g[r["seed"]])
                 for r in rs if r["seed"] in g]
        if not pairs:
            continue
        ds = np.array([d for _, d in pairs])
        print(f"  {v}")
        for s, d in pairs:
            print(f"    seed {s}: {d:+.5f}")
        m = ds.mean()
        verdict = ("PROMOTE: derive a cell from this mechanism" if m >= 0.005
                   else "borderline, add seeds" if m >= 0.003
                   else "stop, the mechanism does not transfer")
        print(f"    mean over {len(ds)} seed(s): {m:+.5f}"
              + (f"  sd {ds.std():.5f}" if len(ds) > 1 else "")
              + f"   -> {verdict}")
        if len(ds) > 1 and (ds > 0).sum() not in (0, len(ds)):
            print(f"    NOTE: sign is inconsistent across seeds "
                  f"({(ds > 0).sum()}/{len(ds)} positive) — not evidence")

    for v in MODELS[1:]:
        for r in runs.get(v, []):
            mech = r.get("mechanism")
            if not mech:
                continue
            print(f"\n  {v} seed {r['seed']}: alpha -> "
                  f"{r.get('alpha_final', float('nan')):.5f}")
            print(f"    similarity s  positives {mech['s_pos']:.5f}  "
                  f"negatives {mech['s_neg']:.5f}")
            print(f"    read gate beta positives {mech['beta_pos']:.5f}  "
                  f"negatives {mech['beta_neg']:.5f}")
            if r.get("alpha_final", 0) < 2 * 0.01:
                print("    -> alpha barely moved: the memory was not adopted")
                print("       and the cell stayed at the GRU.")

    common.save_result(OUT_DIR, "_xgru_summary",
                       {f"{v}__s{r['seed']}": r
                        for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_xgru_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check_determinism", action="store_true")
    ap.add_argument("--unlock_test", action="store_true",
                    help="print test metrics; do NOT use during development")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--seeds", default=None, help="e.g. 42,43,44")
    ap.add_argument("--suffix", default="")
    ap.add_argument("--alpha_init", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.check_determinism:
        check_determinism(args); return
    if args.collect:
        collect(args); return
    if args.model is None:
        print(f"Choose --model {{{','.join(MODELS)}}}, --collect, "
              f"or --check_determinism"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    seed_everything(args.seed)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
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

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])
    for v in (MODELS if args.model == "all" else [args.model]):
        for sd in seeds:
            tag = f"{v}__s{sd}" + (f"__{args.suffix}" if args.suffix else "")
            if (OUT_DIR / f"{tag}.json").exists() and not args.force:
                print(f"\n[skip] {tag} already done")
                continue
            try:
                run_one(v, bw, Y, idx_tr, idx_va, idx_te, device, args, sd)
            except Exception as e:
                print(f"  !! {tag} FAILED: {type(e).__name__}: {e}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    collect(args)


if __name__ == "__main__":
    main()
