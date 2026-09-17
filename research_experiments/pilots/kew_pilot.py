"""
kew_pilot.py
==============
KEW-GRU seed-42 pilot and ablation ladder.

    E0  gru           standard GRU                 H=64   41,188  (-0.9%)
    E1  gru_noreset   reset removed only           H=78   41,381  (-0.5%)
    E2  kw_2action    Keep/Write simplex           H=64   41,188  (-0.9%)
    E3  kew_softmax   Keep/Erase/Write, dense      H=56   41,808  (+0.6%)
    E4  kew_entmax    Keep/Erase/Write, sparse     H=56   41,808  (+0.6%)  <- proposed
    E5  kew_reset     E4 with reset restored       H=50   41,629  (+0.1%)

Hidden width is set PER ARM so all six land within ~1% of the 41,572
baseline. This matters more here than in earlier pilots: the action head is
three times the size of the candidate projection, so at equal width KEW is
+30% and "KEW beats GRU" would be a capacity result.

RUN ORDER
---------
E0 and E4 first — the endpoints. The ladder only earns its GPU time once
there is something to decompose. If E4 does not clear +0.005 at h=15, the
intermediate rungs would cost three hours to explain a null already visible.

    python kew_pilot.py --model gru --seed 42          # ~8 min
    python kew_pilot.py --model kew_entmax --seed 42   # ~45 min
    python kew_pilot.py --collect
    # only if it clears the bar:
    python kew_pilot.py --model kw_2action --seed 42   # E2 -> E3 tests Erase
    python kew_pilot.py --model kew_softmax --seed 42  # E3 -> E4 tests sparsity
    python kew_pilot.py --model gru_noreset --seed 42
    python kew_pilot.py --model kew_reset --seed 42    # was reset removal right?

DIAGNOSTICS
-----------
Thirteen architectures in this project either had their mechanism suppressed
by the optimiser or found it active-but-redundant, so a tie is only
informative with instrumentation:

    keep / erase / write means      is Erase used at all?
    frac_argmax_erase               how often does Erase dominate a coordinate?
    frac_erase_gt_half              how often is it substantial, not just largest?
    exact_zero_frac                 entmax sparsity actually realised
                                    (softmax gives exactly 0 by construction)
    erase by outcome                does Erase fire before events?

If erase_mean stays near its initialisation and frac_argmax_erase is near
zero, the third action was never adopted and E3/E4 collapse to E2.
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
from kew_gru import KEWGRU, VARIANTS, count_parameters, _entmax15_pkg

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "kew_gru"
MODELS = VARIANTS
HIDDEN = {"gru": 64, "gru_noreset": 78, "kw_2action": 64,
          "kew_softmax": 56, "kew_entmax": 56, "kew_reset": 50}


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, E, K, W = [], [], [], []
    for xb, _ in dl:
        if want_diag:
            o, d = model(xb.to(device), return_diagnostics=True)
            E.append(d["_erase_per_window"].cpu().numpy())
            K.append(d["_keep_per_window"].cpu().numpy())
            W.append(d["_write_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        P.append(torch.sigmoid(o).cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    if want_diag and E:
        return out, np.concatenate(E), np.concatenate(K), np.concatenate(W)
    return out


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    print(f"\n{'='*78}\n{variant} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    H = args.hidden if args.hidden else HIDDEN[variant]
    model = KEWGRU(bw.timeline.shape[1], H, variant,
                   n_horizons=len(HORIZONS),
                   keep_bias=args.keep_bias, erase_bias=args.erase_bias,
                   write_bias=args.write_bias).to(device)
    n_par = count_parameters(model)
    print(f"  hidden {H}  params {n_par:,}  train n={len(idx_tr):,}")
    if variant.startswith("kew"):
        print(f"  action biases K {args.keep_bias:+.2f} E {args.erase_bias:+.2f} "
              f"W {args.write_bias:+.2f}   entmax "
              f"{'package' if _entmax15_pkg else 'closed form'}")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)

    def crit(out, y):
        return crit_ce(torch.sigmoid(out), y)      # all arms emit logits

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

        with torch.no_grad():
            xb, _ = next(iter(DataLoader(
                WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                num_workers=0)))
            _, d = model(xb.to(device), return_diagnostics=True)
        print(f"      K {float(d['keep_mean']):.4f}  E "
              f"{float(d['erase_mean']):.4f}  W {float(d['write_mean']):.4f}   "
              f"argmax K/E/W {float(d['frac_argmax_keep']):.3f}/"
              f"{float(d['frac_argmax_erase']):.3f}/"
              f"{float(d['frac_argmax_write']):.3f}")
        print(f"      E>0.5 {float(d['frac_erase_gt_half']):.4f}   "
              f"exact zeros {float(d['exact_zero_frac']):.4f}   "
              f"simplex err {float(d['simplex_err']):.2e}")
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
    pt, e_w, k_w, w_w = predict(model, bw, idx_te, Y, device,
                                workers=args.workers, want_diag=True)

    res = {"model": variant, "seed": args.seed, "hidden": H,
           "n_params": int(n_par), "val_mean_auprc": best,
           "best_epoch": int(best_ep), "horizons": {}}
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

    yj = Y[idx_te, 1].astype(bool)
    ends = bw.starts[idx_te] + bw.window_len - 1
    high = bw.raw_gluc[ends] >= 120
    print(f"\n  memory actions on test windows:")
    print(f"    {'':>10} {'positives':>11} {'negatives':>11}")
    for nm, arr in [("keep", k_w), ("erase", e_w), ("write", w_w)]:
        print(f"    {nm:>10} {arr[yj].mean():>11.5f} {arr[~yj].mean():>11.5f}")
    if (yj & high).any():
        print(f"    within glucose >= 120: erase "
              f"{e_w[yj & high].mean():.5f} vs {e_w[~yj & high].mean():.5f}")
    res["mechanism"] = {"erase_pos": float(e_w[yj].mean()),
                        "erase_neg": float(e_w[~yj].mean()),
                        "keep_pos": float(k_w[yj].mean()),
                        "keep_neg": float(k_w[~yj].mean()),
                        "write_pos": float(w_w[yj].mean()),
                        "write_neg": float(w_w[~yj].mean())}

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
        print("No KEW-GRU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*92}")
    print("# KEW-GRU ABLATION LADDER")
    print("#   E0 gru  E1 gru_noreset  E2 kw_2action")
    print("#   E3 kew_softmax  E4 kew_entmax (proposed)  E5 kew_reset")
    print(f"{'#'*92}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>13} {'H':>4} {'params':>8} {'AUPRC':>9} "
              f"{'PPV':>9} {'Recall':>9}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>13} {lst[0]['hidden']:>4} {lst[0]['n_params']:>8,} "
                  f"{m['auprc']:>9.4f} {m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*92}\n# DECISION\n{'#'*92}")

    def auprc(v, h):
        lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
        return (lst[0]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                if lst else None)

    for h in HORIZONS:
        a = auprc("gru", h)
        if a is None:
            continue
        print(f"\nh={h}  (gru = {a:.4f})")
        for v in MODELS[1:]:
            b = auprc(v, h)
            if b is None:
                continue
            d = b - a
            verdict = ("run seeds 43/44" if d >= 0.010 else
                       "interesting" if d >= 0.005 else "below threshold")
            print(f"  {v:>13} {b:.4f}   vs gru {d:+.4f}   -> {verdict}")

        # the two rungs that carry the argument
        e2, e3, e4 = auprc("kw_2action", h), auprc("kew_softmax", h), \
            auprc("kew_entmax", h)
        if e2 is not None and e3 is not None:
            print(f"    E2 -> E3  (does explicit Erase matter?)   "
                  f"{e3-e2:+.4f}")
        if e3 is not None and e4 is not None:
            print(f"    E3 -> E4  (does sparse competition matter?) "
                  f"{e4-e3:+.4f}")
        e5 = auprc("kew_reset", h)
        if e4 is not None and e5 is not None:
            print(f"    E4 -> E5  (was reset removal justified?)  "
                  f"{e5-e4:+.4f}"
                  f"{'   -> restore reset in the final model' if e5-e4 > 0.005 else ''}")

        if a is not None and e4 is not None and len(runs.get("gru", [])) == 1:
            s = runs["gru"][0]["seed"]
            gr = [r for r in runs["kew_entmax"] if r["seed"] == s]
            if gr:
                dd = common.paired_delta(
                    runs["gru"][0]["horizons"][str(h)]["per_subject"],
                    gr[0]["horizons"][str(h)]["per_subject"],
                    n_boot=args.n_boot)["auprc"]
                sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
                print(f"    paired bootstrap gru vs kew_entmax: "
                      f"{dd['delta']:+.4f} [{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    print(f"\n\n{'#'*92}\n# MEMORY-ACTION BEHAVIOUR\n{'#'*92}")
    for v in MODELS:
        for r in runs.get(v, []):
            mech = r.get("mechanism")
            if not mech or v in ("gru", "gru_noreset"):
                continue
            print(f"\n{v} (seed {r['seed']}):")
            for nm in ["keep", "erase", "write"]:
                print(f"  {nm:>6} positives {mech[nm+'_pos']:.5f}  "
                      f"negatives {mech[nm+'_neg']:.5f}")
            if v == "kw_2action":
                continue
            if mech["erase_pos"] < 0.02:
                print("  -> Erase was never adopted: this collapses to the")
                print("     two-action controller.")
            elif abs(mech["erase_pos"] - mech["erase_neg"]) < 1e-3:
                print("  -> Erase is used but identically before events and")
                print("     non-events: a fixed policy, not an adaptive one.")
            else:
                print("  -> Erase fires differently before events.")

    common.save_result(OUT_DIR, "_kew_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items()
                        for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_kew_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=None,
                    help="override; default is the capacity-matched width")
    ap.add_argument("--keep_bias", type=float, default=0.5)
    ap.add_argument("--erase_bias", type=float, default=-0.5)
    ap.add_argument("--write_bias", type=float, default=0.0)
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
