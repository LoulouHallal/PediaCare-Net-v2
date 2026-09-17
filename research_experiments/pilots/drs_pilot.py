"""
drs_pilot.py
==============
DRS-GRU seed-42 pilot.

    A  gru_abs   GRU, 9 channels                41,572 params
    B  drs_gru   DRS-GRU, same 9 channels       41,878 params  (+0.74%)

Both arms receive the identical 9-channel representation, including the
TA and absolute-state channels. This is deliberate. Eleven earlier
attempts tried to replace or rediscover TA; all of them fell short of
simply supplying it. The question here is narrower and fairer: given the
same inputs as the GRU, does a signed recurrent pathway exploit them
better?

CAPACITY
--------
DRS-GRU carries one extra input matrix per layer (W_lambda) relative to a
GRU, so matching parameters means a slightly smaller hidden size:
hidden 61 gives 41,878 against the baseline's 41,572, within 0.74%.
hidden 60 would be -2.35%, so 61 is the closer match.

STOPPING RULE, fixed before running
-----------------------------------
    delta < +0.005    DRS-GRU stops permanently
    delta >= +0.005   interesting
    delta >= +0.010   run seeds 43 and 44

Usage:
    python drs_pilot.py --model all --seed 42
    python drs_pilot.py --model drs_gru --seed 42 --input_free_z   # ablation
    python drs_pilot.py --collect
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
from drs_gru import DRSGRU, count_trainable_parameters

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "drs_gru"
MODELS = ["gru_abs", "drs_gru"]


class GRUBaseline(nn.Module):
    def __init__(self, in_ch=9, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.variant = "gru_abs"
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x, return_diagnostics=False):
        h, _ = self.gru(x)
        out = self.head(h[:, -1, :])
        return (out, None) if return_diagnostics else out


def build(variant, in_ch, args):
    if variant == "gru_abs":
        return GRUBaseline(in_ch, args.hidden)
    return DRSGRU(in_ch, hidden_size=args.drs_hidden, layers=2,
                  n_horizons=len(HORIZONS), input_free_z=args.input_free_z)


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, RN, LM, CS = [], [], [], []
    for xb, _ in dl:
        if want_diag and model.variant == "drs_gru":
            o, d = model(xb.to(device), return_diagnostics=True)
            RN.append(d["_frac_r_neg_per_window"].cpu().numpy())
            LM.append(d["_lambda_per_window"].cpu().numpy())
            CS.append(d["_cand_sep_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        # DRS emits logits; the GRU head already applies sigmoid
        P.append((torch.sigmoid(o) if model.variant == "drs_gru"
                  else o).cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    if want_diag and RN:
        return out, np.concatenate(RN), np.concatenate(LM), np.concatenate(CS)
    return out


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    tag = variant + ("__ifz" if (variant == "drs_gru" and args.input_free_z)
                     else "")
    print(f"\n{'='*78}\n{tag} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = build(variant, bw.timeline.shape[1], args).to(device)
    n_par = count_trainable_parameters(model)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)

    def crit(out, y):
        return crit_ce(torch.sigmoid(out) if variant == "drs_gru" else out, y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, f"{tag}__s{args.seed}",
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
            raise RuntimeError(f"{tag}: non-finite predictions at epoch {ep}")
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

        if variant == "drs_gru":
            with torch.no_grad():
                xb, _ = next(iter(DataLoader(
                    WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                    num_workers=0)))
                _, d = model(xb.to(device), return_diagnostics=True)
            print(f"      P(r<0) {float(d['frac_r_negative']):.4f}   "
                  f"E|r| {float(d['abs_r_mean']):.4f}   "
                  f"lambda {float(d['lambda_mean']):.4f}   "
                  f"sep {float(d['candidate_separation']):.4f}")
            print(f"      lambda | r<0  {float(d['lambda_where_r_neg']):.4f}   "
                  f"| r>0  {float(d['lambda_where_r_pos']):.4f}   "
                  f"z {float(d['z_mean']):.3f} (sd {float(d['z_sd']):.3f})")
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
                  want_diag=(variant == "drs_gru"))
    if variant == "drs_gru":
        pt, rneg_w, lam_w, sep_w = got
    else:
        pt, rneg_w, lam_w, sep_w = got, None, None, None

    res = {"model": variant, "tag": tag, "seed": args.seed,
           "n_params": int(n_par), "input_free_z": bool(args.input_free_z),
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

    if rneg_w is not None:
        yj = Y[idx_te, 1].astype(bool)
        ends = bw.starts[idx_te] + bw.window_len - 1
        high = bw.raw_gluc[ends] >= 120
        print(f"\n  mechanism on test windows:")
        print(f"    {'':>20} {'positives':>11} {'negatives':>11}")
        for nm, arr in [("P(r<0)", rneg_w), ("lambda", lam_w),
                        ("candidate sep", sep_w)]:
            print(f"    {nm:>20} {arr[yj].mean():>11.5f} {arr[~yj].mean():>11.5f}")
        if (yj & high).any() and (~yj & high).any():
            print(f"    within glucose >= 120 mg/dL:")
            for nm, arr in [("P(r<0)", rneg_w), ("lambda", lam_w)]:
                print(f"    {nm:>20} {arr[yj & high].mean():>11.5f} "
                      f"{arr[~yj & high].mean():>11.5f}")
        res["mechanism"] = {
            "r_neg_pos": float(rneg_w[yj].mean()),
            "r_neg_neg": float(rneg_w[~yj].mean()),
            "lambda_pos": float(lam_w[yj].mean()),
            "lambda_neg": float(lam_w[~yj].mean()),
            "sep_pos": float(sep_w[yj].mean()),
            "sep_neg": float(sep_w[~yj].mean())}

    res["minutes"] = (time.time() - t0) / 60
    name = f"{tag}__s{args.seed}"
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
        print("No DRS-GRU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r.get("tag", r["model"]), []).append(r)

    print(f"\n{'#'*92}")
    print("# DRS-GRU PILOT — A: GRU+TA+Absolute   B: DRS-GRU (same 9 channels)")
    print(f"{'#'*92}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>12} {'params':>9} {'AUPRC':>9} {'PPV':>9} "
              f"{'Recall':>9}")
        for v in sorted(runs):
            lst = [r for r in runs[v] if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>12} {lst[0]['n_params']:>9,} {m['auprc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*92}\n# DECISION\n{'#'*92}")
    for h in HORIZONS:
        A = {r["seed"]: r for r in runs.get("gru_abs", [])
             if str(h) in r["horizons"]}
        for bkey in [k for k in sorted(runs) if k.startswith("drs_gru")]:
            B = {r["seed"]: r for r in runs[bkey] if str(h) in r["horizons"]}
            shared = sorted(set(A) & set(B))
            if not shared:
                continue
            a = np.mean([A[s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            b = np.mean([B[s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            d = b - a
            verdict = ("run seeds 43/44" if d >= 0.010 else
                       "interesting" if d >= 0.005 else "STOP permanently")
            print(f"\nh={h}:  gru_abs {a:.4f}   {bkey} {b:.4f}   "
                  f"delta {d:+.4f}   -> {verdict}")
            if len(shared) == 1:
                s = shared[0]
                dd = common.paired_delta(
                    A[s]["horizons"][str(h)]["per_subject"],
                    B[s]["horizons"][str(h)]["per_subject"],
                    n_boot=args.n_boot)["auprc"]
                sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
                print(f"        paired bootstrap: {dd['delta']:+.4f} "
                      f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    for k in [k for k in sorted(runs) if k.startswith("drs_gru")]:
        for r in runs[k]:
            mech = r.get("mechanism")
            if not mech:
                continue
            print(f"\n{'#'*92}\n# MECHANISM — {k} (seed {r['seed']})\n{'#'*92}")
            print(f"  P(r<0)   positives {mech['r_neg_pos']:.5f}  "
                  f"negatives {mech['r_neg_neg']:.5f}")
            print(f"  lambda   positives {mech['lambda_pos']:.5f}  "
                  f"negatives {mech['lambda_neg']:.5f}")
            print(f"  cand sep positives {mech['sep_pos']:.5f}  "
                  f"negatives {mech['sep_neg']:.5f}")
            if mech["r_neg_pos"] < 0.02:
                print("  -> the signed pathway collapsed: r stayed positive,")
                print("     so the cell reduced to a GRU-like reset.")
            elif mech["sep_pos"] < 0.01:
                print("  -> the two candidates coincide: |r| went to zero and")
                print("     history was simply dropped.")
            elif mech["lambda_pos"] < 0.05:
                print("  -> lambda suppressed signed history entirely.")
            else:
                print("  -> the signed pathway is active and admitted.")

    common.save_result(OUT_DIR, "_drs_summary",
                       {f"{k}__s{r['seed']}": r for k, l in runs.items()
                        for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_drs_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--drs_hidden", type=int, default=61)
    ap.add_argument("--input_free_z", action="store_true",
                    help="ablation: update gate sees only h_{t-1}")
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
        print("Choose --model all or --collect.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: DRS-GRU is a hand-written recurrent loop; "
              "CPU will be slow.")

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
        tag = v + ("__ifz" if (v == "drs_gru" and args.input_free_z) else "")
        if (OUT_DIR / f"{tag}__s{args.seed}.json").exists() and not args.force:
            print(f"\n[skip] {tag}__s{args.seed} already done")
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
