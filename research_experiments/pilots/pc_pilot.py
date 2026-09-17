"""
pc_pilot.py
=============
Pilot experiment: does causal patient context carry information beyond the
seven existing channels?

    A  gru_ta        the 7 TA channels                     (current baseline)
    C  gru_ta_ctx    those channels plus e_p, the 16-dim
                     encoding of the child's previous 48 h

e_p is held constant across the 60 timesteps of a window -- it describes
the child's recent physiological regime, not the moment-to-moment
trajectory, which c_t and v_t already carry.

WHY ONLY TWO ARMS
-----------------
The full programme is A / TAP-GRU / C / PC-GRU / PC-TAP-GRU. Building all
five before knowing whether patient context helps at all would risk five
hours of GPU on a dead direction. If C - A is negligible, no amount of
architectural machinery placed downstream of that context will rescue it,
because the information is not there.

Decision rule agreed in advance:
    C - A  <  0.005   stop the personalisation direction
    C - A  ~  0.010   interesting, not decisive
    C - A  >= 0.020   proceed to the patient-conditioned recurrent cell

FAIRNESS
--------
Both arms are trained on the SAME windows. Requiring 48 h of prior history
drops the earliest windows of every subject, so arm A is re-trained on the
reduced set rather than compared against its existing full-set result --
otherwise the two arms would differ in training data as well as in inputs,
and the comparison would be worthless. The existing seed-42 gru_ta number
from tsl_gru.py is therefore NOT the baseline here; the A run in this
script is.

Usage:
    python pc_pilot.py --model all --seed 42
    python pc_pilot.py --collect
"""

import gc
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import config
import common
from stage2_deep import MultiHorizonLoss, Heads
from ta_gru import attach_ta
import patient_context as PC

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "pc_pilot"
MODELS = ["gru_ta", "gru_ta_ctx"]
CTX_DIM = 16


class CtxWindowDataset(Dataset):
    """Window slice plus that window's patient-context vector."""

    def __init__(self, bw, idx, labels, ctx=None):
        self.tl, self.starts, self.wl = bw.timeline, bw.starts, bw.window_len
        self.idx = np.asarray(idx)
        self.y = labels[self.idx].astype(np.float32)
        self.ctx = None if ctx is None else ctx[self.idx].astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.starts[self.idx[i]]
        x = torch.from_numpy(np.ascontiguousarray(self.tl[s:s + self.wl]))
        y = torch.from_numpy(self.y[i])
        if self.ctx is None:
            return x, y, torch.zeros(1)
        return x, y, torch.from_numpy(self.ctx[i])


class CtxGRU(nn.Module):
    """
    GRU over the TA channels, optionally with a patient-context vector
    appended to every timestep.

    The encoder is deliberately small (K -> 32 -> 16): large enough to
    represent several independent characteristics, small enough that it
    cannot quietly become a second prediction network operating on
    summary statistics.
    """

    def __init__(self, use_ctx, in_ch=7, ctx_in=21, hidden=64, layers=2,
                 dropout=0.2):
        super().__init__()
        self.use_ctx = use_ctx
        if use_ctx:
            self.enc = nn.Sequential(nn.Linear(ctx_in, 32), nn.ReLU(),
                                     nn.Linear(32, CTX_DIM), nn.Tanh())
        self.gru = nn.GRU(in_ch + (CTX_DIM if use_ctx else 0), hidden,
                          num_layers=layers, batch_first=True, dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x, ctx=None):
        if self.use_ctx:
            e = self.enc(ctx).unsqueeze(1).expand(-1, x.shape[1], -1)
            x = torch.cat([x, e], dim=-1)
        h, _ = self.gru(x)
        return self.head(h[:, -1, :])


@torch.no_grad()
def predict(model, bw, idx, labels, ctx, device, batch=4096, workers=2):
    model.eval()
    dl = DataLoader(CtxWindowDataset(bw, idx, labels, ctx),
                    batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for xb, _, cb in dl:
        out.append(model(xb.to(device), cb.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run_one(variant, bw, Y, ctx, idx_tr, idx_va, idx_te, device, args):
    use_ctx = variant == "gru_ta_ctx"
    print(f"\n{'='*78}\n{variant} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    model = CtxGRU(use_ctx, in_ch=bw.timeline.shape[1],
                   ctx_in=ctx.shape[1], hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}  context={use_ctx}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(CtxWindowDataset(bw, idx_tr, Y, ctx),
                    batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    best, best_state, bad, best_ep = -1.0, None, 0, -1
    for ep in range(args.epochs):
        model.train()
        tot = nb = 0
        for xb, yb, cb in dl:
            xb, yb, cb = (xb.to(device, non_blocking=True), yb.to(device),
                          cb.to(device))
            opt.zero_grad()
            loss = crit(model(xb, cb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item(); nb += 1

        pv = predict(model, bw, idx_va, Y, ctx, device, workers=args.workers)
        vauprc = float(np.mean([average_precision_score(Y[idx_va, j].astype(int),
                                                        pv[:, j])
                                for j in range(len(HORIZONS))]))
        sched.step(vauprc)
        mark = ""
        if vauprc > best:
            best, best_ep, bad = vauprc, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"val mean-AUPRC {vauprc:.4f}{mark}")
        if bad >= args.patience:
            print("    early stop"); break

    model.load_state_dict(best_state)
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, ctx, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, ctx, device, workers=args.workers)

    res = {"model": variant, "seed": args.seed, "n_params": int(n_par),
           "use_context": use_ctx, "val_mean_auprc": best,
           "best_epoch": int(best_ep), "train_n": int(len(idx_tr)),
           "horizons": {}}
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, tinfo = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["threshold_info"] = tinfo
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
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No pilot results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*96}")
    print("# PATIENT-CONTEXT PILOT — both arms trained on the same windows")
    print(f"{'#'*96}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>14} {'params':>9} {'n_seeds':>8} "
              f"{'AUPRC mean':>11} {'sd':>8} {'PPV':>8} {'Recall':>8}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            a = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"] for r in lst])
            p = np.array([r["horizons"][str(h)]["per_subject_mean"]["ppv"] for r in lst])
            rc = np.array([r["horizons"][str(h)]["per_subject_mean"]["recall"] for r in lst])
            sd = a.std(ddof=1) if len(a) > 1 else float("nan")
            print(f"  {v:>14} {lst[0]['n_params']:>9,} {len(a):>8} "
                  f"{a.mean():>11.4f} {sd:>8.4f} {p.mean():>8.4f} {rc.mean():>8.4f}")

    print(f"\n\n{'#'*96}")
    print("# DECISION")
    print(f"{'#'*96}")
    for h in HORIZONS:
        A = [r for r in runs.get("gru_ta", []) if str(h) in r["horizons"]]
        C = [r for r in runs.get("gru_ta_ctx", []) if str(h) in r["horizons"]]
        if not A or not C:
            continue
        # paired where seeds match: same subjects, so pairing removes
        # subject-sampling noise from the delta
        pa = {r["seed"]: r for r in A}
        pc = {r["seed"]: r for r in C}
        shared = sorted(set(pa) & set(pc))
        a = np.mean([pa[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        c = np.mean([pc[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        d = c - a
        verdict = ("PROCEED to the patient-conditioned cell" if d >= 0.020 else
                   "interesting, not decisive" if d >= 0.010 else
                   "marginal" if d >= 0.005 else
                   "STOP: patient context adds no usable information")
        print(f"\nh={h}:  A {a:.4f}   C {c:.4f}   C-A = {d:+.4f}   -> {verdict}")
        if len(shared) == 1:
            s = shared[0]
            dd = common.paired_delta(pa[s]["horizons"][str(h)]["per_subject"],
                                     pc[s]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"        paired bootstrap (seed {s}): "
                  f"{dd['delta']:+.4f} [{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    common.save_result(OUT_DIR, "_pc_pilot_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_pc_pilot_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
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
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    Y = bw._labels_any

    cpath = config.DATA_DERIVED / f"patient_context_{args.tag}.npz"
    if not cpath.exists():
        raise FileNotFoundError(
            f"No context at {cpath}. Run: python patient_context.py "
            f"--tag {args.tag}")
    z = np.load(cpath, allow_pickle=True)
    ctx, keep = z["context"], z["keep"]
    print(f"context: {ctx.shape[1]} features, "
          f"{keep.sum():,}/{len(keep):,} windows have 48 h of history")

    tr, va, te = common.get_split(bw.meta)
    # BOTH arms restricted to windows with a full history, so the only
    # difference between A and C is the input, never the training set
    tr, va, te = tr & keep, va & keep, te & keep

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"train {len(idx_tr):,} (shared) val {len(idx_va):,} "
          f"test {len(idx_te):,}  across "
          f"{len(np.unique(bw.meta[idx_te]))} test subjects")

    for v in (MODELS if args.model == "all" else [args.model]):
        name = f"{v}__s{args.seed}"
        if (OUT_DIR / f"{name}.json").exists() and not args.force:
            print(f"\n[skip] {name} already done")
            continue
        try:
            run_one(v, bw, Y, ctx, idx_tr, idx_va, idx_te, device, args)
        except Exception as e:
            print(f"  !! {v} FAILED: {type(e).__name__}: {e}")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    collect(args)


if __name__ == "__main__":
    main()
