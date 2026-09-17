"""
dms_tcn.py
============
Dynamic Multi-Scale receptive-field TCN, and the three-arm pilot that
tests it.

    A  tcn_ta      the existing TCN block, now receiving the TA channels
    B  msd_tcn     four causal depthwise-separable branches spanning
                   different (kernel, dilation) pairs, fused by
                   concatenation + pointwise projection
    C  dms_tcn     the same four branches, combined by a router that
                   produces weights PER TIMESTEP           <- proposed

WHY ARM A MATTERS AS MUCH AS THE PROPOSAL
-----------------------------------------
The TCN reached AUPRC 0.759 on 5 channels and was never given the TA
features, while the GRU went 0.770 -> 0.862 when it received them. If the
TCN gains comparably, its earlier deficit was partly input representation
rather than architecture, and only then is "TCN vs MSD vs DMS" a fair
architectural question. Comparing a redesigned TCN against the old
5-channel number would confound the architecture with the very large TA
effect.

THE BRANCHES
------------
    B1 = DSConv(k=3, d=1)     per-layer receptive field   15 min
    B2 = DSConv(k=5, d=2)                                 45 min
    B3 = DSConv(k=7, d=4)                                125 min
    B4 = DSConv(k=9, d=8)                                325 min

That range is chosen to match the physiology: a rapid fall is local, an
insulin action curve is medium, a meal-plus-correction cycle is long.

All branches are causal (left padding, right trim) and depthwise
separable, so parameter cost is K*C + C*C rather than K*C*C.

THE ROUTER
----------
    F_t     = ReLU(W_f X_t)                 1x1 projection of block input
    alpha_t = softmax(W_a F_t)              (B, T, 4)
    Y_t     = sum_j alpha_{t,j} * B_j(X)_t
    Y'      = PWConv(Y)
    X_out   = LayerNorm(X_in + Y')

The router sees only the block's own learned representation. It is NOT
given s_t or any separate threshold-derived signal.

That choice is deliberate and follows from four earlier failures in this
project. TA-GRU, TRM-GRU and TAP-GRU all conditioned a mechanism on c_t
and v_t -- quantities already present as input channels -- and in every
case the network suppressed the mechanism rather than using it. Handing
the router a redundant signal invites the same outcome. Conditioning on
F_t alone means that if routing helps, the router learned to read the
trajectory from its own representation.

NOTE ON INITIALISATION
----------------------
W_a is initialised at zero, so alpha starts exactly uniform and the
router begins with no preference among receptive fields. This is NOT the
same as saying DMS begins as MSD: MSD concatenates its four branches and
then projects, while DMS forms a weighted sum before projecting, so the
two differ even at uniform weights.

DIAGNOSTICS
-----------
A tie is uninformative unless we know whether the router routed. Logged
every epoch: routing entropy (uniform = log 4 = 1.3863), per-branch mean
weight, the temporal standard deviation of each branch weight, and the
branch means on positive vs negative windows. Entropy pinned at 1.3863
with near-zero temporal SD means the router collapsed to averaging, and
the tie is then explained rather than mysterious.

PILOT RULE (agreed before running)
----------------------------------
Against arm A at h=15:
    < +0.005    stop that architecture
    +0.005..+0.010   mildly interesting
    >= +0.010   serious candidate
    >= +0.020   very promising
Ablations and further seeds only for a candidate that clears the bar.

Usage:
    python dms_tcn.py --model all --seed 42
    python dms_tcn.py --collect
"""

import gc
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
import common
from stage2_deep import WindowDataset, MultiHorizonLoss, Heads
from ta_gru import attach_ta

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "dms_tcn"
MODELS = ["tcn_ta", "msd_tcn", "dms_tcn"]
BRANCHES = [(3, 1), (5, 2), (7, 4), (9, 8)]     # (kernel, dilation)


# ─── BUILDING BLOCKS ──────────────────────────────────────────────────────────

class CausalDSConv(nn.Module):
    """
    Causal depthwise-separable dilated convolution.

    Depthwise: K*C parameters instead of K*C*C. Pointwise mixing is done
    once per block after fusion rather than inside every branch, so four
    branches cost about as much as one ordinary convolution.
    """

    def __init__(self, ch, k, d):
        super().__init__()
        self.pad = (k - 1) * d
        self.dw = nn.Conv1d(ch, ch, k, dilation=d, padding=self.pad, groups=ch)

    def forward(self, x):                       # (B, C, T)
        return self.dw(x)[:, :, :x.size(2)]     # trim right padding -> causal


class VanillaTCNBlock(nn.Module):
    """Arm A: two causal dilated convolutions with a residual connection."""

    def __init__(self, c_in, c_out, k=3, d=1, dropout=0.2):
        super().__init__()
        pad = (k - 1) * d
        self.c1 = nn.Conv1d(c_in, c_out, k, dilation=d, padding=pad)
        self.c2 = nn.Conv1d(c_out, c_out, k, dilation=d, padding=pad)
        self.n1, self.n2 = nn.BatchNorm1d(c_out), nn.BatchNorm1d(c_out)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

    def forward(self, x):
        T = x.size(2)
        r = self.res(x)
        o = self.drop(F.relu(self.n1(self.c1(x)[:, :, :T])))
        o = self.drop(F.relu(self.n2(self.c2(o)[:, :, :T])))
        return F.relu(o + r)


class MultiScaleBlock(nn.Module):
    """
    Arms B and C.

    dynamic=False : concatenate the four branches, then project  (MSD)
    dynamic=True  : weighted sum with per-timestep weights, then
                    project                                       (DMS)
    """

    def __init__(self, c_in, c_out, dynamic, dropout=0.2, route_dim=16):
        super().__init__()
        self.dynamic = dynamic
        self.proj_in = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out
                        else nn.Identity())
        self.branches = nn.ModuleList([CausalDSConv(c_out, k, d)
                                       for k, d in BRANCHES])
        n_b = len(BRANCHES)
        self.pw = nn.Conv1d(c_out * (1 if dynamic else n_b), c_out, 1)
        self.norm = nn.LayerNorm(c_out)
        self.drop = nn.Dropout(dropout)
        self.res = nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity()

        if dynamic:
            # router sees only the block's own representation
            self.route_proj = nn.Conv1d(c_out, route_dim, 1)
            self.route_out = nn.Conv1d(route_dim, n_b, 1)
            # zero init -> alpha starts exactly uniform, no branch preference
            nn.init.zeros_(self.route_out.weight)
            nn.init.zeros_(self.route_out.bias)

    def forward(self, x, return_alpha=False):
        r = self.res(x)
        h = self.proj_in(x)
        outs = [b(h) for b in self.branches]            # each (B, C, T)

        alpha = None
        if self.dynamic:
            Ft = F.relu(self.route_proj(h))
            alpha = torch.softmax(self.route_out(Ft), dim=1)   # (B, n_b, T)
            stacked = torch.stack(outs, dim=1)                 # (B, n_b, C, T)
            y = (alpha.unsqueeze(2) * stacked).sum(dim=1)      # (B, C, T)
        else:
            y = torch.cat(outs, dim=1)

        y = self.drop(self.pw(y))
        o = self.norm((y + r).transpose(1, 2)).transpose(1, 2)
        o = F.relu(o)
        return (o, alpha) if return_alpha else (o, None)


class TCNModel(nn.Module):
    def __init__(self, variant, in_ch=7, hidden=64, n_blocks=4, dropout=0.2):
        super().__init__()
        self.variant = variant
        self.dynamic = variant == "dms_tcn"
        if variant == "tcn_ta":
            dil = [1, 2, 4, 8]
            self.blocks = nn.ModuleList([
                VanillaTCNBlock(in_ch if i == 0 else hidden, hidden,
                                d=dil[i % len(dil)], dropout=dropout)
                for i in range(n_blocks)])
        else:
            self.blocks = nn.ModuleList([
                MultiScaleBlock(in_ch if i == 0 else hidden, hidden,
                                dynamic=self.dynamic, dropout=dropout)
                for i in range(n_blocks)])
        self.head = Heads(hidden)

    def forward(self, x, return_alpha=False):
        o = x.transpose(1, 2)                            # (B, C, T)
        alphas = []
        for b in self.blocks:
            if isinstance(b, VanillaTCNBlock):
                o = b(o)
            else:
                o, a = b(o, return_alpha=return_alpha)
                if a is not None:
                    alphas.append(a)
        z = o[:, :, -1]                                  # last timestep: causal
        out = self.head(z)
        return (out, alphas) if return_alpha else out


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

@torch.no_grad()
def route_stats(model, bw, idx, labels, device, n=20000, batch=1024):
    """
    Did the router route? Entropy at log(4) = 1.3863 with near-zero
    temporal SD means it collapsed to uniform averaging, and a tie in
    AUPRC is then explained rather than ambiguous.
    """
    if model.variant != "dms_tcn":
        return None
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx[:n], labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    ents, means, sds, ys = [], [], [], []
    for xb, yb in dl:
        _, alphas = model(xb.to(device), return_alpha=True)
        if not alphas:
            return None
        a = alphas[0]                                   # first block
        ent = -(a * torch.log(a.clamp_min(1e-12))).sum(1)   # (B, T)
        ents.append(ent.mean(1).cpu().numpy())
        means.append(a.mean(dim=2).cpu().numpy())       # (B, n_b)
        sds.append(a.std(dim=2).cpu().numpy())          # over time
        ys.append(yb[:, 1].numpy())
    ent = np.concatenate(ents)
    m = np.concatenate(means)
    sd = np.concatenate(sds)
    y = np.concatenate(ys).astype(int)
    out = {
        "entropy_mean": float(ent.mean()),
        "entropy_uniform": float(np.log(len(BRANCHES))),
        "branch_means": [float(v) for v in m.mean(0)],
        "branch_temporal_sd": [float(v) for v in sd.mean(0)],
    }
    if (y == 1).any() and (y == 0).any():
        out["branch_means_positive"] = [float(v) for v in m[y == 1].mean(0)]
        out["branch_means_negative"] = [float(v) for v in m[y == 0].mean(0)]
    return out


def print_route(s):
    if s is None:
        return
    bm = " ".join(f"{v:.3f}" for v in s["branch_means"])
    bsd = " ".join(f"{v:.4f}" for v in s["branch_temporal_sd"])
    print(f"      routing entropy {s['entropy_mean']:.4f} "
          f"(uniform {s['entropy_uniform']:.4f})")
    print(f"      branch means  [{bm}]   (15/45/125/325 min)")
    print(f"      temporal sd   [{bsd}]")
    if "branch_means_positive" in s:
        p = " ".join(f"{v:.3f}" for v in s["branch_means_positive"])
        n = " ".join(f"{v:.3f}" for v in s["branch_means_negative"])
        print(f"      pos [{p}]  neg [{n}]")


# ─── TRAIN ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, labels, device, batch=2048, workers=2):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    return np.concatenate([model(xb.to(device)).cpu().numpy()
                           for xb, _ in dl]).astype(np.float32)


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    print(f"\n{'='*78}\n{variant} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    model = TCNModel(variant, in_ch=bw.timeline.shape[1],
                     hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    name_ck = f"{variant}__s{args.seed}"
    ck = common.TrainCheckpoint(OUT_DIR, name_ck,
                                resume=not getattr(args, "no_resume", False))
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state, hist = None, []
    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = skipped = 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(); skipped += 1; continue
            opt.step()
            tot += loss.item(); nb += 1

        pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(f"{variant}: non-finite predictions at epoch {ep}")
        vauprc = float(np.mean([average_precision_score(Y[idx_va, j].astype(int),
                                                        pv[:, j])
                                for j in range(len(HORIZONS))]))
        sched.step(vauprc)
        improved = vauprc > best
        mark = ""
        if improved:
            best, best_ep, bad = vauprc, ep, 0
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            mark = " *"
        else:
            bad += 1
        ck.save(model, opt, sched, ep, best, best_ep, bad, improved)
        note = f"  [{skipped} skipped]" if skipped else ""
        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"val mean-AUPRC {vauprc:.4f}{mark}{note}")
        s = route_stats(model, bw, idx_va, Y, device)
        if s:
            s["epoch"] = ep
            hist.append(s)
            print_route(s)
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final = route_stats(model, bw, idx_va, Y, device)

    res = {"model": variant, "seed": args.seed, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "route_final": final, "route_history": hist, "horizons": {}}
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
        print("No DMS-TCN results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*96}")
    print("# TCN PILOT — A: TCN+TA   B: MSD-TCN   C: DMS-TCN (proposed)")
    print(f"{'#'*96}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>10} {'params':>9} {'n_seeds':>8} "
              f"{'AUPRC mean':>11} {'sd':>8} {'PPV':>8} {'Recall':>8}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            a = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"] for r in lst])
            p = np.array([r["horizons"][str(h)]["per_subject_mean"]["ppv"] for r in lst])
            rc = np.array([r["horizons"][str(h)]["per_subject_mean"]["recall"] for r in lst])
            sd = a.std(ddof=1) if len(a) > 1 else float("nan")
            print(f"  {v:>10} {lst[0]['n_params']:>9,} {len(a):>8} "
                  f"{a.mean():>11.4f} {sd:>8.4f} {p.mean():>8.4f} {rc.mean():>8.4f}")

    print(f"\n\n{'#'*96}")
    print("# DECISION")
    print(f"{'#'*96}")
    for h in HORIZONS:
        got = {v: {r["seed"]: r for r in runs.get(v, [])
                   if str(h) in r["horizons"]} for v in MODELS}
        if not got["tcn_ta"]:
            continue
        for cand in ["msd_tcn", "dms_tcn"]:
            shared = sorted(set(got["tcn_ta"]) & set(got[cand]))
            if not shared:
                continue
            a = np.mean([got["tcn_ta"][s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            c = np.mean([got[cand][s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            gap = c - a
            verdict = ("very promising" if gap >= 0.020 else
                       "serious candidate" if gap >= 0.010 else
                       "mildly interesting" if gap >= 0.005 else
                       "STOP")
            print(f"\nh={h}  {cand} - tcn_ta = {gap:+.4f}   "
                  f"(A {a:.4f}, {cand} {c:.4f})   -> {verdict}")
            if len(shared) == 1:
                s = shared[0]
                d = common.paired_delta(
                    got["tcn_ta"][s]["horizons"][str(h)]["per_subject"],
                    got[cand][s]["horizons"][str(h)]["per_subject"],
                    n_boot=args.n_boot)["auprc"]
                sig = "*" if (d["lo"] > 0 or d["hi"] < 0) else " "
                print(f"        paired bootstrap: {d['delta']:+.4f} "
                      f"[{d['lo']:+.4f}, {d['hi']:+.4f}]{sig}")
        # is DMS better than fixed multi-scale?
        shared = sorted(set(got["msd_tcn"]) & set(got["dms_tcn"]))
        if shared:
            b = np.mean([got["msd_tcn"][s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            c = np.mean([got["dms_tcn"][s]["horizons"][str(h)]["per_subject_mean"]["auprc"]
                         for s in shared])
            print(f"        dms - msd = {c-b:+.4f}   "
                  f"(does dynamic routing beat fixed multi-scale?)")

    print(f"\n\n{'#'*96}")
    print("# ROUTING BEHAVIOUR")
    print(f"{'#'*96}")
    for r in runs.get("dms_tcn", []):
        s = r.get("route_final")
        if not s:
            continue
        print(f"\nseed {r['seed']}:")
        print_route(s)
        gap = s["entropy_uniform"] - s["entropy_mean"]
        tsd = max(s["branch_temporal_sd"])
        if gap < 0.02 and tsd < 0.005:
            print("      -> the router collapsed to uniform averaging: it "
                  "never selected a receptive field, so a tie in AUPRC "
                  "reflects an unused mechanism.")
        elif tsd < 0.005:
            print("      -> the router prefers particular branches but does "
                  "not vary them over time: this is learned fixed weighting, "
                  "not dynamic selection.")
        else:
            print("      -> the router varies its receptive field over time: "
                  "the mechanism is active.")

    common.save_result(OUT_DIR, "_dms_tcn_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_dms_tcn_summary.json'}")


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
    ap.add_argument("--no_resume", action="store_true")
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
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"windows {len(bw):,} | channels {bw.timeline.shape[1]} | "
          f"train {len(idx_tr):,} (shared) val {len(idx_va):,} "
          f"test {len(idx_te):,}")

    for v in (MODELS if args.model == "all" else [args.model]):
        name = f"{v}__s{args.seed}"
        if (OUT_DIR / f"{name}.json").exists() and not args.force:
            print(f"\n[skip] {name} already done")
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
