"""
stage2_deep.py
================
RQ2 stage 2 — deep sequence models, and the first place where
LOSS-LEVEL and SEQUENCE-LEVEL imbalance handling can be tested.

WHY THIS REOPENS RQ1
--------------------
Stage 1 could only test imbalance handling at the DATA level: resample a
feature matrix, or use an ensemble that resamples internally. None of it
improved AUPRC. But two whole families were untestable with classical ML:

    loss level      weighted BCE, focal loss -- change how much each
                    example contributes to the gradient, without
                    fabricating or discarding any data
    sequence level  oversampling REAL windows, and (later) generative
                    augmentation, which operate on trajectories rather
                    than flattened feature vectors

Focal loss is the interesting one. Oversampling changes P(Y) in the
training set; focal loss keeps the natural distribution and instead
down-weights easy examples. With millions of obviously-normoglycaemic
windows, that is a different mechanism, not a variant of the same one.

One caution that the literature gets wrong often: the focal term
(1-p)^gamma emphasises HARD examples, not minority examples. Minority
emphasis comes from alpha. Hard negatives here are exactly the near-miss
trajectories that fall toward 70 and recover -- the ones that generate
false alarms -- so a large gamma can help or hurt, and it must be tuned
rather than assumed.

WHAT IS HELD FIXED
------------------
Same windows, same subject split (170/36/38, seed 42), same label set,
same threshold policy on natural-prevalence validation, same metrics and
subject-level bootstrap as stage 1. Only the model and the imbalance
mechanism change, so stage 1 and stage 2 numbers are directly comparable.

MEMORY
------
1.9M training windows x 60 x 5 channels is ~2.3 GB materialised. The
Dataset slices `timeline[start:start+60]` on demand instead, so peak
memory stays at one batch.

HONEST EXPECTATION
------------------
Ten classical algorithms converged on AUPRC 0.69-0.72 at h=15 -- a
0.03 spread across completely different model families. That pattern
indicates a task ceiling rather than a model limitation, and in the
earlier project phase a plain GRU matched a dual-pathway
TCN-Transformer. Expect a modest gain here, not a transformation. The
experiment is still worth running: "we tested it properly and complexity
did not help" is a result, and it is the question RQ2 asks.

Usage:
    python stage2_deep.py --model gru --loss bce
    python stage2_deep.py --model lstm --loss focal --gamma 2.0 --alpha 0.75
    python stage2_deep.py --model gru --loss bce --balance random_over
    python stage2_deep.py --collect
"""

import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RQ2_STAGES["stage2_dl"]
MODELS = ["lstm", "gru", "mlp", "cnn"]
LOSSES = ["bce", "weighted_bce", "focal"]
SEQ_BALANCE = ["none", "random_over", "random_under"]


# ─── DATA ─────────────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    """
    Lazy view over BuiltWindows. Slices the shared timeline per item, so
    the full window tensor is never materialised.
    """

    def __init__(self, bw, idx, labels):
        self.tl = bw.timeline
        self.starts = bw.starts
        self.wl = bw.window_len
        self.idx = np.asarray(idx)
        self.y = labels[self.idx].astype(np.float32)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.starts[self.idx[i]]
        return (torch.from_numpy(np.ascontiguousarray(self.tl[s:s + self.wl])),
                torch.from_numpy(self.y[i]))


def resample_indices(idx, y, method, ratio=0.3, seed=42):
    """
    Sequence-level balancing: select REAL windows by index.

    Unlike SMOTE on flattened windows, this never interpolates between
    two children's glucose trajectories, so every training example
    remains a real physiological sequence and the multi-horizon labels
    stay intact.
    """
    if method == "none":
        return idx
    from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    R = (RandomOverSampler if method == "random_over" else RandomUnderSampler)
    cur = y.sum() / max((y == 0).sum(), 1)
    if ratio <= cur * 1.02:
        return idx                       # already at or above target
    out, _ = R(sampling_strategy=ratio, random_state=seed).fit_resample(
        idx.reshape(-1, 1), y)
    return out.ravel()


# ─── LOSSES ───────────────────────────────────────────────────────────────────

class MultiHorizonLoss(nn.Module):
    """
    bce            plain binary cross-entropy, natural distribution
    weighted_bce   positives up-weighted by pos_weight (inverse frequency)
    focal          alpha * (1-p)^gamma on positives, (1-alpha) * p^gamma on
                   negatives -- down-weights EASY examples of both classes

    pos_weight is computed per horizon from the training labels, because
    prevalence differs 3-fold between h=15 and h=120.
    """

    def __init__(self, kind="bce", pos_weight=None, alpha=0.75, gamma=2.0,
                 eps=1e-7):
        super().__init__()
        self.kind, self.alpha, self.gamma, self.eps = kind, alpha, gamma, eps
        self.register_buffer("pos_weight",
                             torch.ones(len(HORIZONS)) if pos_weight is None
                             else torch.as_tensor(pos_weight,
                                                  dtype=torch.float32))

    def forward(self, p, y):
        p = torch.clamp(p, self.eps, 1 - self.eps)
        if self.kind == "bce":
            loss = -(y * torch.log(p) + (1 - y) * torch.log(1 - p))
        elif self.kind == "weighted_bce":
            w = self.pos_weight.to(p.device).unsqueeze(0)
            loss = -(w * y * torch.log(p) + (1 - y) * torch.log(1 - p))
        elif self.kind == "focal":
            pos = -self.alpha * y * (1 - p) ** self.gamma * torch.log(p)
            neg = -(1 - self.alpha) * (1 - y) * p ** self.gamma * torch.log(1 - p)
            loss = pos + neg
        else:
            raise ValueError(f"unknown loss {self.kind}")
        return loss.mean()


# ─── MODELS ───────────────────────────────────────────────────────────────────

class Heads(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, d // 2), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(d // 2, len(HORIZONS)))

    def forward(self, z):
        return torch.sigmoid(self.net(z))


class RNNModel(nn.Module):
    def __init__(self, cell, in_ch, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.rnn = rnn(in_ch, hidden, num_layers=layers, batch_first=True,
                       dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])      # last timestep -> causal


class MLPModel(nn.Module):
    def __init__(self, in_ch, window=60, hidden=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(in_ch * window, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Dropout(dropout))
        self.head = Heads(hidden // 2)

    def forward(self, x):
        return self.head(self.net(x))


class CNNModel(nn.Module):
    def __init__(self, in_ch, hidden=64, dropout=0.2):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 3, padding=2),
                                 nn.BatchNorm1d(o), nn.ReLU(), nn.Dropout(dropout))
        self.b = nn.ModuleList([blk(in_ch, hidden), blk(hidden, hidden),
                                blk(hidden, hidden)])
        self.head = Heads(hidden)

    def forward(self, x):
        T = x.size(1)
        o = x.permute(0, 2, 1)
        for b in self.b:
            o = b(o)[:, :, :T]              # trim right padding -> causal
        return self.head(o.max(dim=2).values)


def build(name, in_ch, hidden):
    if name in ("lstm", "gru"):
        return RNNModel(name, in_ch, hidden)
    if name == "mlp":
        return MLPModel(in_ch, hidden=hidden * 2)
    if name == "cnn":
        return CNNModel(in_ch, hidden)
    raise ValueError(f"unknown model {name}")


# ─── TRAIN / PREDICT ──────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, device, batch=4096, workers=2):
    model.eval()
    ds = WindowDataset(bw, idx, np.zeros((len(bw), len(HORIZONS)), np.float32))
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers)
    out = []
    for xb, _ in dl:
        out.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def run_name(model, loss, args):
    """Canonical run name. Seed 42 keeps the legacy name so results computed
    before seeds were tracked are still found; other seeds get a suffix."""
    base = f"{model}__{loss}__{args.balance}__{args.labels}"
    return base if args.seed == config.SPLIT_SEED else f"{base}__s{args.seed}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gru", choices=MODELS + ["all"])
    ap.add_argument("--loss", default="bce", choices=LOSSES + ["all"])
    ap.add_argument("--balance", default="none", choices=SEQ_BALANCE)
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
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
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any
    print(f"windows {len(bw):,} | channels {bw.n_channels} | "
          f"train {tr.sum():,} val {va.sum():,} test {te.sum():,}")

    models = MODELS if args.model == "all" else [args.model]
    losses = LOSSES if args.loss == "all" else [args.loss]

    for mo in models:
        for lo in losses:
            # The seed MUST be in the filename. Without it a second seed
            # overwrites the first and multi-seed runs are impossible --
            # which is how run-to-run variance went unmeasured until two
            # accidental repeat runs revealed it exceeds the differences
            # between architectures.
            name = run_name(mo, lo, args)
            path = OUT_DIR / f"{name}.json"
            if path.exists() and not args.force:
                print(f"\n[skip] {name} already done")
                continue
            try:
                run_one(mo, lo, bw, Y, (tr, va, te), device, args, name)
            except Exception as e:
                print(f"  !! {name} FAILED: {type(e).__name__}: {e}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    collect(args)


def run_one(model_name, loss_kind, bw, Y, masks, device, args, name):
    tr, va, te = masks
    print(f"\n{'='*78}\n{model_name} | loss={loss_kind} | balance={args.balance} "
          f"| labels={args.labels}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    idx_tr = np.flatnonzero(tr)
    # cap BEFORE balancing so the cap cannot undo the resampling
    if args.train_cap and len(idx_tr) > args.train_cap:
        anchor = Y[idx_tr, 1].astype(int)        # stratify on h=30
        from stage1_classical_ml import stratified_subsample
        idx_tr = idx_tr[stratified_subsample(anchor, args.train_cap, args.seed)]
    idx_tr = resample_indices(idx_tr, Y[idx_tr, 1].astype(int), args.balance,
                              seed=args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    print(f"  train n={len(idx_tr):,} | prevalence "
          f"{np.round(prev, 4).tolist()}")
    if loss_kind == "weighted_bce":
        print(f"  pos_weight per horizon: {np.round(pos_weight, 2).tolist()}")

    model = build(model_name, bw.n_channels, args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  params: {n_par:,}")

    crit = MultiHorizonLoss(loss_kind, pos_weight, args.alpha, args.gamma).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)

    ds = WindowDataset(bw, idx_tr, Y)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    Y_va, Y_te = Y[idx_va], Y[idx_te]
    meta_te = bw.meta[idx_te]

    best, best_state, bad, best_ep = -1.0, None, 0, -1
    for ep in range(args.epochs):
        model.train()
        tot = nb = 0
        skipped = 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                skipped += 1
                continue
            loss.backward()
            # clip_grad_norm_ does NOT protect against non-finite gradients:
            # if the total norm is inf or NaN the scaling factor is NaN and
            # every parameter is poisoned in one step. Check before stepping.
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad()
                skipped += 1
                continue
            opt.step()
            tot += loss.item(); nb += 1

        pv = predict(model, bw, idx_va, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(
                f"{model_name}/{loss_kind}: model produced non-finite "
                f"predictions at epoch {ep} ({skipped} batches were already "
                f"skipped this epoch). This is divergence, not a result. Try "
                f"a lower --lr (e.g. 3e-4), and check "
                f"check_feature_range.py for extreme input values.")
        from sklearn.metrics import average_precision_score
        vauprc = float(np.mean([average_precision_score(Y_va[:, j].astype(int),
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
        note = f"  [{skipped} batches skipped: non-finite]" if skipped else ""
        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"val mean-AUPRC {vauprc:.4f}{mark}{note}")
        if bad >= args.patience:
            print(f"    early stop (no improvement for {args.patience} epochs)")
            break

    model.load_state_dict(best_state)
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, device, workers=args.workers)
    pt = predict(model, bw, idx_te, device, workers=args.workers)

    res = {"model": model_name, "loss": loss_kind, "balance": args.balance,
           "label_set": args.labels, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "alpha": args.alpha, "gamma": args.gamma, "seed": args.seed,
           "train_n": int(len(idx_tr)), "horizons": {}}

    for j, h in enumerate(HORIZONS):
        thr, tinfo = common.find_threshold(Y_va[:, j].astype(int), pv[:, j])
        ev = common.evaluate(Y_te[:, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["threshold_info"] = tinfo
        ev["test_prevalence"] = float(Y_te[:, j].mean())
        fixed = common.evaluate(Y_te[:, j].astype(int), pt[:, j], meta_te, 0.5)
        ev["fixed_threshold_0.5"] = {
            "pooled": fixed["pooled"],
            "per_subject_mean": fixed["per_subject_mean"],
            "per_subject": fixed["per_subject"],
            "constraint": fixed["constraint"]}
        res["horizons"][str(h)] = ev

        m, c = ev["per_subject_mean"], ev["constraint"]
        f5 = fixed["per_subject_mean"]
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")
        print(f"        at fixed thr=0.5:  PPV {f5['ppv']:.4f}  "
              f"Rec {f5['recall']:.4f}  F1 {f5['f1']:.4f}")

    res["minutes"] = (time.time() - t0) / 60
    np.savez_compressed(OUT_DIR / f"{name}_probs.npz",
                        val=pv, test=pt, idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state, "config": vars(args),
                "val_mean_auprc": best}, OUT_DIR / f"{name}.pt")
    common.save_result(OUT_DIR, name, res)
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


# ─── COLLECT ──────────────────────────────────────────────────────────────────

def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No stage-2 results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs[(r["model"], r["loss"], r["balance"], r.get("seed",
                                                         config.SPLIT_SEED))] = r

    print(f"\n{'#'*104}")
    print("# RQ2 STAGE 2 — DEEP SEQUENCE MODELS (per-subject mean)")
    print(f"{'#'*104}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'model':>12} {'loss':>14} {'balance':>13} {'seed':>5} "
              f"{'params':>9} {'AUROC':>8} {'AUPRC':>8} {'PPV':>8} "
              f"{'Recall':>8} {'F1':>8}")
        rows = [(r["horizons"][str(h)]["per_subject_mean"]["auprc"], k, r)
                for k, r in runs.items() if str(h) in r["horizons"]]
        for _, (mo, lo, ba, sd), r in sorted(rows, reverse=True):
            m = r["horizons"][str(h)]["per_subject_mean"]
            print(f"  {mo:>12} {lo:>14} {ba:>13} {sd:>5} {r['n_params']:>9,} "
                  f"{m['auroc']:>8.4f} {m['auprc']:>8.4f} {m['ppv']:>8.4f} "
                  f"{m['recall']:>8.4f} {m['f1']:>8.4f}")

    # ---- loss-level imbalance handling, paired against plain BCE ----------
    print(f"\n\n{'#'*104}")
    print("# EFFECT OF LOSS-LEVEL IMBALANCE HANDLING vs plain BCE "
          "(paired subject bootstrap)")
    print(f"{'#'*104}")
    # Loss comparisons are made WITHIN a seed: comparing a weighted-BCE run
    # at one seed against a BCE run at another would confound the loss with
    # run-to-run variance.
    for (mo, sd) in sorted({(m, s_) for m, _, _, s_ in runs}):
        base = runs.get((mo, "bce", "none", sd))
        if base is None:
            continue
        for (m2, lo, ba, s2_), r in sorted(runs.items()):
            if m2 != mo or s2_ != sd or (lo == "bce" and ba == "none"):
                continue
            print(f"\n{mo} (seed {sd}): bce/none -> {lo}/{ba}")
            for h in HORIZONS:
                if str(h) not in base["horizons"] or str(h) not in r["horizons"]:
                    continue
                d = common.paired_delta(base["horizons"][str(h)]["per_subject"],
                                        r["horizons"][str(h)]["per_subject"],
                                        n_boot=args.n_boot)
                parts = [f"{k}={d[k]['delta']:+.4f}"
                         f"[{d[k]['lo']:+.4f},{d[k]['hi']:+.4f}]"
                         f"{'*' if (d[k]['lo'] > 0 or d[k]['hi'] < 0) else ' '}"
                         for k in ["auprc", "f1", "ppv", "recall"]]
                print(f"  h={h:>3}: " + "  ".join(parts))
    print("\n  * = 95% CI excludes zero")
    print("\nNOTE: compare AUPRC against the stage-1 classical models. If deep")
    print("      sequence models do not exceed AUPRC ~0.72 at h=15, that is")
    print("      evidence of a task ceiling rather than a model limitation.")

    # ---- seed variance ----------------------------------------------------
    # Architecture differences here are small. Reporting a ranking from single
    # runs is only meaningful if the between-model gaps exceed the
    # between-seed spread, so both are printed side by side.
    import collections
    by_cfg = collections.defaultdict(list)
    for (mo, lo, ba, sd), r in runs.items():
        by_cfg[(mo, lo, ba)].append((sd, r))

    multi = {k: v for k, v in by_cfg.items() if len(v) > 1}
    print(f"\n\n{'#'*104}")
    print("# SEED VARIANCE (configurations run with more than one seed)")
    print(f"{'#'*104}")
    if not multi:
        print("\n  None yet. Run the same configuration with --seed 43 and 44:")
        print("  a ranking is not interpretable until the between-seed spread")
        print("  is known to be smaller than the between-model differences.")
    else:
        for h in HORIZONS:
            print(f"\nh={h} min")
            print(f"  {'model':>12} {'loss':>14} {'n_seeds':>8} "
                  f"{'AUPRC mean':>11} {'sd':>8} {'min':>8} {'max':>8} {'range':>8}")
            for (mo, lo, ba), lst in sorted(multi.items()):
                v = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"]
                              for _, r in lst if str(h) in r["horizons"]])
                if len(v) < 2:
                    continue
                print(f"  {mo:>12} {lo:>14} {len(v):>8} {v.mean():>11.4f} "
                      f"{v.std(ddof=1):>8.4f} {v.min():>8.4f} {v.max():>8.4f} "
                      f"{v.max()-v.min():>8.4f}")
        print("\n  Compare the `range` column against the gaps between models")
        print("  in the tables above. If they are similar, the ranking is not")
        print("  supported by the data.")

    common.save_result(OUT_DIR, "_stage2_summary",
                       {f"{m}__{l}__{b}__s{sd}": r
                        for (m, l, b, sd), r in runs.items()})
    print(f"\n✓ Saved -> {OUT_DIR / '_stage2_summary.json'}")


if __name__ == "__main__":
    main()
