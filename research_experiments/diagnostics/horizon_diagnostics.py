"""
horizon_diagnostics.py
========================
Two cheap diagnostics that test whether the multi-horizon output design is
costing anything, BEFORE any new architecture is built.

THE STRUCTURAL FACT BEING TESTED
--------------------------------
The four labels are nested by construction. If a reading below 70 mg/dL
occurs within 15 minutes, one has necessarily occurred within 30, 60 and
120 minutes. So the true probabilities must satisfy

    P15 <= P30 <= P60 <= P120

but the model produces four independent sigmoid outputs and nothing
enforces that ordering. It is free to predict P15 = 0.8 and P30 = 0.3,
which is not merely inaccurate -- it is impossible.

DIAGNOSTIC 1 -- monotonicity violations (free, no training)
-----------------------------------------------------------
Counts how often the saved predictions violate the ordering, and by how
much. Reported at three thresholds of severity, because a violation of
0.001 is numerical noise while one of 0.2 means the heads disagree about
what they are predicting.

Also reports the LABEL violation rate, which must be exactly zero. If it
is not, the labels themselves are inconsistent and everything downstream
is suspect -- so this doubles as a data check.

DIAGNOSTIC 2 -- single- vs multi-horizon training (one run)
-----------------------------------------------------------
Trains an identical GRU+TA that predicts ONLY h=15, changing nothing else:
same channels, same training indices, same loss, same seed, same early
stopping. If a dedicated model beats the h=15 head of the shared model,
the four tasks are interfering through the shared representation.

    delta = AUPRC15(single) - AUPRC15(multi)

    delta <  0.005   interference is not the bottleneck; stop here
    delta >= 0.005   a real architectural weakness has been identified

WHY THIS ORDER
--------------
Diagnostic 1 costs nothing and runs on predictions that already exist.
Diagnostic 2 costs one training run. Both are far cheaper than designing
an architecture around a weakness that may not exist -- which is the trap
the previous five architecture attempts fell into.

Usage:
    python horizon_diagnostics.py --monotonicity          # free
    python horizon_diagnostics.py --single_horizon --seed 42
    python horizon_diagnostics.py --report
"""

import gc
import json
import time
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
import common
from stage2_deep import WindowDataset, MultiHorizonLoss, Heads
from ta_gru import attach_ta

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "horizon_diag"

# every folder that stores saved probabilities in the shared format
PROB_DIRS = [
    config.RESULTS / "RQ2_models" / "tsl_gru",
    config.RESULTS / "RQ2_models" / "tap_gru",
    config.RESULTS / "RQ2_models" / "dms_tcn",
    config.RESULTS / "RQ2_models" / "trm_gru",
    config.RQ2_STAGES["stage2_dl"],
]


# ─── DIAGNOSTIC 1 ─────────────────────────────────────────────────────────────

def monotonicity(probs, name="", labels=None, verbose=True):
    """
    probs : (N, 4) predicted probabilities in horizon order.

    Adjacent-pair violations, plus the largest single drop per window.
    Thresholds separate numerical noise from genuine disagreement.
    """
    p = np.asarray(probs, dtype=np.float64)
    pairs = [("P15>P30", 0, 1), ("P30>P60", 1, 2), ("P60>P120", 2, 3)]
    out = {"model": name, "n": int(len(p)), "pairs": {}}

    worst = np.zeros(len(p))
    for lab, i, j in pairs:
        d = p[:, i] - p[:, j]                       # positive => violation
        worst = np.maximum(worst, d)
        out["pairs"][lab] = {
            "frac_any": float((d > 0).mean()),
            "frac_gt_001": float((d > 0.01).mean()),
            "frac_gt_01": float((d > 0.10).mean()),
            "mean_violation": float(d[d > 0].mean()) if (d > 0).any() else 0.0,
            "max_violation": float(d.max()),
        }
    out["any_violation_frac"] = float((worst > 0).mean())
    out["any_violation_gt_01"] = float((worst > 0.10).mean())
    out["mean_worst_violation"] = float(worst[worst > 0].mean()) if (worst > 0).any() else 0.0

    if labels is not None:
        y = np.asarray(labels)
        lv = 0.0
        for _, i, j in pairs:
            lv = max(lv, float((y[:, i] > y[:, j]).mean()))
        out["label_violation_frac"] = lv

    if verbose:
        print(f"\n{name}  (n = {out['n']:,})")
        print(f"  {'pair':>10} {'any':>9} {'>0.01':>9} {'>0.10':>9} "
              f"{'mean':>9} {'max':>9}")
        for lab in out["pairs"]:
            v = out["pairs"][lab]
            print(f"  {lab:>10} {100*v['frac_any']:>8.2f}% "
                  f"{100*v['frac_gt_001']:>8.2f}% {100*v['frac_gt_01']:>8.2f}% "
                  f"{v['mean_violation']:>9.4f} {v['max_violation']:>9.4f}")
        print(f"  any pair violated: {100*out['any_violation_frac']:.2f}%   "
              f"by more than 0.10: {100*out['any_violation_gt_01']:.2f}%")
        if labels is not None:
            print(f"  LABEL violation rate: {100*out['label_violation_frac']:.4f}% "
                  f"(must be 0.0000%)")
    return out


def run_monotonicity(args):
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    Y = bw._labels_any

    print(f"{'#'*88}")
    print("# DIAGNOSTIC 1 — horizon monotonicity of saved predictions")
    print("#   Labels are nested: hypo within 15 min implies hypo within")
    print("#   30, 60 and 120. So P15 <= P30 <= P60 <= P120 must hold.")
    print("#   Four independent sigmoid heads do not enforce this.")
    print(f"{'#'*88}")

    results = []
    for d in PROB_DIRS:
        if not d.exists():
            continue
        for f in sorted(d.glob("*_probs.npz")):
            z = np.load(f)
            if "test" not in z or "idx_test" not in z:
                continue
            name = f"{d.name}/{f.name.replace('_probs.npz','')}"
            lab = Y[z["idx_test"]] if len(z["idx_test"]) == len(z["test"]) else None
            results.append(monotonicity(z["test"], name, lab))

    if not results:
        print("\nNo saved probability files found.")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, "_monotonicity", {"models": results})

    frac = np.array([r["any_violation_frac"] for r in results])
    big = np.array([r["any_violation_gt_01"] for r in results])
    print(f"\n{'#'*88}")
    print("# VERDICT")
    print(f"{'#'*88}")
    print(f"\n  across {len(results)} models: any violation "
          f"{100*frac.mean():.2f}% of windows (range {100*frac.min():.2f}"
          f"–{100*frac.max():.2f}%)")
    print(f"  violations larger than 0.10: {100*big.mean():.2f}% "
          f"(range {100*big.min():.2f}–{100*big.max():.2f}%)")
    if big.mean() > 0.02:
        print("\n  -> Substantial violations. The independent heads disagree")
        print("     about a nested event, so enforcing the ordering")
        print("     architecturally may recover real information.")
    elif frac.mean() > 0.10:
        print("\n  -> Frequent but small violations. The model has largely")
        print("     learned the ordering on its own; enforcing it would")
        print("     mostly tidy the output rather than add information.")
    else:
        print("\n  -> Violations are rare and small. The nested structure is")
        print("     already respected, so a monotonic output layer is")
        print("     unlikely to improve prediction.")
    print(f"\n✓ Saved -> {OUT_DIR / '_monotonicity.json'}")


# ─── DIAGNOSTIC 2 ─────────────────────────────────────────────────────────────

class SingleHorizonGRU(nn.Module):
    """Identical to the GRU+TA baseline except that it has one output."""

    def __init__(self, in_ch=7, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.net = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.ReLU(),
                                 nn.Dropout(0.1), nn.Linear(hidden // 2, 1))

    def forward(self, x):
        h, _ = self.gru(x)
        return torch.sigmoid(self.net(h[:, -1, :])).squeeze(-1)


def run_single_horizon(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    j = HORIZONS.index(args.horizon)

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"train {len(idx_tr):,} (identical indices to the multi-horizon run) "
          f"val {len(idx_va):,} test {len(idx_te):,}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = SingleHorizonGRU(in_ch=bw.timeline.shape[1],
                             hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)

    prev = float(Y[idx_tr, j].mean())
    pw = float(np.clip((1 - prev) / max(prev, 1e-6), 1.0, 50.0))
    print(f"\nh={args.horizon} only | params {n_par:,} | prevalence {prev:.4f} "
          f"| pos_weight {pw:.2f}")

    # same weighted-BCE form as the multi-horizon loss, one head
    def crit(p, y):
        p = torch.clamp(p, 1e-7, 1 - 1e-7)
        return -(pw * y * torch.log(p) + (1 - y) * torch.log(1 - p)).mean()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    @torch.no_grad()
    def predict(idx):
        model.eval()
        d = DataLoader(WindowDataset(bw, idx, Y), batch_size=4096,
                       shuffle=False, num_workers=args.workers)
        return np.concatenate([model(xb.to(device)).cpu().numpy()
                               for xb, _ in d]).astype(np.float32)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, f"single_h{args.horizon}__s{args.seed}",
                                resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state = None
    t0 = time.time()
    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb[:, j].to(device)
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

        pv = predict(idx_va)
        v = float(average_precision_score(Y[idx_va, j].astype(int), pv))
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
              f"val AUPRC {v:.4f}{' *' if improved else ''}")
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
    pv, pt = predict(idx_va), predict(idx_te)

    thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv)
    ev = common.evaluate(Y[idx_te, j].astype(int), pt, bw.meta[idx_te], thr)
    m = ev["per_subject_mean"]
    print(f"\n  h={args.horizon} single-task: AUROC {m['auroc']:.4f}  "
          f"AUPRC {m['auprc']:.4f}  PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}")

    res = {"horizon": args.horizon, "seed": args.seed, "n_params": int(n_par),
           "val_auprc": best, "best_epoch": int(best_ep),
           "per_subject_mean": m, "per_subject": ev["per_subject"],
           "constraint": ev["constraint"], "threshold": float(thr),
           "minutes": (time.time() - t0) / 60}
    common.save_result(OUT_DIR, f"single_h{args.horizon}__s{args.seed}", res)
    ck.cleanup()
    gc.collect()
    return res


# ─── REPORT ───────────────────────────────────────────────────────────────────

def run_report(args):
    """Compare the single-horizon run against the multi-horizon h=15 head."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sp = OUT_DIR / f"single_h{args.horizon}__s{args.seed}.json"
    if not sp.exists():
        print(f"No single-horizon result at {sp.name}. Run --single_horizon first.")
        return
    single = json.load(open(sp))

    # the multi-horizon counterpart: gru_ta from the tap_gru or dms_tcn pilot
    multi = None
    for d, nm in [(config.RESULTS / "RQ2_models" / "tap_gru", "gru_ta__s42.json"),
                  (config.RESULTS / "RQ2_models" / "dms_tcn", "tcn_ta__s42.json")]:
        p = d / nm
        if p.exists():
            r = json.load(open(p))
            if str(args.horizon) in r["horizons"]:
                multi = r
                multi_name = f"{d.name}/{nm}"
                break
    if multi is None:
        print("No multi-horizon gru_ta result found to compare against.")
        return

    ev = multi["horizons"][str(args.horizon)]
    a = ev["per_subject_mean"]["auprc"]
    b = single["per_subject_mean"]["auprc"]
    d = b - a

    print(f"\n{'#'*88}")
    print(f"# DIAGNOSTIC 2 — single vs multi-horizon at h={args.horizon}")
    print(f"{'#'*88}")
    print(f"\n  multi-horizon ({multi_name}): AUPRC {a:.4f}")
    print(f"  single-horizon (this run)    : AUPRC {b:.4f}")
    print(f"  delta = {d:+.4f}")

    pd_ = common.paired_delta(ev["per_subject"], single["per_subject"],
                              n_boot=args.n_boot)["auprc"]
    sig = "*" if (pd_["lo"] > 0 or pd_["hi"] < 0) else " "
    print(f"  paired bootstrap: {pd_['delta']:+.4f} "
          f"[{pd_['lo']:+.4f}, {pd_['hi']:+.4f}]{sig}")

    print(f"\n{'#'*88}")
    if d >= 0.005:
        print("  -> Multi-horizon training is COSTING accuracy at this horizon.")
        print("     A real architectural weakness has been identified, and a")
        print("     horizon-structured output is worth designing.")
    else:
        print("  -> No meaningful interference. The shared representation")
        print("     serves h=15 as well as a dedicated model does, so")
        print("     restructuring the output heads is unlikely to help.")
    print(f"{'#'*88}")
    common.save_result(OUT_DIR, "_single_vs_multi",
                       {"horizon": args.horizon, "multi_auprc": a,
                        "single_auprc": b, "delta": d, "paired": pd_})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--monotonicity", action="store_true")
    ap.add_argument("--single_horizon", action="store_true")
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--horizon", type=int, default=15, choices=HORIZONS)
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
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.monotonicity:
        run_monotonicity(args)
    if args.single_horizon:
        run_single_horizon(args)
        run_report(args)
    elif args.report:
        run_report(args)
    if not (args.monotonicity or args.single_horizon or args.report):
        print("Choose --monotonicity (free), --single_horizon (one run), "
              "or --report.")


if __name__ == "__main__":
    main()
