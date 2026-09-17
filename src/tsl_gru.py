"""
tsl_gru.py
============
Trajectory-Scaled Light GRU (TSL-GRU) and its frozen four-arm ablation.

THE PROPOSED CELL
-----------------
The reset and update gates are removed. What replaces the update gate is
a retention coefficient whose value is conditioned on a causal summary of
the glucose trajectory relative to the clinical threshold:

    h~_t   = ReLU(BN(W_x x_t) + U_h h_{t-1})
    s_t    = rho * s_{t-1} + (1 - rho) * [c_t, v_t, c_t*v_t]
    Delta_t= tanh(W_s s_t + b_s)
    lambda_t = clip(sigmoid(theta_lambda) + kappa * Delta_t, eps, 1-eps)
    h_t    = lambda_t * h_{t-1} + (1 - lambda_t) * h~_t

so the memory timescale itself moves with how the child's glucose is
travelling relative to 70 mg/dL: a stable trajectory retains more of the
previous state, a trajectory approaching the threshold replaces it faster.

WHY THIS DIFFERS FROM THE TWO EARLIER ATTEMPTS
----------------------------------------------
TA-GRU added a correction term on top of an ordinary GRU gate; the network
simply drove that correction to zero (q fell to 0.04 and the betas shrank
every epoch), so the cell reverted to a standard GRU. TRM-GRU added a
retrieval pathway alongside an unchanged GRU, and the shuffle control
showed the retrieval carried no threshold semantics.

Here there is no "off" position that restores a GRU: the reset gate is
gone and the update gate has been replaced. The cell must compute
something different. What the network CAN still neutralise is the
*conditioning* -- driving W_s toward zero leaves lambda_t constant, which
is a leaky-integrator RNN rather than a GRU. That is why sd(lambda_t),
not mean(lambda_t), is the diagnostic that matters, and why arm C exists.

THE FROZEN FOUR-ARM ABLATION
----------------------------
    A  gru_ta      standard GRU + TA inputs            (current baseline)
    B  ligru_ta    reset gate removed, ReLU candidate  (does simplification help?)
    C  tsl_static  B + learned CONSTANT retention      (does scaling-instead-of-
                   lambda = sigmoid(theta), kappa = 0   gating help?)
    D  tsl_gru     C + trajectory-conditioned lambda   (does conditioning help?)

The decisive comparison is **D - C**, not D - A. A and D differ in three
ways at once (gating structure, retention parameterisation, conditioning);
C isolates the last of them. Without C, a D win could be credited to the
retention reformulation rather than to the trajectory conditioning that is
the actual claim.

kappa is learnable and initialised small, so D starts near C and has to
earn any deviation. If it ends near zero, the conditioning was not used --
and that is reported rather than hidden inside a tied AUPRC.

IMPLEMENTATION NOTES
--------------------
BatchNorm is applied to the input projection W_x x_t ONLY, never to the
recurrent term, and gradients are clipped: ReLU candidates inside a
recurrent loop can grow without bound, and normalising the recurrent path
would additionally leak batch statistics across timesteps.

All four arms use the same hand-written loop. Using cuDNN for A and a
Python loop for D would confound the mechanism with the implementation,
so expect ~30 min per arm rather than the ~9 min TRM-GRU needed.

Usage:
    python tsl_gru.py --model all --seed 42
    python tsl_gru.py --collect
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
OUT_DIR = config.RESULTS / "RQ2_models" / "tsl_gru"
MODELS = ["gru_ta", "ligru_ta", "tsl_static", "tsl_gru"]
TA_C, TA_V = 5, 6          # channel indices of c_t and v+_t
EPS = 1e-3


# ─── CELLS ────────────────────────────────────────────────────────────────────

class StdGRUCell(nn.Module):
    """Arm A. Update-gate convention: h_t = (1-z)h_{t-1} + z*h~_t."""

    def __init__(self, in_ch, hidden):
        super().__init__()
        self.x2h = nn.Linear(in_ch, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)

    def forward(self, x, h, s=None, lam_override=None):
        xr, xz, xn = self.x2h(x).chunk(3, -1)
        hr, hz, hn = self.h2h(h).chunk(3, -1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * h + z * n, None


class LightCell(nn.Module):
    """
    Arms B, C and D. Reset gate removed, ReLU candidate, BatchNorm on the
    input projection only.

    mode:
      'gate'    keep a conventional update gate           -> B
      'static'  learned constant retention, kappa = 0     -> C
      'traj'    trajectory-conditioned retention          -> D
    """

    def __init__(self, in_ch, hidden, mode="gate", s_dim=3):
        super().__init__()
        self.mode, self.hidden = mode, hidden
        self.x2n = nn.Linear(in_ch, hidden, bias=False)
        self.bn = nn.BatchNorm1d(hidden)
        self.h2n = nn.Linear(hidden, hidden)

        if mode == "gate":
            self.x2z = nn.Linear(in_ch, hidden)
            self.h2z = nn.Linear(hidden, hidden)
        else:
            # base retention, per hidden unit. -0.4 -> sigmoid ~ 0.40, so the
            # cell starts biased toward NEW information rather than sitting
            # at an arbitrary retention level.
            self.theta_lambda = nn.Parameter(torch.full((hidden,), -0.4))
        if mode == "traj":
            self.Ws = nn.Linear(s_dim, hidden)
            # small init: D starts close to C and must earn any deviation,
            # the same "start neutral" principle used for TA-GRU's q
            self.kappa = nn.Parameter(torch.tensor(0.1))

    def lam(self, s):
        base = torch.sigmoid(self.theta_lambda).unsqueeze(0)
        if self.mode == "static":
            return base.expand(s.shape[0], -1) if s is not None else base
        delta = torch.tanh(self.Ws(s))
        return torch.clamp(base + self.kappa * delta, EPS, 1 - EPS)

    def forward(self, x, h, s=None, lam_override=None):
        """
        lam_override lets an intervention experiment substitute a different
        retention coefficient into an ALREADY-TRAINED cell, without touching
        any other weight. That is what separates "lambda varies over time"
        from "the variation is what the performance depends on".
        """
        n = torch.relu(self.bn(self.x2n(x)) + self.h2n(h))
        if self.mode == "gate":
            z = torch.sigmoid(self.x2z(x) + self.h2z(h))
            return (1 - z) * h + z * n, None
        lam = self.lam(s) if lam_override is None else lam_override
        return lam * h + (1 - lam) * n, lam


class TSLGRU(nn.Module):
    def __init__(self, variant, in_ch=7, hidden=64, layers=2, dropout=0.2,
                 rho=0.9):
        super().__init__()
        self.variant, self.hidden, self.layers = variant, hidden, layers
        self.rho = rho
        self.needs_s = variant in ("tsl_static", "tsl_gru")

        mode = {"ligru_ta": "gate", "tsl_static": "static",
                "tsl_gru": "traj"}.get(variant)
        make = (lambda i, o: StdGRUCell(i, o)) if variant == "gru_ta" \
            else (lambda i, o: LightCell(i, o, mode))
        self.cells = nn.ModuleList([
            make(in_ch if i == 0 else hidden, hidden) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = Heads(hidden)

    def trajectory(self, x):
        """
        s_t = rho * s_{t-1} + (1 - rho) * [c_t, v_t, c_t*v_t]

        Causal EMA, no learnable temporal parameters, so a gain in arm D
        cannot be attributed to a second temporal model learned alongside
        the cell.
        """
        c, v = x[:, :, TA_C], x[:, :, TA_V]
        u = torch.stack([c, v, c * v], dim=-1)          # (B,T,3)
        out = torch.empty_like(u)
        s = u[:, 0, :]
        out[:, 0, :] = s
        for t in range(1, u.shape[1]):
            s = self.rho * s + (1 - self.rho) * u[:, t, :]
            out[:, t, :] = s
        return out

    def forward(self, x, return_lambda=False, shuffle_s=None,
                lam_const=None):
        """
        shuffle_s : LongTensor permutation of the time axis, applied to the
                    trajectory descriptor only. Destroys the temporal
                    alignment between s_t and the input while leaving every
                    weight and every x_t untouched.
        lam_const : (hidden,) tensor substituted for lambda at every
                    timestep. Removes temporal variation while holding the
                    average retention level fixed.

        Both are inference-time interventions on a trained model.
        """
        B, T, _ = x.shape
        S = self.trajectory(x) if self.needs_s else None
        if S is not None and shuffle_s is not None:
            S = S[:, shuffle_s, :]
        lam_ov = (lam_const.unsqueeze(0).expand(B, -1)
                  if lam_const is not None else None)
        hs = [x.new_zeros(B, self.hidden) for _ in range(self.layers)]
        lams = []
        for t in range(T):
            inp = x[:, t, :]
            st = S[:, t, :] if S is not None else None
            for li, cell in enumerate(self.cells):
                hs[li], lam = cell(inp, hs[li], st, lam_override=lam_ov)
                if lam is not None and return_lambda and li == 0:
                    lams.append(lam.detach())
                inp = self.drop(hs[li]) if li < self.layers - 1 else hs[li]
        out = self.head(hs[-1])
        if return_lambda:
            return out, (torch.stack(lams, 1) if lams else None)
        return out


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

@torch.no_grad()
def lambda_stats(model, bw, idx, labels, device, n=20000, batch=1024):
    """
    sd(lambda) across timesteps is the key number. A cell can perform well
    with a constant lambda -- that is arm C -- so a high mean tells us
    nothing about whether the trajectory conditioning was used.
    """
    if model.variant not in ("tsl_static", "tsl_gru"):
        return None
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx[:n], labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    mean_t, sd_t, ys = [], [], []
    for xb, yb in dl:
        _, lam = model(xb.to(device), return_lambda=True)
        if lam is None:
            return None
        lam = lam.mean(-1)                       # average over hidden units
        mean_t.append(lam.mean(1).cpu().numpy())
        sd_t.append(lam.std(1).cpu().numpy())    # variation ACROSS TIME
        ys.append(yb[:, 1].numpy())
    m = np.concatenate(mean_t)
    sd = np.concatenate(sd_t)
    y = np.concatenate(ys).astype(int)
    out = {
        "lambda_mean": float(m.mean()),
        "lambda_sd_over_time": float(sd.mean()),
        "lambda_p05": float(np.percentile(m, 5)),
        "lambda_p50": float(np.percentile(m, 50)),
        "lambda_p95": float(np.percentile(m, 95)),
        "lambda_positive": float(m[y == 1].mean()) if (y == 1).any() else float("nan"),
        "lambda_negative": float(m[y == 0].mean()) if (y == 0).any() else float("nan"),
    }
    c0 = model.cells[0]
    if model.variant == "tsl_gru":
        out["kappa"] = float(c0.kappa.detach().cpu())
        out["mean_abs_kappa_delta"] = float(abs(out["kappa"]) * 0.5)
    return out


def print_lambda(s):
    if s is None:
        return
    extra = (f"  kappa {s['kappa']:+.4f}" if "kappa" in s else "")
    print(f"      lambda: mean {s['lambda_mean']:.4f}  "
          f"sd-over-time {s['lambda_sd_over_time']:.5f}  "
          f"p05/p50/p95 {s['lambda_p05']:.3f}/{s['lambda_p50']:.3f}/"
          f"{s['lambda_p95']:.3f}{extra}")
    print(f"              lambda(pos) {s['lambda_positive']:.4f} vs "
          f"lambda(neg) {s['lambda_negative']:.4f}   delta "
          f"{s['lambda_positive']-s['lambda_negative']:+.5f}")


# ─── TRAIN ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, labels, device, batch=2048, workers=2):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    return np.concatenate([model(xb.to(device)).cpu().numpy()
                           for xb, _ in dl]).astype(np.float32)


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    print(f"\n{'='*78}\n{variant} | seed {args.seed} | labels {args.labels}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    model = TSLGRU(variant, in_ch=bw.timeline.shape[1], hidden=args.hidden,
                   dropout=args.dropout,
                   rho=args.rho).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}  rho={args.rho}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=getattr(args, 'weight_decay', 0.0))
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    best, best_state, bad, best_ep, hist = -1.0, None, 0, -1, []
    for ep in range(args.epochs):
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
            raise RuntimeError(
                f"{variant}: non-finite predictions at epoch {ep}. ReLU "
                f"candidates in a recurrent loop can diverge; try --lr 3e-4.")
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
        note = f"  [{skipped} skipped]" if skipped else ""
        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"val mean-AUPRC {vauprc:.4f}{mark}{note}")
        s = lambda_stats(model, bw, idx_va, Y, device)
        if s:
            s["epoch"] = ep
            hist.append(s)
            print_lambda(s)
        if bad >= args.patience:
            print("    early stop"); break

    model.load_state_dict(best_state)
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final = lambda_stats(model, bw, idx_va, Y, device)

    res = {"model": variant, "loss": "weighted_bce", "balance": "none",
           "label_set": args.labels, "seed": args.seed, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "rho": args.rho, "lambda_final": final, "lambda_history": hist,
           "horizons": {}}

    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, tinfo = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["threshold_info"] = tinfo
        ev["test_prevalence"] = float(Y[idx_te, j].mean())
        fx = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, 0.5)
        ev["fixed_threshold_0.5"] = {k: fx[k] for k in
                                     ["pooled", "per_subject_mean",
                                      "per_subject", "constraint"]}
        res["horizons"][str(h)] = ev
        m, c = ev["per_subject_mean"], ev["constraint"]
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")

    res["minutes"] = (time.time() - t0) / 60
    name = f"{variant}__weighted_bce__none__{args.labels}"
    if args.seed != config.SPLIT_SEED:
        name += f"__s{args.seed}"
    if getattr(args, "suffix", ""):
        # without this every sweep trial writes to the same .json/.pt/.npz
        name += f"__{args.suffix}"
    np.savez_compressed(OUT_DIR / f"{name}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state, "config": vars(args)},
               OUT_DIR / f"{name}.pt")
    common.save_result(OUT_DIR, name, res)
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


# ─── COLLECT ──────────────────────────────────────────────────────────────────

def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No TSL-GRU results yet.")
        return
    runs = [json.load(open(f)) for f in files]
    import collections
    by = collections.defaultdict(list)
    for r in runs:
        by[r["model"]].append(r)

    print(f"\n{'#'*100}")
    print("# TSL-GRU ABLATION")
    print("#   A gru_ta     : standard GRU + TA inputs")
    print("#   B ligru_ta   : reset gate removed, ReLU candidate")
    print("#   C tsl_static : learned CONSTANT retention (kappa = 0)")
    print("#   D tsl_gru    : trajectory-conditioned retention   <- proposed")
    print(f"{'#'*100}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>14} {'params':>9} {'n_seeds':>8} "
              f"{'AUPRC mean':>11} {'sd':>8} {'range':>8} {'PPV':>8} {'Recall':>8}")
        for v in MODELS:
            lst = [r for r in by.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            a = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"] for r in lst])
            p = np.array([r["horizons"][str(h)]["per_subject_mean"]["ppv"] for r in lst])
            rc = np.array([r["horizons"][str(h)]["per_subject_mean"]["recall"] for r in lst])
            sd = a.std(ddof=1) if len(a) > 1 else float("nan")
            rg = a.max() - a.min() if len(a) > 1 else float("nan")
            print(f"  {v:>14} {lst[0]['n_params']:>9,} {len(a):>8} "
                  f"{a.mean():>11.4f} {sd:>8.4f} {rg:>8.4f} "
                  f"{p.mean():>8.4f} {rc.mean():>8.4f}")

    print(f"\n\n{'#'*100}")
    print("# DECISION  (the claim rests on D - C, not D - A)")
    print(f"{'#'*100}")
    for h in HORIZONS:
        got = {v: np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"]
                            for r in by.get(v, []) if str(h) in r["horizons"]])
               for v in MODELS}
        got = {k: v for k, v in got.items() if len(v)}
        if not {"gru_ta", "tsl_static", "tsl_gru"} <= set(got):
            continue
        sds = [v.std(ddof=1) for v in got.values() if len(v) > 1]
        sd = max(sds) if sds else float("nan")
        A, C, D = got["gru_ta"], got["tsl_static"], got["tsl_gru"]
        B = got.get("ligru_ta")
        line = f"\nh={h}:  A {A.mean():.4f}"
        if B is not None:
            line += f"  B {B.mean():.4f}"
        line += f"  C {C.mean():.4f}  D {D.mean():.4f}   (max seed sd {sd:.4f})"
        print(line)
        if B is not None:
            print(f"  B-A = {B.mean()-A.mean():+.4f}   does simplification help?")
            print(f"  C-B = {C.mean()-B.mean():+.4f}   does scaling beat gating?")
        dDC = D.mean() - C.mean()
        verdict = ("trajectory conditioning helps" if dDC > sd
                   else "conditioning adds nothing over constant retention"
                   if abs(dDC) <= sd or np.isnan(sd)
                   else "conditioning HURTS")
        print(f"  D-C = {dDC:+.4f}   -> {verdict}   <-- THE CLAIM")
        print(f"  D-A = {D.mean()-A.mean():+.4f}   (vs the current baseline)")
        if np.isnan(sd):
            print("  (single seed: run 43 and 44 before concluding)")

    print(f"\n\n{'#'*100}")
    print("# RETENTION BEHAVIOUR")
    print(f"{'#'*100}")
    for v in ["tsl_static", "tsl_gru"]:
        for r in by.get(v, []):
            s = r.get("lambda_final")
            if not s:
                continue
            print(f"\n{v} (seed {r['seed']}):")
            print_lambda(s)
            if v == "tsl_gru":
                # 1e-3 was too coarse: seed 42 produced sd 0.00088 with
                # kappa growing 0.10 -> 0.60, which is an ACTIVE mechanism,
                # yet the old threshold reported it as collapsed. What
                # matters is whether kappa was reinforced or driven out.
                if s["lambda_sd_over_time"] < 1e-5 or abs(s.get("kappa", 0)) < 0.02:
                    print("      -> lambda is effectively constant and kappa "
                          "was driven out: the conditioning collapsed, and D "
                          "is C with extra parameters.")
                elif abs(s.get("kappa", 0)) < 1e-3:
                    print("      -> kappa ~ 0: the conditioning term was "
                          "driven out during training.")
                elif abs(s["lambda_positive"] - s["lambda_negative"]) < 1e-3:
                    print("      -> lambda varies, but not differently before "
                          "hypoglycaemia.")
                else:
                    print("      -> lambda varies over time AND differs before "
                          "hypoglycaemia: the mechanism is active.")

    common.save_result(OUT_DIR, "_tsl_gru_summary",
                       {f"{r['model']}__s{r['seed']}": r for r in runs})
    print(f"\n✓ Saved -> {OUT_DIR / '_tsl_gru_summary.json'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.9,
                    help="EMA decay for the trajectory descriptor s_t")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.2,
                    help="dropout between recurrent layers")
    ap.add_argument("--weight_decay", type=float, default=0.0,
                    help="Adam weight decay")
    ap.add_argument("--suffix", default="",
                    help="appended to the run name. REQUIRED when "
                         "sweeping, or trials overwrite each other.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args)
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: hand-written recurrent loop; CPU is impractical.")

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"windows {len(bw):,} | channels {bw.timeline.shape[1]} | "
          f"train {len(idx_tr):,} (shared across arms) val {len(idx_va):,} "
          f"test {len(idx_te):,}")

    for v in (MODELS if args.model == "all" else [args.model]):
        name = f"{v}__weighted_bce__none__{args.labels}"
        if args.seed != config.SPLIT_SEED:
            name += f"__s{args.seed}"
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
