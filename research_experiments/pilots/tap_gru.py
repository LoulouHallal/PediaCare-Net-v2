"""
tap_gru.py
============
Trajectory-Adaptive Product GRU (TAP-GRU v1).

Two things inside the GRU cell are redesigned. Nothing is bolted on
alongside it: no attention, no second pathway, no auxiliary model.

    1. GATE SHARPNESS becomes trajectory-dependent.
       Contrast-enhanced GRUs raise the gate activation to a fixed power
       (sigma(a)^2) to make the update/reset decision more selective.
       Fixing the exponent at 2 would simply reproduce that. Here the
       exponent is learned per timestep from the glucose trajectory, so
       a stable stretch can behave like an ordinary GRU while a rapidly
       falling one sharpens its gates.

    2. THE CANDIDATE STATE gains a product-unit branch.
       An additive transformation represents multiplicative dependencies
       inefficiently. A product unit computes its output in the log domain,
       so a weighted sum of logs becomes a product of powers.

FROZEN EQUATIONS
----------------
    s_t   = EMA[c_t, v+_t, c_t*v+_t]                 (rho = 0.9)
    p^z_t = 1 + 2*sigmoid(W_pz s_t + b_pz)
    p^r_t = 1 + 2*sigmoid(W_pr s_t + b_pr)
    z_t   = sigmoid(a^z_t) ** p^z_t
    r_t   = sigmoid(a^r_t) ** p^r_t
    q_t   = [x_t , r_t * h_{t-1}]
    l_t   = clamp(log(eps + |q_t|), -8, 8)
    g_t   = softplus(W_pu l_t + b_pu)
    h~_t  = tanh(W_a x_t + U_a (r_t * h_{t-1}) + V_a g_t)
    h_t   = (1 - z_t) * h_{t-1} + z_t * h~_t

NUMERICAL GUARDS -- and why each is needed
------------------------------------------
* softplus(u) replaces log(1 + exp(u)). The two are identical, but the
  naive form overflows to inf for u >= ~89 in float32 -- verified: at
  u = 100 the naive expression returns inf while softplus returns 100.
  With |q| near eps, log gives about -7, so a row of W_pu summing to -3
  would produce exp(21) ~ 1.3e9 and grow from there. The intermediate
  exp is never materialised here.

* l_t is clamped to [-8, 8] so the pre-activation cannot run away as
  h_{t-1} grows across timesteps.

* sigma(a)^p is computed as exp(p * logsigmoid(a)) rather than by
  exponentiating the sigmoid directly. Both give the same value, but the
  direct form has a gradient of ~2.6e-26 when sigma(a) ~ 2e-9, which
  silently kills learning for saturated gates. The log-space form keeps
  the gradient well-scaled.

* b_pz and b_pr start at -3, so p ~ 1.09 at initialisation and the gates
  behave almost exactly like an ordinary GRU. W_pu and V_a start small so
  the product branch does not dominate at epoch 0. The cell therefore
  begins as a GRU and must EARN both deviations -- the same principle
  that made TSL-GRU's kappa growth interpretable and TA-GRU's collapse
  unambiguous.

THE PILOT
---------
Two arms on seed 42 only:
    A  gru_ta    standard GRU + TA channels     (current baseline)
    D  tap_gru   proposed cell, same inputs

Decision rule agreed in advance, at h=15:
    < +0.003    stop
    +0.003..+0.010  interesting, probably insufficient
    >= +0.010   continue to the full ablation
    >= +0.020   very promising

Only if D succeeds do we train the ablations that separate the two
mechanisms (product units alone, fixed-exponent sharpening alone).

DIAGNOSTICS
-----------
Logged every epoch so that a tie is diagnosable rather than mute:
p^z and p^r distributions, their values on positive vs negative windows,
and the norm of the product branch relative to the additive branch. Three
distinct failure modes are then separable -- sharpening collapsed
(p -> 1), product branch collapsed (ratio -> 0), or both active but not
helpful.

Usage:
    python tap_gru.py --model all --seed 42
    python tap_gru.py --collect
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
OUT_DIR = config.RESULTS / "RQ2_models" / "tap_gru"
MODELS = ["gru_ta", "tap_gru"]
TA_C, TA_V = 5, 6
P_MAX = 2.0
EPS = 1e-6
L_CLAMP = 8.0


class StdGRUCell(nn.Module):
    """Arm A, written in the update-gate convention for comparability."""

    def __init__(self, in_ch, hidden):
        super().__init__()
        self.x2h = nn.Linear(in_ch, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)

    def forward(self, x, h, s=None):
        xr, xz, xn = self.x2h(x).chunk(3, -1)
        hr, hz, hn = self.h2h(h).chunk(3, -1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * h + z * n, None


class TAPCell(nn.Module):
    """Arm D. Trajectory-adaptive gate exponents + product-unit candidate."""

    def __init__(self, in_ch, hidden, s_dim=3):
        super().__init__()
        self.hidden = hidden
        self.x2h = nn.Linear(in_ch, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)

        # exponents: bias -3 => p ~ 1.09 at init, i.e. almost a plain GRU
        self.Wpz = nn.Linear(s_dim, hidden)
        self.Wpr = nn.Linear(s_dim, hidden)
        nn.init.zeros_(self.Wpz.weight); nn.init.constant_(self.Wpz.bias, -3.0)
        nn.init.zeros_(self.Wpr.weight); nn.init.constant_(self.Wpr.bias, -3.0)

        # product-unit branch, started small so it cannot dominate epoch 0
        self.Wpu = nn.Linear(in_ch + hidden, hidden)
        self.Va = nn.Linear(hidden, hidden, bias=False)
        nn.init.normal_(self.Wpu.weight, std=0.01); nn.init.zeros_(self.Wpu.bias)
        nn.init.normal_(self.Va.weight, std=0.01)

    def exponents(self, s):
        return (1.0 + P_MAX * torch.sigmoid(self.Wpz(s)),
                1.0 + P_MAX * torch.sigmoid(self.Wpr(s)))

    def forward(self, x, h, s=None, return_diag=False):
        xr, xz, xn = self.x2h(x).chunk(3, -1)
        hr, hz, hn = self.h2h(h).chunk(3, -1)
        pz, pr = self.exponents(s)

        # sigma(a)**p via exp(p * logsigmoid(a)): identical value, but the
        # direct form's gradient vanishes to ~1e-26 for saturated gates
        z = torch.exp(pz * F.logsigmoid(xz + hz))
        r = torch.exp(pr * F.logsigmoid(xr + hr))

        rh = r * h
        q = torch.cat([x, rh], dim=-1)
        l = torch.clamp(torch.log(EPS + q.abs()), -L_CLAMP, L_CLAMP)
        g = F.softplus(self.Wpu(l) + self.Wpu.bias * 0)   # bias already inside
        add = xn + hn * r
        prod = self.Va(g)
        n = torch.tanh(add + prod)

        h_new = (1 - z) * h + z * n
        if return_diag:
            return h_new, {"pz": pz.detach(), "pr": pr.detach(),
                           "add_norm": add.detach().abs().mean(),
                           "prod_norm": prod.detach().abs().mean()}
        return h_new, None


class TAPGRU(nn.Module):
    def __init__(self, variant, in_ch=7, hidden=64, layers=2, dropout=0.2,
                 rho=0.9):
        super().__init__()
        self.variant, self.hidden, self.layers, self.rho = (variant, hidden,
                                                            layers, rho)
        self.needs_s = variant == "tap_gru"
        make = (TAPCell if variant == "tap_gru" else StdGRUCell)
        self.cells = nn.ModuleList([
            make(in_ch if i == 0 else hidden, hidden) for i in range(layers)])
        self.drop = nn.Dropout(dropout)
        self.head = Heads(hidden)

    def trajectory(self, x):
        """s_t = rho*s_{t-1} + (1-rho)*[c_t, v_t, c_t*v_t] -- causal EMA."""
        c, v = x[:, :, TA_C], x[:, :, TA_V]
        u = torch.stack([c, v, c * v], dim=-1)
        out = torch.empty_like(u)
        s = u[:, 0, :]
        out[:, 0, :] = s
        for t in range(1, u.shape[1]):
            s = self.rho * s + (1 - self.rho) * u[:, t, :]
            out[:, t, :] = s
        return out

    def forward(self, x, return_diag=False):
        B, T, _ = x.shape
        S = self.trajectory(x) if self.needs_s else None
        hs = [x.new_zeros(B, self.hidden) for _ in range(self.layers)]
        diags = []
        for t in range(T):
            inp = x[:, t, :]
            st = S[:, t, :] if S is not None else None
            for li, cell in enumerate(self.cells):
                if isinstance(cell, TAPCell):
                    hs[li], d = cell(inp, hs[li], st,
                                     return_diag=return_diag and li == 0)
                    if d is not None:
                        diags.append(d)
                else:
                    hs[li], _ = cell(inp, hs[li], st)
                inp = self.drop(hs[li]) if li < self.layers - 1 else hs[li]
        out = self.head(hs[-1])
        return (out, diags) if return_diag else out


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

@torch.no_grad()
def tap_stats(model, bw, idx, labels, device, n=20000, batch=1024):
    """
    A tie is only informative if we know which mechanism was used. Three
    failure modes are separable here: p -> 1 (sharpening collapsed),
    prod/add -> 0 (product branch collapsed), or both active but unhelpful.
    """
    if model.variant != "tap_gru":
        return None
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx[:n], labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    pz_m, pr_m, an, pn, ys = [], [], [], [], []
    for xb, yb in dl:
        _, diags = model(xb.to(device), return_diag=True)
        if not diags:
            return None
        pz = torch.stack([d["pz"].mean(-1) for d in diags], 1)   # (B,T)
        pr = torch.stack([d["pr"].mean(-1) for d in diags], 1)
        pz_m.append(pz.mean(1).cpu().numpy())
        pr_m.append(pr.mean(1).cpu().numpy())
        an.append(float(np.mean([d["add_norm"].item() for d in diags])))
        pn.append(float(np.mean([d["prod_norm"].item() for d in diags])))
        ys.append(yb[:, 1].numpy())
    pz = np.concatenate(pz_m); pr = np.concatenate(pr_m)
    y = np.concatenate(ys).astype(int)
    a, p = float(np.mean(an)), float(np.mean(pn))
    return {
        "pz_mean": float(pz.mean()), "pz_sd": float(pz.std()),
        "pr_mean": float(pr.mean()), "pr_sd": float(pr.std()),
        "pz_positive": float(pz[y == 1].mean()) if (y == 1).any() else float("nan"),
        "pz_negative": float(pz[y == 0].mean()) if (y == 0).any() else float("nan"),
        "pr_positive": float(pr[y == 1].mean()) if (y == 1).any() else float("nan"),
        "pr_negative": float(pr[y == 0].mean()) if (y == 0).any() else float("nan"),
        "add_norm": a, "prod_norm": p,
        "prod_over_add": p / max(a, 1e-9),
    }


def print_tap(s):
    if s is None:
        return
    print(f"      p_z {s['pz_mean']:.4f} (sd {s['pz_sd']:.4f})  "
          f"p_r {s['pr_mean']:.4f} (sd {s['pr_sd']:.4f})   "
          f"[1.0 = ordinary GRU, max 3.0]")
    print(f"      p_z pos/neg {s['pz_positive']:.4f}/{s['pz_negative']:.4f}  "
          f"p_r pos/neg {s['pr_positive']:.4f}/{s['pr_negative']:.4f}")
    print(f"      product/additive norm ratio {s['prod_over_add']:.4f}  "
          f"(prod {s['prod_norm']:.4f}, add {s['add_norm']:.4f})")


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
    model = TAPGRU(variant, in_ch=bw.timeline.shape[1], hidden=args.hidden,
                   rho=args.rho).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    # Per-epoch checkpointing: a disconnected session costs one epoch
    # rather than the entire run.
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
            raise RuntimeError(
                f"{variant}: non-finite predictions at epoch {ep}. The "
                f"product branch can diverge; try --lr 3e-4.")
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
        s = tap_stats(model, bw, idx_va, Y, device)
        if s:
            s["epoch"] = ep
            hist.append(s)
            print_tap(s)
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)          # resumed run: best weights are on disk
        best_state = {k: v.detach().cpu().clone()
                      for k, v in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final = tap_stats(model, bw, idx_va, Y, device)

    res = {"model": variant, "seed": args.seed, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "tap_final": final, "tap_history": hist, "horizons": {}}
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
    ck.cleanup()                        # run completed; resume files no longer needed
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No TAP-GRU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*96}")
    print("# TAP-GRU v1 PILOT — A: GRU+TA   D: TAP-GRU (proposed)")
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
        A = [r for r in runs.get("gru_ta", []) if str(h) in r["horizons"]]
        D = [r for r in runs.get("tap_gru", []) if str(h) in r["horizons"]]
        if not A or not D:
            continue
        pa = {r["seed"]: r for r in A}
        pd_ = {r["seed"]: r for r in D}
        shared = sorted(set(pa) & set(pd_))
        if not shared:
            continue
        a = np.mean([pa[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        d = np.mean([pd_[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        gap = d - a
        verdict = ("very promising" if gap >= 0.020 else
                   "continue to the full ablation" if gap >= 0.010 else
                   "interesting, probably insufficient" if gap >= 0.003 else
                   "STOP: TAP-GRU does not improve prediction")
        print(f"\nh={h}:  A {a:.4f}   D {d:.4f}   D-A = {gap:+.4f}   -> {verdict}")
        if len(shared) == 1:
            s = shared[0]
            dd = common.paired_delta(pa[s]["horizons"][str(h)]["per_subject"],
                                     pd_[s]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"        paired bootstrap (seed {s}): "
                  f"{dd['delta']:+.4f} [{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    print(f"\n\n{'#'*96}")
    print("# MECHANISM BEHAVIOUR")
    print(f"{'#'*96}")
    for r in runs.get("tap_gru", []):
        s = r.get("tap_final")
        if not s:
            continue
        print(f"\nseed {r['seed']}:")
        print_tap(s)
        sharp = max(s["pz_mean"], s["pr_mean"]) - 1.0
        if sharp < 0.15 and s["prod_over_add"] < 0.05:
            print("      -> BOTH mechanisms collapsed: the cell reverted to "
                  "an ordinary GRU.")
        elif sharp < 0.15:
            print("      -> gate sharpening collapsed (p ~ 1); only the "
                  "product branch is active.")
        elif s["prod_over_add"] < 0.05:
            print("      -> product branch collapsed; only gate sharpening "
                  "is active.")
        else:
            print("      -> both mechanisms are active.")

    common.save_result(OUT_DIR, "_tap_gru_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_tap_gru_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.9)
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
    ap.add_argument("--no_resume", action="store_true",
                    help="ignore any partial checkpoint and train from scratch")
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
