"""
ar_rhu.py
===========
Anticipatory-Reactive Residual Hazard Unit (AR-RHU).

THE PROBLEM THIS ADDRESSES
--------------------------
The error-regime audit found performance collapsing as glucose rises:
AUPRC 0.879 below 90 mg/dL against 0.105 above 150 at h=30, with GRU and
TCN correlating at 0.975 and failing on the same windows. Adding absolute
state helped, and the gain was concentrated in exactly that hard regime
(+0.039 at 120-150 mg/dL, versus +0.003 below 90).

So the task contains two different prediction problems: one where the
fall is already visible, and one where risk is developing before the
trajectory shows it. A single hidden state is asked to serve both.

WHY THIS DIFFERS FROM THE FOUR EARLIER ATTEMPTS
-----------------------------------------------
TA-GRU added a gate correction; the network drove q from 0.056 to 0.041
and the betas shrank every epoch. TRM-GRU added retrieval; the shuffle
control was indistinguishable from the real thing. TAP-GRU added gate
sharpening and a product branch; the exponents stayed pinned at 1.08 out
of a possible 3.0 and the product branch collapsed to 7%. DMS-TCN added
dynamic routing; it routed, but gained nothing.

Every one of them could be suppressed, and every one was. The common
cause: each mechanism was free to duplicate what the ordinary state
already computed, and duplication is the cheapest solution.

AR-RHU removes that option structurally. The anticipatory state is
explicitly projected off the reactive state at every timestep, so it
CANNOT simply re-encode the same information. That is a constraint on the
representation, not a suggestion the optimiser may ignore.

FROZEN EQUATIONS
----------------
Reactive pathway, on the immediate clinical channels
x^R = [c_t, v+_t, g_abs, rate_abs]:

    r~_t  = tanh(W_R x^R_t + U_R r_{t-1})
    z^R_t = sigmoid(W_zR x^R_t + U_zR r_{t-1})
    r_t   = (1 - z^R_t) * r_{t-1} + z^R_t * r~_t

Anticipatory proposal, on the broader physiological channels
x^A = [glucose_z, basal, bolus, carbs, carbs_observed, g_abs, rate_abs]:

    a*_t  = tanh(W_A x^A_t + U_A a_{t-1} + C_A r_{t-1})

Safe residualisation:

    rhat_t = r_t / max(||r_t||_2, eps)
    a'_t   = a*_t - gamma * (a*_t . rhat_t) rhat_t,   gamma = sigmoid(theta)

Dividing by max(||r||, eps) rather than by ||r||^2 + eps keeps the
coefficient bounded: at t=0 the hidden state is zero, and a squared-norm
denominator makes the projection coefficient explode there. With this
form, r_t = 0 gives rhat_t = 0 and the projection simply does nothing.

Anticipatory memory then updates through its own gate, so it persists
across time rather than being recomputed each step:

    z^A_t = sigmoid(W_zA x^A_t + U_zA a_{t-1})
    a_t   = (1 - z^A_t) * a_{t-1} + z^A_t * a'_t

Per-horizon risk, combined as complementary contributions:

    q^R_h = sigmoid(w^R_h . r_T + b^R_h)
    q^A_h = sigmoid(w^A_h . a_T + b^A_h)
    p_h   = 1 - (1 - q^R_h)(1 - q^A_h)

so (1 - q^R) q^A is the additional risk NOT already explained by the
reactive pathway. These are called risk contributions rather than
hazards, because the labels are cumulative ("any reading below 70 within
the horizon") rather than instantaneous hazard targets.

gamma IS LEARNABLE
------------------
Initialised at 0.90 -- strong residualisation, but with enough sigmoid
gradient left to move. Its trajectory is the primary diagnostic:

    gamma -> 1    the model wants the separation
    gamma ~ 0.5   partial overlap is useful
    gamma -> 0    the architecture is trying to recover an ordinary
                  overlapping representation, which would explain a tie

Hard-setting gamma = 1 would make a null result uninterpretable: we could
not tell whether the constraint failed to help or actively hurt.

Usage:
    python ar_rhu.py --model all --seed 42
    python ar_rhu.py --collect
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

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "ar_rhu"
MODELS = ["gru_abs", "ar_rhu"]

CH_REACTIVE = [5, 6, 7, 8]                     # c, v+, g_abs, rate_abs
CH_ANTICIP = [0, 1, 2, 3, 4, 7, 8]             # glucose_z, basal, bolus,
                                               # carbs, carbs_obs, g_abs, rate
EPS = 1e-6
GAMMA_INIT = 2.1972                            # sigmoid(2.1972) = 0.90


class GRUBaseline(nn.Module):
    """Arm A: ordinary GRU on all 9 channels."""

    def __init__(self, in_ch=9, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.variant = "gru_abs"
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x, return_diag=False):
        h, _ = self.gru(x)
        out = self.head(h[:, -1, :])
        return (out, None) if return_diag else out


class ARRHU(nn.Module):
    """Arm B: the proposed cell."""

    def __init__(self, hidden=64, dropout=0.2):
        super().__init__()
        self.variant = "ar_rhu"
        self.hidden = hidden
        nR, nA = len(CH_REACTIVE), len(CH_ANTICIP)

        # reactive pathway
        self.WR = nn.Linear(nR, hidden); self.UR = nn.Linear(hidden, hidden, bias=False)
        self.WzR = nn.Linear(nR, hidden); self.UzR = nn.Linear(hidden, hidden, bias=False)

        # anticipatory pathway
        self.WA = nn.Linear(nA, hidden); self.UA = nn.Linear(hidden, hidden, bias=False)
        self.CA = nn.Linear(hidden, hidden, bias=False)
        self.WzA = nn.Linear(nA, hidden); self.UzA = nn.Linear(hidden, hidden, bias=False)

        self.theta_gamma = nn.Parameter(torch.tensor(GAMMA_INIT))
        self.drop = nn.Dropout(dropout)

        # separate per-horizon readouts for the two contributions
        self.headR = nn.Linear(hidden, len(HORIZONS))
        self.headA = nn.Linear(hidden, len(HORIZONS))

    def gamma(self):
        return torch.sigmoid(self.theta_gamma)

    def forward(self, x, return_diag=False):
        B, T, _ = x.shape
        xR, xA = x[:, :, CH_REACTIVE], x[:, :, CH_ANTICIP]
        r = x.new_zeros(B, self.hidden)
        a = x.new_zeros(B, self.hidden)
        g = self.gamma()
        cos_before, cos_after = [], []

        for t in range(T):
            xr, xa = xR[:, t, :], xA[:, t, :]

            # reactive
            r_cand = torch.tanh(self.WR(xr) + self.UR(r))
            zR = torch.sigmoid(self.WzR(xr) + self.UzR(r))
            r_new = (1 - zR) * r + zR * r_cand

            # anticipatory proposal, conditioned on the PREVIOUS reactive state
            a_star = torch.tanh(self.WA(xa) + self.UA(a) + self.CA(r))

            # safe residualisation against the CURRENT reactive state
            nrm = r_new.norm(dim=-1, keepdim=True)
            rhat = r_new / torch.clamp(nrm, min=EPS)
            rhat = torch.where(nrm > EPS, rhat, torch.zeros_like(rhat))
            proj = (a_star * rhat).sum(-1, keepdim=True)
            a_res = a_star - g * proj * rhat

            zA = torch.sigmoid(self.WzA(xa) + self.UzA(a))
            a_new = (1 - zA) * a + zA * a_res

            if return_diag:
                cos_before.append(self._cos(a_star, r_new).detach())
                cos_after.append(self._cos(a_res, r_new).detach())
            r, a = r_new, a_new

        qR = torch.sigmoid(self.headR(self.drop(r)))
        qA = torch.sigmoid(self.headA(self.drop(a)))
        p = 1.0 - (1.0 - qR) * (1.0 - qA)

        if return_diag:
            return p, {"qR": qR.detach(), "qA": qA.detach(),
                       "gamma": float(g.detach()),
                       "cos_before": torch.stack(cos_before, 1).mean().item(),
                       "cos_after": torch.stack(cos_after, 1).mean().item()}
        return p

    @staticmethod
    def _cos(u, v):
        return (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1) + EPS)


def build(variant, in_ch, hidden):
    return ARRHU(hidden) if variant == "ar_rhu" else GRUBaseline(in_ch, hidden)


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

@torch.no_grad()
def ar_stats(model, bw, idx, labels, device, g_raw, n=20000, batch=1024):
    """
    Does the anticipatory pathway do what it is named for -- signal risk
    while the reactive pathway is still quiet?
    """
    if model.variant != "ar_rhu":
        return None
    model.eval()
    sub = idx[:n]
    dl = DataLoader(WindowDataset(bw, sub, labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    QR, QA, Y, CB, CA = [], [], [], [], []
    for xb, yb in dl:
        _, d = model(xb.to(device), return_diag=True)
        QR.append(d["qR"].cpu().numpy()); QA.append(d["qA"].cpu().numpy())
        Y.append(yb.numpy()); CB.append(d["cos_before"]); CA.append(d["cos_after"])
    qR, qA = np.concatenate(QR), np.concatenate(QA)
    y = np.concatenate(Y)
    j = 1                                        # h=30 for the summary
    gl = g_raw[:len(qR)]

    quiet = qR[:, j] < 0.2
    pos = y[:, j] == 1
    high = gl >= 120

    out = {"gamma": float(model.gamma().detach().cpu()),
           "cos_before": float(np.mean(CB)), "cos_after": float(np.mean(CA)),
           "qR_mean": float(qR[:, j].mean()), "qA_mean": float(qA[:, j].mean()),
           "qR_pos": float(qR[pos, j].mean()) if pos.any() else float("nan"),
           "qA_pos": float(qA[pos, j].mean()) if pos.any() else float("nan"),
           "qR_neg": float(qR[~pos, j].mean()), "qA_neg": float(qA[~pos, j].mean())}

    # THE key statistic: anticipatory signal when reactive is quiet, before a real event
    m = quiet & pos
    out["qA_when_quiet_and_positive"] = float(qA[m, j].mean()) if m.sum() > 10 else float("nan")
    m2 = quiet & ~pos
    out["qA_when_quiet_and_negative"] = float(qA[m2, j].mean()) if m2.sum() > 10 else float("nan")
    out["n_quiet_positive"] = int(m.sum())

    m3 = pos & high
    out["frac_qA_gt_qR_pos_high_glucose"] = (
        float((qA[m3, j] > qR[m3, j]).mean()) if m3.sum() > 10 else float("nan"))
    return out


def print_ar(s):
    if s is None:
        return
    print(f"      gamma {s['gamma']:.4f}   cos(a,r) before {s['cos_before']:+.4f} "
          f"-> after {s['cos_after']:+.4f}")
    print(f"      qR pos/neg {s['qR_pos']:.4f}/{s['qR_neg']:.4f}   "
          f"qA pos/neg {s['qA_pos']:.4f}/{s['qA_neg']:.4f}")
    print(f"      qA when reactive quiet: positives "
          f"{s['qA_when_quiet_and_positive']:.4f} vs negatives "
          f"{s['qA_when_quiet_and_negative']:.4f}  "
          f"(n={s['n_quiet_positive']})")


# ─── TRAIN ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, labels, device, batch=2048, workers=2):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    return np.concatenate([model(xb.to(device)).cpu().numpy()
                           for xb, _ in dl]).astype(np.float32)


def run_name(variant, args):
    """
    Hidden size is part of the name so that a capacity-matched run cannot
    overwrite the original. AR-RHU at hidden=64 has 22,665 parameters
    against the GRU's 41,572; hidden=88 gives 41,721, within 0.4%.
    """
    base = f"{variant}__s{args.seed}"
    return base if args.hidden == 64 else f"{base}__h{args.hidden}"


def run_one(variant, bw, Y, idx_tr, idx_va, idx_te, device, args):
    print(f"\n{'='*78}\n{variant} | hidden {args.hidden} | seed {args.seed}"
          f"\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    model = build(variant, bw.timeline.shape[1], args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")
    if variant == "ar_rhu":
        print(f"  gamma init {float(model.gamma()):.4f}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    ends_va = bw.starts[idx_va] + bw.window_len - 1
    g_va = bw.raw_gluc[ends_va]

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, run_name(variant, args),
                                resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state, hist = None, []
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
            raise RuntimeError(f"{variant}: non-finite predictions at epoch {ep}")
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
        s = ar_stats(model, bw, idx_va, Y, device, g_va)
        if s:
            s["epoch"] = ep
            hist.append(s)
            print_ar(s)
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
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final = ar_stats(model, bw, idx_va, Y, device, g_va)

    res = {"model": variant, "seed": args.seed, "hidden": args.hidden,
           "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "ar_final": final, "ar_history": hist, "horizons": {}}
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, tinfo = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        ev["test_prevalence"] = float(Y[idx_te, j].mean())
        res["horizons"][str(h)] = ev
        m, c = ev["per_subject_mean"], ev["constraint"]
        print(f"  h={h:>3}  AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")

    res["minutes"] = (time.time() - t0) / 60
    name = run_name(variant, args)
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
        print("No AR-RHU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        key = r["model"] if r.get("hidden", 64) == 64 else \
            f"{r['model']}_h{r['hidden']}"
        runs.setdefault(key, []).append(r)

    print(f"\n{'#'*92}")
    print("# AR-RHU PILOT — A: GRU+TA+Absolute   B: AR-RHU (proposed)")
    print(f"{'#'*92}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>10} {'params':>9} {'AUPRC':>9} {'PPV':>9} {'Recall':>9}")
        for v in sorted(runs):
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>10} {lst[0]['n_params']:>9,} {m['auprc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*92}\n# DECISION\n{'#'*92}")
    for h in HORIZONS:
      A = {r["seed"]: r for r in runs.get("gru_abs", []) if str(h) in r["horizons"]}
      for bkey in [k for k in sorted(runs) if k.startswith("ar_rhu")]:
        B = {r["seed"]: r for r in runs.get(bkey, []) if str(h) in r["horizons"]}
        shared = sorted(set(A) & set(B))
        if not shared:
            continue
        a = np.mean([A[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        b = np.mean([B[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        d = b - a
        pa = A[shared[0]]["n_params"]; pb = B[shared[0]]["n_params"]
        verdict = ("strong result" if d >= 0.020 else
                   "run ablations and further seeds" if d >= 0.010 else
                   "interesting" if d >= 0.005 else
                   "STOP")
        print(f"\nh={h}:  gru_abs {a:.4f} ({pa:,}p)   {bkey} {b:.4f} "
              f"({pb:,}p)   delta {d:+.4f}   -> {verdict}")
        if len(shared) == 1:
            s = shared[0]
            dd = common.paired_delta(A[s]["horizons"][str(h)]["per_subject"],
                                     B[s]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"        paired bootstrap: {dd['delta']:+.4f} "
                  f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    print(f"\n\n{'#'*92}\n# MECHANISM BEHAVIOUR\n{'#'*92}")
    for r in [x for k in runs if k.startswith("ar_rhu") for x in runs[k]]:
        s = r.get("ar_final")
        if not s:
            continue
        print(f"\nseed {r['seed']}, hidden {r.get('hidden', 64)} "
              f"({r['n_params']:,} params):")
        print_ar(s)
        gq = s.get("qA_when_quiet_and_positive", float("nan"))
        gn = s.get("qA_when_quiet_and_negative", float("nan"))
        if s["gamma"] < 0.2:
            print("      -> gamma collapsed: the model wanted an ordinary")
            print("         overlapping representation, so the separation")
            print("         constraint was working against it.")
        elif abs(s["cos_after"]) > 0.3:
            print("      -> the states are still strongly aligned despite")
            print("         residualisation; the constraint is not binding.")
        elif np.isfinite(gq) and np.isfinite(gn) and gq > gn * 1.5:
            print("      -> the anticipatory pathway fires ahead of the")
            print("         reactive one on true events: the mechanism does")
            print("         what it was designed to do.")
        else:
            print("      -> the states separate as intended, but the")
            print("         anticipatory pathway does not signal earlier than")
            print("         the reactive one.")

    common.save_result(OUT_DIR, "_ar_rhu_summary",
                       {f"{v}__s{r['seed']}": r
                        for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_ar_rhu_summary.json'}")


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
    if device == "cpu":
        print("  WARNING: AR-RHU is a hand-written recurrent loop; "
              "CPU will be impractically slow.")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    print(f"  channels: {bw.features}")
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"train {len(idx_tr):,} (shared) val {len(idx_va):,} test {len(idx_te):,}")

    for v in (MODELS if args.model == "all" else [args.model]):
        if (OUT_DIR / f"{run_name(v, args)}.json").exists() and not args.force:
            print(f"\n[skip] {run_name(v, args)} already done")
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
