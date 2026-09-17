"""
ta_gru.py
===========
Threshold-Approach GRU (TA-GRU) and its controlled ablation.

THE EXPERIMENT
--------------
Three models, run at identical seeds on identical training indices:

    A  gru            the 5 normalised channels, ordinary GRU cell
    B  gru_ta_feat    those channels PLUS c_t and v+_t, ordinary GRU cell
    C  ta_gru         exactly the same inputs as B, and c_t / v+_t
                      additionally modulate the update gate

B and C receive **identical information**. That is the whole point: if
C > B, the gain is attributable to the recurrent mechanism; if C ~= B,
the threshold/slope features were doing the work and the architectural
change adds nothing. Comparing TA-GRU only against A could not separate
those two explanations.

THE MECHANISM
-------------
Under causal per-subject z-scoring the network cannot locate the clinical
threshold: 70 mg/dL maps to a different normalised value for every child,
and that value drifts as the child's running statistics update. The TA
features restore it, expressed in units of that child's own variability:

    delta_t = (G_t - 70) / sigma_t          signed distance to threshold
    c_t     = sigmoid(-delta_t)             bounded proximity, in [0,1]
    v_t     = (G_{t-1} - G_t) / sigma_t     positive when falling
    v+_t    = softplus(v_t)                 non-negative downward motion

    q_t = sigmoid(b0 + b1*c_t + b2*v+_t + b3*c_t*v+_t),  b_i = softplus(theta_i)

Constraining b1, b2, b3 to be non-negative makes q_t monotonically
non-decreasing in both proximity and downward speed, so "closer to the
threshold while falling" can never *reduce* the approach signal. Without
that constraint the gate could learn an inverted meaning and still fit,
leaving the result uninterpretable.

    z_t^TA = z_t + q_t * (1 - z_t)          one-sided: z^TA >= z always

so when q_t ~ 0 the cell is exactly an ordinary GRU (the model that
currently performs best), and when q_t is large the hidden state is more
willing to overwrite itself with the incoming trajectory.

NOTE ON GATE CONVENTION
-----------------------
PyTorch's nn.GRU computes h_t = (1-z)*n_t + z*h_{t-1}, i.e. its z is a
KEEP gate. The formulation above uses z as an UPDATE gate. The cell here
is written in the update-gate convention so that z^TA = z + q(1-z) means
what it says; silently applying it to PyTorch's z would invert the
mechanism.

All three variants use this same hand-written cell. Using cuDNN for A and
a Python loop for C would confound the mechanism with the implementation.

DIAGNOSTICS
-----------
A tied AUPRC is uninformative unless we know whether the gate was used at
all. Every epoch logs mean/sd/percentiles of q, the fraction saturated at
either end, and mean q on positive vs negative windows. Three failure
modes are then distinguishable:

    q -> 0 everywhere        the model does not need the mechanism
    q -> 1 everywhere        degenerate; the cell has become memoryless
    q(pos) ~= q(neg)         the gate fires, but not preferentially
                             before hypoglycaemia

Two further controls are available: --ablate_q zero (force q=0, which
should reproduce B exactly) and --ablate_q noise (feed random q, which
should destroy any benefit if the signal was meaningful).

Usage:
    python ta_gru.py --model all --labels any --seed 42
    python ta_gru.py --model all --labels any --seed 43
    python ta_gru.py --collect
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
import stage2_deep as S2
from stage2_deep import WindowDataset, MultiHorizonLoss, Heads

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "ta_gru"
MODELS = ["gru", "gru_ta_feat", "ta_gru"]
HYPO_MGDL = 70.0
TA_CLAMP = 10.0


# ─── TA FEATURE CONSTRUCTION ──────────────────────────────────────────────────

def compute_ta_channels(bw, verbose=True):
    """
    Build c_t and v+_t for every reading.

    sigma_t is recomputed here with the same causal expanding formula
    build_windows used, rather than read from the file (it was never
    saved). Recomputing from raw_gluc reproduces it exactly and keeps the
    features strictly causal: sigma_t uses only readings before t.
    """
    n = len(bw.raw_gluc)
    c = np.zeros(n, dtype=np.float32)
    vp = np.zeros(n, dtype=np.float32)

    for si in range(len(bw.subjects)):
        m = np.flatnonzero(bw.reading_subject == si)
        if len(m) < 2:
            continue
        g = bw.raw_gluc[m].astype(np.float64)

        # expanding mean/std, shifted by one so reading t is never used
        cnt = np.arange(1, len(g) + 1)
        c1, c2 = np.cumsum(g), np.cumsum(g ** 2)
        mean = c1 / cnt
        var = np.maximum(c2 / cnt - mean ** 2, 0.0)
        std = np.sqrt(var)
        std = np.concatenate([std[:1], std[:-1]])
        std = np.maximum(std, 1e-6)

        delta = (g - HYPO_MGDL) / std                 # signed distance
        prev = np.concatenate([g[:1], g[:-1]])
        v = (prev - g) / std                          # positive when falling

        c[m] = 1.0 / (1.0 + np.exp(np.clip(delta, -60, 60)))   # sigmoid(-delta)
        vp[m] = np.log1p(np.exp(np.clip(v, -30, 30)))          # softplus(v)

    # same clamp the main features receive: the 1e-6 std floor can still
    # produce extreme values in the first readings of a subject
    n_clip = int(((np.abs(c) > TA_CLAMP) | (np.abs(vp) > TA_CLAMP)).sum())
    c = np.clip(c, -TA_CLAMP, TA_CLAMP)
    vp = np.clip(vp, -TA_CLAMP, TA_CLAMP)
    if verbose:
        print(f"  TA channels: c in [{c.min():.3f},{c.max():.3f}]  "
              f"v+ in [{vp.min():.3f},{vp.max():.3f}]  ({n_clip:,} clipped)")
    return np.stack([c, vp], axis=1).astype(np.float32)


def attach_ta(bw, verbose=True):
    """Append c_t, v+_t as channels 5 and 6 of the shared timeline."""
    ta = compute_ta_channels(bw, verbose)
    bw.timeline = np.concatenate([bw.timeline, ta], axis=1)
    bw.features = list(bw.features) + ["ta_proximity_c", "ta_downslope_vplus"]
    return bw


# ─── CELL ─────────────────────────────────────────────────────────────────────

class TAGRUCell(nn.Module):
    """
    GRU cell in the UPDATE-gate convention:

        h_t = (1 - z_t) * h_{t-1} + z_t * h~_t

    With use_ta=True the update gate is raised one-sidedly:

        z_t^TA = z_t + q_t * (1 - z_t)
    """

    def __init__(self, in_ch, hidden, use_ta=False):
        super().__init__()
        self.hidden, self.use_ta = hidden, use_ta
        self.x2h = nn.Linear(in_ch, 3 * hidden)
        self.h2h = nn.Linear(hidden, 3 * hidden)
        if use_ta:
            # betas pass through softplus so all three effects are
            # monotonically non-decreasing; theta init is negative so the
            # cell STARTS as an ordinary GRU (q ~ 0) and must learn to
            # deviate, rather than starting in an arbitrary regime
            self.theta = nn.Parameter(torch.full((3,), -2.0))
            self.b0 = nn.Parameter(torch.tensor(-3.0))

    def betas(self):
        return F.softplus(self.theta)

    def approach(self, c, vplus):
        b = self.betas()
        return torch.sigmoid(self.b0 + b[0] * c + b[1] * vplus
                             + b[2] * c * vplus)

    def forward(self, x, h, c=None, vplus=None, q_override=None):
        gx, gh = self.x2h(x), self.h2h(h)
        xr, xz, xn = gx.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)

        q = None
        if self.use_ta:
            q = (self.approach(c, vplus).unsqueeze(-1)
                 if q_override is None else q_override.unsqueeze(-1))
            z = z + q * (1.0 - z)          # one-sided: z^TA >= z

        h_new = (1.0 - z) * h + z * n
        return h_new, (q.squeeze(-1) if q is not None else None)


class TAGRU(nn.Module):
    """
    Stacked TAGRUCell + the shared 4-horizon head.

    variant:
      'gru'          channels 0:5, no TA features, no gate
      'gru_ta_feat'  channels 0:7, TA features as ordinary inputs
      'ta_gru'       channels 0:7 AND the gate reads c_t, v+_t
    """

    def __init__(self, variant, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.variant = variant
        self.n_in = 5 if variant == "gru" else 7
        self.use_ta = variant == "ta_gru"
        self.hidden, self.layers = hidden, layers
        self.cells = nn.ModuleList([
            TAGRUCell(self.n_in if i == 0 else hidden, hidden,
                      use_ta=self.use_ta and i == 0)   # gate on layer 0 only:
            for i in range(layers)])                   # c_t/v+_t are inputs,
        self.drop = nn.Dropout(dropout)                # not hidden states
        self.head = Heads(hidden)

    def forward(self, x, return_q=False, q_mode=None):
        B, T, _ = x.shape
        c_ch, v_ch = (x[:, :, 5], x[:, :, 6]) if self.n_in == 7 else (None, None)
        xin = x[:, :, :self.n_in]
        hs = [x.new_zeros(B, self.hidden) for _ in range(self.layers)]
        qs = []
        for t in range(T):
            inp = xin[:, t, :]
            for li, cell in enumerate(self.cells):
                qov = None
                if cell.use_ta and q_mode is not None:
                    qov = (torch.zeros(B, device=x.device) if q_mode == "zero"
                           else torch.rand(B, device=x.device))
                hs[li], q = cell(inp, hs[li],
                                 c_ch[:, t] if c_ch is not None else None,
                                 v_ch[:, t] if v_ch is not None else None,
                                 q_override=qov)
                if q is not None and return_q:
                    qs.append(q.detach())
                inp = self.drop(hs[li]) if li < self.layers - 1 else hs[li]
        out = self.head(hs[-1])
        return (out, torch.stack(qs, 1) if qs else None) if return_q else out


# ─── GATE DIAGNOSTICS ─────────────────────────────────────────────────────────

@torch.no_grad()
def gate_stats(model, bw, idx, labels, device, n=20000, batch=1024):
    """
    Distribution of q, and whether it fires preferentially before
    hypoglycaemia. A tied AUPRC is uninterpretable without this.
    """
    if not getattr(model, "use_ta", False):
        return None
    model.eval()
    sub = idx[:n]
    ds = WindowDataset(bw, sub, labels)
    dl = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    allq, ally = [], []
    for xb, yb in dl:
        _, q = model(xb.to(device), return_q=True)
        allq.append(q.mean(dim=1).cpu().numpy())   # mean over timesteps
        ally.append(yb[:, 1].numpy())              # h=30 label
    q = np.concatenate(allq)
    y = np.concatenate(ally).astype(int)
    b = model.cells[0].betas().detach().cpu().numpy()
    return {
        "q_mean": float(q.mean()), "q_sd": float(q.std()),
        "q_p05": float(np.percentile(q, 5)),
        "q_p50": float(np.percentile(q, 50)),
        "q_p95": float(np.percentile(q, 95)),
        "frac_q_lt_005": float((q < 0.05).mean()),
        "frac_q_gt_095": float((q > 0.95).mean()),
        "q_mean_positive": float(q[y == 1].mean()) if (y == 1).any() else float("nan"),
        "q_mean_negative": float(q[y == 0].mean()) if (y == 0).any() else float("nan"),
        "beta_proximity": float(b[0]), "beta_downslope": float(b[1]),
        "beta_interaction": float(b[2]),
        "b0": float(model.cells[0].b0.detach().cpu()),
    }


def print_gate(g):
    if g is None:
        return
    print(f"      q: mean {g['q_mean']:.4f} sd {g['q_sd']:.4f}  "
          f"p05/p50/p95 {g['q_p05']:.3f}/{g['q_p50']:.3f}/{g['q_p95']:.3f}  "
          f"<0.05 {100*g['frac_q_lt_005']:.1f}%  >0.95 {100*g['frac_q_gt_095']:.1f}%")
    print(f"      q(pos) {g['q_mean_positive']:.4f} vs q(neg) "
          f"{g['q_mean_negative']:.4f}   delta "
          f"{g['q_mean_positive']-g['q_mean_negative']:+.4f}   "
          f"betas c/v/cv {g['beta_proximity']:.3f}/{g['beta_downslope']:.3f}/"
          f"{g['beta_interaction']:.3f}")


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
    model = TAGRU(variant, hidden=args.hidden).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  channels={model.n_in}  params={n_par:,}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(WindowDataset(bw, idx_tr, Y), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    best, best_state, bad, best_ep, gate_hist = -1.0, None, 0, -1, []
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
            raise RuntimeError(f"{variant}: non-finite predictions at epoch {ep}")
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

        g = gate_stats(model, bw, idx_va, Y, device)
        if g:
            g["epoch"] = ep
            gate_hist.append(g)
            print_gate(g)
        if bad >= args.patience:
            print(f"    early stop"); break

    model.load_state_dict(best_state)
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final_gate = gate_stats(model, bw, idx_va, Y, device)

    res = {"model": variant, "loss": "weighted_bce", "balance": "none",
           "label_set": args.labels, "seed": args.seed, "n_params": int(n_par),
           "n_channels": model.n_in, "val_mean_auprc": best,
           "best_epoch": int(best_ep), "train_n": int(len(idx_tr)),
           "gate_final": final_gate, "gate_history": gate_hist,
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
        print("No TA-GRU results yet.")
        return
    runs = [json.load(open(f)) for f in files]

    import collections
    by = collections.defaultdict(list)
    for r in runs:
        by[r["model"]].append(r)

    print(f"\n{'#'*100}")
    print("# TA-GRU ABLATION — A (gru) / B (gru+TA features) / C (ta_gru)")
    print("#   B and C receive identical inputs; only C modulates the gate.")
    print(f"{'#'*100}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>14} {'ch':>3} {'params':>9} {'n_seeds':>8} "
              f"{'AUPRC mean':>11} {'sd':>8} {'range':>8} {'PPV':>8} {'Recall':>8}")
        for v in MODELS:
            lst = [r for r in by.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            a = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"]
                          for r in lst])
            p = np.array([r["horizons"][str(h)]["per_subject_mean"]["ppv"]
                          for r in lst])
            rc = np.array([r["horizons"][str(h)]["per_subject_mean"]["recall"]
                           for r in lst])
            sd = a.std(ddof=1) if len(a) > 1 else float("nan")
            print(f"  {v:>14} {lst[0]['n_channels']:>3} "
                  f"{lst[0]['n_params']:>9,} {len(a):>8} {a.mean():>11.4f} "
                  f"{sd:>8.4f} {a.max()-a.min() if len(a)>1 else float('nan'):>8.4f} "
                  f"{p.mean():>8.4f} {rc.mean():>8.4f}")

    # ---- the decision ------------------------------------------------------
    print(f"\n\n{'#'*100}")
    print("# DECISION")
    print(f"{'#'*100}")
    for h in HORIZONS:
        got = {}
        for v in MODELS:
            lst = [r for r in by.get(v, []) if str(h) in r["horizons"]]
            if lst:
                got[v] = np.array([r["horizons"][str(h)]["per_subject_mean"]["auprc"]
                                   for r in lst])
        if len(got) < 3:
            continue
        A, B, C = got["gru"], got["gru_ta_feat"], got["ta_gru"]
        sd = max([x.std(ddof=1) for x in (A, B, C) if len(x) > 1] or [np.nan])
        dCB, dBA = C.mean() - B.mean(), B.mean() - A.mean()
        verdict = ("mechanism helps beyond the features"
                   if dCB > sd else
                   "mechanism adds nothing beyond the features"
                   if abs(dCB) <= sd else
                   "mechanism HURTS relative to the features")
        print(f"\nh={h}:  A {A.mean():.4f}   B {B.mean():.4f}   C {C.mean():.4f}"
              f"   (max seed sd {sd:.4f})")
        print(f"  B-A = {dBA:+.4f}  (do the TA features add information?)")
        print(f"  C-B = {dCB:+.4f}  -> {verdict}")

    print(f"\n\n{'#'*100}")
    print("# GATE BEHAVIOUR (ta_gru only)")
    print(f"{'#'*100}")
    for r in by.get("ta_gru", []):
        g = r.get("gate_final")
        if not g:
            continue
        print(f"\nseed {r['seed']}:")
        print_gate(g)
        if g["q_mean"] < 0.05:
            print("      -> q ~ 0: the model did not use the mechanism.")
        elif g["q_mean"] > 0.95:
            print("      -> q ~ 1: degenerate, the cell is close to memoryless.")
        elif abs(g["q_mean_positive"] - g["q_mean_negative"]) < 0.01:
            print("      -> the gate fires, but not preferentially before hypo.")
        else:
            print("      -> the gate fires preferentially before hypoglycaemia.")

    common.save_result(OUT_DIR, "_ta_gru_summary",
                       {f"{r['model']}__s{r['seed']}": r for r in runs})
    print(f"\n✓ Saved -> {OUT_DIR / '_ta_gru_summary.json'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
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
    if device == "cpu":
        print("  WARNING: the TA cell is a Python loop over 60 timesteps. "
              "On CPU this is impractically slow.")

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any

    # Identical training indices for A, B and C at a given seed. Computed
    # ONCE here rather than inside each run, so the architecture comparison
    # cannot be contaminated by different subsamples.
    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"windows {len(bw):,} | channels {bw.timeline.shape[1]} | "
          f"train {len(idx_tr):,} (shared across variants) "
          f"val {len(idx_va):,} test {len(idx_te):,}")

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
