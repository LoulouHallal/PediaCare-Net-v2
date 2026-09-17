"""
trm_gru.py
============
Threshold-Relative Memory GRU (TRM-GRU) and its frozen four-way ablation.

THE PROPOSED MECHANISM
----------------------
An ordinary GRU predicts from the final hidden state h_T alone. TRM-GRU
additionally retrieves earlier hidden states, and the retrieval is
controlled by the threshold-approach state rather than by content:

    k_t = phi(c_t, v_t, c_t * v_t)          keys, from TA state ONLY
    q_T = psi(c_T, v_T)                     query, from the CURRENT TA state
    a_t = softmax(q_T . k_t / sqrt(d))
    m_T = sum_t a_t * h_t
    head input = [h_T ; m_T ; h_T * m_T]

Intuitively the model asks: "which moments earlier in this window had a
threshold-approach condition similar to the one I am in now, and what was
the recurrent state then?" The TA signal becomes a coordinate for memory
access rather than another input channel.

WHY KEYS COME FROM (c, v) AND NOT FROM h
----------------------------------------
This is the design decision the experiment turns on. Because the TA
features are also GRU inputs, h_t already encodes c_t and v_t -- so
generic content attention (variant B) could in principle learn any
retrieval pattern C implements. C is therefore NOT more expressive than
B; it is more CONSTRAINED. It can only win by generalising better from
170 training subjects, which is exactly what an inductive bias is for.
Letting keys see h_t would destroy that distinction and make C a strict
superset of B.

THE FROZEN ABLATION
-------------------
    A  gru_ta        TA inputs, prediction from h_T only
    B  gru_attn      TA inputs, generic content attention (keys from h_t)
    C  trm_gru       TA inputs, retrieval keyed on TA state      <- proposed
    D  trm_shuffled  identical to C, but TA keys permuted across timesteps

Readings:
    A -> B   does retrieving history help at all?
    B -> C   does threshold-relative retrieval beat content retrieval?
    C -> D   does the temporal ALIGNMENT of TA states matter, or is the
             benefit merely from having a second readout vector?

D is the control that makes a null result interpretable. Without it, C ~= B
could mean either "TA retrieval is no better than content retrieval" or
"any extra pooled vector would have done this", and those are different
conclusions.

D's permutation is seeded and deterministic, and shuffles ONLY the keys --
the hidden states h_t retain their true order, so the retrieval targets
are unchanged and only the addressing is scrambled.

DIAGNOSTICS
-----------
A tie in AUPRC is uninformative unless we know what the attention did.
Logged every epoch:
    entropy of a_t          (uniform = 4.09 nats at T=60; low = peaked)
    mass on the last 12 steps (is it just reading the recent past?)
    entropy on positive vs negative windows
    mean |TA-distance| between the query state and attended states,
        weighted by attention -- if TRM-GRU works as designed, attended
        states should have SIMILAR threshold conditions to the query, so
        this should be lower than under uniform attention

Usage:
    python trm_gru.py --model all --seed 42
    python trm_gru.py --model all --seed 43
    python trm_gru.py --collect
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
OUT_DIR = config.RESULTS / "RQ2_models" / "trm_gru"
MODELS = ["gru_ta", "gru_attn", "trm_gru", "trm_shuffled"]
TA_C, TA_V = 5, 6          # channel indices of c_t and v+_t


# ─── MODEL ────────────────────────────────────────────────────────────────────

class TRMGRU(nn.Module):
    def __init__(self, variant, in_ch=7, hidden=64, layers=2, key_dim=32,
                 dropout=0.2, seed=42, max_len=60):
        super().__init__()
        self.variant = variant
        self.hidden, self.key_dim = hidden, key_dim
        self.use_mem = variant != "gru_ta"
        self.shuffle_keys = variant == "trm_shuffled"
        self.ta_keys = variant in ("trm_gru", "trm_shuffled")
        self.seed = seed

        self.gru = nn.GRU(in_ch, hidden, num_layers=layers,
                          batch_first=True, dropout=dropout)

        if self.use_mem:
            if self.ta_keys:
                # keys and query from the TA state ONLY -- see module docstring
                self.phi = nn.Sequential(nn.Linear(3, key_dim), nn.Tanh(),
                                         nn.Linear(key_dim, key_dim))
                self.psi = nn.Sequential(nn.Linear(2, key_dim), nn.Tanh(),
                                         nn.Linear(key_dim, key_dim))
            else:
                # generic content attention: keys and query from h
                self.phi = nn.Linear(hidden, key_dim)
                self.psi = nn.Linear(hidden, key_dim)
            # [h_T ; m_T ; h_T * m_T]
            self.head = Heads(hidden * 3)
        else:
            self.head = Heads(hidden)

        if self.shuffle_keys:
            # fixed at construction from the run seed, so the control is a
            # clean counterfactual rather than a noisy one
            g = torch.Generator().manual_seed(seed)
            self.register_buffer("key_perm",
                                 torch.randperm(max_len, generator=g))

    def _keys_query(self, h, x):
        if self.ta_keys:
            c, v = x[:, :, TA_C], x[:, :, TA_V]
            k = self.phi(torch.stack([c, v, c * v], dim=-1))     # (B,T,d)
            q = self.psi(torch.stack([c[:, -1], v[:, -1]], dim=-1))  # (B,d)
        else:
            k = self.phi(h)
            q = self.psi(h[:, -1, :])
        return k, q

    def forward(self, x, return_attn=False):
        h, _ = self.gru(x)                       # (B,T,H)
        hT = h[:, -1, :]
        if not self.use_mem:
            out = self.head(hT)
            return (out, None) if return_attn else out

        k, q = self._keys_query(h, x)
        if self.shuffle_keys:
            # Permute the KEYS only. The hidden states keep their true
            # order, so the retrieval targets are unchanged and only the
            # addressing is scrambled.
            #
            # The permutation is a registered buffer, fixed at construction
            # from the run seed. An earlier version derived it per forward
            # pass from the batch size, which made the last (short) batch
            # use a different permutation and the model non-deterministic
            # at evaluation time -- the control would then have been noisy
            # rather than a clean counterfactual.
            k = k[:, self.key_perm[:k.shape[1]], :]

        scores = torch.einsum("bd,btd->bt", q, k) / (self.key_dim ** 0.5)
        a = torch.softmax(scores, dim=1)         # (B,T)
        m = torch.einsum("bt,bth->bh", a, h)     # (B,H)

        out = self.head(torch.cat([hT, m, hT * m], dim=-1))
        return (out, a) if return_attn else out


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

@torch.no_grad()
def attn_stats(model, bw, idx, labels, device, n=20000, batch=1024):
    """
    What did the attention actually do? Without this a tied AUPRC cannot
    be told apart from "the mechanism was never used".
    """
    if not model.use_mem:
        return None
    model.eval()
    sub = idx[:n]
    dl = DataLoader(WindowDataset(bw, sub, labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    ents, recent, ys, tadist = [], [], [], []
    for xb, yb in dl:
        xb = xb.to(device)
        _, a = model(xb, return_attn=True)
        ent = -(a * torch.log(a.clamp_min(1e-12))).sum(1)
        ents.append(ent.cpu().numpy())
        recent.append(a[:, -12:].sum(1).cpu().numpy())
        ys.append(yb[:, 1].numpy())
        # attention-weighted distance in TA space between the query state
        # and the attended states: low = retrieving similar conditions
        c, v = xb[:, :, TA_C], xb[:, :, TA_V]
        d = ((c - c[:, -1:]) ** 2 + (v - v[:, -1:]) ** 2).sqrt()
        tadist.append((a * d).sum(1).cpu().numpy())
    ent = np.concatenate(ents)
    rec = np.concatenate(recent)
    y = np.concatenate(ys).astype(int)
    td = np.concatenate(tadist)
    T = bw.window_len
    return {
        "entropy_mean": float(ent.mean()),
        "entropy_uniform": float(np.log(T)),
        "mass_last12_mean": float(rec.mean()),
        "mass_last12_uniform": float(12 / T),
        "entropy_positive": float(ent[y == 1].mean()) if (y == 1).any() else float("nan"),
        "entropy_negative": float(ent[y == 0].mean()) if (y == 0).any() else float("nan"),
        "ta_dist_attended": float(td.mean()),
        "ta_dist_positive": float(td[y == 1].mean()) if (y == 1).any() else float("nan"),
        "ta_dist_negative": float(td[y == 0].mean()) if (y == 0).any() else float("nan"),
    }


def print_attn(s):
    if s is None:
        return
    print(f"      attn: entropy {s['entropy_mean']:.3f} "
          f"(uniform {s['entropy_uniform']:.3f})  "
          f"mass last-12 {s['mass_last12_mean']:.3f} "
          f"(uniform {s['mass_last12_uniform']:.3f})")
    print(f"            entropy pos/neg {s['entropy_positive']:.3f}/"
          f"{s['entropy_negative']:.3f}   TA-dist attended "
          f"{s['ta_dist_attended']:.4f} (pos {s['ta_dist_positive']:.4f} / "
          f"neg {s['ta_dist_negative']:.4f})")


# ─── TRAIN ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, labels, device, batch=4096, workers=2):
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
    model = TRMGRU(variant, in_ch=bw.timeline.shape[1], hidden=args.hidden,
                   key_dim=args.key_dim, seed=args.seed).to(device)
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  train n={len(idx_tr):,}  channels={bw.timeline.shape[1]}  "
          f"params={n_par:,}  memory={model.use_mem}  ta_keys={model.ta_keys}")

    crit = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
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
        s = attn_stats(model, bw, idx_va, Y, device)
        if s:
            s["epoch"] = ep
            hist.append(s)
            print_attn(s)
        if bad >= args.patience:
            print("    early stop"); break

    model.load_state_dict(best_state)
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt = predict(model, bw, idx_te, Y, device, workers=args.workers)
    final_attn = attn_stats(model, bw, idx_va, Y, device)

    res = {"model": variant, "loss": "weighted_bce", "balance": "none",
           "label_set": args.labels, "seed": args.seed, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "train_n": int(len(idx_tr)), "attn_final": final_attn,
           "attn_history": hist, "horizons": {}}

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
        print("No TRM-GRU results yet.")
        return
    runs = [json.load(open(f)) for f in files]
    import collections
    by = collections.defaultdict(list)
    for r in runs:
        by[r["model"]].append(r)

    print(f"\n{'#'*100}")
    print("# TRM-GRU ABLATION")
    print("#   A gru_ta      : h_T only")
    print("#   B gru_attn    : generic content attention (keys from h_t)")
    print("#   C trm_gru     : retrieval keyed on TA state      <- proposed")
    print("#   D trm_shuffled: C with TA keys permuted in time  (control)")
    print(f"{'#'*100}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>16} {'params':>9} {'n_seeds':>8} "
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
            print(f"  {v:>16} {lst[0]['n_params']:>9,} {len(a):>8} "
                  f"{a.mean():>11.4f} {sd:>8.4f} {rg:>8.4f} "
                  f"{p.mean():>8.4f} {rc.mean():>8.4f}")

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
        if not {"gru_ta", "gru_attn", "trm_gru"} <= set(got):
            continue
        A, B, C = got["gru_ta"], got["gru_attn"], got["trm_gru"]
        D = got.get("trm_shuffled")
        sds = [x.std(ddof=1) for x in got.values() if len(x) > 1]
        sd = max(sds) if sds else float("nan")
        print(f"\nh={h}:  A {A.mean():.4f}  B {B.mean():.4f}  C {C.mean():.4f}"
              + (f"  D {D.mean():.4f}" if D is not None else "")
              + f"   (max seed sd {sd:.4f})")
        print(f"  B-A = {B.mean()-A.mean():+.4f}   does retrieving history help at all?")
        dCB = C.mean() - B.mean()
        v1 = ("threshold-relative retrieval beats content retrieval" if dCB > sd
              else "no better than generic attention" if abs(dCB) <= sd or np.isnan(sd)
              else "WORSE than generic attention")
        print(f"  C-B = {dCB:+.4f}   -> {v1}")
        if D is not None:
            dCD = C.mean() - D.mean()
            v2 = ("temporal alignment of TA states matters" if dCD > sd
                  else "alignment does NOT matter -- the gain (if any) is from "
                       "having a second readout, not from TA semantics")
            print(f"  C-D = {dCD:+.4f}   -> {v2}")
        if np.isnan(sd):
            print("  (single seed: run 43 and 44 before drawing conclusions)")

    print(f"\n\n{'#'*100}")
    print("# ATTENTION BEHAVIOUR")
    print(f"{'#'*100}")
    for v in ["gru_attn", "trm_gru", "trm_shuffled"]:
        for r in by.get(v, []):
            s = r.get("attn_final")
            if not s:
                continue
            print(f"\n{v} (seed {r['seed']}):")
            print_attn(s)
            if abs(s["entropy_mean"] - s["entropy_uniform"]) < 0.05:
                print("      -> attention is essentially uniform: the "
                      "retrieval is just mean-pooling the hidden states.")
            elif s["mass_last12_mean"] > 0.8:
                print("      -> attention collapsed onto the recent past, "
                      "which h_T already summarises.")
            else:
                print("      -> attention is selective and not purely recency-based.")

    common.save_result(OUT_DIR, "_trm_gru_summary",
                       {f"{r['model']}__s{r['seed']}": r for r in runs})
    print(f"\n✓ Saved -> {OUT_DIR / '_trm_gru_summary.json'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all", choices=MODELS + ["all"])
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--key_dim", type=int, default=32)
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
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any

    # identical training indices across all four variants at a given seed
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
