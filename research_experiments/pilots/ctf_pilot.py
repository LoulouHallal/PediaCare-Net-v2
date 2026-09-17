"""
ctf_pilot.py
==============
CTF-RU seed-42 pilot, and the grid audit that must precede it.

    A  gru_abs   GRU on 9 channels, incl. TA and absolute   41,572 params
    C  ctf_ru    CTF-RU, 5 context channels + PHYSICAL
                 glucose and rate, NO TA channels           41,551 params

THE INTEGRATION POINT THAT MATTERS MOST
---------------------------------------
CTF-RU's grid is defined in mg/dL. It must receive real glucose values.
The dataset's `glucose_absolute` channel is (G - 100)/50 and spans about
[-1.2, 7.6]; feeding that in would place every reading in the bottom bin
and silently destroy the clinical grid. The raw values come from
bw.raw_gluc, which is the only array in the pipeline holding true mg/dL.

Rate sign convention: rate_t = (G_t - G_{t-1}) / 5, so FALLING glucose
gives a NEGATIVE rate, hence a negative displacement, hence transport
toward lower-glucose states. This is the opposite sign to the TA channel
v+, which is positive when falling -- an easy and invisible error, so the
audit prints both to confirm.

--audit
-------
Reports the raw glucose distribution and how much of it the grid covers,
plus the displacement distribution. The grid range should be chosen from
the data rather than assumed: since the bin count does not affect the
parameter count (update_gate and candidate are shared across bins), there
is no reason to truncate the range and push every hyperglycaemic reading
into a single edge state -- particularly since the error-regime audit
found high glucose to be exactly where the model was weakest.

Usage:
    python ctf_pilot.py --audit
    python ctf_pilot.py --model all --seed 42 --grid_max 220
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
from absolute_state import attach_absolute
from ctf_ru import ClinicalThresholdFluxRU, count_trainable_parameters

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "ctf_ru"
MODELS = ["gru_abs", "ctf_ru"]
CTX_CHANNELS = [0, 1, 2, 3, 4]          # glucose_z, basal, bolus, carbs, obs


def build_rate(bw):
    """
    rate_t = (G_t - G_{t-1}) / sample_min, per subject, never across a
    subject boundary. NEGATIVE when falling.
    """
    g = bw.raw_gluc.astype(np.float32)
    rate = np.zeros_like(g)
    for si in range(len(bw.subjects)):
        m = np.flatnonzero(bw.reading_subject == si)
        if len(m) < 2:
            continue
        gg = g[m]
        rate[m] = np.diff(gg, prepend=gg[0]) / bw.sample_min
    return g, rate


class CTFDataset(Dataset):
    """Window slice plus the physical glucose and rate for that window."""

    def __init__(self, bw, idx, labels, g_raw, rate, channels=None):
        self.tl, self.starts, self.wl = bw.timeline, bw.starts, bw.window_len
        self.idx = np.asarray(idx)
        self.y = labels[self.idx].astype(np.float32)
        self.g, self.r = g_raw, rate
        self.ch = channels

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        s = self.starts[self.idx[i]]
        sl = slice(s, s + self.wl)
        x = self.tl[sl] if self.ch is None else self.tl[sl][:, self.ch]
        return (torch.from_numpy(np.ascontiguousarray(x)),
                torch.from_numpy(np.ascontiguousarray(self.g[sl])),
                torch.from_numpy(np.ascontiguousarray(self.r[sl])),
                torch.from_numpy(self.y[i]))


class GRUBaseline(nn.Module):
    def __init__(self, in_ch=9, hidden=64, layers=2, dropout=0.2):
        super().__init__()
        self.variant = "gru_abs"
        self.gru = nn.GRU(in_ch, hidden, num_layers=layers, batch_first=True,
                          dropout=dropout)
        self.head = Heads(hidden)

    def forward(self, x, g=None, r=None, return_diagnostics=False):
        h, _ = self.gru(x)
        out = self.head(h[:, -1, :])
        return (out, None) if return_diagnostics else out


# ─── AUDIT ────────────────────────────────────────────────────────────────────

def run_audit(args, bw):
    g, rate = build_rate(bw)
    print(f"{'#'*90}")
    print("# RAW GLUCOSE AUDIT — does the clinical grid cover the data?")
    print(f"{'#'*90}")
    print(f"\n  raw glucose, mg/dL  ({len(g):,} readings)")
    for q in [0, 1, 5, 50, 95, 99, 99.9]:
        lab = "min" if q == 0 else f"p{q}"
        print(f"    {lab:<6} {np.percentile(g, q):>8.1f}")
    print(f"    max    {g.max():>8.1f}")

    print(f"\n  these are real mg/dL, not the normalised channel:")
    print(f"    timeline[:,0] (glucose_z)  range "
          f"[{bw.timeline[:,0].min():.2f}, {bw.timeline[:,0].max():.2f}]")
    print(f"    bw.raw_gluc                range "
          f"[{g.min():.1f}, {g.max():.1f}]  <- what CTF-RU receives")

    print(f"\n  rate, mg/dL/min   (negative = falling)")
    for q in [0.1, 1, 50, 99, 99.9]:
        print(f"    p{q:<5} {np.percentile(rate, q):>8.3f}")
    print(f"    mean {rate.mean():>8.4f}   (should be ~0 over long records)")
    print(f"    sign check: fraction negative {100*(rate < 0).mean():.1f}%, "
          f"positive {100*(rate > 0).mean():.1f}%")

    print(f"\n  grid coverage")
    print(f"    {'grid':>14} {'below min':>11} {'above max':>11} {'outside':>10}")
    for lo, hi in [(20, 220), (20, 300), (20, 400), (40, 400)]:
        b = float((g < lo).mean())
        a = float((g > hi).mean())
        n_bins = int((hi - lo) / args.dx) + 1
        print(f"    {f'{lo}-{hi} ({n_bins}b)':>14} {100*b:>10.3f}% "
              f"{100*a:>10.3f}% {100*(a+b):>9.3f}%")

    d = rate * (bw.sample_min / args.dx)
    print(f"\n  displacement delta = v * {bw.sample_min}/{args.dx}, in bins")
    print(f"    p99 |delta| {np.percentile(np.abs(d), 99):>7.3f}   "
          f"p99.9 {np.percentile(np.abs(d), 99.9):>7.3f}   "
          f"max {np.abs(d).max():>7.3f}")
    print(f"    fraction |delta| > 1: {100*(np.abs(d) > 1).mean():.3f}%  "
          f"(handled by multi-cell transport, not clipped)")

    print(f"\n{'#'*90}\nCHOOSING THE GRID\n{'#'*90}")
    over220 = float((g > 220).mean())
    if over220 > 0.005:
        print(f"\n  {100*over220:.2f}% of readings exceed 220 mg/dL. Those would")
        print("  all collapse into the top state, losing resolution exactly")
        print("  where the error-regime audit found the model weakest.")
        print("  Widen the grid -- it costs compute per timestep but NOT")
        print("  parameters, since the cell weights are shared across bins.")
    else:
        print(f"\n  only {100*over220:.3f}% of readings exceed 220 mg/dL, so the")
        print("  20-220 grid is adequate.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, "_grid_audit", {
        "glucose": {f"p{q}": float(np.percentile(g, q))
                    for q in [0, 1, 5, 50, 95, 99, 99.9]},
        "glucose_max": float(g.max()),
        "frac_above_220": over220,
        "frac_below_20": float((g < 20).mean()),
        "delta_p999": float(np.percentile(np.abs(d), 99.9)),
        "frac_delta_gt_1": float((np.abs(d) > 1).mean())})
    print(f"\n✓ Saved -> {OUT_DIR / '_grid_audit.json'}")


# ─── TRAIN ────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, bw, idx, Y, g, r, device, ch, batch=2048, workers=2,
            want_diag=False):
    model.eval()
    dl = DataLoader(CTFDataset(bw, idx, Y, g, r, ch), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, A, Fx = [], [], []
    for xb, gb, rb, _ in dl:
        if want_diag and model.variant == "ctf_ru":
            o, d = model(xb.to(device), gb.to(device), rb.to(device),
                         return_diagnostics=True)
            A.append(d["_alpha_per_window"].cpu().numpy())
            Fx.append(d["_flux_norm_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device), gb.to(device), rb.to(device))
        P.append(torch.sigmoid(o).cpu().numpy() if model.variant == "ctf_ru"
                 else o.cpu().numpy())
    out = np.concatenate(P).astype(np.float32)
    if want_diag and A:
        return out, np.concatenate(A), np.concatenate(Fx)
    return out


def run_one(variant, bw, Y, g, r, idx_tr, idx_va, idx_te, device, args):
    ch = None if variant == "gru_abs" else CTX_CHANNELS
    print(f"\n{'='*78}\n{variant} | seed {args.seed}\n{'='*78}")
    t0 = time.time()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if variant == "gru_abs":
        model = GRUBaseline(bw.timeline.shape[1], args.hidden).to(device)
    else:
        model = ClinicalThresholdFluxRU(
            input_size=len(CTX_CHANNELS), hidden_per_bin=args.hidden_per_bin,
            head_hidden=args.head_hidden, n_horizons=len(HORIZONS),
            grid_min_mgdl=args.grid_min, grid_max_mgdl=args.grid_max,
            dx_mgdl=args.dx, dt_min=float(bw.sample_min)).to(device)
        print(f"  grid {args.grid_min:.0f}-{args.grid_max:.0f} mg/dL, "
              f"{model.n_bins} states, dx {args.dx:.0f}")
        print(f"  threshold 70 between index {model.below_idx} and "
              f"{model.above_idx} (coord {model.boundary_coord:.2f})")
    n_par = count_trainable_parameters(model)
    print(f"  train n={len(idx_tr):,}  params={n_par:,}")

    prev = Y[idx_tr].mean(0)
    pos_weight = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pos_weight).to(device)
    pw = torch.tensor(pos_weight, dtype=torch.float32, device=device)

    def crit(out, y):
        if variant == "ctf_ru":                 # model emits logits
            return crit_ce(torch.sigmoid(out), y)
        return crit_ce(out, y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    dl = DataLoader(CTFDataset(bw, idx_tr, Y, g, r, ch), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, f"{variant}__s{args.seed}",
                                resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state = None

    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0
        for xb, gb, rb, yb in dl:
            xb, gb, rb, yb = (xb.to(device, non_blocking=True), gb.to(device),
                              rb.to(device), yb.to(device))
            opt.zero_grad()
            loss = crit(model(xb, gb, rb), yb)
            if not torch.isfinite(loss):
                continue
            loss.backward()
            gn = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                opt.zero_grad(); continue
            opt.step()
            tot += loss.item(); nb += 1

        pv = predict(model, bw, idx_va, Y, g, r, device, ch,
                     workers=args.workers)
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

        if variant == "ctf_ru":
            with torch.no_grad():
                xb, gb, rb, _ = next(iter(DataLoader(
                    CTFDataset(bw, idx_va[:2048], Y, g, r, ch),
                    batch_size=2048, num_workers=0)))
                _, d = model(xb.to(device), gb.to(device), rb.to(device),
                             return_diagnostics=True)
            print(f"      alpha {float(d['alpha_mean']):.4f} "
                  f"(sd {float(d['alpha_sd']):.4f})   flux "
                  f"{float(d['boundary_flux_norm_mean']):.5f}   rho "
                  f"{float(d['flux_decay']):.4f}")
            print(f"      |delta| mean {float(d['delta_abs_mean']):.3f} max "
                  f"{float(d['delta_abs_max']):.2f}   edge lo/hi "
                  f"{float(d['edge_hit_rate_lower']):.4f}/"
                  f"{float(d['edge_hit_rate_upper']):.4f}   cons.err "
                  f"{float(d['conservation_err_bar']):.2e}")
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best val mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv = predict(model, bw, idx_va, Y, g, r, device, ch, workers=args.workers)
    got = predict(model, bw, idx_te, Y, g, r, device, ch,
                  workers=args.workers, want_diag=(variant == "ctf_ru"))
    if variant == "ctf_ru":
        pt, alpha_w, flux_w = got
    else:
        pt, alpha_w, flux_w = got, None, None

    res = {"model": variant, "seed": args.seed, "n_params": int(n_par),
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

    # did the mechanism behave differently before events?
    if alpha_w is not None:
        yj = Y[idx_te, 1].astype(bool)
        ends = bw.starts[idx_te] + bw.window_len - 1
        high = bw.raw_gluc[ends] >= 120
        print(f"\n  mechanism on test windows:")
        print(f"    {'':>22} {'positives':>11} {'negatives':>11}")
        for nm, arr in [("alpha", alpha_w), ("boundary flux norm", flux_w)]:
            print(f"    {nm:>22} {arr[yj].mean():>11.5f} {arr[~yj].mean():>11.5f}")
        if (yj & high).any() and (~yj & high).any():
            print(f"    within glucose >= 120 mg/dL:")
            for nm, arr in [("alpha", alpha_w), ("boundary flux norm", flux_w)]:
                print(f"    {nm:>22} {arr[yj & high].mean():>11.5f} "
                      f"{arr[~yj & high].mean():>11.5f}")
        res["mechanism"] = {
            "alpha_pos": float(alpha_w[yj].mean()),
            "alpha_neg": float(alpha_w[~yj].mean()),
            "flux_pos": float(flux_w[yj].mean()),
            "flux_neg": float(flux_w[~yj].mean()),
            "flux_pos_high": float(flux_w[yj & high].mean()) if (yj & high).any() else None,
            "flux_neg_high": float(flux_w[~yj & high].mean()) if (~yj & high).any() else None}

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
        print("No CTF-RU results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*90}")
    print("# CTF-RU PILOT — A: GRU+TA+Absolute   C: CTF-RU (no TA channels)")
    print(f"{'#'*90}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'variant':>10} {'params':>9} {'AUPRC':>9} {'PPV':>9} {'Recall':>9}")
        for v in MODELS:
            lst = [r for r in runs.get(v, []) if str(h) in r["horizons"]]
            if not lst:
                continue
            m = lst[0]["horizons"][str(h)]["per_subject_mean"]
            print(f"  {v:>10} {lst[0]['n_params']:>9,} {m['auprc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")

    print(f"\n\n{'#'*90}\n# DECISION\n{'#'*90}")
    for h in HORIZONS:
        A = {r["seed"]: r for r in runs.get("gru_abs", []) if str(h) in r["horizons"]}
        C = {r["seed"]: r for r in runs.get("ctf_ru", []) if str(h) in r["horizons"]}
        shared = sorted(set(A) & set(C))
        if not shared:
            continue
        a = np.mean([A[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        c = np.mean([C[s]["horizons"][str(h)]["per_subject_mean"]["auprc"] for s in shared])
        d = c - a
        if h == 15:
            verdict = ("excellent" if c >= 0.884 else
                       "serious result: run seeds and ablations" if c >= 0.874 else
                       "promising architecture" if c >= 0.869 else
                       "matched TA without receiving TA" if c >= 0.864 else
                       "no accuracy advantage")
        else:
            verdict = "better" if d > 0 else "behind"
        print(f"\nh={h}:  gru_abs {a:.4f}   ctf_ru {c:.4f}   delta {d:+.4f}"
              f"   -> {verdict}")
        if len(shared) == 1:
            s = shared[0]
            dd = common.paired_delta(A[s]["horizons"][str(h)]["per_subject"],
                                     C[s]["horizons"][str(h)]["per_subject"],
                                     n_boot=args.n_boot)["auprc"]
            sig = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
            print(f"        paired bootstrap: {dd['delta']:+.4f} "
                  f"[{dd['lo']:+.4f}, {dd['hi']:+.4f}]{sig}")

    for r in runs.get("ctf_ru", []):
        mech = r.get("mechanism")
        if not mech:
            continue
        print(f"\n{'#'*90}\n# MECHANISM (seed {r['seed']})\n{'#'*90}")
        print(f"  alpha  positives {mech['alpha_pos']:.5f}  "
              f"negatives {mech['alpha_neg']:.5f}")
        print(f"  flux   positives {mech['flux_pos']:.5f}  "
              f"negatives {mech['flux_neg']:.5f}")
        if mech.get("flux_pos_high") is not None:
            print(f"  flux (glucose >= 120)  positives {mech['flux_pos_high']:.5f}  "
                  f"negatives {mech['flux_neg_high']:.5f}")
        if mech["alpha_pos"] < 0.05:
            print("  -> alpha collapsed: the network rejected transport.")
        elif mech["flux_pos"] > 1.5 * mech["flux_neg"]:
            print("  -> boundary flux is markedly higher before events: the")
            print("     proposed mechanism is operating as intended.")
        else:
            print("  -> transport is active but boundary flux does not")
            print("     distinguish events from non-events.")

    common.save_result(OUT_DIR, "_ctf_summary",
                       {f"{v}__s{r['seed']}": r for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_ctf_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--hidden_per_bin", type=int, default=55)
    ap.add_argument("--head_hidden", type=int, default=114)
    ap.add_argument("--grid_min", type=float, default=20.0)
    ap.add_argument("--grid_max", type=float, default=220.0)
    ap.add_argument("--dx", type=float, default=20.0)
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

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    if args.audit:
        run_audit(args, bw)
        return
    if args.model is None:
        print("Choose --audit, --model all, or --collect.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: CTF-RU is a hand-written recurrent loop over "
              f"{int((args.grid_max-args.grid_min)/args.dx)+1} states; "
              "CPU will be impractical.")

    attach_ta(bw)
    attach_absolute(bw)
    g, rate = build_rate(bw)
    print(f"  physical glucose [{g.min():.1f}, {g.max():.1f}] mg/dL   "
          f"rate [{rate.min():.2f}, {rate.max():.2f}] mg/dL/min")
    print(f"  CTF-RU context channels: "
          f"{[bw.features[i] for i in CTX_CHANNELS]}")

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
        if (OUT_DIR / f"{v}__s{args.seed}.json").exists() and not args.force:
            print(f"\n[skip] {v}__s{args.seed} already done")
            continue
        try:
            run_one(v, bw, Y, g, rate, idx_tr, idx_va, idx_te, device, args)
        except Exception as e:
            print(f"  !! {v} FAILED: {type(e).__name__}: {e}")
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    collect(args)


if __name__ == "__main__":
    main()
