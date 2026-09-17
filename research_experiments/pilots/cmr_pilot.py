"""
cmr_pilot.py
==============
CMR-GRU ablation — validation-first, deterministic, paired multi-seed.

    gru        M=1, no FiLM, no routing    H=64   41,188   reference
    modular    M=4 block-diagonal          H=88   41,848  (+1.6%)
    film       M=1 + FiLM                  H=64   41,829  (+1.6%)
    mod_film   M=4 + FiLM                  H=88   42,729  (+3.7%)
    cmr        M=4 + FiLM + routing        H=84   40,141  (-2.5%)

Block-diagonal recurrence is ~45% cheaper at equal width, so the modular
arms run wider to stay within a few percent of the baseline. mod_film at
+3.7% is the loosest match; if it wins, rerun it at a smaller width before
believing the gain.

WHAT EACH RUNG ANSWERS
----------------------
    gru      -> modular     does splitting the state help on its own?
    gru      -> film        does context modulation help on its own?
    film     -> mod_film    do they compose?
    mod_film -> cmr         does sparse routing add anything?

Running only gru vs cmr would show a number and explain nothing. This is
the structure requested: each mechanism tested alone and in combination.

RUN ORDER
---------
Endpoints first. The intermediate rungs only earn GPU time once there is
something to decompose.

    python cmr_pilot.py --model gru --seed 42
    python cmr_pilot.py --model cmr --seed 42
    python cmr_pilot.py --collect
    # only if cmr clears the bar:
    python cmr_pilot.py --model modular --seed 42
    python cmr_pilot.py --model film --seed 42
    python cmr_pilot.py --model mod_film --seed 42

PROTOCOL
--------
Validation-only decisions. Test metrics are computed and saved but printed
only under --unlock_test. Seventeen architectures in this project were
chosen after inspecting the 38 test subjects, which makes them development
evidence rather than a clean confirmatory estimate; the write-up must say
so.

Determinism via seed_everything and make_loader. NOTE the cost: with
torch.use_deterministic_algorithms(True) a GRU run took 111 minutes here
against ~8 without. --fast_determinism keeps seeds, seeded DataLoader
workers and cudnn.deterministic but drops the strict algorithm flag. The
earlier non-reproducibility was almost certainly unseeded DataLoader
workers, which make_loader fixes, so the fast mode may well be
bit-reproducible too -- verify with --check_determinism before trusting it.

PROMOTION RULE, fixed before running
------------------------------------
Primary quantity: mean VALIDATION AUPRC over the four horizons.

    delta < +0.003     stop
    +0.003 to +0.005   borderline, add seeds
    delta >= +0.005    promote, run the full ladder and seeds 43/44

+0.003 is a project decision threshold: the baseline itself has moved by
0.00305 across nominally comparable runs.
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
from stage2_deep import WindowDataset, MultiHorizonLoss
from ta_gru import attach_ta
from absolute_state import attach_absolute
from cmr_gru import CMRGRU, CMRConfig, ARMS, count_parameters
from interaction_screen import seed_everything, make_loader
from nadir_head import NadirDataset, WithNadirHead, NadirLoss

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "cmr_gru"
MODELS = ["gru", "modular", "film", "mod_film", "cmr"]
HID = {"gru": 64, "modular": 88, "film": 64, "mod_film": 88, "cmr": 84}


@torch.no_grad()
def predict(model, bw, idx, Y, device, batch=2048, workers=2, want_diag=False):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, Y), batch_size=batch,
                    shuffle=False, num_workers=workers)
    P, R, G = [], [], []
    for xb, _ in dl:
        if want_diag:
            o, d = model(xb.to(device), return_diagnostics=True)
            if "_route_per_window" in d:
                R.append(d["_route_per_window"].cpu().numpy())
            if "_gamma_per_window" in d:
                G.append(d["_gamma_per_window"].cpu().numpy())
        else:
            o = model(xb.to(device))
        P.append(torch.sigmoid(o).cpu().numpy())
    return (np.concatenate(P).astype(np.float32),
            np.concatenate(R) if R else None,
            np.concatenate(G) if G else None)


def run_one(name, bw, Y, idx_tr, idx_va, idx_te, device, args, seed):
    H = args.hidden or HID[name]
    tag = f"{name}__s{seed}" + (f"__{args.suffix}" if args.suffix else "")
    if getattr(args, "lam", 0.0) > 0:
        # keep auxiliary runs on their own checkpoint; TrainCheckpoint
        # resumes by tag and would otherwise continue the baseline run
        tag += f"__lam{args.lam:g}"
    if getattr(args, "label_set", "any") != "any":
        # a consensus run is a different task, not a variant of the
        # legacy run -- it must never share a checkpoint or a filename
        tag += f"__{args.label_set}"
    print(f"\n{'='*78}\n{tag}\n{'='*78}")
    t0 = time.time()
    seed_everything(seed, strict=not args.fast_determinism)

    cfg = CMRConfig(bw.timeline.shape[1], hidden_size=H,
                    n_horizons=len(HORIZONS), anchor_init=args.anchor_init,
                    **ARMS[name])
    model = CMRGRU(cfg).to(device)
    if args.lam > 0:
        model = WithNadirHead(model, n_out=len(HORIZONS)).to(device)
        print(f'  + nadir head on {model.head_name} '
              f'({model.in_features} -> {len(HORIZONS)}), lam {args.lam:g}')
    n_par = count_parameters(model)
    print(f"  H {H}  modules {cfg.n_modules}  FiLM {cfg.use_film}  "
          f"routing {cfg.use_routing}  params {n_par:,}")
    if cfg.use_film or cfg.use_routing:
        print(f"  anchors init {args.anchor_init} (bounded [0,1), cannot go "
              f"negative)")
        print(f"  context channels: "
              f"{[bw.features[i] for i in cfg.context_idx]}")

    prev = Y[idx_tr].mean(0)
    pw = np.clip((1 - prev) / np.maximum(prev, 1e-6), 1.0, 50.0)
    crit_ce = MultiHorizonLoss("weighted_bce", pw).to(device)

    def crit(o, y):
        return crit_ce(torch.sigmoid(o), y)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max",
                                                       factor=0.5, patience=3)
    crit_n = NadirLoss(crit, lam=args.lam, delta=args.nadir_delta)

    _train_ds = WindowDataset(bw, idx_tr, Y)
    if args.lam > 0:
        _nz = np.load(config.DATA_DERIVED / f"nadir_targets_{args.tag}.npz")
        _train_ds = NadirDataset(_train_ds, idx_tr, _nz["nadir_z"], _nz["valid"])
        print(f"  nadir targets: {_nz['valid'][idx_tr].mean():.4%} of train "
              f"slots valid, clip {float(_nz['clip_hi']):.0f} mg/dL")
    dl = make_loader(_train_ds, args.batch, shuffle=True,
                     seed=seed, num_workers=args.workers, drop_last=True)

    from sklearn.metrics import average_precision_score
    ck = common.TrainCheckpoint(OUT_DIR, tag, resume=not args.no_resume)
    start_ep, best, best_ep, bad = ck.load_into(model, opt, sched)
    best_state = None

    for ep in range(start_ep, args.epochs):
        model.train()
        tot = nb = 0
        for _batch in dl:
            if args.lam > 0:
                xb, yb, gb, mb = _batch
                xb, yb = xb.to(device, non_blocking=True), yb.to(device)
                gb, mb = gb.to(device), mb.to(device)
                opt.zero_grad()
                _logits, _nad = model(xb, want_nadir=True)
                loss = crit_n(_logits, yb, _nad, gb, mb)
            else:
                xb, yb = _batch
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

        pv, _, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
        if not np.isfinite(pv).all():
            raise RuntimeError(f"{tag}: non-finite predictions at epoch {ep}")
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
              f"VAL mean-AUPRC {v:.4f}{' *' if improved else ''}")
        if args.lam > 0:
            # if the nadir term dwarfs the bce term the classifier is being
            # drowned -- a scaling problem, not a failed mechanism
            print(f"      bce {crit_n.last['bce']:.5f}  "
                  f"nadir {crit_n.last['nadir']:.5f}  "
                  f"(lam*nadir {args.lam * crit_n.last['nadir']:.5f})")

        if cfg.use_film or cfg.use_routing:
            with torch.no_grad():
                xb, _ = next(iter(DataLoader(
                    WindowDataset(bw, idx_va[:2048], Y), batch_size=2048,
                    num_workers=0)))
                _, d = model(xb.to(device), return_diagnostics=True)
            bits = []
            if "lam" in d:
                bits.append(f"lambda {float(d['lam']):.5f}  |gamma-1| "
                            f"{float(d['gamma_dev_mean']):.5f}  |beta| "
                            f"{float(d['beta_abs_mean']):.5f}")
            if "mu" in d:
                bits.append(f"mu {float(d['mu']):.5f}  route entropy "
                            f"{float(d['route_entropy_mean']):.4f} of "
                            f"{float(d['uniform_entropy']):.4f}  max share "
                            f"{float(d['route_max_mean']):.4f}")
            for b in bits:
                print(f"      {b}")
        if bad >= args.patience:
            print("    early stop"); break

    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        ck.restore_best(model)
        best_state = {k: t.detach().cpu().clone()
                      for k, t in model.state_dict().items()}
    print(f"  best VAL mean-AUPRC {best:.4f} @ epoch {best_ep}")

    pv, _, _ = predict(model, bw, idx_va, Y, device, workers=args.workers)
    pt, route, gam = predict(model, bw, idx_te, Y, device,
                             workers=args.workers,
                             want_diag=(cfg.use_film or cfg.use_routing))

    from sklearn.metrics import average_precision_score
    res = {"model": name, "seed": seed, "hidden": H,
           "n_modules": cfg.n_modules, "use_film": cfg.use_film,
           "use_routing": cfg.use_routing, "n_params": int(n_par),
           "val_mean_auprc": best, "best_epoch": int(best_ep),
           "val_horizons": {}, "horizons": {}}
    for j, h in enumerate(HORIZONS):
        res["val_horizons"][str(h)] = float(average_precision_score(
            Y[idx_va, j].astype(int), pv[:, j]))
    meta_te = bw.meta[idx_te]
    for j, h in enumerate(HORIZONS):
        thr, _ = common.find_threshold(Y[idx_va, j].astype(int), pv[:, j])
        ev = common.evaluate(Y[idx_te, j].astype(int), pt[:, j], meta_te, thr)
        ev["threshold"] = float(thr)
        res["horizons"][str(h)] = ev
    res["mean_auprc"] = float(np.mean(
        [res["horizons"][str(h)]["per_subject_mean"]["auprc"]
         for h in HORIZONS]))

    print("  validation by horizon: " +
          "  ".join(f"h{h} {res['val_horizons'][str(h)]:.4f}"
                    for h in HORIZONS))
    if args.unlock_test:
        for h in HORIZONS:
            m = res["horizons"][str(h)]["per_subject_mean"]
            print(f"  [test] h={h:>3}  AUPRC {m['auprc']:.4f}  "
                  f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}")
    else:
        print("  (test computed and saved but withheld; --unlock_test)")

    if route is not None or gam is not None:
        yj = Y[idx_te, 1].astype(bool)
        mech = {}
        if route is not None:
            # does routing differ before events? that is the whole claim
            mech["route_pos"] = route[yj].mean(0).tolist()
            mech["route_neg"] = route[~yj].mean(0).tolist()
            print(f"\n  module routing, positives "
                  f"{[round(v,4) for v in mech['route_pos']]}")
            print(f"                  negatives "
                  f"{[round(v,4) for v in mech['route_neg']]}")
        if gam is not None:
            mech["gamma_dev"] = float(gam.mean())
        res["mechanism"] = mech
        with torch.no_grad():
            c0 = model.cells[0]
            if hasattr(c0, "rho_lam"):
                res["lambda_final"] = float(c0.lam())
            if hasattr(c0, "rho_mu"):
                res["mu_final"] = float(c0.mu())

    res["minutes"] = (time.time() - t0) / 60
    np.savez_compressed(OUT_DIR / f"{tag}_probs.npz", val=pv, test=pt,
                        idx_val=idx_va, idx_test=idx_te)
    torch.save({"model_state": best_state}, OUT_DIR / f"{tag}.pt")
    common.save_result(OUT_DIR, tag, res)
    ck.cleanup()
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


def check_determinism(args):
    import itertools
    files = sorted(OUT_DIR.glob("gru__s*_probs.npz"))
    if len(files) < 2:
        print("Need two gru runs:\n"
              "  python cmr_pilot.py --model gru --seed 42\n"
              "  python cmr_pilot.py --model gru --seed 42 --force "
              "--suffix rep --no_resume")
        return
    print(f"{'#'*88}\n# DETERMINISM CHECK\n{'#'*88}\n")
    for a, b in itertools.combinations(files, 2):
        za, zb = np.load(a), np.load(b)
        if za["test"].shape != zb["test"].shape:
            continue
        d = float(np.abs(za["test"] - zb["test"]).max())
        print(f"  {a.name} vs {b.name}\n    max |p_A - p_B| = {d:.3e}"
              + ("   reproducible\n" if d < 1e-6 else
                 "   NOT reproducible -- find the first divergence\n"))


def collect(args):
    files = [f for f in sorted(OUT_DIR.glob("*.json"))
             if not f.name.startswith("_")]
    if not files:
        print("No CMR results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs.setdefault(r["model"], []).append(r)

    print(f"\n{'#'*88}\n# CMR-GRU ABLATION — VALIDATION\n{'#'*88}\n")
    print(f"  {'arm':>10} {'seed':>5} {'H':>4} {'M':>3} {'FiLM':>5} "
          f"{'route':>6} {'params':>8} {'val mean':>10}")
    for v in MODELS:
        for r in sorted(runs.get(v, []), key=lambda x: x["seed"]):
            print(f"  {v:>10} {r['seed']:>5} {r['hidden']:>4} "
                  f"{r['n_modules']:>3} {str(r['use_film']):>5} "
                  f"{str(r['use_routing']):>6} {r['n_params']:>8,} "
                  f"{r['val_mean_auprc']:>10.5f}")

    g = {r["seed"]: r["val_mean_auprc"] for r in runs.get("gru", [])}
    if not g:
        print("\n  run the gru arm first")
        return

    print(f"\n\n{'#'*88}\n# DECISION — mean VALIDATION AUPRC, paired by seed"
          f"\n{'#'*88}\n")
    means = {"gru": float(np.mean(list(g.values())))}
    for v in MODELS[1:]:
        pairs = [(r["seed"], r["val_mean_auprc"] - g[r["seed"]])
                 for r in runs.get(v, []) if r["seed"] in g]
        if not pairs:
            continue
        ds = np.array([d for _, d in pairs])
        means[v] = float(np.mean([r["val_mean_auprc"]
                                  for r in runs[v] if r["seed"] in g]))
        m = ds.mean()
        verdict = ("PROMOTE, run the full ladder and seeds 43/44"
                   if m >= 0.005 else "borderline, add seeds"
                   if m >= 0.003 else "below threshold")
        print(f"  {v:>10} vs gru: " +
              "  ".join(f"s{s} {d:+.5f}" for s, d in pairs) +
              f"   mean {m:+.5f}   -> {verdict}")
        if len(ds) > 1 and 0 < (ds > 0).sum() < len(ds):
            print(f"             sign inconsistent across seeds "
                  f"({(ds > 0).sum()}/{len(ds)}) — not evidence")

    print(f"\n  incremental rungs (what each mechanism adds):")
    for a, b, q in [("gru", "modular", "modular states alone"),
                    ("gru", "film", "context modulation alone"),
                    ("film", "mod_film", "modular on top of FiLM"),
                    ("mod_film", "cmr", "sparse routing on top")]:
        if a in means and b in means:
            print(f"    {a:>9} -> {b:<9} {means[b]-means[a]:+.5f}   {q}")

    print(f"\n\n{'#'*88}\n# WERE THE MECHANISMS ADOPTED?\n{'#'*88}")
    for v in MODELS[1:]:
        for r in runs.get(v, []):
            if "lambda_final" not in r and "mu_final" not in r:
                continue
            print(f"\n  {v} seed {r['seed']}:")
            init = args.anchor_init
            if "lambda_final" in r:
                lam = r["lambda_final"]
                print(f"    lambda {init:.3f} -> {lam:.5f}"
                      + ("   barely moved: FiLM not adopted"
                         if lam < 2 * init else ""))
            if "mu_final" in r:
                mu = r["mu_final"]
                print(f"    mu     {init:.3f} -> {mu:.5f}"
                      + ("   barely moved: routing not adopted"
                         if mu < 2 * init else ""))
            mech = r.get("mechanism", {})
            if "route_pos" in mech:
                dp = np.abs(np.array(mech["route_pos"]) -
                            np.array(mech["route_neg"])).max()
                print(f"    max routing difference positives vs negatives: "
                      f"{dp:.5f}")
                if dp < 0.01:
                    print("    -> routing is active but identical before "
                          "events and non-events:")
                    print("       a fixed policy, not a context-adaptive one.")

    common.save_result(OUT_DIR, "_cmr_summary",
                       {f"{v}__s{r['seed']}": r
                        for v, l in runs.items() for r in l})
    print(f"\n✓ Saved -> {OUT_DIR / '_cmr_summary.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=MODELS + ["all"])
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--check_determinism", action="store_true")
    ap.add_argument("--unlock_test", action="store_true")
    ap.add_argument("--fast_determinism", action="store_true",
                    help="keep seeds and seeded workers but drop the strict "
                         "algorithm flag; ~10x faster, verify first")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--suffix", default="")
    ap.add_argument("--hidden", type=int, default=None)
    ap.add_argument("--anchor_init", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no_resume", action="store_true")
    ap.add_argument("--lam", type=float, default=0.0,
                    help="weight on the future-nadir auxiliary loss. "
                         "0 reproduces the baseline exactly.")
    ap.add_argument("--nadir_delta", type=float, default=1.0,
                    help="Huber delta on the z-scored nadir target")
    ap.add_argument("--label_set", default="any",
                    choices=["any", "consensus"],
                    help="any: 1 if ANY future reading <70 (legacy). "
                         "consensus: episode onset, >=3 consecutive "
                         "readings <70 (called primary in "
                         "build_windows.py). Prevalence differs ~3x, so "
                         "the two are not comparable.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.check_determinism:
        check_determinism(args); return
    if args.collect:
        collect(args); return
    if args.model is None:
        print(f"Choose --model {{{','.join(MODELS)}}}, --collect, "
              f"or --check_determinism"); return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}"
          + ("   [fast determinism]" if args.fast_determinism else ""))
    seed_everything(args.seed, strict=not args.fast_determinism)
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any if args.label_set == "any" else bw._labels
    print(f"  endpoint: {args.label_set}  "
          f"prevalence {Y.mean(0).round(5).tolist()}")

    from stage1_classical_ml import stratified_subsample
    idx_tr = np.flatnonzero(tr)
    if args.train_cap and len(idx_tr) > args.train_cap:
        idx_tr = idx_tr[stratified_subsample(Y[idx_tr, 1].astype(int),
                                             args.train_cap, args.seed)]
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    print(f"  train {len(idx_tr):,} (shared) val {len(idx_va):,} "
          f"test {len(idx_te):,}")

    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else [args.seed])
    for v in (MODELS if args.model == "all" else [args.model]):
        for sd in seeds:
            tag = f"{v}__s{sd}" + (f"__{args.suffix}" if args.suffix else "")
            if args.lam > 0:
                tag += f"__lam{args.lam:g}"
            if args.label_set != "any":
                tag += f"__{args.label_set}"
            if (OUT_DIR / f"{tag}.json").exists() and not args.force:
                print(f"\n[skip] {tag} already done")
                continue
            try:
                run_one(v, bw, Y, idx_tr, idx_va, idx_te, device, args, sd)
            except Exception as e:
                print(f"  !! {tag} FAILED: {type(e).__name__}: {e}")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    collect(args)


if __name__ == "__main__":
    main()
