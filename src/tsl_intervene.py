"""
tsl_intervene.py
==================
Intervention test on a TRAINED TSL-GRU. No retraining.

THE QUESTION THIS ANSWERS
-------------------------
Seed 42 gave D - C = +0.005 at every horizon, and kappa grew from 0.10 to
0.60 rather than being driven out. So the trajectory conditioning was
reinforced during training. But sd(lambda_t) over time is only 0.00088 --
lambda moves between roughly 0.336 and 0.339.

Running more seeds tells us whether D beats C *consistently*. It does not
tell us whether the *temporal variation in lambda* is what D's advantage
depends on. D could be winning simply because it happened to settle at a
better average retention level than C did, in which case the mechanism is
decorative and a tuned constant would do the same job.

Three evaluations of the SAME checkpoint separate those explanations:

    normal     lambda_t as learned
    constant   lambda_t replaced by its mean, per hidden unit
               -> removes temporal variation, HOLDS THE AVERAGE FIXED
    shuffled   trajectory descriptor s_t permuted across time
               -> keeps lambda's distribution, destroys its alignment
                  with the input

Every weight is identical in all three. Only the retention signal changes.

    normal > constant  =>  the temporal variation matters, not just the level
    normal > shuffled  =>  the variation must be aligned with the trajectory
    normal ~ constant  =>  D's advantage is a favourable constant timescale,
                           and the conditioning is not doing the work

The constant is computed on VALIDATION, never on test, so the control is
not defined using the data it is scored on.

Deltas carry paired subject-level bootstrap CIs: the three conditions run
on the identical subjects, so pairing removes subject-sampling noise and
makes small differences resolvable.

Usage:
    python tsl_intervene.py --seed 42
    python tsl_intervene.py --seed 42 --n_boot 10000
"""

import json
import time
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

import config
import common
from stage2_deep import WindowDataset
from ta_gru import attach_ta
from tsl_gru import TSLGRU, OUT_DIR as TSL_DIR

HORIZONS = config.HORIZONS
CONDITIONS = ["normal", "constant", "shuffled"]


@torch.no_grad()
def mean_lambda(model, bw, idx, labels, device, n=50000, batch=1024):
    """Per-hidden-unit average of lambda, computed on VALIDATION."""
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx[:n], labels), batch_size=batch,
                    shuffle=False, num_workers=0)
    tot, cnt = None, 0
    for xb, _ in dl:
        _, lam = model(xb.to(device), return_lambda=True)
        s = lam.sum(dim=(0, 1))                 # sum over batch and time
        tot = s if tot is None else tot + s
        cnt += lam.shape[0] * lam.shape[1]
    return tot / cnt


@torch.no_grad()
def predict(model, bw, idx, labels, device, batch=2048, workers=2, **kw):
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    return np.concatenate([model(xb.to(device), **kw).cpu().numpy()
                           for xb, _ in dl]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="tsl_gru")
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--rho", type=float, default=0.9)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    args = ap.parse_args()

    name = f"{args.variant}__weighted_bce__none__{args.labels}"
    if args.seed != config.SPLIT_SEED:
        name += f"__s{args.seed}"
    ckpt_path = TSL_DIR / f"{name}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint at {ckpt_path}. "
                                f"Run tsl_gru.py --model {args.variant} first.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    t0 = time.time()

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)
    meta_te = bw.meta[idx_te]
    meta_va = bw.meta[idx_va]

    model = TSLGRU(args.variant, in_ch=bw.timeline.shape[1],
                   hidden=args.hidden, rho=args.rho).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
    model.eval()
    print(f"Loaded {name}  (kappa = "
          f"{float(model.cells[0].kappa.detach().cpu()):+.4f})")

    lam_c = mean_lambda(model, bw, idx_va, Y, device)
    print(f"Constant lambda from VALIDATION: mean {float(lam_c.mean()):.4f}  "
          f"min {float(lam_c.min()):.4f}  max {float(lam_c.max()):.4f}")

    g = torch.Generator().manual_seed(args.seed)
    perm = torch.randperm(bw.window_len, generator=g).to(device)

    kwargs = {"normal": {},
              "constant": {"lam_const": lam_c},
              "shuffled": {"shuffle_s": perm}}

    print(f"\nScoring VALIDATION ({len(idx_va):,}) and test "
          f"({len(idx_te):,}) under each condition...")
    probs, val_probs = {}, {}
    for cond in CONDITIONS:
        print(f"  {cond}...")
        val_probs[cond] = predict(model, bw, idx_va, Y, device,
                                  batch=args.batch, workers=args.workers,
                                  **kwargs[cond])
        probs[cond] = predict(model, bw, idx_te, Y, device,
                              batch=args.batch, workers=args.workers,
                              **kwargs[cond])

    # ---- evaluate ----------------------------------------------------------
    res = {"checkpoint": str(ckpt_path), "seed": args.seed,
           "kappa": float(model.cells[0].kappa.detach().cpu()),
           "lambda_const_mean": float(lam_c.mean()), "horizons": {}}

    print(f"\n{'#'*96}")
    print("# INTERVENTION ON A TRAINED CHECKPOINT — identical weights, "
          "different retention signal")
    print(f"{'#'*96}")
    for j, h in enumerate(HORIZONS):
        yv = Y[idx_va, j].astype(int)
        yt = Y[idx_te, j].astype(int)
        entry, entry_val = {}, {}
        print(f"\nh={h} min")
        print(f"  {'condition':>10} {'VAL AUPRC':>10} {'test AUPRC':>10} "
              f"{'VAL rec':>9} {'test rec':>9}")
        for cond in CONDITIONS:
            # threshold re-selected on validation per condition: the score
            # distribution shifts under intervention, so reusing one
            # threshold would confound calibration with discrimination
            thr, _ = common.find_threshold(yv, val_probs[cond][:, j])
            ev = common.evaluate(yt, probs[cond][:, j], meta_te, thr)
            ev["threshold"] = float(thr)
            evv = common.evaluate(yv, val_probs[cond][:, j], meta_va, thr)
            evv["threshold"] = float(thr)
            entry[cond] = ev
            entry_val[cond] = evv
            m, mv = ev["per_subject_mean"], evv["per_subject_mean"]
            print(f"  {cond:>10} {mv['auprc']:>10.4f} {m['auprc']:>10.4f} "
                  f"{mv['recall']:>9.4f} {m['recall']:>9.4f}")

        print(f"  {'-'*66}")
        for other in ["constant", "shuffled"]:
            d = common.paired_delta(entry[other]["per_subject"],
                                    entry["normal"]["per_subject"],
                                    n_boot=args.n_boot)
            dv = common.paired_delta(entry_val[other]["per_subject"],
                                     entry_val["normal"]["per_subject"],
                                     n_boot=args.n_boot)
            a, av = d["auprc"], dv["auprc"]
            sig = "*" if (a["lo"] > 0 or a["hi"] < 0) else " "
            sigv = "*" if (av["lo"] > 0 or av["hi"] < 0) else " "
            verdict = ("temporal variation MATTERS" if av["lo"] > 0 else
                       "no evidence it matters" if av["hi"] > 0 else
                       "intervention IMPROVED the model")
            print(f"  normal - {other:<9}:")
            print(f"      VAL  dAUPRC {av['delta']:+.4f} "
                  f"[{av['lo']:+.4f}, {av['hi']:+.4f}]{sigv}  -> {verdict}")
            print(f"      test dAUPRC {a['delta']:+.4f} "
                  f"[{a['lo']:+.4f}, {a['hi']:+.4f}]{sig}")
            entry[f"delta_normal_minus_{other}"] = d
            entry_val[f"delta_normal_minus_{other}"] = dv
        def _strip(dd):
            return {k: ({kk: vv for kk, vv in v.items() if kk != "per_subject"}
                        if isinstance(v, dict) and "per_subject" in v else v)
                    for k, v in dd.items()}
        res["horizons"][str(h)] = _strip(entry)
        res["horizons"][str(h)]["_validation"] = _strip(entry_val)

    print(f"\n{'#'*96}")
    print("# READING THIS")
    print(f"{'#'*96}")
    print("""
  normal > constant, CI excludes zero
      the temporal variation in lambda is doing real work; a tuned
      constant retention rate would not reproduce it. Proceed to seeds
      43 and 44.

  normal ~ constant
      D's advantage over C is a favourable average retention level, not
      the conditioning. Running more seeds would not change that, and the
      honest claim becomes the parameter reduction rather than the
      mechanism.

  normal > shuffled but normal ~ constant
      lambda's variation matters only through its distribution, not its
      alignment with the trajectory -- weaker than the intended claim.
""")
    res["minutes"] = (time.time() - t0) / 60
    common.save_result(TSL_DIR, f"_intervention_{name}", res)
    print(f"✓ Saved -> {TSL_DIR / ('_intervention_' + name + '.json')}  "
          f"({res['minutes']:.1f} min)")


if __name__ == "__main__":
    main()
