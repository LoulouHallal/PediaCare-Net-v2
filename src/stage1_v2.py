"""
stage1_v2.py
==============
RQ2 stage 1 on the REBUILT dataset (build_windows.py output).

Differences from the original stage1_classical_ml.py:

  * reads the compact timeline format via common.load_built()
  * sweeps BOTH label definitions in the same run:
        consensus -- episode ONSET within the horizon (primary)
        any       -- ANY reading < 70 within the horizon (legacy)
    so "does the label definition matter more than the model?" is
    answered by one table rather than two incomparable experiments
  * caches summary features to disk (computing them for 2.76M windows
    takes minutes and is identical across every model and balancing arm)
  * caps training rows, because RF on 510k rows already took 41 min and
    the rebuilt training split is 1.9M

WHY THE TRAINING CAP
--------------------
The rebuilt dataset has 3.8x more windows, but they are heavily
overlapping views of the same 244 children -- at stride 6 consecutive
windows share 54 of 60 readings. More rows therefore buy far less than
the row count suggests, while costing linearly in fit time. The cap is a
stratified subsample (prevalence preserved) and is recorded in every
result. Raise it with --train_cap 0 to use everything.

Usage:
    python stage1_v2.py --model rf --balance none --labels both
    python stage1_v2.py --model rf --balance all  --labels consensus
    python stage1_v2.py --collect
"""

import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path

import config
import common
import balancing as B
from stage1_classical_ml import (build_model, NEEDS_SCALING,
                                 SUBSAMPLE_DEFAULT, stratified_subsample,
                                 MODELS, BALANCE)
from imbalance_ensembles import ENSEMBLE_MODELS, check_no_double_balancing

HORIZONS = config.HORIZONS
LABEL_SETS = ["consensus", "any"]
OUT_DIR = config.RQ2_STAGES["stage1_ml"] / "v2_rebuilt"


# ─── FEATURE CACHE ────────────────────────────────────────────────────────────

def cached_features(bw, tag, batch=200_000, verbose=True):
    """
    Summary features for every window, cached to data_derived.

    Windows are materialised in batches: the full tensor would be
    2.76M x 60 x 5 floats (~3.3 GB), while one batch is ~240 MB.
    """
    path = config.DATA_DERIVED / f"features_{tag}.npy"
    if path.exists():
        if verbose:
            print(f"Loading cached features -> {path.name}")
        return np.load(path, mmap_mode="r")

    if verbose:
        print(f"Computing summary features for {len(bw):,} windows "
              f"(batched, this runs once)...")
    chunks = []
    for b in range(0, len(bw), batch):
        idx = np.arange(b, min(b + batch, len(bw)))
        chunks.append(common.summary_features(bw.windows(idx),
                                              n_ch=bw.n_channels))
        if verbose:
            print(f"  {min(b + batch, len(bw)):>10,} / {len(bw):,}")
    F = np.concatenate(chunks).astype(np.float32)
    del chunks
    np.save(path, F)
    if verbose:
        print(f"Saved -> {path} ({F.nbytes/1e6:.0f} MB)")
    return F


# ─── ONE RUN ──────────────────────────────────────────────────────────────────

def run_one(model_name, method, label_set, F, bw, masks, args):
    tr, va, te = masks
    Y = bw._labels if label_set == "consensus" else bw._labels_any
    meta_te = bw.meta[te]

    # ensembles with internal resampling must not also be fed resampled data
    check_no_double_balancing(model_name, method)
    print(f"\n{'='*78}\n{model_name} | balance={method} | labels={label_set}\n{'='*78}")
    t0 = time.time()
    res = {"model": model_name, "balance": method, "label_set": label_set,
           "seed": args.seed, "horizons": {}, "n_features": int(F.shape[1]),
           "dataset_tag": args.tag}

    F_va = np.asarray(F[va])
    F_te = np.asarray(F[te])
    idx_tr = np.flatnonzero(tr)
    # Saved so that downstream analyses (operating-point sweeps, event-level
    # metrics) can reuse the predictions instead of retraining every model.
    saved_val = np.zeros((int(va.sum()), len(HORIZONS)), dtype=np.float32)
    saved_test = np.zeros((int(te.sum()), len(HORIZONS)), dtype=np.float32)

    for j, h in enumerate(HORIZONS):
        y_tr_full = Y[tr, j].astype(int)

        # cap BEFORE balancing so the cap does not undo the resampling
        if args.train_cap and len(y_tr_full) > args.train_cap:
            keep = stratified_subsample(y_tr_full, args.train_cap, args.seed)
            sel = idx_tr[keep]
            cap_note = {"to": int(len(keep)),
                        "note": "stratified subsample, prevalence preserved"}
        else:
            sel = idx_tr
            cap_note = None
        F_tr = np.asarray(F[sel])
        y_tr = Y[sel, j].astype(int)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xb, yb, binfo = B.balance_features(
                F_tr, y_tr, method=method,
                sampling_strategy=args.sampling_strategy,
                seed=args.seed, split_name="train")

        n_max = args.subsample or SUBSAMPLE_DEFAULT.get(model_name)
        if n_max and len(yb) > n_max:
            k = stratified_subsample(yb, n_max, args.seed)
            Xb, yb = Xb[k], yb[k]

        Xb_f, Fva_f, Fte_f = Xb, F_va, F_te
        if model_name in NEEDS_SCALING:
            from sklearn.preprocessing import StandardScaler
            sc = StandardScaler().fit(Xb)
            Xb_f, Fva_f, Fte_f = sc.transform(Xb), sc.transform(F_va), sc.transform(F_te)

        cw = "balanced" if method == "class_weight" else None
        clf = build_model(model_name, class_weight=cw, seed=args.seed,
                          n_jobs=args.n_jobs)
        clf.fit(Xb_f, yb)

        p_va = clf.predict_proba(Fva_f)[:, 1]
        p_te = clf.predict_proba(Fte_f)[:, 1]

        saved_val[:, j] = p_va.astype(np.float32)
        saved_test[:, j] = p_te.astype(np.float32)

        thr, tinfo = common.find_threshold(Y[va, j].astype(int), p_va)
        ev = common.evaluate(Y[te, j].astype(int), p_te, meta_te, thr)
        ev["threshold"] = float(thr)
        ev["threshold_info"] = tinfo

        # SECOND evaluation at a FIXED 0.5 threshold.
        #
        # Most published class-imbalance comparisons evaluate at 0.5. Under
        # severe imbalance an untuned model predicts the majority class
        # almost always, so 0.5 gives near-zero recall -- and resampling
        # appears to produce a large improvement. That apparent gain is a
        # threshold effect, not a modelling one. Recording both operating
        # points lets us show directly that balancing helps at 0.5 and not
        # at a tuned threshold, which reconciles our result with the
        # literature rather than contradicting it.
        ev_fixed = common.evaluate(Y[te, j].astype(int), p_te, meta_te, 0.5)
        ev["fixed_threshold_0.5"] = {
            "pooled": ev_fixed["pooled"],
            "per_subject_mean": ev_fixed["per_subject_mean"],
            "per_subject": ev_fixed["per_subject"],
            "constraint": ev_fixed["constraint"],
        }
        ev["balance_info"] = binfo
        if cap_note:
            ev["balance_info"]["train_cap"] = cap_note
        ev["train_prevalence"] = float(yb.mean())
        ev["test_prevalence"] = float(Y[te, j].mean())
        res["horizons"][str(h)] = ev

        m, c = ev["per_subject_mean"], ev["constraint"]
        f5 = ev["fixed_threshold_0.5"]["per_subject_mean"]
        print(f"  h={h:>3}  n={len(yb):>8,} prev={yb.mean():.4f}  |  "
              f"AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"{c['n_meeting']}/{ev['n_subjects']}")
        print(f"        at fixed thr=0.5:  PPV {f5['ppv']:.4f}  "
              f"Rec {f5['recall']:.4f}  F1 {f5['f1']:.4f}")

        # free per-horizon arrays before the next loop
        del F_tr, Xb, Xb_f

    res["minutes"] = (time.time() - t0) / 60
    name = f"{model_name}__{method}__{label_set}"
    np.savez_compressed(OUT_DIR / f"{name}_probs.npz",
                        val=saved_val, test=saved_test,
                        idx_val=np.flatnonzero(va), idx_test=np.flatnonzero(te))
    common.save_result(OUT_DIR, name, res)
    print(f"  saved ({res['minutes']:.1f} min)")
    return res


# ─── COLLECT ──────────────────────────────────────────────────────────────────

def collect(args):
    files = sorted(OUT_DIR.glob("*__*__*.json"))
    if not files:
        print("No results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs[(r["model"], r["balance"], r["label_set"])] = r

    print(f"\n{'#'*104}")
    print("# RQ2 STAGE 1 (rebuilt data) — per-subject mean")
    print(f"{'#'*104}")
    for ls in LABEL_SETS:
        sub = {k: v for k, v in runs.items() if k[2] == ls}
        if not sub:
            continue
        print(f"\n\n### label set: {ls}")
        for h in HORIZONS:
            print(f"\nh={h} min   (test prevalence "
                  f"{list(sub.values())[0]['horizons'][str(h)]['test_prevalence']:.4f})")
            print(f"  {'model':>12} {'balance':>18} {'AUROC':>8} {'AUPRC':>8} "
                  f"{'PPV':>8} {'Recall':>8} {'F1':>8} {'constraint':>11}")
            rows = [(r["horizons"][str(h)]["per_subject_mean"]["auprc"], mo, me, r)
                    for (mo, me, _), r in sub.items() if str(h) in r["horizons"]]
            for _, mo, me, r in sorted(rows, reverse=True):
                ev = r["horizons"][str(h)]
                m, c = ev["per_subject_mean"], ev["constraint"]
                print(f"  {mo:>12} {me:>18} {m['auroc']:>8.4f} {m['auprc']:>8.4f} "
                      f"{m['ppv']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                      f"{c['n_meeting']:>3}/{ev['n_subjects']:<7}")

    # ---- label definition vs balancing: which moves the numbers more? -------
    print(f"\n\n{'#'*104}")
    print("# LABEL DEFINITION vs BALANCING — paired subject-level bootstrap")
    print(f"{'#'*104}")
    for (mo, me, ls), r in sorted(runs.items()):
        if ls != "consensus":
            continue
        other = runs.get((mo, me, "any"))
        if other is None:
            continue
        print(f"\n{mo} / {me}:  consensus -> any  (label definition changed)")
        for h in HORIZONS:
            a = r["horizons"][str(h)]["per_subject"]
            b = other["horizons"][str(h)]["per_subject"]
            d = common.paired_delta(a, b, n_boot=args.n_boot)
            parts = [f"{k}={d[k]['delta']:+.4f}"
                     f"[{d[k]['lo']:+.4f},{d[k]['hi']:+.4f}]"
                     f"{'*' if (d[k]['lo']>0 or d[k]['hi']<0) else ' '}"
                     for k in ["auprc", "f1", "ppv"]]
            print(f"  h={h:>3}: " + "  ".join(parts))

    print(f"\n\n{'#'*104}")
    print("# EFFECT OF BALANCING (within each label set) vs balance='none'")
    print(f"{'#'*104}")
    for ls in LABEL_SETS:
        for mo in sorted({m for m, _, l in runs if l == ls}):
            base = runs.get((mo, "none", ls))
            if base is None:
                continue
            for me in BALANCE:
                if me == "none" or (mo, me, ls) not in runs:
                    continue
                print(f"\n[{ls}] {mo}: none -> {me}")
                for h in HORIZONS:
                    a = base["horizons"][str(h)]["per_subject"]
                    b = runs[(mo, me, ls)]["horizons"][str(h)]["per_subject"]
                    d = common.paired_delta(a, b, n_boot=args.n_boot)
                    parts = [f"{k}={d[k]['delta']:+.4f}"
                             f"[{d[k]['lo']:+.4f},{d[k]['hi']:+.4f}]"
                             f"{'*' if (d[k]['lo']>0 or d[k]['hi']<0) else ' '}"
                             for k in ["auprc", "f1", "ppv", "recall"]]
                    print(f"  h={h:>3}: " + "  ".join(parts))
    print("\n  * = 95% CI excludes zero")

    common.save_result(OUT_DIR, "_stage1_v2_summary",
                       {f"{m}__{b}__{l}": r for (m, b, l), r in runs.items()})
    print(f"\n✓ Saved -> {OUT_DIR / '_stage1_v2_summary.json'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=MODELS + ["all", "core"])
    ap.add_argument("--balance", default="none", choices=BALANCE + ["all", "core"])
    ap.add_argument("--labels", default="both",
                    choices=LABEL_SETS + ["both"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--train_cap", type=int, default=500_000,
                    help="cap training rows (stratified). 0 = no cap.")
    ap.add_argument("--sampling_strategy", type=float, default=0.3)
    ap.add_argument("--subsample", type=int, default=None)
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args)
        return

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    masks = common.get_split(bw.meta)
    print(f"windows {len(bw):,} | channels {bw.n_channels} | "
          f"train {masks[0].sum():,} val {masks[1].sum():,} test {masks[2].sum():,}")

    F = cached_features(bw, args.tag)
    print(f"features: {F.shape}")

    models = {"all": MODELS, "core": ["rf", "svm_linear"]}.get(args.model, [args.model])
    methods = {"all": BALANCE, "core": ["none", "class_weight", "smote"]}.get(
        args.balance, [args.balance])
    label_sets = LABEL_SETS if args.labels == "both" else [args.labels]

    for ls in label_sets:
        for mo in models:
            for me in methods:
                p = OUT_DIR / f"{mo}__{me}__{ls}.json"
                if p.exists() and not args.force:
                    print(f"\n[skip] {mo}/{me}/{ls} already done")
                    continue
                try:
                    run_one(mo, me, ls, F, bw, masks, args)
                except Exception as e:
                    print(f"  !! {mo}/{me}/{ls} FAILED: {type(e).__name__}: {e}")

    collect(args)


if __name__ == "__main__":
    main()
