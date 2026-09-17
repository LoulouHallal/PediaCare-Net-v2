"""
stage1_classical_ml.py
========================
RQ2 stage 1 -- classical machine learning, evaluated BEFORE and AFTER
class balancing (RQ1).

This is the first rung of the model progression. It establishes the floor
that every later stage (DL, TCN, Transformer, hybrid) must beat to justify
its complexity.

WHAT IT DOES
------------
For each (model x balancing method) pair:
  1. Build summary features from the raw windows (common.summary_features)
  2. Balance the TRAINING set only, per horizon
  3. Fit one binary classifier per horizon
  4. Choose the threshold on the NATURAL-prevalence validation set
  5. Evaluate on test at both aggregation scales, with per-subject detail
  6. Save JSON

Then `--collect` builds the comparison table and the paired
before-vs-after bootstrap deltas.

WHY THE THRESHOLD COMES FROM UNBALANCED VALIDATION DATA
-------------------------------------------------------
A model trained on resampled data outputs probabilities calibrated to the
RESAMPLED prevalence. Its raw scores are not comparable to an unbalanced
model's. Selecting the operating point on the untouched validation set
re-anchors every model to real prevalence, which is what makes the
before/after comparison meaningful rather than an artifact of calibration.

SVM AND DATASET SIZE
--------------------
The training split is ~510,000 windows. Kernel SVM is roughly quadratic
in sample count and will not finish at that size. `svm_rbf` therefore
trains on a stratified subsample (default 30,000) and says so in its
results; `svm_linear` (LinearSVC) scales and runs on everything. Both are
reported so the subsampling caveat is visible rather than hidden.

Usage:
    python stage1_classical_ml.py --model rf --balance none
    python stage1_classical_ml.py --model all --balance all
    python stage1_classical_ml.py --collect
"""

import sys
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path

import config
import common
import balancing as B

HORIZONS = config.HORIZONS
from imbalance_ensembles import (ENSEMBLE_MODELS, NATURAL_COUNTERPARTS,
                                 COUNTERPART_OF, build_ensemble,
                                 check_no_double_balancing)

MODELS = (["logreg", "dt", "rf", "svm_linear", "svm_rbf", "xgboost"]
          + ENSEMBLE_MODELS + NATURAL_COUNTERPARTS)
BALANCE = ["none", "class_weight", "random_over", "random_under",
           "smote", "adasyn", "smote_then_under"]

OUT_DIR = config.RQ2_STAGES["stage1_ml"]


# ─── MODELS ───────────────────────────────────────────────────────────────────

def build_model(name, class_weight=None, seed=42, n_jobs=-1):
    if name in ENSEMBLE_MODELS + NATURAL_COUNTERPARTS:
        # class_weight is ignored: these resample internally, and adding a
        # weighting term on top would be a second, unattributable correction
        return build_ensemble(name, seed=seed, n_jobs=n_jobs)
    """
    class_weight is passed ONLY when the balancing method is
    'class_weight'; otherwise it stays None so that resampling and
    weighting are never applied at the same time (which would double-count
    the correction and make the arms incomparable).
    """
    if name == "logreg":
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=2000, class_weight=class_weight,
                                  random_state=seed, n_jobs=n_jobs)
    if name == "dt":
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(max_depth=12, min_samples_leaf=50,
                                      class_weight=class_weight,
                                      random_state=seed)
    if name == "rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=200, max_depth=None,
                                      min_samples_leaf=20, n_jobs=n_jobs,
                                      class_weight=class_weight,
                                      random_state=seed)
    if name == "svm_linear":
        # LinearSVC has no predict_proba; calibrate to get scores usable
        # with a probability threshold and with AUPRC.
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(C=1.0, class_weight=class_weight, random_state=seed,
                         dual="auto", max_iter=5000)
        return CalibratedClassifierCV(base, method="sigmoid", cv=3)
    if name == "svm_rbf":
        from sklearn.svm import SVC
        return SVC(C=1.0, kernel="rbf", gamma="scale", probability=True,
                   class_weight=class_weight, random_state=seed)
    if name == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError("pip install xgboost") from e
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric="aucpr", random_state=seed,
                             n_jobs=n_jobs, tree_method="hist")
    raise ValueError(f"unknown model {name}")


NEEDS_SCALING = {"logreg", "svm_linear", "svm_rbf"}
SUBSAMPLE_DEFAULT = {"svm_rbf": 30_000}


def stratified_subsample(y, n_max, seed):
    """Preserve prevalence while shrinking. Returns indices."""
    if len(y) <= n_max:
        return np.arange(len(y))
    rng = np.random.default_rng(seed)
    pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
    frac = n_max / len(y)
    keep = np.concatenate([
        rng.choice(pos, size=max(int(len(pos) * frac), 2), replace=False),
        rng.choice(neg, size=max(int(len(neg) * frac), 2), replace=False)])
    return np.sort(keep)


# ─── ONE (MODEL, BALANCE) RUN ─────────────────────────────────────────────────

def run_one(model_name, method, data, args):
    F_tr, Y_tr, F_va, Y_va, F_te, Y_te, meta_te = data
    print(f"\n{'='*78}\n{model_name}  |  balance={method}\n{'='*78}")
    t0 = time.time()

    res = {"model": model_name, "balance": method, "seed": args.seed,
           "horizons": {}, "n_features": int(F_tr.shape[1])}

    for j, h in enumerate(HORIZONS):
        y_tr = Y_tr[:, j].astype(int)
        y_va = Y_va[:, j].astype(int)
        y_te = Y_te[:, j].astype(int)

        # ---- balance TRAIN only, anchored on this horizon ----------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            Xb, yb, binfo = B.balance_features(
                F_tr, y_tr, method=method,
                sampling_strategy=args.sampling_strategy,
                seed=args.seed, split_name="train")

        n_max = args.subsample or SUBSAMPLE_DEFAULT.get(model_name)
        sub_note = None
        if n_max and len(yb) > n_max:
            keep = stratified_subsample(yb, n_max, args.seed)
            Xb, yb = Xb[keep], yb[keep]
            sub_note = {"to": int(len(yb)),
                        "note": "stratified subsample (prevalence preserved)"}

        # ---- scale if the model needs it (fit on TRAIN only) -------------
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

        # ---- threshold on NATURAL-prevalence val -------------------------
        thr, tinfo = common.find_threshold(y_va, p_va)
        ev = common.evaluate(y_te, p_te, meta_te, thr)
        ev["threshold"] = float(thr)
        ev["threshold_info"] = tinfo
        ev["balance_info"] = {k: v for k, v in binfo.items() if k != "class_weights"}
        ev["balance_info"]["class_weights"] = binfo.get("class_weights")
        if sub_note:
            ev["balance_info"]["train_subsample"] = sub_note
        res["horizons"][str(h)] = ev

        m = ev["per_subject_mean"]
        c = ev["constraint"]
        print(f"  h={h:>3}  train n={len(yb):>8,} prev={yb.mean():.4f}  |  "
              f"AUROC {m['auroc']:.4f}  AUPRC {m['auprc']:.4f}  "
              f"PPV {m['ppv']:.4f}  Rec {m['recall']:.4f}  |  "
              f"constraint {c['n_meeting']}/{ev['n_subjects']}")

    res["minutes"] = (time.time() - t0) / 60
    common.save_result(OUT_DIR, f"{model_name}__{method}", res)
    print(f"  saved -> {model_name}__{method}.json  ({res['minutes']:.1f} min)")
    return res


# ─── COLLECT ──────────────────────────────────────────────────────────────────

def collect(args):
    files = sorted(OUT_DIR.glob("*__*.json"))
    if not files:
        print("No results yet.")
        return
    runs = {}
    for f in files:
        r = json.load(open(f))
        runs[(r["model"], r["balance"])] = r

    models = sorted({m for m, _ in runs})
    methods = [b for b in BALANCE if any(b == bb for _, bb in runs)]

    print(f"\n{'#'*100}")
    print("# RQ2 STAGE 1 — CLASSICAL ML (per-subject mean)")
    print(f"{'#'*100}")
    for h in HORIZONS:
        print(f"\nh={h} min")
        print(f"  {'model':>12} {'balance':>18} {'AUROC':>8} {'AUPRC':>8} "
              f"{'PPV':>8} {'Recall':>8} {'F1':>8} {'constraint':>11}")
        rows = []
        for (mo, me), r in runs.items():
            if str(h) not in r["horizons"]:
                continue
            ev = r["horizons"][str(h)]
            rows.append((ev["per_subject_mean"]["auprc"], mo, me, ev))
        for _, mo, me, ev in sorted(rows, reverse=True):
            m, c = ev["per_subject_mean"], ev["constraint"]
            print(f"  {mo:>12} {me:>18} {m['auroc']:>8.4f} {m['auprc']:>8.4f} "
                  f"{m['ppv']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f} "
                  f"{c['n_meeting']:>3}/{ev['n_subjects']:<7}")

    # ---- paired before/after balancing deltas -------------------------------
    print(f"\n\n{'#'*100}")
    print("# EFFECT OF BALANCING — paired subject-level bootstrap vs balance='none'")
    print("#   CI excluding zero = the balancing method made a real difference")
    print(f"{'#'*100}")
    deltas = {}
    for mo in models:
        base = runs.get((mo, "none"))
        if base is None:
            continue
        for me in methods:
            if me == "none" or (mo, me) not in runs:
                continue
            deltas.setdefault(mo, {})[me] = {}
            print(f"\n{mo}  :  none -> {me}")
            for h in HORIZONS:
                a = base["horizons"][str(h)]["per_subject"]
                b = runs[(mo, me)]["horizons"][str(h)]["per_subject"]
                d = common.paired_delta(a, b, n_boot=args.n_boot)
                deltas[mo][me][str(h)] = d
                parts = []
                for k in ["auprc", "f1", "ppv", "recall"]:
                    dd = d[k]
                    star = "*" if (dd["lo"] > 0 or dd["hi"] < 0) else " "
                    parts.append(f"{k}={dd['delta']:+.4f}[{dd['lo']:+.4f},{dd['hi']:+.4f}]{star}")
                print(f"  h={h:>3}: " + "  ".join(parts))
    print("\n  * = 95% CI excludes zero")

    common.save_result(OUT_DIR, "_stage1_summary", {
        "runs": {f"{m}__{b}": r for (m, b), r in runs.items()},
        "balancing_deltas": deltas})
    print(f"\n✓ Saved -> {OUT_DIR / '_stage1_summary.json'}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf",
                    choices=MODELS + ["all", "core"])
    ap.add_argument("--balance", default="none",
                    choices=BALANCE + ["all", "core"])
    ap.add_argument("--dataset", default="metabonet")
    ap.add_argument("--sampling_strategy", type=float, default=0.3)
    ap.add_argument("--subsample", type=int, default=None,
                    help="cap training rows (stratified). Default: none, "
                         "except svm_rbf which caps at 30,000.")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--collect", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args)
        return

    X, Y, meta = common.load_windows(args.dataset)
    common.verify_split(meta)
    tr, va, te = common.get_split(meta)
    print(f"Building summary features ({X.shape[0]:,} windows)...")
    F = common.summary_features(X)
    data = (F[tr], Y[tr], F[va], Y[va], F[te], Y[te], meta[te])
    print(f"Features: {F.shape[1]} | train {tr.sum():,} | val {va.sum():,} "
          f"| test {te.sum():,} across {len(np.unique(meta[te]))} subjects")
    del X

    models = {"all": MODELS, "core": ["rf", "svm_linear"]}.get(
        args.model, [args.model])
    methods = {"all": BALANCE,
               "core": ["none", "class_weight", "smote"]}.get(
        args.balance, [args.balance])

    for mo in models:
        for me in methods:
            try:
                run_one(mo, me, data, args)
            except Exception as e:
                print(f"  !! {mo} / {me} FAILED: {type(e).__name__}: {e}")

    collect(args)


if __name__ == "__main__":
    main()
