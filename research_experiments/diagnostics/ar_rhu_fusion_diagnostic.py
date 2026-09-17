"""
ar_rhu_fusion_diagnostic.py
=============================
Did AR-RHU learn useful complementary representations that our
hand-designed union rule then combined badly?

WHY ASK
-------
AR-RHU's mechanism did not collapse. gamma rose from 0.90 to 0.9807, the
two states became almost orthogonal (cos -0.155 -> -0.003), and when the
reactive branch was quiet the anticipatory branch separated true events
from non-events by 8.8x (0.598 against 0.068). That is the first
mechanism in this project the optimiser reinforced rather than suppressed.

Yet the fused output was slightly WORSE than a plain GRU at every horizon
(-0.0045 to -0.0051).

Exactly one component of the architecture was never tested: the fusion

    p = 1 - (1 - q^R)(1 - q^A)

That rule assumes q^R and q^A behave as probabilities of complementary
independent risks. They are trained under weighted BCE with pos_weight up
to 23, which produces cost-sensitive risk scores rather than calibrated
probabilities -- so there is no reason the noisy-OR is the right
combination. The diagnostics also show the branches operating at very
different scales (q^A 0.821/0.084 against q^R 0.550/0.018), which a fixed
rule cannot reconcile.

WHAT IS COMPARED
----------------
    reactive only        q^R
    anticipatory only    q^A
    fixed union          1 - (1 - q^R)(1 - q^A)      the current output
    learned fusion       sigmoid(b + w_R logit(q^R) + w_A logit(q^A))

The learned fusion is fit on VALIDATION subjects only, then frozen and
applied to the 38 test subjects. Nothing is fitted on test. Only three
parameters per horizon are learned, so this cannot rescue the model by
overfitting -- it can only reveal whether the information was already
there and the fixed rule was discarding it.

READING THE RESULT
------------------
    learned fusion beats the fixed union by >= 0.005
        the recurrent decomposition worked and the output composition was
        wrong; redesign the head and retrain

    learned fusion ~ fixed union
        the two branches carry the same discriminative information however
        they are combined, and AR-RHU stops here

Usage:
    python ar_rhu_fusion_diagnostic.py --seed 42
"""

import json
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import config
import common
from stage2_deep import WindowDataset
from ta_gru import attach_ta
from absolute_state import attach_absolute
from ar_rhu import ARRHU, OUT_DIR as AR_DIR, CH_REACTIVE, CH_ANTICIP

HORIZONS = config.HORIZONS
EPS = 1e-6


@torch.no_grad()
def branch_scores(model, bw, idx, labels, device, batch=2048, workers=2):
    """
    Recover q^R and q^A separately. The saved probability file holds only
    the fused p, so the branches have to be recomputed from the
    checkpoint -- no training, one forward pass.
    """
    model.eval()
    dl = DataLoader(WindowDataset(bw, idx, labels), batch_size=batch,
                    shuffle=False, num_workers=workers)
    QR, QA = [], []
    for xb, _ in dl:
        _, d = model(xb.to(device), return_diag=True)
        QR.append(d["qR"].cpu().numpy())
        QA.append(d["qA"].cpu().numpy())
    return (np.concatenate(QR).astype(np.float32),
            np.concatenate(QA).astype(np.float32))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    args = ap.parse_args()

    ck_path = AR_DIR / f"ar_rhu__s{args.seed}.pt"
    if not ck_path.exists():
        print(f"  missing checkpoint {ck_path}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    attach_ta(bw)
    attach_absolute(bw)
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)

    model = ARRHU(args.hidden).to(device)
    model.load_state_dict(torch.load(ck_path, map_location=device)["model_state"])
    print(f"Loaded ar_rhu__s{args.seed}  (gamma {float(model.gamma()):.4f})")

    print(f"\nrecovering branch scores ({len(idx_va):,} val + "
          f"{len(idx_te):,} test windows)...")
    qR_va, qA_va = branch_scores(model, bw, idx_va, Y, device,
                                 workers=args.workers)
    qR_te, qA_te = branch_scores(model, bw, idx_te, Y, device,
                                 workers=args.workers)

    meta_te = bw.meta[idx_te]
    results = {"seed": args.seed, "gamma": float(model.gamma()), "horizons": {}}

    print(f"\n{'#'*94}")
    print("# AR-RHU FUSION DIAGNOSTIC")
    print("#   is the hand-designed union rule discarding what the branches learned?")
    print(f"{'#'*94}")

    for j, h in enumerate(HORIZONS):
        yv = Y[idx_va, j].astype(int)
        yt = Y[idx_te, j].astype(int)

        cand = {
            "reactive only   q^R": (qR_va[:, j], qR_te[:, j]),
            "anticipatory    q^A": (qA_va[:, j], qA_te[:, j]),
            "fixed union (current)": (
                1 - (1 - qR_va[:, j]) * (1 - qA_va[:, j]),
                1 - (1 - qR_te[:, j]) * (1 - qA_te[:, j])),
        }

        # learned fusion: three parameters, fit on VALIDATION only
        Xv = np.column_stack([logit(qR_va[:, j]), logit(qA_va[:, j])])
        Xt = np.column_stack([logit(qR_te[:, j]), logit(qA_te[:, j])])
        lr = LogisticRegression(max_iter=2000).fit(Xv, yv)
        cand["learned fusion"] = (lr.predict_proba(Xv)[:, 1],
                                  lr.predict_proba(Xt)[:, 1])

        print(f"\nh={h} min   (test base rate {yt.mean():.4f})")
        print(f"  {'combination':>22} {'AUPRC':>9} {'AUROC':>9} {'PPV':>9} "
              f"{'Recall':>9}")
        entry, per_subj = {}, {}
        for name, (pv, pt) in cand.items():
            thr, _ = common.find_threshold(yv, pv)
            ev = common.evaluate(yt, pt, meta_te, thr)
            m = ev["per_subject_mean"]
            print(f"  {name:>22} {m['auprc']:>9.4f} {m['auroc']:>9.4f} "
                  f"{m['ppv']:>9.4f} {m['recall']:>9.4f}")
            entry[name] = {k: float(m[k]) for k in
                           ["auroc", "auprc", "ppv", "recall", "f1"]}
            per_subj[name] = ev["per_subject"]

        d = common.paired_delta(per_subj["fixed union (current)"],
                                per_subj["learned fusion"],
                                n_boot=args.n_boot)["auprc"]
        sig = "*" if (d["lo"] > 0 or d["hi"] < 0) else " "
        print(f"  {'-'*62}")
        print(f"  learned fusion - fixed union: {d['delta']:+.4f} "
              f"[{d['lo']:+.4f}, {d['hi']:+.4f}]{sig}")
        print(f"  fitted weights: w_R {lr.coef_[0][0]:+.3f}  "
              f"w_A {lr.coef_[0][1]:+.3f}  b {lr.intercept_[0]:+.3f}")

        entry["delta_learned_minus_fixed"] = d
        entry["weights"] = {"w_R": float(lr.coef_[0][0]),
                            "w_A": float(lr.coef_[0][1]),
                            "b": float(lr.intercept_[0])}
        results["horizons"][str(h)] = entry

    print(f"\n{'#'*94}\nREADING THIS\n{'#'*94}")
    d15 = results["horizons"]["15"]["delta_learned_minus_fixed"]
    gains = [results["horizons"][str(h)]["delta_learned_minus_fixed"]["delta"]
             for h in HORIZONS]
    mean_gain = float(np.mean(gains))
    print(f"\n  mean gain from learned fusion across horizons: {mean_gain:+.4f}")

    # A calibration on synthetic data showed that even two FULLY redundant
    # branches yield about +0.005 from learned fusion, simply because three
    # free parameters fit the validation set slightly better than a fixed
    # rule. The threshold for "the fusion was the problem" is therefore set
    # above that floor rather than at it.
    if mean_gain >= 0.010 and d15["lo"] > 0:
        print("\n  -> The branches carry more information than the fixed union")
        print("     rule extracts. The recurrent decomposition worked and the")
        print("     output composition was the weak point. Worth redesigning")
        print("     the head with a learned fusion and retraining.")
    elif mean_gain >= 0.005:
        print("\n  -> A gain of roughly the size two REDUNDANT branches would")
        print("     produce from three extra free parameters (calibrated at")
        print("     ~+0.005 on synthetic data). Not evidence that the fusion")
        print("     rule was the problem.")
    else:
        print("\n  -> The fixed union is not the problem. However the two")
        print("     branches are combined, they carry the same discriminative")
        print("     information -- the anticipatory signal is real but")
        print("     redundant with the reactive one. AR-RHU stops here.")

    # is either branch alone better than the fusion?
    best_alone = max(
        max(results["horizons"][str(h)]["reactive only   q^R"]["auprc"],
            results["horizons"][str(h)]["anticipatory    q^A"]["auprc"])
        for h in [15])
    fused15 = results["horizons"]["15"]["fixed union (current)"]["auprc"]
    if best_alone > fused15:
        print(f"\n  NOTE: at h=15 a single branch alone ({best_alone:.4f}) beats")
        print(f"  the fusion ({fused15:.4f}). Combining them is actively")
        print(f"  hurting, which points at the fusion rather than the states.")

    common.save_result(AR_DIR, f"_fusion_diagnostic_s{args.seed}", results)
    print(f"\n✓ Saved -> {AR_DIR / f'_fusion_diagnostic_s{args.seed}.json'}")


if __name__ == "__main__":
    main()
