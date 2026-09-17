"""
run_sweep.py  --  execute the pre-registered tuning study
==========================================================

Reads results/tuning/protocol.json (written by tune_protocol.py BEFORE
any training) and runs it. Nothing about the search space, the trial
count, the selection metric or the decision rule is decided here -- this
script only executes what was already fixed.

    phase 1   8 configs x 2 arms, 15 epochs, seed 42
    phase 2   each arm's best config rerun on seeds 43, 44, 45, 40 epochs
    phase 3   compare ONLY the phase-2 runs

WHY SELECTION AND MEASUREMENT USE DIFFERENT RUNS
------------------------------------------------
Taking the best of 8 trials is taking the maximum of 8 noisy draws. The
measured seed range is 0.00207, so best-of-8 sits roughly that far above
typical EVEN IF THE ARCHITECTURES ARE IDENTICAL. A difference computed
on the same runs used to pick the winner is therefore not evidence of
anything. Re-running the chosen configs on fresh seeds removes that
bias: selection noise does not survive re-randomisation, a real effect
does.

Both arms get the SAME 8 configurations, so neither gets a luckier draw
of the space.

SELECTION METRIC
----------------
Per-subject mean validation AUPRC, computed here from each trial's saved
_probs.npz. Epoch selection inside tsl_gru.py stays POOLED, unchanged
from every existing result -- only the choice of config uses per-subject.

RESUME POLICY
-------------
A trial whose .json exists is skipped. A trial interrupted mid-run is
rerun from scratch, not resumed: resuming restores optimiser state but
not sampler position, so a resumed run is not determinism-clean and
would not be comparable to the others.

    python run_sweep.py --phase 1
    python run_sweep.py --phase 1 --dry_run
    python run_sweep.py --phase 2
    python run_sweep.py --phase 3
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import config  # noqa: E402

TSL_DIR = str(config.RESULTS / "RQ2_models" / "tsl_gru")
PROTO = "../results/tuning/protocol.json"


def load_proto(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found -- run tune_protocol.py --write first")
    p = json.load(open(path))
    print(f"protocol {p.get('sha256', '?')} | {p['n_trials']} trials "
          f"| arms {p['arms']} | confirm {p['confirmation_seeds']}")
    return p


def run_name(arm, seed, suffix):
    n = f"{arm}__weighted_bce__none__any"
    if seed != 42:
        n += f"__s{seed}"
    if suffix:
        n += f"__{suffix}"
    return n


def per_subject_val_auprc(name, bw, Y, idx_va):
    """Mean over horizons of per-subject mean validation AUPRC."""
    from sklearn.metrics import average_precision_score
    f = os.path.join(TSL_DIR, f"{name}_probs.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    p, idx = z["val"], z["idx_val"]
    subj = np.asarray(bw.meta)[idx].astype(str)
    y = Y[idx]
    per_h = []
    for k in range(y.shape[1]):
        vals = []
        for s in np.unique(subj):
            m = subj == s
            yy = y[m, k].astype(int)
            if len(np.unique(yy)) > 1:
                vals.append(average_precision_score(yy, p[m, k]))
        per_h.append(float(np.mean(vals)) if vals else np.nan)
    return float(np.nanmean(per_h)), per_h


def launch(arm, cfg, seed, epochs, patience, suffix, dry):
    cmd = [sys.executable, "tsl_gru.py", "--model", arm,
           "--seed", str(seed), "--epochs", str(epochs),
           "--patience", str(patience),
           "--lr", f"{cfg['lr']:.6g}",
           "--dropout", f"{cfg['dropout']:.6g}",
           "--batch", str(cfg["batch"]),
           "--weight_decay", f"{cfg['weight_decay']:.6g}",
           "--suffix", suffix, "--force"]
    print("  $ " + " ".join(cmd))
    if dry:
        return True
    r = subprocess.run(cmd)
    return r.returncode == 0


def phase1(p, args, bw, Y, idx_va):
    print(f"\n{'='*78}\nPHASE 1 -- search, {p['n_trials']} configs x "
          f"{len(p['arms'])} arms, {args.epochs} epochs\n{'='*78}")
    for cfg in p["configs"]:
        for arm in p["arms"]:
            sfx = f"tune_t{cfg['trial']}"
            name = run_name(arm, 42, sfx)
            if os.path.exists(os.path.join(TSL_DIR, f"{name}.json")):
                print(f"\n[skip] {name} already done")
                continue
            print(f"\n--- {arm} trial {cfg['trial']}"
                  + ("  (INCUMBENT)" if cfg["trial"] == 0 else "")
                  + f"  lr {cfg['lr']:.2e} drop {cfg['dropout']:.3f} "
                    f"batch {cfg['batch']} wd {cfg['weight_decay']:.2e} ---")
            ok = launch(arm, cfg, 42, args.epochs, args.patience, sfx,
                        args.dry_run)
            if not ok:
                print(f"  FAILED -- {name} will be retried on the next run "
                      f"(no .json written)")


def rank(p, bw, Y, idx_va):
    """Per-arm ranking on per-subject validation AUPRC."""
    out = {}
    for arm in p["arms"]:
        rows = []
        for cfg in p["configs"]:
            name = run_name(arm, 42, f"tune_t{cfg['trial']}")
            r = per_subject_val_auprc(name, bw, Y, idx_va)
            if r is None:
                continue
            rows.append({"trial": cfg["trial"], "score": r[0],
                         "per_h": r[1], **{k: cfg[k] for k in
                         ("lr", "dropout", "batch", "weight_decay")}})
        out[arm] = sorted(rows, key=lambda x: -x["score"])
    return out


def phase2(p, args, bw, Y, idx_va):
    r = rank(p, bw, Y, idx_va)
    print(f"\n{'='*78}\nPHASE 2 -- confirmation on seeds "
          f"{p['confirmation_seeds']}\n{'='*78}")
    for arm in p["arms"]:
        if not r[arm]:
            sys.exit(f"no completed trials for {arm} -- finish phase 1")
        best = r[arm][0]
        cfg = next(c for c in p["configs"] if c["trial"] == best["trial"])
        print(f"\n{arm}: best is trial {best['trial']} "
              f"(val per-subject {best['score']:.5f})"
              + ("  -- the INCUMBENT" if best["trial"] == 0 else ""))
        for seed in p["confirmation_seeds"]:
            sfx = f"best_t{best['trial']}"
            name = run_name(arm, seed, sfx)
            if os.path.exists(os.path.join(TSL_DIR, f"{name}.json")):
                print(f"  [skip] {name}")
                continue
            launch(arm, cfg, seed, p["fixed"]["confirm_epochs"],
                   p["fixed"]["confirm_patience"], sfx, args.dry_run)


def phase3(p, bw, Y, idx_va):
    r = rank(p, bw, Y, idx_va)
    print(f"\n{'='*78}\nPHASE 1 RESULTS -- per-subject mean validation "
          f"AUPRC\n{'='*78}")
    for arm in p["arms"]:
        print(f"\n{arm}")
        print(f"  {'trial':>5} {'score':>9} {'lr':>10} {'drop':>7} "
              f"{'batch':>6} {'wd':>10}")
        for x in r[arm]:
            print(f"  {x['trial']:>5} {x['score']:>9.5f} {x['lr']:>10.2e} "
                  f"{x['dropout']:>7.3f} {x['batch']:>6} "
                  f"{x['weight_decay']:>10.2e}"
                  + ("   <- INCUMBENT" if x["trial"] == 0 else "")
                  + ("   <- best" if x is r[arm][0] else ""))
        inc = next((x for x in r[arm] if x["trial"] == 0), None)
        if inc and r[arm][0]["trial"] != 0:
            print(f"  best - incumbent: "
                  f"{r[arm][0]['score'] - inc['score']:+.5f}")

    print(f"\n{'='*78}\nPHASE 2 -- CONFIRMATION SEEDS (this is the "
          f"comparison)\n{'='*78}")
    conf = {}
    for arm in p["arms"]:
        if not r[arm]:
            continue
        b = r[arm][0]["trial"]
        vals = {}
        for seed in p["confirmation_seeds"]:
            name = run_name(arm, seed, f"best_t{b}")
            v = per_subject_val_auprc(name, bw, Y, idx_va)
            if v:
                vals[seed] = v[0]
        conf[arm] = vals
        if vals:
            a = np.array(list(vals.values()))
            print(f"  {arm:<10} trial {b}  " +
                  " ".join(f"s{s} {v:.5f}" for s, v in sorted(vals.items())) +
                  f"   mean {a.mean():.5f}")

    if len(conf) == 2 and all(conf.values()):
        a, b = p["arms"]
        shared = sorted(set(conf[a]) & set(conf[b]))
        if shared:
            d = np.array([conf[b][s] - conf[a][s] for s in shared])
            thr = p["decision_threshold"]
            v = ("TSL-GRU better -- report it" if d.mean() >= thr else
                 "TSL-GRU worse under tuning" if d.mean() <= -thr else
                 "equivalent -- the existing efficiency claim stands")
            print(f"\n  {b} - {a}, paired by seed: " +
                  " ".join(f"s{s}{conf[b][s]-conf[a][s]:+.5f}" for s in shared))
            print(f"  mean {d.mean():+.5f}   threshold +/-{thr}   -> {v}")
            print(f"\n  Measured noise floor {p['noise_floor_measured']}. "
                  f"Nothing smaller is interpretable.")
    else:
        print("\n  (phase 2 incomplete)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, required=True, choices=[1, 2, 3])
    ap.add_argument("--protocol", default=PROTO)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    p = load_proto(args.protocol)
    bw = common.load_built(p["fixed"]["tag"])
    tr, va, te = common.get_split(bw.meta)
    Y = bw._labels_any
    idx_va = np.flatnonzero(va)

    if args.phase == 1:
        phase1(p, args, bw, Y, idx_va)
        print("\nPhase 1 done (or partially done -- rerun to continue).")
        print("Then: python run_sweep.py --phase 3   to see the ranking")
    elif args.phase == 2:
        phase2(p, args, bw, Y, idx_va)
    else:
        phase3(p, bw, Y, idx_va)


if __name__ == "__main__":
    main()
