"""
tune_protocol.py  --  pre-registration for the GRU / TSL-GRU tuning study
==========================================================================

RUN THIS FIRST. It writes the search space, the seed plan, and the
decision rule to a JSON before any model is trained, so the analysis
cannot be adjusted after the numbers are seen.

DESIGN
------
  space      identical for both arms: lr, dropout, batch, weight decay
  hidden     FIXED at 64, layers FIXED at 2, NOT searched
  trials     8 per arm, same 8 sampled configurations for both
  epochs     20 in the search phase, patience 5
  select     per-subject mean validation AUPRC (the project's primary
             metric), NOT the pooled value cmr_pilot selects epochs on
  confirm    each arm's winning config rerun on seeds 43, 44, 45
  compare    the confirmation seeds only
  test       untouched

WHY HIDDEN SIZE IS EXCLUDED
---------------------------
The whole claim is 61% fewer parameters at equal accuracy, measured at
matched width. Searching hidden size would let TSL-GRU grow to 128 and
the comparison would no longer be about the cell.

WHY CONFIRMATION SEEDS ARE NOT OPTIONAL
---------------------------------------
Taking the best of 8 trials means taking the maximum of 8 noisy draws.
The 5-seed baseline range is 0.00207, so best-of-8 sits roughly that far
above typical EVEN IF THE TWO ARCHITECTURES ARE IDENTICAL. A difference
measured on the same runs used for selection is therefore not evidence.
Re-running the chosen configs on fresh seeds removes that bias: selection
noise does not survive re-randomisation, a real effect does.

The same argument is why both arms get the SAME 8 configurations. Giving
one arm a luckier draw of the space would bias the comparison before
training starts.

DECISION RULE, FIXED IN ADVANCE
-------------------------------
On the three confirmation seeds, paired by seed:

  mean delta >= +0.003   TSL-GRU is better; report it
  |mean delta| < 0.003   equivalent; the existing efficiency claim stands
  mean delta <= -0.003   TSL-GRU is worse under tuning; report that too

+0.003 is the threshold already used for CMR. The measured noise floor is
0.00207, so nothing smaller is interpretable.

THIS STUDY CANNOT PRODUCE "TSL BEATS GRU" BY ITSELF
---------------------------------------------------
If tuning lifts both arms equally, the conclusion is that the original
fixed settings were suboptimal for both, not that either architecture
won. That outcome is reported, not discarded.

    python tune_protocol.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime

import numpy as np

PROTOCOL = {
    "study": "gru_vs_tsl_hyperparameter_tuning",
    "arms": ["gru_ta", "tsl_gru"],
    "fixed": {
        "hidden": 64,
        "layers": 2,
        "optimizer": "adam",
        "scheduler": "ReduceLROnPlateau(mode=max, factor=0.5, patience=3)",
        "loss": "weighted_bce",
        "endpoint": "any",
        "tag": "stride6",
        "search_epochs": 20,
        "search_patience": 5,
        "confirm_epochs": 40,
        "confirm_patience": 6,
    },
    "space": {
        "lr":           {"type": "loguniform", "low": 3e-4, "high": 3e-3},
        "dropout":      {"type": "uniform",    "low": 0.0,  "high": 0.4},
        "batch":        {"type": "choice",     "values": [256, 512, 1024]},
        "weight_decay": {"type": "loguniform", "low": 1e-6, "high": 1e-3},
    },
    "n_trials": 8,
    "trial_0_is_incumbent": True,
    "shared_configs": True,
    "search_seed": 42,
    "sampler_seed": 20260831,
    "confirmation_seeds": [43, 44, 45],
    "selection_metric": "per_subject_mean_auprc_validation",
    "decision_threshold": 0.003,
    "noise_floor_measured": 0.00207,
    "test_set": "SEALED -- not used in this study",
    "outcomes": {
        "tsl_better": "mean paired delta >= +0.003 on confirmation seeds",
        "equivalent": "|mean paired delta| < 0.003; existing efficiency "
                      "claim stands unchanged",
        "tsl_worse": "mean paired delta <= -0.003; reported as such",
        "both_improve": "if tuning lifts both arms, the finding is that the "
                        "original fixed settings were suboptimal for both, "
                        "not that either architecture won",
    },
    "reporting_commitment": "All 16 search trials and all 6 confirmation "
                            "runs are reported regardless of outcome. No "
                            "run is dropped after the fact.",
}


INCUMBENT = {"lr": 1e-3, "dropout": 0.2, "batch": 512, "weight_decay": 0.0,
             "trial": 0, "note": "current fixed settings, included so the "
                                 "search can be judged against the status quo"}


def sample_configs(space, n, seed):
    """
    The SAME n configurations are used for both arms.

    Trial 0 is the INCUMBENT (the settings every result so far used).
    Without it there is no way to tell whether the best of the search
    actually beats what the project already had -- the search would only
    show which random draw won among itself.
    """
    rng = np.random.default_rng(seed)
    out = [dict(INCUMBENT)]
    for i in range(1, n):
        c = {}
        for k, s in space.items():
            if s["type"] == "loguniform":
                c[k] = float(np.exp(rng.uniform(np.log(s["low"]),
                                                np.log(s["high"]))))
            elif s["type"] == "uniform":
                c[k] = float(rng.uniform(s["low"], s["high"]))
            elif s["type"] == "choice":
                c[k] = s["values"][int(rng.integers(len(s["values"])))]
        c["trial"] = i
        out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default="results/tuning/protocol.json")
    a = ap.parse_args()

    cfgs = sample_configs(PROTOCOL["space"], PROTOCOL["n_trials"],
                          PROTOCOL["sampler_seed"])
    doc = dict(PROTOCOL)
    doc["configs"] = cfgs
    doc["written_at"] = datetime.now().isoformat(timespec="seconds")
    body = json.dumps(doc, sort_keys=True).encode()
    doc["sha256"] = hashlib.sha256(body).hexdigest()[:16]

    print("=" * 78)
    print("PRE-REGISTERED PROTOCOL")
    print("=" * 78)
    print(f"  arms        {doc['arms']}")
    print(f"  fixed       hidden {doc['fixed']['hidden']}, "
          f"layers {doc['fixed']['layers']}  (NOT searched)")
    print(f"  trials      {doc['n_trials']} per arm, identical configs")
    print(f"  select on   {doc['selection_metric']}")
    print(f"  confirm on  seeds {doc['confirmation_seeds']}")
    print(f"  decide at   +/-{doc['decision_threshold']} "
          f"(noise floor {doc['noise_floor_measured']})")
    print(f"  test set    {doc['test_set']}")
    print(f"  hash        {doc['sha256']}")

    print(f"\n{'trial':>5} {'lr':>10} {'dropout':>8} {'batch':>6} {'wd':>10}"
          f"   {'note':<12}")
    for c in cfgs:
        print(f"{c['trial']:>5} {c['lr']:>10.2e} {c['dropout']:>8.3f} "
              f"{c['batch']:>6} {c['weight_decay']:>10.2e}"
              f"   {'INCUMBENT' if c['trial'] == 0 else ''}")

    print(f"\n  compute: 16 search runs x 20 epochs + 6 confirmation runs")
    print(f"  x 40 epochs. At roughly 1-2 h per full run that is 2-4 days.")

    if a.write:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        if os.path.exists(a.out):
            print(f"\n  {a.out} already exists -- NOT overwritten. "
                  f"A pre-registration that can be rewritten after seeing "
                  f"results is not a pre-registration.")
            return
        with open(a.out, "w") as f:
            json.dump(doc, f, indent=2)
        print(f"\n  written -> {a.out}")
        print("  Commit this file before the first training run.")
    else:
        print("\n  (dry run -- pass --write to save)")


if __name__ == "__main__":
    main()
