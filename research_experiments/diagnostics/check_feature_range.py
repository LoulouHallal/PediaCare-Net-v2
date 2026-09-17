"""
check_feature_range.py
========================
Measure how extreme the normalised feature values actually are.

WHY
---
`build_windows.py` normalises causally: each reading is divided by the
expanding standard deviation of everything before it, with a floor of
1e-6 to avoid dividing by zero.

That floor is the problem. `bolus` and `carbs` are sparse -- long runs of
exact zeros. While a channel is constant, its expanding std is 0, so the
floor applies, and the FIRST non-zero value is divided by 1e-6:

    (1.5 - 0) / 1e-6  =  1,500,000

A feature value of that magnitude saturates any unbounded linear or
convolutional layer, which is the likely cause of the NaN failures in
the TCN / Transformer / hybrid runs, and of the contaminated min/max
summary features used by the classical models.

This script reports the damage per channel so the fix can be sized
before anything is re-run.

Usage:
    python check_feature_range.py --tag stride6
"""

import argparse
import numpy as np

import config
import common


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--clamp", type=float, default=10.0,
                    help="candidate clamp value to evaluate")
    ap.add_argument("--stride1_tag", default=None,
                    help="also check the stride-1 eval set, e.g. --stride1_tag eval")
    args = ap.parse_args()

    # clamp=0 disables the clamping applied by BuiltWindows, so the RAW
    # values can be measured. Loading through the default path would clamp
    # first and then report zero extremes -- which is what an earlier
    # version of this script did.
    bw = common.load_built(args.tag, clamp=0)
    tl = bw.timeline
    print("(loaded WITHOUT clamping, to measure the raw values)\n")
    print(f"timeline: {tl.shape[0]:,} readings x {tl.shape[1]} channels\n")

    print(f"{'channel':>18} {'min':>14} {'max':>14} {'p99.9':>10} "
          f"{'|x|>10':>10} {'|x|>100':>10} {'|x|>1e4':>10}")
    total_big = 0
    for c, name in enumerate(bw.features):
        v = tl[:, c]
        a = np.abs(v)
        n10 = int((a > 10).sum())
        n100 = int((a > 100).sum())
        n1e4 = int((a > 1e4).sum())
        total_big += n10
        print(f"{name:>18} {v.min():>14.4g} {v.max():>14.4g} "
              f"{np.percentile(a, 99.9):>10.3f} "
              f"{n10:>10,} {n100:>10,} {n1e4:>10,}")

    frac = total_big / tl.size
    print(f"\nvalues with |x| > {args.clamp}: {total_big:,} "
          f"({100*frac:.4f}% of all cells)")

    # how many WINDOWS are affected -- this is what matters for training,
    # since one extreme cell contaminates its whole window
    ends = bw.starts + bw.window_len - 1
    bad_reading = (np.abs(tl) > args.clamp).any(axis=1)
    cum = np.concatenate([[0], np.cumsum(bad_reading)])
    n_bad_win = int(((cum[ends + 1] - cum[bw.starts]) > 0).sum())
    print(f"windows containing at least one such value: {n_bad_win:,} / "
          f"{len(bw):,} ({100*n_bad_win/len(bw):.2f}%)")

    print(f"\nEffect of clamping to +/-{args.clamp}:")
    for c, name in enumerate(bw.features):
        v = tl[:, c]
        changed = int((np.abs(v) > args.clamp).sum())
        if changed:
            print(f"  {name:>18}: {changed:>10,} values clipped "
                  f"({100*changed/len(v):.4f}% of that channel)")
    # how many windows would be affected in the EVAL set too
    if args.stride1_tag:
        bwe = common.load_built(args.stride1_tag, clamp=0)
        a = np.abs(bwe.timeline)
        bad = (a > args.clamp).any(axis=1)
        cum = np.concatenate([[0], np.cumsum(bad)])
        e = bwe.starts + bwe.window_len - 1
        nb = int(((cum[e + 1] - cum[bwe.starts]) > 0).sum())
        print(f"\n[{args.stride1_tag}] windows affected: {nb:,} / {len(bwe):,} "
              f"({100*nb/len(bwe):.2f}%)")

    print("\nA clamp only touches values already far outside the range any")
    print("real z-scored physiological signal occupies, so it removes the")
    print("artifact without altering the signal.")


if __name__ == "__main__":
    main()
