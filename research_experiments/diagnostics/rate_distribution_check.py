"""
rate_distribution_check.py
============================
Is CFL-limited nearest-neighbour transport sufficient for CTF-RU, or is
multi-cell transport required?

THE QUESTION
------------
The revised transport moves a fraction of the latent state one bin per
timestep:

    C_t = |v_t| * dt / dx = |v_t| * 5 / 20

Nearest-neighbour transport is only valid while C_t <= 1, which means
|v_t| <= 4 mg/dL/min. Above that the latent state would need to cross
more than one bin in a single 5-minute step, and clipping C to 1 silently
understates fast falls -- exactly the trajectories that matter most.

So the design decision hinges on one number: how often does |v| exceed
4 mg/dL/min?

    negligible (<< 1%)   clip C to 1, log the clipping rate, proceed
    substantial          implement multi-cell or semi-Lagrangian transport

This costs no GPU time and no training. It reads the raw glucose timeline
already on disk.

WHAT IS MEASURED
----------------
The rate is computed the same way the model will see it -- a one-step
difference on the 5-minute grid, per subject, never crossing a subject
boundary. Rates are reported over all readings and, separately, over
readings preceding a hypoglycaemic event, since the tail matters more
there.

Usage:
    python rate_distribution_check.py
    python rate_distribution_check.py --dx 20 --dt 5
"""

import argparse
import numpy as np

import config
import common

HORIZONS = config.HORIZONS
OUT_DIR = config.RESULTS / "RQ2_models" / "ctf_ru"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--dx", type=float, default=20.0, help="bin spacing, mg/dL")
    ap.add_argument("--dt", type=float, default=5.0, help="timestep, minutes")
    args = ap.parse_args()

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    g = bw.raw_gluc.astype(np.float64)

    # one-step difference per subject, never across a subject boundary
    rate = np.zeros(len(g), dtype=np.float64)
    for si in range(len(bw.subjects)):
        m = np.flatnonzero(bw.reading_subject == si)
        if len(m) < 2:
            continue
        gg = g[m]
        d = np.diff(gg, prepend=gg[0]) / bw.sample_min
        rate[m] = d
    a = np.abs(rate)
    v_crit = args.dx / args.dt

    print(f"{'#'*88}")
    print(f"# GLUCOSE RATE DISTRIBUTION  ({len(a):,} readings, "
          f"{bw.sample_min}-min cadence)")
    print(f"#   nearest-neighbour transport is valid while C = |v|*dt/dx <= 1,")
    print(f"#   i.e. |v| <= {v_crit:.1f} mg/dL/min at dx={args.dx:.0f}, dt={args.dt:.0f}")
    print(f"{'#'*88}")

    print(f"\n  |rate| in mg/dL/min")
    for q in [50, 75, 90, 95, 99, 99.9]:
        print(f"    p{q:<5} {np.percentile(a, q):>8.3f}")
    print(f"    max    {a.max():>8.3f}")
    print(f"    mean   {a.mean():>8.3f}")

    print(f"\n  Courant number C = |v| * {args.dt:.0f} / {args.dx:.0f}")
    for q in [50, 90, 95, 99, 99.9]:
        print(f"    p{q:<5} {np.percentile(a, q)*args.dt/args.dx:>8.3f}")
    print(f"    max    {a.max()*args.dt/args.dx:>8.3f}")

    over = float((a > v_crit).mean())
    print(f"\n  fraction with |v| > {v_crit:.1f} (C > 1): {100*over:.4f}%  "
          f"({int((a > v_crit).sum()):,} readings)")

    # the tail matters more where it matters clinically
    ends = bw.starts + bw.window_len - 1
    y30 = bw._labels_any[:, HORIZONS.index(30)].astype(bool)
    a_win = a[ends]
    print(f"\n  restricted to window endpoints ({len(a_win):,}):")
    print(f"    all windows      p99 {np.percentile(a_win, 99):>6.3f}   "
          f"> {v_crit:.0f}: {100*(a_win > v_crit).mean():.4f}%")
    if y30.any():
        ap_ = a_win[y30]
        print(f"    before an event  p99 {np.percentile(ap_, 99):>6.3f}   "
              f"> {v_crit:.0f}: {100*(ap_ > v_crit).mean():.4f}%")
    an = a_win[~y30]
    print(f"    no event         p99 {np.percentile(an, 99):>6.3f}   "
          f"> {v_crit:.0f}: {100*(an > v_crit).mean():.4f}%")

    # how much of a window is affected, not just how many readings
    wl = bw.window_len
    bad = a > v_crit
    cum = np.concatenate([[0], np.cumsum(bad)])
    per_win = cum[bw.starts + wl] - cum[bw.starts]
    print(f"\n  windows containing at least one clipped step: "
          f"{100*(per_win > 0).mean():.2f}%")
    print(f"  mean clipped steps per affected window: "
          f"{per_win[per_win > 0].mean():.2f} of {wl}"
          if (per_win > 0).any() else "")

    print(f"\n{'#'*88}\nREADING THIS\n{'#'*88}")
    if over < 0.001:
        print(f"\n  |v| exceeds {v_crit:.0f} mg/dL/min in {100*over:.4f}% of readings.")
        print("  Clipping C to 1 affects a negligible fraction, so")
        print("  nearest-neighbour transport is sufficient. Log the clipping")
        print("  rate during training and proceed.")
    elif over < 0.01:
        print(f"\n  |v| exceeds {v_crit:.0f} mg/dL/min in {100*over:.4f}% of readings --")
        print("  small but not negligible, and concentrated in fast falls.")
        print("  Clipping is defensible if the affected windows are reported,")
        print("  but multi-cell transport would be more faithful.")
    else:
        print(f"\n  |v| exceeds {v_crit:.0f} mg/dL/min in {100*over:.4f}% of readings.")
        print("  Clipping would systematically understate the fastest falls,")
        print("  which are the trajectories the architecture exists to model.")
        print("  Implement multi-cell or semi-Lagrangian transport instead.")

    print(f"\n  (a larger dx would reduce clipping: at dx=30, C>1 needs "
          f"|v| > {30/args.dt:.0f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    common.save_result(OUT_DIR, "_rate_distribution", {
        "dx": args.dx, "dt": args.dt, "v_critical": v_crit,
        "percentiles": {f"p{q}": float(np.percentile(a, q))
                        for q in [50, 75, 90, 95, 99, 99.9]},
        "max": float(a.max()),
        "frac_over_critical": over,
        "frac_windows_affected": float((per_win > 0).mean())})
    print(f"\n✓ Saved -> {OUT_DIR / '_rate_distribution.json'}")


if __name__ == "__main__":
    main()
