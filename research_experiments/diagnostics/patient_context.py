"""
patient_context.py
====================
Causal 48-hour patient-context features.

WHAT THIS IS FOR
----------------
Per-patient PPV on the test set ranges from 0.07 to 0.87. A single set of
weights is being asked to serve 244 physiologically different children.
The model currently has no way to know WHICH child it is looking at.

This module builds a compact description of the child's recent
physiological regime -- variability, hypoglycaemia tendency, distribution
shape, rate-of-change behaviour -- from the 48 hours immediately BEFORE
each prediction window.

WHY NOT A PATIENT-ID EMBEDDING
------------------------------
Test subjects are different children from training subjects. A learned
ID embedding has no entry for an unseen child, so it either fails at test
time or turns the experiment into transductive personalisation. Deriving
the context from unlabelled prior history instead means a completely new
child is handled with no fine-tuning and no labels -- which is both the
deployable formulation and the stronger claim.

CAUSALITY
---------
The context for a window is computed from readings strictly BEFORE that
window's first sample. Nothing from inside the prediction window, and
nothing after it, can enter. `verify_causal()` checks this directly by
corrupting future readings and confirming earlier contexts are bit-identical.

The context is recomputed PER WINDOW, not once per subject. Computing it
once per subject would let late-timeline data leak into early windows.

WHAT THE FEATURES DELIBERATELY AVOID
------------------------------------
They do not restate the current trajectory. c_t and v_t already carry
threshold proximity and instantaneous slope; duplicating those would make
any gain uninterpretable. These statistics describe the child's stable
character over two days, which is a different timescale.

COST
----
A naive implementation is 576 readings x 20 statistics x 2.76M windows.
Sum-based statistics use cumulative sums over each subject's timeline, so
each window costs O(1) after one pass. Percentiles and skewness cannot be
computed that way and use a strided cache: exact quantiles on a subsample
of the lookback, at a stride that keeps the estimate stable.

Windows without a full 48 h of prior history are dropped from every split
alike, and the count is reported.

Usage:
    python patient_context.py --tag stride6          # build and cache
    python patient_context.py --tag stride6 --verify # leakage check
"""

import argparse
import numpy as np

import config
import common

LOOKBACK = 576              # 48 h at 5-minute sampling
HYPO, SEVERE, HYPER = 70.0, 54.0, 180.0
QUANT_STRIDE = 4            # subsample for percentile estimates
RAPID = 1.0                 # mg/dL per minute, |rate| above this is "rapid"

FEATURE_NAMES = [
    "gl_mean", "gl_sd", "gl_cv", "gl_median",
    "gl_p10", "gl_p25", "gl_p75", "gl_p90", "gl_iqr", "gl_skew",
    "frac_below_70", "frac_below_54", "frac_above_180",
    "n_hypo_episodes", "mean_hypo_duration_min", "mins_since_last_hypo",
    "roc_mean_abs", "roc_sd", "frac_falling_fast", "frac_rising_fast",
    "coverage",
]


def _episode_stats(low, times):
    """Count of hypo runs, mean duration, minutes since the last one."""
    if not low.any():
        return 0.0, 0.0, float(len(low) * 5)
    d = np.diff(low.astype(np.int8), prepend=0)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(np.diff(low.astype(np.int8), append=0) == -1)
    n = len(starts)
    if n == 0:
        return 0.0, 0.0, float(len(low) * 5)
    dur = float(np.mean((ends[:n] - starts + 1) * 5))
    since = float((len(low) - 1 - ends[-1]) * 5)
    return float(n), dur, since


def context_for_subject(gluc, times, starts_local):
    """
    Context for every window start of one subject.

    `starts_local` are indices into this subject's timeline. Each window's
    context uses gluc[start - LOOKBACK : start] -- strictly before the
    window's first reading.
    """
    n = len(gluc)
    g = gluc.astype(np.float64)

    # O(1) sum-based statistics
    c1 = np.concatenate([[0.0], np.cumsum(g)])
    c2 = np.concatenate([[0.0], np.cumsum(g ** 2)])
    c3 = np.concatenate([[0.0], np.cumsum(g ** 3)])
    below70 = np.concatenate([[0.0], np.cumsum(g < HYPO)])
    below54 = np.concatenate([[0.0], np.cumsum(g < SEVERE)])
    above180 = np.concatenate([[0.0], np.cumsum(g > HYPER)])

    roc = np.diff(g, prepend=g[0]) / 5.0                # mg/dL per minute
    r1 = np.concatenate([[0.0], np.cumsum(np.abs(roc))])
    r2 = np.concatenate([[0.0], np.cumsum(roc ** 2)])
    fall = np.concatenate([[0.0], np.cumsum(roc < -RAPID)])
    rise = np.concatenate([[0.0], np.cumsum(roc > RAPID)])

    out = np.zeros((len(starts_local), len(FEATURE_NAMES)), dtype=np.float32)
    keep = np.zeros(len(starts_local), dtype=bool)

    for k, s in enumerate(starts_local):
        lo = s - LOOKBACK
        if lo < 0:
            continue                                    # insufficient history
        keep[k] = True
        m = LOOKBACK
        su = c1[s] - c1[lo]
        su2 = c2[s] - c2[lo]
        su3 = c3[s] - c3[lo]
        mean = su / m
        var = max(su2 / m - mean ** 2, 1e-12)
        sd = np.sqrt(var)
        skew = (su3 / m - 3 * mean * var - mean ** 3) / (sd ** 3)

        w = g[lo:s]
        q = np.sort(w[::QUANT_STRIDE])
        p10, p25, med, p75, p90 = np.percentile(q, [10, 25, 50, 75, 90])

        low = w < HYPO
        n_ep, dur, since = _episode_stats(low, None)

        out[k] = [
            mean, sd, sd / max(mean, 1e-6), med,
            p10, p25, p75, p90, p75 - p25, skew,
            (below70[s] - below70[lo]) / m,
            (below54[s] - below54[lo]) / m,
            (above180[s] - above180[lo]) / m,
            n_ep, dur, since,
            (r1[s] - r1[lo]) / m,
            np.sqrt(max((r2[s] - r2[lo]) / m, 0.0)),
            (fall[s] - fall[lo]) / m,
            (rise[s] - rise[lo]) / m,
            1.0,                                        # coverage placeholder
        ]
    return out, keep


def build(bw, verbose=True):
    """
    Context for every window in the dataset.
    Returns (context (N, F) float32, keep (N,) bool).
    """
    n_win = len(bw)
    ctx = np.zeros((n_win, len(FEATURE_NAMES)), dtype=np.float32)
    keep = np.zeros(n_win, dtype=bool)

    # map global start indices back to per-subject local indices
    ends = bw.starts + bw.window_len - 1
    win_subj = bw.reading_subject[ends]

    for si in range(len(bw.subjects)):
        rmask = np.flatnonzero(bw.reading_subject == si)
        if len(rmask) == 0:
            continue
        base = rmask[0]
        wsel = np.flatnonzero(win_subj == si)
        if len(wsel) == 0:
            continue
        local = bw.starts[wsel] - base
        c, k = context_for_subject(bw.raw_gluc[rmask],
                                   bw.reading_times[rmask], local)
        ctx[wsel] = c
        keep[wsel] = k
        if verbose and (si + 1) % 50 == 0:
            print(f"  {si+1}/{len(bw.subjects)} subjects")

    if verbose:
        print(f"\nwindows with a full {LOOKBACK}-reading (48 h) history: "
              f"{keep.sum():,} / {n_win:,} ({100*keep.mean():.1f}%)")
        print(f"dropped {(~keep).sum():,} windows lacking prior history "
              f"(applied identically to train, val and test)")
    return ctx, keep


def standardise(ctx, fit_mask, verbose=True):
    """Z-score using TRAINING windows only."""
    mu = ctx[fit_mask].mean(0)
    sd = np.maximum(ctx[fit_mask].std(0), 1e-6)
    out = np.clip((ctx - mu) / sd, -10, 10).astype(np.float32)
    if verbose:
        print(f"standardised on {int(fit_mask.sum()):,} training windows")
    return out, mu, sd


# ─── CAUSALITY CHECK ──────────────────────────────────────────────────────────

def verify_causal(bw, n_check=200, seed=0):
    """
    Corrupt readings AFTER a cutoff and confirm the contexts of windows
    that start before it are bit-identical. This is the check that
    distinguishes a causal feature from a leaking one.
    """
    rng = np.random.default_rng(seed)
    si = int(rng.integers(0, len(bw.subjects)))
    rmask = np.flatnonzero(bw.reading_subject == si)
    ends = bw.starts + bw.window_len - 1
    wsel = np.flatnonzero(bw.reading_subject[ends] == si)
    if len(wsel) < 20 or len(rmask) < LOOKBACK + 200:
        return None

    base = rmask[0]
    local = bw.starts[wsel] - base
    g = bw.raw_gluc[rmask].copy()
    cut = int(len(g) * 0.6)

    c_clean, k_clean = context_for_subject(g, None, local)
    g_bad = g.copy()
    g_bad[cut:] = 400.0                                # corrupt the future
    c_bad, k_bad = context_for_subject(g_bad, None, local)

    before = (local <= cut) & k_clean                  # windows starting before
    same = np.allclose(c_clean[before], c_bad[before], atol=0, rtol=0)
    after = (local > cut + LOOKBACK) & k_clean
    differs = (not np.allclose(c_clean[after], c_bad[after])
               if after.any() else None)
    return {"subject": str(bw.subjects[si]),
            "n_windows_before_cut": int(before.sum()),
            "contexts_before_cut_identical": bool(same),
            "contexts_after_cut_differ": differs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)

    if args.verify:
        print("Causality check: corrupt the future, contexts before the cut "
              "must be unchanged\n")
        for s in range(3):
            r = verify_causal(bw, seed=s)
            if r:
                print(f"  subject {r['subject']}: "
                      f"{r['n_windows_before_cut']} windows before cut | "
                      f"identical: {r['contexts_before_cut_identical']} | "
                      f"after-cut contexts differ: {r['contexts_after_cut_differ']}")
        return

    print(f"Building {LOOKBACK}-reading (48 h) causal patient context "
          f"for {len(bw):,} windows...")
    ctx, keep = build(bw)
    tr, _, _ = common.get_split(bw.meta)
    ctx_z, mu, sd = standardise(ctx, tr & keep)

    out = config.DATA_DERIVED / f"patient_context_{args.tag}.npz"
    np.savez_compressed(out, context=ctx_z, keep=keep, mu=mu, sd=sd,
                        features=np.array(FEATURE_NAMES),
                        lookback=np.int64(LOOKBACK))
    print(f"\n✓ Saved -> {out}")
    print(f"  shape {ctx_z.shape}, {len(FEATURE_NAMES)} features")


if __name__ == "__main__":
    main()
