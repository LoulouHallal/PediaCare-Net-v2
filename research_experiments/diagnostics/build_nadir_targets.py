"""
build_nadir_targets.py  --  Phase 3 auxiliary targets
======================================================

Produces, per window and per horizon:

    nadir_raw   min(raw_gluc[end+1 .. end+H/5])        mg/dL
    nadir_z     the same, z-scored with TRAIN statistics
    t2h         minutes until the first reading < 70    (Phase 4)
    valid       False where the future leaves the subject or segment

WHY THIS TARGET
---------------
Phase 2 measured the supervision collapse directly. At h=120 the positives
span nadir 39-70 (iqr 54-66) and the negatives span 70-401 (iqr 88-141),
and every one of them is a single bit. A window at glucose 143 heading for
a nadir of 62 gets the same gradient as a window at 143 that levels off at
130 -- and 22.9% of h=120 false negatives are exactly that "fast drop from
high" case.

NORMALISATION USES TRAIN SUBJECTS ONLY. Using all subjects would leak
validation distribution into the target scale. The statistics are saved so
the identical transform can be applied at test time.

MASKING IS NOT OPTIONAL. 539 windows have no future readings but still
carry a classification label. Those must be excluded from the auxiliary
loss, not imputed -- a mean-filled nadir is a fabricated training signal.

    python build_nadir_targets.py --tag stride6
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import config  # noqa: E402

HYPO = 70.0
CHUNK = 100_000


def window_end(bw):
    for cand in (bw.starts + bw.window_len - 1, bw.starts + bw.window_len):
        if np.array_equal(bw.raw_gluc[cand], bw.raw_gluc_at_pred):
            return cand
    raise AssertionError("cannot reproduce raw_gluc_at_pred from starts")


def future_block(bw, end_idx, n_steps):
    n = end_idx.size
    out = np.full((n, n_steps), np.nan, dtype=np.float32)
    limit = bw.raw_gluc.shape[0] - 1
    off = np.arange(1, n_steps + 1, dtype=np.int64)
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        e = end_idx[a:b]
        fi = e[:, None] + off[None, :]
        oob = fi > limit
        fic = np.clip(fi, 0, limit)
        same = ((bw.reading_subject[fic] == bw.reading_subject[e][:, None]) &
                (bw.seg[fic] == bw.seg[e][:, None]) & ~oob)
        v = bw.raw_gluc[fic].astype(np.float32)
        v[~same] = np.nan
        out[a:b] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--label_set", default="any")
    ap.add_argument("--clip_hi", type=float, default=180.0,
                    help="clip the nadir target above this (mg/dL). The "
                         "regression should spend capacity separating 62 "
                         "from 78, not 250 from 300. Set 0 to disable.")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    bw = common.load_built(a.tag, label_set=a.label_set)
    end = window_end(bw)
    n, H = bw.starts.size, len(bw.horizons)
    meta = np.asarray(bw.meta).astype(str)

    # train mask over WINDOWS -- statistics must not see val or test
    tr, va, te = common.get_split(meta, seed=42)
    tr = np.asarray(tr)
    tr_mask = tr if tr.dtype == bool else np.isin(np.arange(n), tr)
    print(f"{n:,} windows | train {tr_mask.sum():,} "
          f"({len(np.unique(meta[tr_mask]))} subjects)")

    nadir = np.full((n, H), np.nan, dtype=np.float32)
    t2h = np.full((n, H), np.nan, dtype=np.float32)
    valid = np.zeros((n, H), dtype=bool)

    for k, h in enumerate(bw.horizons):
        steps = h // bw.sample_min
        fut = future_block(bw, end, steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nd = np.nanmin(fut, axis=1)
        ok = np.isfinite(nd)
        nadir[:, k] = nd
        valid[:, k] = ok

        below = np.where(np.isnan(fut), False, fut < HYPO)
        any_b = below.any(axis=1)
        first = np.where(any_b, below.argmax(axis=1), -1)
        tt = np.full(n, np.nan, dtype=np.float32)
        tt[any_b] = (first[any_b] + 1) * bw.sample_min
        t2h[:, k] = tt
        del fut

        y = bw.labels[:, k].astype(int)
        agree = ((nd[ok] < HYPO).astype(int) == y[ok]).mean()
        print(f"  h={h:>3}  valid {ok.mean():.4%}  "
              f"nadir mean {np.nanmean(nd):6.1f}  sd {np.nanstd(nd):5.1f}  "
              f"label agreement {agree:.4%}")

    if a.clip_hi and a.clip_hi > 0:
        frac = np.nanmean(nadir > a.clip_hi)
        nadir = np.minimum(nadir, a.clip_hi)
        print(f"\nclipped nadir target at {a.clip_hi:.0f} mg/dL "
              f"({frac:.2%} of valid windows). The classification label is "
              f"untouched; only the auxiliary target is clipped.")

    # z-score with TRAIN statistics only, per horizon
    mu = np.zeros(H, dtype=np.float32)
    sd = np.ones(H, dtype=np.float32)
    for k in range(H):
        m = tr_mask & valid[:, k]
        mu[k] = float(np.mean(nadir[m, k]))
        sd[k] = float(np.std(nadir[m, k]))
        if sd[k] < 1e-6:
            raise ValueError(f"degenerate nadir sd at horizon index {k}")
    nadir_z = (nadir - mu[None, :]) / sd[None, :]

    print("\ntrain-only normalisation (mg/dL):")
    for k, h in enumerate(bw.horizons):
        print(f"  h={h:>3}  mu {mu[k]:6.2f}  sd {sd[k]:5.2f}")

    # sanity: within positives, does the target actually vary?
    print("\nvariance the binary label throws away:")
    for k, h in enumerate(bw.horizons):
        m = valid[:, k]
        y = bw.labels[:, k].astype(int)
        pos = nadir[m & (y == 1), k]
        neg = nadir[m & (y == 0), k]
        print(f"  h={h:>3}  positives sd {pos.std():5.2f} "
              f"(iqr {np.percentile(pos,25):.0f}-{np.percentile(pos,75):.0f})   "
              f"negatives sd {neg.std():6.2f} "
              f"(iqr {np.percentile(neg,25):.0f}-{np.percentile(neg,75):.0f})")

    out = a.out or str(config.DATA_DERIVED / f"nadir_targets_{a.tag}.npz")
    np.savez_compressed(
        out, nadir_raw=nadir, nadir_z=nadir_z, t2h=t2h, valid=valid,
        mu=mu, sd=sd, clip_hi=float(a.clip_hi),
        horizons=np.array(bw.horizons),
        sample_min=bw.sample_min, tag=a.tag, label_set=a.label_set)
    print(f"\nwrote -> {out}")
    print("Apply the SAME mu/sd at test time. Never refit them.")


if __name__ == "__main__":
    main()
