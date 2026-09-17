"""
phase2_run.py  --  Phase 2 error analysis, wired to this project's schema
==========================================================================

No GPU, no model load, no training. Saved validation probabilities are
joined to bw.raw_gluc, so this is a table operation.

    windows are bw.starts (2,761,827 of them), each 60 readings long
    end        = starts + window_len - 1        the prediction instant
    history    = raw_gluc[start .. end]         mg/dL, unnormalised
    future     = raw_gluc[end+1 .. end+H/5]     the horizon
    probs      = <run>_probs.npz["val"]         (n_val, 4)
    which      = <run>_probs.npz["idx_val"]     window indices

FUTURE VALUES ARE MASKED AT SEGMENT AND SUBJECT BOUNDARIES. A window near
the end of a subject's recording has no future readings; those become NaN
rather than silently picking up the next subject's glucose. The count of
masked windows is reported -- if it is large at h=120 that is itself a
finding, since those windows were still given a label.

Usage
-----
    python phase2_run.py --list
    python phase2_run.py \
        --probs results/RQ2_models/ta_gru/gru_ta_feat__weighted_bce__none__any_probs.npz \
        --tag stride6
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common  # noqa: E402
import config  # noqa: E402
from error_analysis import (  # noqa: E402
    HYPO_MGDL, RECENT_N, _slope_accel, _time_to_first_hypo,
    subject_panel, subject_bootstrap_ci, characterize,
)

CHUNK = 100_000


def list_runs() -> None:
    root = str(config.RESULTS if hasattr(config, "RESULTS")
               else "results")
    hits = sorted(glob.glob(f"{root}/**/*_probs.npz", recursive=True))
    print(f"{len(hits)} probability files under {root}\n")
    for h in hits:
        z = np.load(h)
        n = z["val"].shape[0] if "val" in z else "?"
        print(f"  {os.path.relpath(h, root):<70} n_val={n}")
    print("\nPick the frozen GRU-Baseline (GRU + TA + Absolute) and pass it "
          "with --probs.")


def _window_end(bw) -> np.ndarray:
    """
    end index of each window. Verified against bw.raw_gluc_at_pred, which
    common.py computes internally -- if the convention is off by one this
    assertion fires immediately rather than silently shifting every
    trajectory by 5 minutes.
    """
    for cand in (bw.starts + bw.window_len - 1, bw.starts + bw.window_len):
        if np.array_equal(bw.raw_gluc[cand], bw.raw_gluc_at_pred):
            return cand
    raise AssertionError(
        "could not reproduce raw_gluc_at_pred from starts/window_len; "
        "inspect the end-index convention in common.BuiltWindows"
    )


def _future_block(bw, end_idx: np.ndarray, n_steps: int):
    """
    raw_gluc over the next n_steps readings, NaN where the span leaves the
    subject or the recording segment. Chunked to bound peak memory.
    """
    n = end_idx.size
    out = np.full((n, n_steps), np.nan, dtype=np.float32)
    limit = bw.raw_gluc.shape[0] - 1
    off = np.arange(1, n_steps + 1, dtype=np.int64)

    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        e = end_idx[a:b]
        fi = e[:, None] + off[None, :]
        oob = fi > limit
        fi_c = np.clip(fi, 0, limit)
        same = ((bw.reading_subject[fi_c] == bw.reading_subject[e][:, None]) &
                (bw.seg[fi_c] == bw.seg[e][:, None]) & ~oob)
        v = bw.raw_gluc[fi_c].astype(np.float32)
        v[~same] = np.nan
        out[a:b] = v
    return out


def _history_tail(bw, end_idx: np.ndarray, n: int = RECENT_N) -> np.ndarray:
    off = np.arange(-(n - 1), 1, dtype=np.int64)
    out = np.empty((end_idx.size, n), dtype=np.float32)
    for a in range(0, end_idx.size, CHUNK):
        b = min(a + CHUNK, end_idx.size)
        out[a:b] = bw.raw_gluc[end_idx[a:b][:, None] + off[None, :]]
    return out


def load_thresholds(json_path: str | None, horizons) -> dict | None:
    if not json_path or not os.path.exists(json_path):
        return None
    with open(json_path) as f:
        res = json.load(f)
    out = {}
    for h in horizons:
        node = res.get("horizons", {}).get(str(h), {})
        if "threshold" in node:
            out[h] = float(node["threshold"])
    return out or None


def build(probs_path: str, tag: str, label_set: str, thr_json: str | None):
    bw = common.load_built(tag, label_set=label_set)
    z = np.load(probs_path)
    P, idx = z["val"], z["idx_val"]

    if idx.max() >= bw.starts.shape[0]:
        raise SystemExit(
            f"idx_val max {idx.max()} exceeds {bw.starts.shape[0]} windows in "
            f"tag='{tag}'. These probabilities were produced on a different "
            f"window build -- try --tag eval."
        )

    end = _window_end(bw)[idx]
    subj = bw.meta[idx]
    Y = bw.labels[idx]

    # split check. cmr_pilot does np.flatnonzero(tr), so get_split returns
    # masks over WINDOWS (bw.meta). Handle a per-subject return too, since a
    # silent mismatch here would mean analysing the wrong subjects.
    va_set = None
    for arg in (bw.meta, bw.subjects):
        try:
            _tr, va, _te = common.get_split(arg, seed=42)
        except Exception:
            continue
        va = np.asarray(va)
        pool = np.asarray(arg)
        picked = pool[va] if va.dtype != bool else pool[va]
        va_set = set(np.unique(picked).tolist())
        if 30 <= len(va_set) <= 45:      # expect 36 validation subjects
            break
    got = set(pd.unique(subj).tolist())
    if va_set is None:
        print("  WARNING: could not resolve the split; skipping subject check.",
              file=sys.stderr)
    elif not got.issubset(va_set):
        print(f"  WARNING: {len(got - va_set)} subjects in the probs file are "
              f"not in the validation split -- verify this run.", file=sys.stderr)
    print(f"  {len(idx):,} windows | {len(got)} subjects"
          + (f" (validation split has {len(va_set)})" if va_set else ""))

    tail = _history_tail(bw, end)
    slope, accel = _slope_accel(tail)
    g_now = tail[:, -1]

    thr = load_thresholds(thr_json, bw.horizons)
    frames = []
    for k, h in enumerate(bw.horizons):
        steps = h // bw.sample_min
        fut = _future_block(bw, end, steps)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            nadir = np.nanmin(fut, axis=1)
        n_masked = int(np.isnan(nadir).sum())

        y = Y[:, k].astype(int)
        p = P[:, k].astype(float)
        if thr and h in thr:
            t = thr[h]
        else:
            t, _ = common.find_threshold(y, p)
        print(f"  h={h:>3}  threshold {t:.4f}  "
              f"future masked {n_masked:,} ({n_masked/len(y):.2%})")

        yp = (p >= t).astype(int)
        frames.append(pd.DataFrame({
            "subject_id": subj,
            "horizon": h,
            "y_true": y,
            "y_prob": p,
            "y_pred": yp,
            "outcome": np.where((yp == 1) & (y == 1), "TP",
                       np.where((yp == 1) & (y == 0), "FP",
                       np.where((yp == 0) & (y == 1), "FN", "TN"))),
            "glucose_now": g_now,
            "recent_min": np.nanmin(tail, axis=1),
            "recent_max": np.nanmax(tail, axis=1),
            "recent_std": np.nanstd(tail, axis=1),
            "slope_per5min": slope,
            "accel": accel,
            "dist_from_70": g_now - HYPO_MGDL,
            "future_nadir": nadir,
            "min_to_first_hypo": _time_to_first_hypo(fut, bw.sample_min),
        }))
        del fut

    df = pd.concat(frames, ignore_index=True)
    df["near_miss_negative"] = ((df.y_true == 0) &
                                (df.future_nadir < 85.0) &
                                (df.future_nadir >= HYPO_MGDL))

    # label audit: does the stored label agree with the raw trajectory?
    ok = df.future_nadir.notna()
    agree = ((df.loc[ok, "future_nadir"] < HYPO_MGDL).astype(int)
             == df.loc[ok, "y_true"]).mean()
    print(f"\n  label vs raw-trajectory agreement: {agree:.4%}")
    if agree < 0.98:
        print("  NOTE: labels use a stricter rule than 'any reading <70' "
              "(consensus/persistence). future_nadir stays diagnostic only.")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--probs")
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--label_set", default="any", choices=["any", "consensus"])
    ap.add_argument("--thr_json", default=None,
                    help="run .json holding the frozen per-horizon thresholds")
    ap.add_argument("--outdir", default="results/error_analysis")
    args = ap.parse_args()

    if args.list or not args.probs:
        return list_runs()

    os.makedirs(args.outdir, exist_ok=True)
    df = build(args.probs, args.tag, args.label_set, args.thr_json)
    panel = subject_panel(df)
    report = characterize(df, panel)

    name = os.path.basename(args.probs).replace("_probs.npz", "")
    df.to_csv(f"{args.outdir}/windows_{name}.csv.gz", index=False,
              compression="gzip")
    panel.to_csv(f"{args.outdir}/subject_panel_{name}.csv", index=False)
    with open(f"{args.outdir}/report_{name}.txt", "w") as f:
        f.write(report)

    summary = {str(h): {c: subject_bootstrap_ci(panel, c, h)[0]
                        for c in ("auprc", "auroc", "recall", "precision",
                                  "f1", "false_alarm_rate", "brier")}
               for h in sorted(df.horizon.unique())}
    with open(f"{args.outdir}/baseline_panel_{name}.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(report)
    print(f"\nwrote -> {args.outdir}/  (baseline_panel_{name}.json is the "
          "frozen reference for every later phase)")


if __name__ == "__main__":
    main()
