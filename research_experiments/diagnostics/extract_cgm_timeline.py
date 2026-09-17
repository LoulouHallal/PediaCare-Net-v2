"""
extract_cgm_timeline.py
=========================
Extract per-subject CGM timelines (real mg/dL + timestamps) from the raw
MetaboNet parquet, so that EVENT-LEVEL metrics can be computed.

WHY THIS IS NEEDED
------------------
The windowed dataset (`metabonet_windows_pediatric.npz`) cannot support
event-level evaluation:

  1. Channel 0 is per-subject z-scored, so it cannot be thresholded at
     70 mg/dL.
  2. It contains exactly 3,000 windows per subject -- a cap, not a
     continuous timeline. Consecutive rows do not overlap at any stride
     (checked at 1, 2, 3, 6, 12), so the original time order cannot be
     reconstructed from it.
  3. It has no timestamps, so "false alarms per patient-day" has no
     denominator.

Episode definitions and alarm burden both require real glucose values on
a real clock. This script rebuilds that from the source parquet.

MEMORY
------
The parquet has ~154.8M rows across 149 row groups. Reading it whole
exhausts Colab RAM. This script streams row group by row group, keeps
only three columns, immediately drops rows with no CGM reading, and
filters to the requested subjects before anything accumulates.

OUTPUT
------
`data_derived/cgm_timeline_<dataset>.parquet` with columns:
    id (str), date (datetime64[ns]), cgm (float32)
sorted by (id, date), plus a JSON summary of per-subject coverage.

Usage:
    # subjects taken from the windowed npz, so the cohort matches exactly
    python extract_cgm_timeline.py --subjects_from metabonet

    # or all subjects in the parquet
    python extract_cgm_timeline.py --all_subjects
"""

import gc
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

import pyarrow.parquet as pq

import config
import common

RAW_PARQUET = config.RAW_DATA["metabonet"]
NEEDED = ["id", "date", "CGM"]

# Physiologically possible CGM range. Values outside this are sensor
# artifacts or fill values, not readings; keeping them would create
# spurious hypoglycemia episodes.
CGM_MIN, CGM_MAX = 20.0, 600.0


def subject_ids_from_windows(dataset):
    """Exact cohort used by every existing result."""
    _, _, meta = common.load_windows(dataset)
    return sorted(set(meta.tolist()))


def extract(parquet_path, keep_ids=None, verbose=True):
    f = pq.ParquetFile(str(parquet_path))
    n_groups = f.metadata.num_row_groups
    if verbose:
        print(f"Parquet: {f.metadata.num_rows:,} rows in {n_groups} row groups")
        print(f"Reading columns {NEEDED} only\n")

    keep = set(keep_ids) if keep_ids else None
    parts = []
    n_seen = n_kept = 0

    for g in range(n_groups):
        tbl = f.read_row_group(g, columns=NEEDED)
        df = tbl.to_pandas()
        del tbl
        n_seen += len(df)

        df = df[df["CGM"].notna()]
        if keep is not None and len(df):
            df = df[df["id"].astype(str).isin(keep)]
        if len(df):
            df["id"] = df["id"].astype(str)
            df["cgm"] = df["CGM"].astype(np.float32)
            parts.append(df[["id", "date", "cgm"]])
            n_kept += len(df)

        del df
        if verbose and (g + 1) % 20 == 0:
            print(f"  group {g+1:>3}/{n_groups}  seen {n_seen:>12,}  "
                  f"kept {n_kept:>11,}")
        gc.collect()

    if not parts:
        raise RuntimeError("No CGM rows matched. Check the subject IDs.")

    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    if verbose:
        print(f"\nBefore cleaning: {len(out):,} CGM rows")

    # Drop implausible readings, then exact duplicate timestamps per subject
    bad = (out["cgm"] < CGM_MIN) | (out["cgm"] > CGM_MAX)
    if bad.any() and verbose:
        print(f"  dropping {int(bad.sum()):,} readings outside "
              f"[{CGM_MIN}, {CGM_MAX}] mg/dL")
    out = out[~bad]

    before = len(out)
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"],
                                                          keep="first")
    if verbose and before != len(out):
        print(f"  dropping {before - len(out):,} duplicate (id, date) rows")

    return out.reset_index(drop=True)


def coverage_summary(df, verbose=True):
    """
    Per-subject coverage. The median sampling interval tells us the CGM
    grid (should be ~5 min); monitoring days is the denominator for
    false-alarms-per-day.
    """
    summary = {}
    for pid, g in df.groupby("id", sort=True):
        t = g["date"].values.astype("datetime64[m]").astype(np.int64)
        gaps = np.diff(t) if len(t) > 1 else np.array([np.nan])
        span_days = (t[-1] - t[0]) / (60 * 24) if len(t) > 1 else 0.0
        med_gap = float(np.nanmedian(gaps)) if len(gaps) else float("nan")
        # Monitored time = sum of gaps that look like real sampling, so a
        # multi-week sensor outage is not counted as monitored time.
        valid = gaps[(gaps > 0) & (gaps <= 30)] if len(gaps) else np.array([])
        monitored_days = float(valid.sum()) / (60 * 24) if valid.size else 0.0
        summary[pid] = {
            "n_readings": int(len(g)),
            "median_gap_min": med_gap,
            "span_days": float(span_days),
            "monitored_days": monitored_days,
            "cgm_mean": float(g["cgm"].mean()),
            "cgm_min": float(g["cgm"].min()),
            "pct_below_70": float((g["cgm"] < 70).mean() * 100),
        }

    if verbose:
        meds = [v["median_gap_min"] for v in summary.values()]
        mon = [v["monitored_days"] for v in summary.values()]
        low = [v["pct_below_70"] for v in summary.values()]
        print(f"\nCoverage across {len(summary)} subjects:")
        print(f"  median sampling interval : {np.nanmedian(meds):.1f} min "
              f"(range {np.nanmin(meds):.1f}-{np.nanmax(meds):.1f})")
        print(f"  monitored days / subject : median {np.median(mon):.1f}  "
              f"(total {np.sum(mon):,.0f} patient-days)")
        print(f"  % readings < 70 mg/dL    : median {np.median(low):.2f}%  "
              f"(range {np.min(low):.2f}-{np.max(low):.2f}%)")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default=str(RAW_PARQUET))
    ap.add_argument("--subjects_from", default="metabonet",
                    help="take the subject list from this windowed dataset")
    ap.add_argument("--all_subjects", action="store_true")
    ap.add_argument("--out_name", default=None)
    args = ap.parse_args()

    config.DATA_DERIVED.mkdir(parents=True, exist_ok=True)

    keep_ids = None
    tag = "all"
    if not args.all_subjects:
        keep_ids = subject_ids_from_windows(args.subjects_from)
        tag = args.subjects_from
        print(f"Restricting to {len(keep_ids)} subjects from '{tag}' windows")

    df = extract(Path(args.parquet), keep_ids)

    found = sorted(df["id"].unique().tolist())
    print(f"\nSubjects with CGM data: {len(found)}")
    if keep_ids:
        missing = sorted(set(keep_ids) - set(found))
        if missing:
            print(f"  WARNING: {len(missing)} requested subjects have no CGM "
                  f"rows: {missing[:10]}{' ...' if len(missing) > 10 else ''}")

    summary = coverage_summary(df)

    name = args.out_name or f"cgm_timeline_{tag}"
    out_pq = config.DATA_DERIVED / f"{name}.parquet"
    df.to_parquet(out_pq, index=False)
    with open(config.DATA_DERIVED / f"{name}_coverage.json", "w") as f_:
        json.dump(summary, f_, indent=2)

    print(f"\n✓ Saved {len(df):,} rows -> {out_pq}")
    print(f"✓ Saved coverage    -> {config.DATA_DERIVED / (name + '_coverage.json')}")
    print("\nNOTE: this is REAL mg/dL on a real clock, unlike the windowed")
    print("      npz (z-scored, no timestamps, capped at 3,000 windows per")
    print("      subject). Event-level metrics must be computed from here.")


if __name__ == "__main__":
    main()
