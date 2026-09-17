"""
build_windows.py
==================
Rebuild the windowed dataset from the RAW parquet, fixing four problems
in the existing `metabonet_windows_pediatric.npz`.

WHAT WAS WRONG WITH THE OLD WINDOWS
-----------------------------------
1. No timestamps -> event-level metrics impossible (no denominator for
   false-alarms-per-day, no way to match an alarm to an episode).
2. Glucose stored z-scored only -> cannot threshold at 70 mg/dL.
3. Exactly 3,000 windows per subject -- a cap. The subjects have ~74,000
   readings each, so roughly 4% of the available data was used, and the
   retained windows are not contiguous (no stride from 1 to 12 reproduces
   the overlap).
4. Per-subject normalisation used statistics from each subject's first
   70% INCLUDING held-out test subjects, whose evaluation windows span
   the full timeline. Transductive, and a reviewer will find it.

WHAT THIS PRODUCES
------------------
A compact representation, not materialised windows:

    timeline  (N_readings, C)  float32   normalised features
    raw_gluc  (N_readings,)    float32   glucose in mg/dL
    times     (N_readings,)    int64     minutes since epoch
    subject   (N_readings,)    int32     index into `subjects`
    seg       (N_readings,)    int32     contiguous-segment id
    starts    (N_windows,)     int64     index of each window's FIRST row
    labels    (N_windows, 4)   float32   consensus labels
    labels_any(N_windows, 4)   float32   legacy "any reading <70" labels

Materialising 3M windows x 60 x 5 channels would be ~3.6 GB and crash
Colab. Slicing `timeline[s : s+60]` on demand costs nothing and keeps the
file around 360 MB. `common.WindowDataset` does this for training.

CHANNELS
--------
The public MetaboNet release contains NO wearable signals (steps,
heartrate, skin_temp, air_temp, galvanic_skin_response, workout_intensity
are all 0.00% present, 0/244 subjects). This is why channels 4-11 were
dead. We therefore build 4 real channels plus one missingness indicator:

    0 glucose
    1 basal
    2 bolus
    3 carbs
    4 carbs_observed   (1 if this subject records carbs at all, else 0)

`carbs` is present for only 171/244 subjects, and the 73 without it are
exactly the 73 with `calories_burned`/`workout_duration` -- MetaboNet
merges at least two studies with different protocols. The indicator makes
"not recorded" distinguishable from "ate nothing", but it also encodes
which study a subject came from, so the model could learn a study
shortcut. Train with and without it (--no_carb_indicator) and report both.

`insulin` is excluded: it equals basal + bolus. The script verifies this
on the real data and reports the agreement rate rather than assuming it.

CAUSAL NORMALISATION
--------------------
Each reading is normalised using an expanding mean/std computed from that
subject's data STRICTLY BEFORE the window ends -- never from the future.
A warm-up period (default 1 day) is required before any window is emitted,
so early statistics are not estimated from a handful of points.

LABELS
------
Two label sets, both computed on RAW mg/dL:

  labels_any  (legacy, matches the old dataset): 1 if ANY reading in
              (t, t+h] is < 70. Kept for continuity with existing results.

  labels      (consensus, primary): 1 if a hypoglycaemic EPISODE ONSET
              falls in (t, t+h]. Onset = first reading of a run of >= 3
              consecutive readings < 70 (>= 15 min at 5-min sampling),
              per the international CGM consensus. An episode ends only
              after >= 3 consecutive readings >= 70, so brief rebounds do
              not split one episode into several.

Usage:
    python build_windows.py --stride 6                 # training set
    python build_windows.py --stride 1 --tag eval      # evaluation set
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
COLS = ["id", "date", "CGM", "basal", "bolus", "carbs", "insulin"]

SAMPLE_MIN = 5            # CGM grid, verified at 5.0 min for all 244 subjects
WINDOW = config.WINDOW_LEN
HORIZONS = config.HORIZONS

HYPO_MGDL = 70.0
EPISODE_MIN_READINGS = 3   # >= 15 min below 70
RECOVERY_MIN_READINGS = 3  # >= 15 min at/above 70 to end an episode

MAX_INTERP_MIN = 15        # interpolate CGM gaps up to this
MAX_GAP_MIN = 30           # gaps beyond this split the timeline
CGM_MIN, CGM_MAX = 20.0, 600.0

FEATURES = ["glucose", "basal", "bolus", "carbs", "carbs_observed"]


# ─── LOAD ─────────────────────────────────────────────────────────────────────

def load_raw(keep_ids, raw_parquet=None, verbose=True):
    raw_path = Path(raw_parquet) if raw_parquet is not None else RAW_PARQUET
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Raw MetaboNet parquet not found at {raw_path}. "
            "See data/README.md or pass --raw_parquet PATH.")
    f = pq.ParquetFile(str(raw_path))
    n_groups = f.metadata.num_row_groups
    keep = set(keep_ids)
    parts, ins_agree, ins_total = [], 0, 0

    if verbose:
        print(f"Streaming {f.metadata.num_rows:,} rows in {n_groups} groups")
    for g in range(n_groups):
        df = f.read_row_group(g, columns=COLS).to_pandas()
        df = df[df["CGM"].notna()]
        df["id"] = df["id"].astype(str)
        df = df[df["id"].isin(keep)]
        if len(df):
            # verify insulin == basal + bolus rather than assuming it
            b = df[["basal", "bolus"]].fillna(0.0).sum(axis=1)
            i = df["insulin"].fillna(0.0)
            ins_agree += int(np.isclose(b, i, atol=1e-6).sum())
            ins_total += len(df)
            parts.append(df[["id", "date", "CGM", "basal", "bolus", "carbs"]])
        del df
        if verbose and (g + 1) % 30 == 0:
            print(f"  {g+1:>3}/{n_groups}")
        gc.collect()

    out = pd.concat(parts, ignore_index=True)
    del parts
    gc.collect()

    if verbose and ins_total:
        print(f"\ninsulin == basal + bolus in "
              f"{100*ins_agree/ins_total:.2f}% of rows "
              f"-> {'redundant, excluded' if ins_agree/ins_total > 0.99 else 'NOT redundant, review'}")

    out = out[(out["CGM"] >= CGM_MIN) & (out["CGM"] <= CGM_MAX)]
    out = out.sort_values(["id", "date"]).drop_duplicates(["id", "date"])
    return out.reset_index(drop=True), (ins_agree / ins_total if ins_total else np.nan)


# ─── EPISODES ─────────────────────────────────────────────────────────────────

def episode_onsets(gluc, min_low=EPISODE_MIN_READINGS,
                   min_rec=RECOVERY_MIN_READINGS):
    """
    Consensus hypoglycaemic episodes on one contiguous segment.

    An episode starts at the first of >= min_low consecutive readings
    < 70 mg/dL, and ends only once >= min_rec consecutive readings are
    >= 70. A brief rebound shorter than the recovery requirement stays
    inside the same episode, so one long unstable low counts once rather
    than several times.

    Returns an int array of onset indices.
    """
    low = gluc < HYPO_MGDL
    n = len(low)
    onsets = []
    i = 0
    in_ep = False
    rec = 0
    while i < n:
        if not in_ep:
            if low[i] and i + min_low <= n and low[i:i + min_low].all():
                onsets.append(i)
                in_ep = True
                rec = 0
            i += 1
        else:
            rec = rec + 1 if not low[i] else 0
            if rec >= min_rec:
                in_ep = False
                rec = 0
            i += 1
    return np.asarray(onsets, dtype=np.int64)


def make_labels(gluc, seg_start, seg_end, horizons=HORIZONS):
    """
    Both label sets for every index of one segment.
    Returns (consensus (n,4), legacy_any (n,4)).
    """
    n = seg_end - seg_start
    g = gluc[seg_start:seg_end]
    cons = np.zeros((n, len(horizons)), dtype=np.float32)
    anyl = np.zeros((n, len(horizons)), dtype=np.float32)

    onsets = episode_onsets(g)
    onset_mask = np.zeros(n, dtype=bool)
    onset_mask[onsets] = True
    low = (g < HYPO_MGDL)

    # cumulative counts let us answer "any in (t, t+k]" in O(1)
    c_on = np.concatenate([[0], np.cumsum(onset_mask)])
    c_low = np.concatenate([[0], np.cumsum(low)])

    for j, h in enumerate(horizons):
        k = h // SAMPLE_MIN                       # readings ahead
        hi = np.minimum(np.arange(n) + 1 + k, n)  # exclusive end
        lo = np.minimum(np.arange(n) + 1, n)      # strictly future
        cons[:, j] = (c_on[hi] - c_on[lo]) > 0
        anyl[:, j] = (c_low[hi] - c_low[lo]) > 0
    return cons, anyl


# ─── BUILD ────────────────────────────────────────────────────────────────────

def build_subject(df_s, causal=True, warmup=288, carb_indicator=True):
    """
    Regularise one subject onto the 5-minute grid, interpolate short CGM
    gaps, split on long gaps, normalise causally, and label.

    Returns dict of arrays, or None if the subject yields no usable data.
    """
    df_s = df_s.sort_values("date")
    t = df_s["date"].values.astype("datetime64[m]").astype(np.int64)
    gluc = df_s["CGM"].to_numpy(np.float32)
    basal = df_s["basal"].fillna(0.0).to_numpy(np.float32)
    bolus = df_s["bolus"].fillna(0.0).to_numpy(np.float32)
    carbs_raw = df_s["carbs"]
    has_carbs = bool(carbs_raw.notna().any())
    carbs = carbs_raw.fillna(0.0).to_numpy(np.float32)

    # segment on gaps that are too long to bridge
    gaps = np.diff(t, prepend=t[0])
    seg_id = np.cumsum(gaps > MAX_GAP_MIN).astype(np.int32)

    # interpolate CGM across short gaps only (physiology is continuous at
    # this scale; longer gaps are sensor outages and must not be invented)
    s = pd.Series(gluc)
    max_steps = MAX_INTERP_MIN // SAMPLE_MIN
    gluc = s.interpolate(limit=max_steps, limit_area="inside").to_numpy(np.float32)
    ok = np.isfinite(gluc)
    if ok.sum() < WINDOW + max(HORIZONS) // SAMPLE_MIN + warmup:
        return None
    t, gluc, basal, bolus, carbs, seg_id = (a[ok] for a in
                                           (t, gluc, basal, bolus, carbs, seg_id))

    feats = np.stack([gluc, basal, bolus, carbs], axis=1).astype(np.float32)

    # ---- causal (expanding) normalisation --------------------------------
    if causal:
        c1 = np.cumsum(feats, axis=0)
        c2 = np.cumsum(feats ** 2, axis=0)
        cnt = np.arange(1, len(feats) + 1)[:, None].astype(np.float32)
        mean = c1 / cnt
        var = np.maximum(c2 / cnt - mean ** 2, 0.0)
        std = np.sqrt(var)
        # shift by one so a reading is never normalised using itself
        mean = np.vstack([mean[:1], mean[:-1]])
        std = np.vstack([std[:1], std[:-1]])
    else:
        mean = feats.mean(0, keepdims=True) * np.ones_like(feats)
        std = feats.std(0, keepdims=True) * np.ones_like(feats)
    norm = (feats - mean) / np.maximum(std, 1e-6)

    if carb_indicator:
        ind = np.full((len(norm), 1), 1.0 if has_carbs else 0.0, dtype=np.float32)
        norm = np.concatenate([norm, ind], axis=1)

    # ---- labels per segment ----------------------------------------------
    cons = np.zeros((len(gluc), len(HORIZONS)), dtype=np.float32)
    anyl = np.zeros_like(cons)
    bounds = np.flatnonzero(np.diff(seg_id, prepend=seg_id[0] - 1)) if len(seg_id) else []
    edges = list(bounds) + [len(seg_id)]
    prev = 0
    for e in edges:
        if e > prev:
            c, a = make_labels(gluc, prev, e)
            cons[prev:e], anyl[prev:e] = c, a
        prev = e

    return {"times": t, "raw_gluc": gluc, "norm": norm, "seg": seg_id,
            "cons": cons, "anyl": anyl, "has_carbs": has_carbs,
            "warmup": warmup}


def valid_starts(n, seg, stride, warmup, max_future):
    """
    Window start indices that are fully inside one segment, have the full
    label horizon available, and sit after the normalisation warm-up.
    """
    starts = np.arange(warmup, n - WINDOW - max_future + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        return starts
    same_seg = seg[starts] == seg[starts + WINDOW - 1]
    return starts[same_seg]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stride", type=int, default=6,
                    help="6 = one window per 30 min (training); 1 = every "
                         "5-min reading (evaluation / event metrics)")
    ap.add_argument("--subjects_from", default="metabonet")
    ap.add_argument("--raw_parquet", default=None,
                    help="optional path to metabonet_public.parquet; defaults "
                         "to config.RAW_DATA['metabonet']")
    ap.add_argument("--warmup", type=int, default=288,
                    help="readings required before the first window "
                         "(288 = 1 day) so causal stats are stable")
    ap.add_argument("--no_causal", action="store_true",
                    help="use whole-timeline stats (the OLD transductive "
                         "behaviour) -- for the normalisation ablation")
    ap.add_argument("--no_carb_indicator", action="store_true")
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    config.DATA_DERIVED.mkdir(parents=True, exist_ok=True)
    keep_ids = sorted(set(common.load_windows(args.subjects_from)[2].tolist()))
    print(f"Subjects: {len(keep_ids)}")

    df, ins_rate = load_raw(keep_ids, raw_parquet=args.raw_parquet)
    print(f"CGM rows after cleaning: {len(df):,}")

    max_future = max(HORIZONS) // SAMPLE_MIN
    subjects, tl, rg, tm, sb, sg, ST, LC, LA = [], [], [], [], [], [], [], [], []
    offset = 0
    skipped = []

    for si, (pid, g) in enumerate(df.groupby("id", sort=True)):
        out = build_subject(g, causal=not args.no_causal, warmup=args.warmup,
                            carb_indicator=not args.no_carb_indicator)
        if out is None:
            skipped.append(pid)
            continue
        n = len(out["times"])
        st = valid_starts(n, out["seg"], args.stride, args.warmup, max_future)
        if len(st) == 0:
            skipped.append(pid)
            continue

        subjects.append(pid)
        tl.append(out["norm"]); rg.append(out["raw_gluc"])
        tm.append(out["times"]); sg.append(out["seg"])
        sb.append(np.full(n, len(subjects) - 1, dtype=np.int32))
        # label index = last reading of the window (prediction is made there)
        lab_idx = st + WINDOW - 1
        ST.append(st + offset)
        LC.append(out["cons"][lab_idx]); LA.append(out["anyl"][lab_idx])
        offset += n
        if (si + 1) % 40 == 0:
            print(f"  {si+1}/{len(keep_ids)} subjects, {offset:,} readings, "
                  f"{sum(len(x) for x in ST):,} windows")

    pack = dict(
        timeline=np.concatenate(tl).astype(np.float32),
        raw_gluc=np.concatenate(rg).astype(np.float32),
        times=np.concatenate(tm).astype(np.int64),
        subject=np.concatenate(sb).astype(np.int32),
        seg=np.concatenate(sg).astype(np.int32),
        starts=np.concatenate(ST).astype(np.int64),
        labels=np.concatenate(LC).astype(np.float32),
        labels_any=np.concatenate(LA).astype(np.float32),
        subjects=np.array(subjects),
        features=np.array(FEATURES if not args.no_carb_indicator
                          else FEATURES[:-1]),
        window_len=np.int64(WINDOW), stride=np.int64(args.stride),
        horizons=np.array(HORIZONS), sample_min=np.int64(SAMPLE_MIN),
    )

    tag = args.tag or f"stride{args.stride}"
    if args.no_causal:
        tag += "_transductive"
    out_path = config.DATA_DERIVED / f"windows_{tag}.npz"
    np.savez_compressed(out_path, **pack)

    n_w = len(pack["starts"])
    print(f"\n{'='*72}")
    print(f"readings {len(pack['timeline']):,} | windows {n_w:,} | "
          f"subjects {len(subjects)}")
    if skipped:
        print(f"skipped {len(skipped)} subjects with insufficient data: {skipped[:8]}")
    print(f"\n{'h':>5} {'consensus':>11} {'legacy any':>11}  (positive rate)")
    for j, h in enumerate(HORIZONS):
        print(f"{h:>5} {pack['labels'][:, j].mean():>11.4f} "
              f"{pack['labels_any'][:, j].mean():>11.4f}")
    print(f"\nfeatures: {list(pack['features'])}")
    print(f"normalisation: {'TRANSDUCTIVE (ablation)' if args.no_causal else 'causal/expanding'}")
    print(f"\n✓ Saved -> {out_path}")

    with open(config.DATA_DERIVED / f"windows_{tag}_meta.json", "w") as f:
        json.dump({"n_windows": int(n_w), "n_readings": int(len(pack["timeline"])),
                   "n_subjects": len(subjects), "skipped": skipped,
                   "stride": args.stride, "causal": not args.no_causal,
                   "carb_indicator": not args.no_carb_indicator,
                   "insulin_redundancy_rate": float(ins_rate),
                   "positive_rate_consensus": {str(h): float(pack["labels"][:, j].mean())
                                               for j, h in enumerate(HORIZONS)},
                   "positive_rate_any": {str(h): float(pack["labels_any"][:, j].mean())
                                         for j, h in enumerate(HORIZONS)}}, f, indent=2)


if __name__ == "__main__":
    main()
