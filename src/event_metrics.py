"""
event_metrics.py
==================
Event-level (episode) evaluation of the alarm system.

WHY WINDOW-LEVEL PPV UNDERSTATES PERFORMANCE
--------------------------------------------
A child goes low at 10:00. The model fires at 9:30, 9:35, 9:40 and 9:45.
Window-level scoring counts three of those as false positives, even
though the warning was correct and early. Clinically the child received
ONE correct alarm, 30 minutes ahead.

Event-level scoring groups repeated predictions into alarm episodes,
matches them to true hypoglycaemic episodes, and reports what a patient
actually experiences:

    episode recall          fraction of true episodes warned about
    false alarms / day      unmatched alarm episodes per patient-day
    alarm precision         matched alarms / all alarms
    median lead time        minutes of warning before onset

This changes the MEASUREMENT, not the model. It is the fairer measure,
not a more generous one -- an alarm that fires 45 min before an episode
and keeps firing is one useful warning, not four errors.

DEFINITIONS (international CGM consensus)
-----------------------------------------
True episode : starts at the first of >= 3 consecutive readings < 70
               mg/dL (>= 15 min at 5-min sampling); ends only after >= 3
               consecutive readings >= 70, so brief rebounds do not split
               one episode into several.

Alarm episode: consecutive positive predictions collapsed using a
               REFRACTORY period (default = the prediction horizon).
               After an alarm fires, further positives are suppressed for
               that period.

Match        : an alarm episode is matched to a true episode if it occurs
               in the warning window
                   onset - PH  <=  alarm  <=  onset - min_lead
               Each alarm matches at most one episode and each episode at
               most one alarm; the FIRST matched alarm sets the lead time.

Alarms that fire while glucose is ALREADY below 70 are excluded by
default: those are detection, not early warning, and counting them as
false alarms would be unfair while counting them as successes would be
misleading.

THREE SEPARATE OPERATIONS -- do not conflate
--------------------------------------------
    persistence filter   before the first alarm  -- stops isolated
                         positives from firing (m of the last n)
    refractory period    after an alarm          -- stops repeat alerts
    episode grouping     at evaluation time      -- how outputs are counted

STRIDE
------
Event metrics require predictions on the real 5-minute grid, so this
script uses the stride-1 `windows_eval` dataset, not the stride-6
training set. Alarm timing at 30-minute resolution would be meaningless
for a 30-minute prediction horizon.

OPERATING POINT
---------------
Threshold and persistence are selected TOGETHER on the validation
subjects, minimising false alarms per day subject to
    episode recall >= min_event_recall
and a minimum median lead time. They are then LOCKED and applied to the
test subjects. Selecting them on test would be tuning on the test set.

Usage:
    python event_metrics.py --model xgboost --labels any --horizon 30
    python event_metrics.py --model rf --labels any --horizon 30 --min_event_recall 0.80
"""

import gc
import json
import time
import argparse
import warnings
import numpy as np
from pathlib import Path

import config
import common
import balancing as B
from stage1_classical_ml import build_model, NEEDS_SCALING, stratified_subsample
from build_windows import episode_onsets, HYPO_MGDL, SAMPLE_MIN

OUT_DIR = config.RESULTS / "event_level"


# ─── GROUND-TRUTH EPISODES ────────────────────────────────────────────────────

def true_episodes(raw_gluc, times, seg):
    """
    Consensus episodes for one subject's timeline.
    Returns array of onset TIMES (minutes since epoch).
    """
    onsets = []
    if len(seg) == 0:
        return np.asarray(onsets, dtype=np.int64)
    bounds = np.flatnonzero(np.diff(seg, prepend=seg[0] - 1))
    edges = list(bounds) + [len(seg)]
    prev = 0
    for e in edges:
        if e > prev:
            o = episode_onsets(raw_gluc[prev:e])
            onsets.extend(times[prev:e][o].tolist())
        prev = e
    return np.asarray(sorted(onsets), dtype=np.int64)


# ─── ALARM CONSTRUCTION ───────────────────────────────────────────────────────

def persistence_filter(flags, m, n):
    """
    m-of-last-n rule. flags is a boolean array in time order.

    m=1, n=1 is no filtering. m=2, n=2 is "two consecutive". m=2, n=3
    tolerates one interruption. A strict k-consecutive rule is m=k, n=k.
    """
    if m <= 1 and n <= 1:
        return flags.copy()
    f = flags.astype(np.int32)
    csum = np.concatenate([[0], np.cumsum(f)])
    out = np.zeros(len(f), dtype=bool)
    for i in range(len(f)):
        lo = max(0, i - n + 1)
        out[i] = (csum[i + 1] - csum[lo]) >= m
    return out


def alarm_times(times, fired, refractory_min):
    """
    Collapse a boolean firing series into alarm episodes using a
    refractory period. Returns the times at which alarms are issued.
    """
    out = []
    last = -np.inf
    idx = np.flatnonzero(fired)
    for i in idx:
        t = times[i]
        if t - last >= refractory_min:
            out.append(t)
            last = t
    return np.asarray(out, dtype=np.int64)


def match_alarms(alarms, onsets, horizon, min_lead=0):
    """
    Greedy one-to-one matching inside the warning window
        onset - horizon <= alarm <= onset - min_lead

    Earliest alarms are matched first, and the first alarm matched to an
    episode determines its lead time. Returns (n_matched, lead_times,
    n_false_alarms, n_episodes).
    """
    onsets = np.asarray(onsets)
    alarms = np.asarray(alarms)
    used_alarm = np.zeros(len(alarms), dtype=bool)
    leads = []
    matched = 0

    for s in onsets:
        lo, hi = s - horizon, s - min_lead
        cand = np.flatnonzero((alarms >= lo) & (alarms <= hi) & (~used_alarm))
        if len(cand):
            j = cand[0]                 # earliest -> longest useful warning
            used_alarm[j] = True
            leads.append(int(s - alarms[j]))
            matched += 1
    return matched, np.asarray(leads), int((~used_alarm).sum()), len(onsets)


# ─── PER-SUBJECT EVALUATION ───────────────────────────────────────────────────

def evaluate_subject(prob, times, raw_gluc, seg, onsets, thr, m, n,
                     horizon, refractory, min_lead, exclude_during_hypo):
    order = np.argsort(times)
    t, p, g = times[order], prob[order], raw_gluc[order]

    fired = p >= thr
    if exclude_during_hypo:
        # an alarm raised while glucose is already < 70 is detection, not
        # early warning; counting it either way would distort the result
        fired &= (g >= HYPO_MGDL)
    fired = persistence_filter(fired, m, n)
    alarms = alarm_times(t, fired, refractory)

    matched, leads, n_fa, n_ep = match_alarms(alarms, onsets, horizon, min_lead)

    gaps = np.diff(np.sort(t))
    days = float(gaps[(gaps > 0) & (gaps <= 30)].sum()) / (60 * 24)
    return {
        "n_episodes": int(n_ep),
        "n_detected": int(matched),
        "n_alarms": int(len(alarms)),
        "n_false_alarms": int(n_fa),
        "monitored_days": days,
        "episode_recall": float(matched / n_ep) if n_ep else float("nan"),
        "alarm_precision": float(matched / len(alarms)) if len(alarms) else float("nan"),
        "fa_per_day": float(n_fa / days) if days > 0 else float("nan"),
        "median_lead_min": float(np.median(leads)) if len(leads) else float("nan"),
    }


def aggregate(per_subject):
    """
    Macro (per-subject mean) and micro (pooled) aggregation.

    Macro weights every child equally; micro weights by monitored time.
    They differ when subjects have very different monitoring durations,
    so both are reported.
    """
    vals = list(per_subject.values())
    ep = np.array([v["n_episodes"] for v in vals], float)
    det = np.array([v["n_detected"] for v in vals], float)
    fa = np.array([v["n_false_alarms"] for v in vals], float)
    days = np.array([v["monitored_days"] for v in vals], float)
    alm = np.array([v["n_alarms"] for v in vals], float)
    leads = np.array([v["median_lead_min"] for v in vals], float)

    return {
        "macro_episode_recall": float(np.nanmean([v["episode_recall"] for v in vals])),
        "macro_fa_per_day": float(np.nanmean([v["fa_per_day"] for v in vals])),
        "macro_alarm_precision": float(np.nanmean([v["alarm_precision"] for v in vals])),
        "macro_median_lead_min": float(np.nanmedian(leads)),
        "micro_episode_recall": float(det.sum() / ep.sum()) if ep.sum() else float("nan"),
        "micro_fa_per_day": float(fa.sum() / days.sum()) if days.sum() else float("nan"),
        "micro_alarm_precision": float(det.sum() / alm.sum()) if alm.sum() else float("nan"),
        "total_episodes": int(ep.sum()),
        "total_detected": int(det.sum()),
        "total_alarms": int(alm.sum()),
        "total_false_alarms": int(fa.sum()),
        "total_patient_days": float(days.sum()),
        "n_subjects": len(vals),
    }


def bootstrap_event(per_subject, n_boot=10000, seed=0):
    """Cluster bootstrap over subjects for the two headline event metrics."""
    rng = np.random.default_rng(seed)
    subs = list(per_subject)
    n = len(subs)
    picks = rng.integers(0, n, size=(n_boot, n))
    out = {}
    for key in ["episode_recall", "fa_per_day", "alarm_precision", "median_lead_min"]:
        v = np.array([per_subject[s][key] for s in subs], float)
        reps = np.nanmean(v[picks], axis=1)
        reps = reps[~np.isnan(reps)]
        out[key] = {
            "mean": float(np.nanmean(v)),
            "lo": float(np.percentile(reps, 2.5)) if reps.size else float("nan"),
            "hi": float(np.percentile(reps, 97.5)) if reps.size else float("nan"),
        }
    return out


# ─── SPLIT-LEVEL DRIVER ───────────────────────────────────────────────────────

def build_subject_arrays(bw, mask):
    """Per-subject prediction times, raw glucose, segment ids and episodes."""
    end = bw.starts + bw.window_len - 1
    idx = np.flatnonzero(mask)
    subj = bw.subjects[bw.reading_subject[end[idx]]]
    out = {}
    for pid in np.unique(subj):
        sel = idx[subj == pid]
        e = end[sel]
        out[pid] = {
            "win_idx": sel,
            "times": bw.reading_times[e],
            "raw_gluc": bw.raw_gluc[e],
            "seg": bw.seg[e],
        }
        out[pid]["onsets"] = true_episodes(out[pid]["raw_gluc"],
                                           out[pid]["times"], out[pid]["seg"])
    return out


def sweep(subject_arrays, probs, thresholds, persistences, horizon,
          refractory, min_lead, exclude_during_hypo):
    """Evaluate every (threshold, persistence) combination."""
    rows = []
    for thr in thresholds:
        for (m, n) in persistences:
            per = {}
            for pid, d in subject_arrays.items():
                per[pid] = evaluate_subject(
                    probs[d["win_idx"]], d["times"], d["raw_gluc"], d["seg"],
                    d["onsets"], thr, m, n, horizon, refractory, min_lead,
                    exclude_during_hypo)
            agg = aggregate(per)
            rows.append({"threshold": float(thr), "persistence": f"{m}of{n}",
                         "m": m, "n": n, **agg, "_per_subject": per})
    return rows


def pareto_front(rows, min_recall=1e-9):
    """
    Configurations not dominated on (higher recall, fewer false alarms).
    A configuration is dominated if another achieves >= recall AND
    <= false alarms per day.

    Configurations that raise NO alarms are excluded first. They are
    trivially non-dominated -- zero false alarms, zero recall -- so they
    crowd the front with rows whose precision and lead time are undefined
    (NaN), and they are never usable operating points.
    """
    rows = [r for r in rows
            if r["macro_episode_recall"] > min_recall
            and np.isfinite(r.get("macro_alarm_precision", np.nan))]
    front = []
    for r in rows:
        dominated = any(
            (o["macro_episode_recall"] >= r["macro_episode_recall"]) and
            (o["macro_fa_per_day"] <= r["macro_fa_per_day"]) and
            (o is not r) and
            ((o["macro_episode_recall"] > r["macro_episode_recall"]) or
             (o["macro_fa_per_day"] < r["macro_fa_per_day"]))
            for o in rows)
        if not dominated:
            front.append(r)
    return sorted(front, key=lambda r: r["macro_fa_per_day"])


def select_operating_point(rows, min_event_recall, min_lead_min):
    """
    Fewest false alarms per day subject to the clinical constraints.
    Falls back to max recall if no configuration satisfies them.
    """
    ok = [r for r in rows
          if r["macro_episode_recall"] >= min_event_recall
          and (np.isnan(r["macro_median_lead_min"])
               or r["macro_median_lead_min"] >= min_lead_min)]
    if ok:
        return min(ok, key=lambda r: r["macro_fa_per_day"]), "constrained"
    return max(rows, key=lambda r: r["macro_episode_recall"]), "fallback_max_recall"



# ─── COLLECT (T7) ─────────────────────────────────────────────────────────────

def collect(args):
    """
    Build the event-level comparison table from every saved run.

    IMPORTANT CAVEAT, stated in the table itself: each model's threshold and
    persistence rule were selected independently on validation, so models sit
    at DIFFERENT points on their own recall/false-alarm curves. Comparing the
    selected points directly conflates model quality with operating-point
    choice. The table therefore also reports each model's recall interpolated
    to a COMMON false-alarm budget, taken from its validation Pareto front,
    which is the like-for-like comparison.
    """
    import pandas as pd

    files = sorted(OUT_DIR.glob("event_*.json"))
    if not files:
        print(f"No event-level results in {OUT_DIR}.")
        return

    rows, pareto, skipped = [], {}, []
    for f in files:
        r = json.load(open(f))
        cfg, ci, te = r["config"], r["test_ci"], r["test"]
        # Runs on a different label set are a DIFFERENT prediction task
        # (episode onset vs any low reading) with a different base rate.
        # Putting them in one comparison table would invite exactly the
        # cross-task comparison this project has argued against.
        if cfg.get("labels", "any") != args.labels:
            skipped.append(f.stem)
            continue
        name = f.stem.replace("event_", "")
        rec = ci["episode_recall"]["mean"]
        prec = ci["alarm_precision"]["mean"]
        rows.append({
            "model": name,
            "horizon": cfg["horizon"],
            "threshold": r["selected"]["threshold"],
            "persistence": r["selected"]["persistence"],
            "episode_recall": rec,
            "recall_lo": ci["episode_recall"]["lo"],
            "recall_hi": ci["episode_recall"]["hi"],
            "fa_per_day": ci["fa_per_day"]["mean"],
            "fa_lo": ci["fa_per_day"]["lo"],
            "fa_hi": ci["fa_per_day"]["hi"],
            "alarm_precision": prec,
            "prec_lo": ci["alarm_precision"]["lo"],
            "prec_hi": ci["alarm_precision"]["hi"],
            # harmonic mean of episode recall and alarm precision -- the event
            # analogue of F1. AUROC/AUPRC/specificity have no event-level
            # equivalent (no scored population, no true-negative episodes).
            "event_f1": (2 * rec * prec / (rec + prec)) if (rec + prec) > 0 else float("nan"),
            "median_lead_min": ci["median_lead_min"]["mean"],
            "episodes_detected": te["total_detected"],
            "episodes_total": te["total_episodes"],
            "patient_days": te["total_patient_days"],
            "window_ppv_same_thr": r["window_level_at_same_threshold"]["ppv"],
            "window_recall_same_thr": r["window_level_at_same_threshold"]["recall"],
        })
        pareto[name] = r.get("val_pareto", [])

    if not rows:
        print(f"No event runs with labels='{args.labels}'.")
        return
    if skipped:
        print(f"Excluded {len(skipped)} run(s) on a different label set: "
              f"{', '.join(skipped)}")
    df = pd.DataFrame(rows).sort_values(["horizon", "event_f1"],
                                        ascending=[True, False])

    # ---- like-for-like: recall at a common false-alarm budget -------------
    budgets = [1.0, 2.0, 3.0, 4.0]
    matched = []
    for name, front in pareto.items():
        if not front:
            continue
        fa = np.array([p["macro_fa_per_day"] for p in front])
        rc = np.array([p["macro_episode_recall"] for p in front])
        o = np.argsort(fa)
        fa, rc = fa[o], rc[o]
        row = {"model": name}
        for b in budgets:
            row[f"recall@{b}FA"] = (float(np.interp(b, fa, rc))
                                    if fa.min() <= b <= fa.max() else float("nan"))
        matched.append(row)
    mdf = pd.DataFrame(matched)

    OUT_T = config.RESULTS / "tables"
    OUT_T.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_T / "T7_event_level.csv", index=False, float_format="%.4f")
    if len(mdf):
        mdf.to_csv(OUT_T / "T7_event_matched_budget.csv", index=False,
                   float_format="%.4f")

    lines = ["# T7 — Event-level comparison", "",
             "Episodes: >= 3 consecutive CGM readings < 70 mg/dL. Alarms are "
             "grouped with a refractory period equal to the prediction "
             "horizon and matched to an episode if they fall in the warning "
             "window before onset. CIs are subject-level cluster bootstrap.",
             "",
             "`event_f1` is the harmonic mean of episode recall and alarm "
             "precision. AUROC, AUPRC and specificity have no event-level "
             "equivalent: there is no scored population and no well-defined "
             "true-negative episode.", "",
             "Runs on a different label set are excluded: episode-onset "
             "labels define a different prediction task with a different "
             "base rate, so they do not belong in the same comparison.", "",
             "## Selected operating points", "",
             "**Caveat:** each model's threshold was chosen independently on "
             "validation, so these rows sit at different points on their own "
             "curves. Use the matched-budget table below for a like-for-like "
             "model comparison.", ""]
    keep = ["model", "horizon", "threshold", "persistence", "episode_recall",
            "fa_per_day", "alarm_precision", "event_f1", "median_lead_min",
            "window_ppv_same_thr"]
    lines += [df[keep].to_markdown(index=False, floatfmt=".4f"), ""]

    if len(mdf):
        lines += ["## Episode recall at a matched false-alarm budget", "",
                  "Interpolated from each model's validation Pareto front. "
                  "This is the fair comparison: same alarm burden for every "
                  "model.", "",
                  mdf.to_markdown(index=False, floatfmt=".4f"), ""]
    (OUT_T / "T7_event_level.md").write_text("\n".join(lines))

    print(f"\n{'#'*100}")
    print("# T7 — EVENT LEVEL (selected operating points)")
    print(f"{'#'*100}")
    print(df[keep].to_string(index=False))
    if len(mdf):
        print(f"\n{'#'*100}")
        print("# EPISODE RECALL AT MATCHED FALSE-ALARM BUDGET (like-for-like)")
        print(f"{'#'*100}")
        print(mdf.to_string(index=False))
    print(f"\n✓ Saved -> {OUT_T / 'T7_event_level.md'} (+ CSVs)")



# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="xgboost")
    ap.add_argument("--loss", default=None,
                    help="stage-2 deep model: load its saved checkpoint "
                         "instead of fitting a classical model. e.g. "
                         "--model gru --loss weighted_bce")
    ap.add_argument("--balance", default="none")
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--horizon", type=int, default=30, choices=config.HORIZONS)
    ap.add_argument("--train_tag", default="stride6")
    ap.add_argument("--eval_tag", default="eval")
    ap.add_argument("--train_cap", type=int, default=200_000)
    ap.add_argument("--refractory", type=int, default=None,
                    help="minutes; default = the prediction horizon")
    ap.add_argument("--min_lead", type=int, default=0,
                    help="minimum useful warning time, minutes")
    ap.add_argument("--min_event_recall", type=float, default=0.80)
    ap.add_argument("--min_median_lead", type=float, default=10.0)
    ap.add_argument("--include_during_hypo", action="store_true",
                    help="count alarms raised while glucose is already <70")
    ap.add_argument("--n_boot", type=int, default=10_000)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--collect", action="store_true",
                    help="rebuild the T7 comparison table from saved runs")
    ap.add_argument("--recall_floor_note", action="store_true",
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.collect:
        collect(args)
        return
    refractory = args.refractory or args.horizon
    j = config.HORIZONS.index(args.horizon)
    t0 = time.time()

    deep = args.loss is not None

    # ---- deep model: reuse the stage-2 checkpoint, no retraining ----------
    if deep:
        import torch
        from stage2_deep import build, WindowDataset
        from torch.utils.data import DataLoader
        ckpt_name = f"{args.model}__{args.loss}__{args.balance}__{args.labels}"
        ckpt_path = config.RQ2_STAGES["stage2_dl"] / f"{ckpt_name}.pt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"No stage-2 checkpoint at {ckpt_path}. Run stage2_deep.py "
                f"with --model {args.model} --loss {args.loss} first.")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        bw = common.load_built(args.eval_tag)
        common.verify_split(bw.meta)
        masks = dict(zip(["train", "val", "test"], common.get_split(bw.meta)))
        Y = bw._labels if args.labels == "consensus" else bw._labels_any
        net = build(args.model, bw.n_channels, 64).to(device)
        net.load_state_dict(torch.load(ckpt_path, map_location=device)["model_state"])
        net.eval()
        print(f"Loaded {ckpt_name} (device {device})")
        print(f"Eval set: {len(bw):,} windows at stride {bw.stride} "
              f"({bw.stride * SAMPLE_MIN} min)")

        probs = {}
        with torch.no_grad():
            for split in ["val", "test"]:
                idx = np.flatnonzero(masks[split])
                print(f"  scoring {split}: {len(idx):,} windows...")
                ds = WindowDataset(bw, idx, np.zeros((len(bw), len(config.HORIZONS)),
                                                     np.float32))
                dl = DataLoader(ds, batch_size=4096, shuffle=False, num_workers=2)
                out = []
                for xb, _ in dl:
                    out.append(net(xb.to(device)).cpu().numpy()[:, j])
                p = np.zeros(len(bw), dtype=np.float32)
                p[idx] = np.concatenate(out).astype(np.float32)
                probs[split] = p
                del out
                gc.collect()
        return _finish(args, bw, masks, Y, probs, j, refractory, t0,
                       f"{args.model}__{args.loss}__{args.balance}")

    # ---- classical model: fit on the stride-6 set (cached features) -------
    from stage1_v2 import cached_features
    bw_tr = common.load_built(args.train_tag)
    common.verify_split(bw_tr.meta)
    tr, _, _ = common.get_split(bw_tr.meta)
    F_tr_all = cached_features(bw_tr, args.train_tag)
    Y_tr = bw_tr._labels if args.labels == "consensus" else bw_tr._labels_any

    idx_tr = np.flatnonzero(tr)
    y_tr = Y_tr[tr, j].astype(int)
    if args.train_cap and len(y_tr) > args.train_cap:
        keep = stratified_subsample(y_tr, args.train_cap, args.seed)
        idx_tr = idx_tr[keep]
    X_tr = np.asarray(F_tr_all[idx_tr])
    y_tr = Y_tr[idx_tr, j].astype(int)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_tr, y_tr, binfo = B.balance_features(
            X_tr, y_tr, method=args.balance, seed=args.seed, split_name="train")

    scaler = None
    if args.model in NEEDS_SCALING:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(X_tr)
        X_tr = scaler.transform(X_tr)

    cw = "balanced" if args.balance == "class_weight" else None
    clf = build_model(args.model, class_weight=cw, seed=args.seed)
    print(f"Training {args.model} (balance={args.balance}, labels={args.labels}, "
          f"h={args.horizon}) on {len(y_tr):,} rows...")
    clf.fit(X_tr, y_tr)
    del X_tr, F_tr_all, bw_tr
    gc.collect()

    # ---- predict on the stride-1 evaluation set ---------------------------
    bw = common.load_built(args.eval_tag)
    masks = dict(zip(["train", "val", "test"], common.get_split(bw.meta)))
    Y = bw._labels if args.labels == "consensus" else bw._labels_any
    print(f"Eval set: {len(bw):,} windows at stride {bw.stride} "
          f"({bw.stride * SAMPLE_MIN} min)")

    probs = {}
    for split in ["val", "test"]:
        idx = np.flatnonzero(masks[split])
        print(f"  scoring {split}: {len(idx):,} windows...")
        chunks = []
        for b in range(0, len(idx), 200_000):
            Xb = common.summary_features(bw.windows(idx[b:b + 200_000]),
                                         n_ch=bw.n_channels)
            if scaler is not None:
                Xb = scaler.transform(Xb)
            chunks.append(clf.predict_proba(Xb)[:, 1].astype(np.float32))
            del Xb
            gc.collect()
        p = np.zeros(len(bw), dtype=np.float32)
        p[idx] = np.concatenate(chunks)
        probs[split] = p
        del chunks
        gc.collect()

    return _finish(args, bw, masks, Y, probs, j, refractory, t0,
                   f"{args.model}__{args.balance}")


def _finish(args, bw, masks, Y, probs, j, refractory, t0, tag):
    """
    Operating-point selection and test evaluation.

    Shared by the classical and deep paths so that BOTH are evaluated with
    identical episode definitions, alarm grouping, persistence rules and
    selection criteria -- otherwise the models would not be comparable.
    """
    # ---- select the operating point on VALIDATION -------------------------
    thresholds = np.round(np.arange(0.05, 0.96, 0.05), 3)
    persistences = [(1, 1), (2, 2), (2, 3), (3, 3), (3, 4)]
    print(f"\nSweeping {len(thresholds)} thresholds x {len(persistences)} "
          f"persistence rules on validation subjects...")

    val_subj = build_subject_arrays(bw, masks["val"])
    val_rows = sweep(val_subj, probs["val"], thresholds, persistences,
                     args.horizon, refractory, args.min_lead,
                     not args.include_during_hypo)
    best, strategy = select_operating_point(val_rows, args.min_event_recall,
                                            args.min_median_lead)
    print(f"\nSelected on validation ({strategy}): threshold="
          f"{best['threshold']:.2f}, persistence={best['persistence']}")
    print(f"  val episode recall {best['macro_episode_recall']:.3f} | "
          f"FA/day {best['macro_fa_per_day']:.2f} | "
          f"precision {best['macro_alarm_precision']:.3f} | "
          f"lead {best['macro_median_lead_min']:.0f} min")

    front = pareto_front(val_rows)
    print(f"\nValidation Pareto front ({len(front)} non-dominated points):")
    print(f"  {'thr':>6} {'persist':>9} {'recall':>8} {'FA/day':>8} "
          f"{'precision':>10} {'lead':>6}")
    for r in front:
        print(f"  {r['threshold']:>6.2f} {r['persistence']:>9} "
              f"{r['macro_episode_recall']:>8.3f} {r['macro_fa_per_day']:>8.2f} "
              f"{r['macro_alarm_precision']:>10.3f} "
              f"{r['macro_median_lead_min']:>5.0f}m")

    # ---- apply the LOCKED operating point to TEST -------------------------
    test_subj = build_subject_arrays(bw, masks["test"])
    test_rows = sweep(test_subj, probs["test"], [best["threshold"]],
                      [(best["m"], best["n"])], args.horizon, refractory,
                      args.min_lead, not args.include_during_hypo)
    test = test_rows[0]
    ci = bootstrap_event(test["_per_subject"], n_boot=args.n_boot, seed=args.seed)

    print(f"\n{'#'*90}")
    print(f"# TEST — event level (threshold and persistence locked from validation)")
    print(f"{'#'*90}")
    print(f"  episode recall   {ci['episode_recall']['mean']:.3f} "
          f"[{ci['episode_recall']['lo']:.3f}, {ci['episode_recall']['hi']:.3f}]")
    print(f"  FA / patient-day {ci['fa_per_day']['mean']:.2f} "
          f"[{ci['fa_per_day']['lo']:.2f}, {ci['fa_per_day']['hi']:.2f}]")
    print(f"  alarm precision  {ci['alarm_precision']['mean']:.3f} "
          f"[{ci['alarm_precision']['lo']:.3f}, {ci['alarm_precision']['hi']:.3f}]")
    print(f"  median lead      {ci['median_lead_min']['mean']:.1f} min "
          f"[{ci['median_lead_min']['lo']:.1f}, {ci['median_lead_min']['hi']:.1f}]")
    print(f"\n  episodes {test['total_detected']}/{test['total_episodes']} detected "
          f"| {test['total_alarms']:,} alarms | {test['total_false_alarms']:,} false "
          f"| {test['total_patient_days']:.0f} patient-days")
    print(f"  micro: recall {test['micro_episode_recall']:.3f}  "
          f"FA/day {test['micro_fa_per_day']:.2f}  "
          f"precision {test['micro_alarm_precision']:.3f}")

    # window-level PPV at the same threshold, for the contrast
    yt = Y[masks["test"], j].astype(int)
    pt = probs["test"][masks["test"]]
    wm = common.compute_metrics(yt, pt, best["threshold"])
    print(f"\n  For contrast, WINDOW-level at the same threshold: "
          f"PPV {wm['ppv']:.3f}, recall {wm['recall']:.3f}")
    print(f"  Event-level precision is higher because repeated early warnings")
    print(f"  about one real episode count once, not many times.")

    def strip(rows):
        return [{k: v for k, v in r.items() if k != "_per_subject"} for r in rows]

    common.save_result(OUT_DIR,
                       f"event_{tag}__{args.labels}__h{args.horizon}",
                       {"config": {**vars(args), "refractory": refractory},
                        "selected": {k: v for k, v in best.items() if k != "_per_subject"},
                        "selection_strategy": strategy,
                        "val_sweep": strip(val_rows),
                        "val_pareto": strip(front),
                        "test": {k: v for k, v in test.items() if k != "_per_subject"},
                        "test_ci": ci,
                        "test_per_subject": test["_per_subject"],
                        "window_level_at_same_threshold": wm,
                        "minutes": (time.time() - t0) / 60})
    print(f"\n✓ Saved -> {OUT_DIR}")


if __name__ == "__main__":
    main()
