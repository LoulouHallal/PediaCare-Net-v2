"""
common.py
=========
Shared data, split, metric, bootstrap, and training utilities.

The original workspace imported ``subject_level_split`` from a file in an
older Google Drive repository. The exact historical split logic now lives in
``src/splits.py`` so this repository is standalone. ``verify_split`` always
checks the 170/36/38 counts and can optionally check exact validation/test
assignments against a private split-reference file supplied by an authorized
user.
"""

import json
import numpy as np
from pathlib import Path

from sklearn.metrics import (precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score,
                             precision_recall_curve, confusion_matrix)

import config

from splits import subject_level_split

HORIZONS = config.HORIZONS
METRICS = config.METRICS


# ─── DATA ─────────────────────────────────────────────────────────────────────

def load_windows(name):
    """
    Load a windowed dataset.
    Returns (X, Y, meta) with X float32 (N, 60, 12), Y float32 (N, 4),
    meta str array (N,) of subject IDs.
    """
    path = config.DATA[name]
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset '{name}' not found at {path}. "
            f"Run config.check_data() to see what is available.")
    d = np.load(path, allow_pickle=True)
    X = d["X"].astype(np.float32)
    Y = d["Y"].astype(np.float32)
    meta = np.array([str(m) for m in d["meta"]])
    return X, Y, meta


def get_split(meta, seed=None):
    """Subject-level train/validation/test masks using the frozen historical logic."""
    seed = config.SPLIT_SEED if seed is None else seed
    return subject_level_split(meta, seed=seed)


def verify_split(meta, expect=(170, 36, 38), verify_assignments=True):
    """
    Confirm that a dataset uses the historical thesis subject split.

    The count check always runs. If ``PEDIACARE_SPLIT_REFERENCE`` points to
    an authorized local JSON file and ``meta`` contains the complete
    244-subject cohort, the validation and test ID sets are checked exactly as
    well. The training assignment is then fixed as the complement.
    """
    tr, va, te = get_split(meta)
    got = tuple(len(np.unique(meta[m])) for m in (tr, va, te))
    if got != expect:
        raise RuntimeError(
            f"Split mismatch: got {got} subjects, expected {expect}. "
            f"New results would NOT be comparable to previous ones.")

    if verify_assignments and config.SPLIT_REFERENCE.exists():
        try:
            ref = json.loads(config.SPLIT_REFERENCE.read_text())
            all_subjects = {str(x) for x in np.unique(meta)}
            ref_total = int(ref.get("n_total", 244))
            if len(all_subjects) == ref_total:
                got_val = {str(x) for x in np.unique(meta[va])}
                got_test = {str(x) for x in np.unique(meta[te])}
                exp_val = {str(x) for x in ref["validation_subjects"]}
                exp_test = {str(x) for x in ref["test_subjects"]}
                if got_val != exp_val or got_test != exp_test:
                    missing_val = sorted(exp_val - got_val)
                    extra_val = sorted(got_val - exp_val)
                    missing_test = sorted(exp_test - got_test)
                    extra_test = sorted(got_test - exp_test)
                    raise RuntimeError(
                        "Historical subject assignment mismatch. "
                        f"val missing={missing_val[:5]} extra={extra_val[:5]}; "
                        f"test missing={missing_test[:5]} extra={extra_test[:5]}.")
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Invalid split reference file: {config.SPLIT_REFERENCE}") from exc
    return got

def base_rates(Y):
    """Positive-class prevalence per horizon. The imbalance we are studying."""
    return {h: float(Y[:, j].mean()) for j, h in enumerate(HORIZONS)}


def summary_features(X, n_ch=None):
    """
    Per-window summary features, for classical ML models and for
    feature-space SMOTE/ADASYN.

    Uses only the live channels by default (4-11 are all-zero in
    MetaboNet, so including them would add constant columns that break
    variance-based methods and inflate the feature count).
    """
    n_ch = config.LIVE_CHANNELS if n_ch is None else n_ch
    Xl = X[:, :, :n_ch]
    last = Xl[:, -1, :]
    feats = [
        Xl.mean(1), Xl.std(1), Xl.min(1), Xl.max(1), last,
        last - Xl[:, -4, :],        # 15-min delta
        last - Xl[:, -7, :],        # 30-min delta
        last - Xl[:, 0, :],         # full-window delta
        Xl[:, -12:, :].mean(1),     # last-hour mean
        Xl[:, -12:, :].std(1),      # last-hour volatility
    ]
    return np.concatenate(feats, axis=1).astype(np.float32)


def feature_names(n_ch=None):
    n_ch = config.LIVE_CHANNELS if n_ch is None else n_ch
    ch = config.FEATURE_NAMES[:n_ch]
    blocks = ["mean", "std", "min", "max", "last", "d15", "d30", "d_full",
              "hr_mean", "hr_std"]
    return [f"{b}_{c}" for b in blocks for c in ch]


# ─── METRICS ──────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_prob, threshold):
    """All six metrics at one operating point. NaN-safe for degenerate slices."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return {m: float("nan") for m in METRICS}
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        auroc=float(roc_auc_score(y_true, y_prob)),
        auprc=float(average_precision_score(y_true, y_prob)),
        ppv=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        specificity=float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan"),
    )


def find_threshold(y_true, y_prob, min_recall=None):
    """
    Max PPV subject to recall >= min_recall, using the exact PR curve.

    Defaults to CALIB_MIN_RECALL (0.85), not MIN_RECALL (0.80): a
    threshold selected to meet 0.80 exactly on a calibration slice
    undershoots it on held-out data roughly half the time. See handoff
    15.3 -- this cost 6 of 12 patients their safety constraint.
    """
    min_recall = config.CALIB_MIN_RECALL if min_recall is None else min_recall
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    if len(np.unique(y_true)) < 2:
        return 0.5, {"strategy": "no_positives"}

    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    prec, rec = prec[:-1], rec[:-1]          # drop the threshold=inf point
    ok = np.where(rec >= min_recall)[0]
    if len(ok):
        best = ok[np.argmax(prec[ok])]
        strategy = "recall_constrained"
    else:
        best = int(np.argmax(rec))
        strategy = "fallback_max_recall"
    return float(thr[best]), {
        "ppv": float(prec[best]), "recall": float(rec[best]),
        "strategy": strategy, "min_recall": float(min_recall)}


def evaluate(y_true, y_prob, meta, threshold):
    """
    Evaluate at both aggregation scales.

    Per-subject means are the PRIMARY scale. Pooling subjects with
    different score distributions distorts ranking metrics: on the
    previous phase's test set, pooled AUPRC was 0.14 LOWER than the
    per-subject mean at h=15 for the identical model and predictions.
    Pooled is reported alongside for comparability with prior work.
    """
    y_true = np.asarray(y_true).astype(int)
    pooled = compute_metrics(y_true, y_prob, threshold)

    subjects = sorted(np.unique(meta))
    per = {m: [] for m in METRICS}
    for s in subjects:
        ix = np.where(meta == s)[0]
        mm = compute_metrics(y_true[ix], y_prob[ix], threshold)
        for m in METRICS:
            per[m].append(mm[m])
    per_subject_mean = {m: float(np.nanmean(per[m])) for m in METRICS}

    rec = np.array(per["recall"], dtype=float)
    return {
        "pooled": pooled,
        "per_subject_mean": per_subject_mean,
        "per_subject": {s: {m: per[m][i] for m in METRICS}
                        for i, s in enumerate(subjects)},
        "n_subjects": len(subjects),
        "constraint": {
            "target": config.MIN_RECALL,
            "n_meeting": int(np.nansum(rec >= config.MIN_RECALL)),
            "worst_recall": float(np.nanmin(rec)) if rec.size else float("nan"),
            "worst_subject": (subjects[int(np.nanargmin(rec))]
                              if rec.size else None),
        },
    }


# ─── SUBJECT-LEVEL BOOTSTRAP ──────────────────────────────────────────────────

def bootstrap_per_subject(per_subject, n_boot=None, seed=0):
    """
    Cluster bootstrap over subjects, per-subject-mean scale.

    Resampling SUBJECTS, not windows: windows within a subject overlap in
    both features and labels, so window-level resampling treats correlated
    samples as independent and yields intervals that are far too narrow.

    `per_subject` is the dict returned by evaluate()["per_subject"].
    Returns {metric: {mean, lo, hi}}.
    """
    n_boot = config.N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    subjects = list(per_subject)
    n = len(subjects)
    picks = rng.integers(0, n, size=(n_boot, n))

    out = {}
    for m in METRICS:
        v = np.array([per_subject[s][m] for s in subjects], dtype=float)
        reps = np.nanmean(v[picks], axis=1)
        reps = reps[~np.isnan(reps)]
        out[m] = {
            "mean": float(np.nanmean(v)),
            "lo": float(np.percentile(reps, 2.5)) if reps.size else float("nan"),
            "hi": float(np.percentile(reps, 97.5)) if reps.size else float("nan"),
        }
    return out


def paired_delta(per_subject_a, per_subject_b, n_boot=None, seed=0):
    """
    Paired cluster bootstrap for (B - A), e.g. after-balancing minus
    before-balancing. Paired = the same resampled subjects for both arms,
    which cancels subject-sampling noise and makes the interval far
    tighter than comparing two independent CIs by eye.
    """
    n_boot = config.N_BOOT if n_boot is None else n_boot
    rng = np.random.default_rng(seed)
    subjects = [s for s in per_subject_a if s in per_subject_b]
    n = len(subjects)
    picks = rng.integers(0, n, size=(n_boot, n))

    out = {}
    for m in METRICS:
        a = np.array([per_subject_a[s][m] for s in subjects], dtype=float)
        b = np.array([per_subject_b[s][m] for s in subjects], dtype=float)
        d = np.nanmean(b[picks], axis=1) - np.nanmean(a[picks], axis=1)
        d = d[~np.isnan(d)]
        out[m] = {
            "delta": float(np.nanmean(b) - np.nanmean(a)),
            "lo": float(np.percentile(d, 2.5)) if d.size else float("nan"),
            "hi": float(np.percentile(d, 97.5)) if d.size else float("nan"),
            "p_gt_0": float(np.mean(d > 0)) if d.size else float("nan"),
        }
    return out


# ─── RESULTS I/O ──────────────────────────────────────────────────────────────

def save_result(out_dir, name, payload):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{name}.json"
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    return p


def print_table(rows, title, scale="per_subject_mean", metrics=None):
    """
    rows: list of (label, result_dict_per_horizon) where result_dict_per_horizon
          maps str(h) -> evaluate() output.
    """
    metrics = metrics or ["auroc", "auprc", "ppv", "recall", "f1"]
    print(f"\n{'='*100}\n{title}\n{'='*100}")
    for h in HORIZONS:
        print(f"\nh={h} min   ({scale})")
        print(f"  {'model':>22} " + " ".join(f"{m:>9}" for m in metrics))
        for label, res in rows:
            if str(h) not in res:
                continue
            m = res[str(h)][scale]
            print(f"  {label:>22} " + " ".join(f"{m[k]:>9.4f}" for k in metrics))


# ─── BUILT WINDOWS (new format from build_windows.py) ─────────────────────────

class BuiltWindows:
    """
    Loader for the compact timeline + index format produced by
    build_windows.py.

    The file stores a continuous per-subject timeline plus the start index
    of each window, rather than materialised windows: 2.76M windows x 60 x
    5 channels would be ~3.3 GB, while the timeline is ~360 MB. Windows are
    sliced on demand.

    Attributes
    ----------
    meta        (n_windows,) str  subject id per window -- same role as the
                                  `meta` array in the old npz, so get_split()
                                  and evaluate() work unchanged
    labels      (n_windows, 4)    consensus labels (primary)
    labels_any  (n_windows, 4)    legacy "any reading < 70" labels
    times       (n_windows,)      minutes-since-epoch of the LAST reading in
                                  each window, i.e. the moment the prediction
                                  is made -- this is what event-level metrics
                                  need
    """

    # Causal normalisation in build_windows divides by an expanding std with
    # a 1e-6 floor. `bolus` and `carbs` are sparse, so while a channel is
    # still all zeros its std is 0, the floor applies, and the FIRST non-zero
    # value becomes (1.5 - 0) / 1e-6 = 1.5e6. Values of that magnitude
    # saturate unbounded linear/conv layers (the TCN, Transformer and hybrid
    # runs all produced NaN) and dominate the min/max summary features used
    # by the classical models. Clamping removes the artifact; no genuine
    # z-scored physiological value lies beyond +/-10.
    CLAMP = 10.0

    def __init__(self, path, label_set="consensus", clamp=None):
        z = np.load(path, allow_pickle=True)
        self.timeline = z["timeline"]
        clamp = self.CLAMP if clamp is None else clamp
        if clamp:
            n_clipped = int((np.abs(self.timeline) > clamp).sum())
            if n_clipped:
                self.timeline = np.clip(self.timeline, -clamp, clamp)
                print(f"  clamped {n_clipped:,} extreme feature values to "
                      f"+/-{clamp} ({100*n_clipped/self.timeline.size:.4f}% "
                      f"of cells; normalisation artifact, see BuiltWindows.CLAMP)")
        self.clamp = clamp
        self.raw_gluc = z["raw_gluc"]
        self.reading_times = z["times"]
        self.reading_subject = z["subject"]
        self.seg = z["seg"]
        self.starts = z["starts"]
        self._labels = z["labels"]
        self._labels_any = z["labels_any"]
        self.subjects = np.array([str(s) for s in z["subjects"]])
        self.features = [str(f) for f in z["features"]]
        self.window_len = int(z["window_len"])
        self.stride = int(z["stride"])
        self.horizons = [int(h) for h in z["horizons"]]
        self.sample_min = int(z["sample_min"])
        self.label_set = label_set

        end = self.starts + self.window_len - 1
        self.meta = self.subjects[self.reading_subject[end]]
        self.times = self.reading_times[end]
        self.raw_gluc_at_pred = self.raw_gluc[end]

    @property
    def labels(self):
        return self._labels if self.label_set == "consensus" else self._labels_any

    def __len__(self):
        return len(self.starts)

    @property
    def n_channels(self):
        return self.timeline.shape[1]

    def window(self, i):
        s = self.starts[i]
        return self.timeline[s:s + self.window_len]

    def windows(self, idx):
        """Materialise a subset. Only call this on a subset that fits in RAM."""
        idx = np.asarray(idx)
        out = np.empty((len(idx), self.window_len, self.n_channels), dtype=np.float32)
        for k, i in enumerate(idx):
            s = self.starts[i]
            out[k] = self.timeline[s:s + self.window_len]
        return out

    def summary_features(self, idx=None, batch=200_000):
        """
        Summary features for the classical-ML path, computed in batches so
        the full window tensor is never held in memory at once.
        """
        idx = np.arange(len(self)) if idx is None else np.asarray(idx)
        chunks = []
        for b in range(0, len(idx), batch):
            chunks.append(summary_features(self.windows(idx[b:b + batch]),
                                           n_ch=self.n_channels))
        return np.concatenate(chunks) if chunks else np.empty((0, 0), np.float32)

    def subject_days(self):
        """
        Monitored days per subject -- the denominator for false alarms per
        patient-day. Counts only gaps that look like real sampling, so a
        multi-week sensor outage is not counted as monitored time.
        """
        out = {}
        for si, pid in enumerate(self.subjects):
            m = self.reading_subject == si
            t = self.reading_times[m]
            if len(t) < 2:
                out[pid] = 0.0
                continue
            g = np.diff(t)
            out[pid] = float(g[(g > 0) & (g <= 30)].sum()) / (60 * 24)
        return out


def load_built(tag="stride6", label_set="consensus", clamp=None):
    return BuiltWindows(config.DATA_DERIVED / f"windows_{tag}.npz",
                        label_set=label_set, clamp=clamp)


# ─── RESUMABLE TRAINING ───────────────────────────────────────────────────────

class TrainCheckpoint:
    """
    Per-epoch checkpointing so a disconnected session costs one epoch
    instead of the whole run.

    The scripts in this project previously saved only after training
    finished, so a Colab drop at epoch 23 of 30 discarded ~90 minutes of
    GPU. This writes after every epoch and can resume mid-run.

    Two files are kept per run:
        <name>.ckpt       latest epoch, for resuming
        <name>.best.ckpt  best-so-far weights, for the final model

    Writes are atomic -- to a .tmp path then renamed -- because a crash
    DURING a save would otherwise leave a truncated file that cannot be
    loaded, which is worse than having no checkpoint at all.

    Cost is roughly 3-5 s per epoch for a model of this size (~250 KB),
    about 2-3% of a 90-minute run.

    Usage
    -----
        ck = TrainCheckpoint(OUT_DIR, name, resume=args.resume)
        start, best, best_ep, bad = ck.load_into(model, opt, sched)
        for ep in range(start, args.epochs):
            ...
            improved = vauprc > best
            if improved: best, best_ep, bad = vauprc, ep, 0
            else: bad += 1
            ck.save(model, opt, sched, ep, best, best_ep, bad, improved)
            if bad >= patience: break
        ck.restore_best(model)
        ck.cleanup()
    """

    def __init__(self, out_dir, name, resume=True, verbose=True):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / f"{name}.ckpt"
        self.best_path = self.dir / f"{name}.best.ckpt"
        self.resume = resume
        self.verbose = verbose

    def _atomic_save(self, obj, path):
        import torch
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp)
        tmp.replace(path)                 # atomic on the same filesystem

    def load_into(self, model, opt=None, sched=None):
        """Returns (start_epoch, best_score, best_epoch, bad_epochs)."""
        import torch
        if not (self.resume and self.path.exists()):
            return 0, -1.0, -1, 0
        try:
            ck = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as e:
            if self.verbose:
                print(f"  checkpoint at {self.path.name} unreadable ({e}); "
                      f"starting fresh")
            return 0, -1.0, -1, 0
        model.load_state_dict(ck["model_state"])
        if opt is not None and ck.get("opt_state"):
            opt.load_state_dict(ck["opt_state"])
        if sched is not None and ck.get("sched_state"):
            sched.load_state_dict(ck["sched_state"])
        if self.verbose:
            print(f"  RESUMING from epoch {ck['epoch']+1} "
                  f"(best {ck['best']:.4f} @ epoch {ck['best_epoch']})")
        return ck["epoch"] + 1, ck["best"], ck["best_epoch"], ck["bad"]

    def save(self, model, opt, sched, epoch, best, best_epoch, bad,
             improved, extra=None):
        state = {k: v.detach().cpu().clone()
                 for k, v in model.state_dict().items()}
        payload = {"model_state": state, "epoch": epoch, "best": best,
                   "best_epoch": best_epoch, "bad": bad,
                   "opt_state": opt.state_dict() if opt is not None else None,
                   "sched_state": sched.state_dict() if sched is not None else None}
        if extra:
            payload.update(extra)
        self._atomic_save(payload, self.path)
        if improved:
            self._atomic_save({"model_state": state, "epoch": epoch,
                               "best": best}, self.best_path)

    def restore_best(self, model):
        """Load the best-scoring weights seen during training."""
        import torch
        if self.best_path.exists():
            ck = torch.load(self.best_path, map_location="cpu",
                            weights_only=False)
            model.load_state_dict(ck["model_state"])
            return ck.get("best")
        return None

    def cleanup(self):
        """Remove resume files once the run has completed successfully."""
        for p in (self.path, self.best_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
