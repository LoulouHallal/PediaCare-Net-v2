"""
balancing.py
==============
Class-imbalance methods for RQ1, with the two traps in this problem
handled explicitly rather than silently.

TRAP 1 -- RESAMPLING MUST TOUCH TRAINING DATA ONLY
--------------------------------------------------
If validation or test data is rebalanced, PPV and AUPRC stop meaning
anything: they would be measured at a prevalence that does not occur in
reality. Base rates here are 4.2% / 5.8% / 8.8% / 14.0% (h=15/30/60/120);
resampling test data to 50% would inflate PPV several-fold for free.

Every function here takes ONLY training arrays and says so in its
signature. `assert_train_only()` is available as an explicit guard.

A second consequence: a model trained on rebalanced data outputs
probabilities calibrated to the RESAMPLED prevalence, not the real one.
Its raw scores are therefore not comparable to an unbalanced model's.
This is handled by selecting the decision threshold on the NATURAL-
prevalence validation set (common.find_threshold), which re-anchors the
operating point. Never select a threshold on resampled data.

TRAP 2 -- FOUR LABELS, ONE DATASET
----------------------------------
SMOTE and friends balance ONE binary target. We have four horizons.
Sequence models are multi-head and need a single training set, so a
choice is unavoidable:

  anchor='30'   balance w.r.t. the h=30 label (default; matches the
                stratified sampler used in the previous phase, so the
                comparison to earlier results is like-for-like)
  anchor='any'  positive if ANY horizon is positive (highest prevalence,
                mildest resampling)
  anchor='15'/'60'/'120'  balance w.r.t. that horizon

Classical models are trained one-per-horizon, so they can balance each
horizon independently -- pass anchor=str(h) in that loop.

The anchor is recorded in every result so the choice is visible.

TRAP 3 -- SMOTE ON RAW WINDOWS IS PHYSIOLOGICALLY DUBIOUS
---------------------------------------------------------
SMOTE interpolates between two samples. Interpolating two 60x12 glucose
trajectories from different children produces a curve no child produced,
and the smoothing artifact can be learnable. For sequence models the
honest options are random oversampling of REAL windows, class weighting,
or a sequence-aware generator (TimeGAN, later).

`smote`/`adasyn` on 3-D input is supported (it is what several papers in
this area do) but emits a warning and operates only on the live channels
to keep the interpolation space from being padded with constant zeros.
Report it as one arm among several, not as the method.

COST NOTE
---------
The training split is ~510,000 windows. Fully balancing h=15 (4.2%)
would synthesise ~450,000 new samples and run k-NN in 240 dimensions
over half a million points -- slow and memory-hungry in Colab. Hence
`sampling_strategy` defaults to 0.3 (minority raised to 30% of the
majority), not 1.0. Full balance is available but should be a deliberate
choice.

Requires: pip install imbalanced-learn
"""

import warnings
import numpy as np

import config

HORIZONS = config.HORIZONS
LIVE = config.LIVE_CHANNELS

METHODS = ["none", "random_over", "random_under", "smote", "adasyn",
           "smote_then_under", "class_weight"]


# ─── GUARDS ───────────────────────────────────────────────────────────────────

def assert_train_only(split_name):
    """Fail loudly if someone tries to resample val or test."""
    if str(split_name).lower() not in ("train", "training", "tr"):
        raise ValueError(
            f"Refusing to resample split '{split_name}'. Resampling is valid "
            f"for TRAINING data only -- rebalancing val/test would make PPV "
            f"and AUPRC meaningless (they would be measured at a prevalence "
            f"that does not exist).")


# ─── LABEL SELECTION ──────────────────────────────────────────────────────────

def anchor_label(Y, anchor="30"):
    """Reduce the (N, 4) multi-horizon label matrix to one binary target."""
    Y = np.asarray(Y)
    if anchor == "any":
        return (Y.sum(axis=1) > 0).astype(int)
    h = int(anchor)
    if h not in HORIZONS:
        raise ValueError(f"anchor must be 'any' or one of {HORIZONS}, got {anchor}")
    return Y[:, HORIZONS.index(h)].astype(int)


def imbalance_report(Y, label=""):
    """Prevalence per horizon plus imbalance ratio (majority:minority)."""
    Y = np.asarray(Y)
    rep = {}
    for j, h in enumerate(HORIZONS):
        p = float(Y[:, j].mean())
        rep[str(h)] = {
            "prevalence": p,
            "n_pos": int(Y[:, j].sum()),
            "n_neg": int((Y[:, j] == 0).sum()),
            "imbalance_ratio": float((1 - p) / p) if p > 0 else float("inf"),
        }
    if label:
        print(f"\n{label}")
        print(f"  {'h':>5} {'prevalence':>11} {'n_pos':>9} {'n_neg':>9} {'ratio':>8}")
        for h in HORIZONS:
            r = rep[str(h)]
            print(f"  {h:>5} {r['prevalence']:>11.4f} {r['n_pos']:>9,} "
                  f"{r['n_neg']:>9,} {r['imbalance_ratio']:>7.1f}:1")
    return rep


# ─── CLASS WEIGHTS (no resampling) ────────────────────────────────────────────

def class_weights(y):
    """
    Inverse-frequency weights {0: w0, 1: w1}, normalised to mean 1.
    Used for loss weighting or WeightedRandomSampler -- changes the
    training signal without fabricating or discarding data.
    """
    y = np.asarray(y).astype(int)
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return {0: 1.0, 1: 1.0}
    w0, w1 = n / (2.0 * n_neg), n / (2.0 * n_pos)
    return {0: float(w0), 1: float(w1)}


def sample_weights(y):
    """Per-sample weights for torch's WeightedRandomSampler."""
    y = np.asarray(y).astype(int)
    cw = class_weights(y)
    return np.where(y == 1, cw[1], cw[0]).astype(np.float64)


# ─── RESAMPLING ───────────────────────────────────────────────────────────────


def _over_strategy(y, sampling_strategy):
    """
    Validate an oversampling ratio against the data.

    imblearn's float `sampling_strategy` means n_minority_after =
    ratio * n_majority. If the minority is ALREADY at or above that
    ratio, imblearn raises rather than no-op. That happens easily here
    after majority subsampling, so we detect it and skip oversampling
    instead of crashing.

    Returns (strategy_or_None, current_ratio).
    """
    y = np.asarray(y).astype(int)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return None, float("inf") if n_neg == 0 else 0.0
    cur = n_pos / n_neg
    # 2% headroom: imblearn errors when the requested minority count rounds
    # to at or below the current count, which happens when cur is only
    # marginally under the target.
    if sampling_strategy is None or sampling_strategy <= cur * 1.02:
        return None, cur
    return sampling_strategy, cur


def _resampler(method, sampling_strategy, seed, k_neighbors=5):
    from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    if method == "random_over":
        return RandomOverSampler(sampling_strategy=sampling_strategy,
                                 random_state=seed)
    if method == "random_under":
        return RandomUnderSampler(sampling_strategy=sampling_strategy,
                                  random_state=seed)
    if method == "smote":
        return SMOTE(sampling_strategy=sampling_strategy, random_state=seed,
                     k_neighbors=k_neighbors)
    if method == "adasyn":
        return ADASYN(sampling_strategy=sampling_strategy, random_state=seed,
                      n_neighbors=k_neighbors)
    raise ValueError(f"unknown method {method}")


def balance_features(X_feat, y, method="none", sampling_strategy=0.3,
                     seed=42, split_name="train", k_neighbors=5):
    """
    Balance a 2-D feature matrix (classical ML path).

    X_feat : (N, F) summary features from common.summary_features
    y      : (N,) binary
    Returns (X_res, y_res, info).
    """
    assert_train_only(split_name)
    X_feat = np.asarray(X_feat, dtype=np.float32)
    y = np.asarray(y).astype(int)
    before = {"n": int(len(y)), "n_pos": int(y.sum()),
              "prevalence": float(y.mean())}

    if method in ("none", "class_weight"):
        info = {"method": method, "before": before, "after": before,
                "sampling_strategy": None,
                "class_weights": class_weights(y) if method == "class_weight" else None}
        return X_feat, y, info

    eff, cur = _over_strategy(y, sampling_strategy)
    if method in ("smote", "adasyn", "random_over", "smote_then_under") and eff is None:
        info = {"method": method, "before": before, "after": before,
                "sampling_strategy": sampling_strategy, "class_weights": None,
                "skipped": f"minority already at ratio {cur:.3f} >= target "
                           f"{sampling_strategy}; no oversampling applied"}
        return X_feat, y, info

    if method == "smote_then_under":
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE
        # Oversample minority to `sampling_strategy`, then trim the majority
        # to 1:1 with the enlarged minority. Keeps the final set small enough
        # to train on while reaching full balance.
        pipe = Pipeline([
            ("over", SMOTE(sampling_strategy=sampling_strategy,
                           random_state=seed, k_neighbors=k_neighbors)),
            ("under", RandomUnderSampler(sampling_strategy=1.0,
                                         random_state=seed)),
        ])
        Xr, yr = pipe.fit_resample(X_feat, y)
    else:
        Xr, yr = _resampler(method, sampling_strategy, seed,
                            k_neighbors).fit_resample(X_feat, y)

    after = {"n": int(len(yr)), "n_pos": int(yr.sum()),
             "prevalence": float(yr.mean())}
    return Xr.astype(np.float32), yr.astype(int), {
        "method": method, "before": before, "after": after,
        "sampling_strategy": sampling_strategy, "class_weights": None}


def balance_sequences(X, Y, method="none", anchor="30", sampling_strategy=0.3,
                      seed=42, split_name="train", k_neighbors=5,
                      max_smote_samples=200_000):
    """
    Balance 3-D windowed data (sequence-model path).

    X : (N, 60, 12)   Y : (N, 4)
    Returns (X_res, Y_res, info).

    Index-based methods (random_over / random_under) select REAL windows,
    so multi-horizon labels stay intact and physiologically valid --
    these are the preferred options for sequence models.

    Synthetic methods (smote / adasyn) interpolate trajectories. They
    operate on the live channels only, dead channels are re-zeroed, and
    the non-anchor horizon labels of synthetic samples are copied from
    the nearest real minority sample rather than invented. A warning is
    emitted. Use as a comparison arm, not the default.
    """
    assert_train_only(split_name)
    X = np.asarray(X, dtype=np.float32)
    Y = np.asarray(Y, dtype=np.float32)
    y = anchor_label(Y, anchor)
    before = {"n": int(len(y)), "n_pos": int(y.sum()),
              "prevalence": float(y.mean())}
    base_info = {"method": method, "anchor": anchor, "before": before,
                 "sampling_strategy": sampling_strategy,
                 "class_weights": None, "synthetic": False}

    if method == "none":
        return X, Y, {**base_info, "after": before}

    if method == "class_weight":
        return X, Y, {**base_info, "after": before,
                      "class_weights": class_weights(y)}

    # ---- index-based: real windows only -------------------------------------
    if method in ("random_over", "random_under"):
        idx = np.arange(len(y)).reshape(-1, 1)
        idx_r, _ = _resampler(method, sampling_strategy, seed,
                              k_neighbors).fit_resample(idx, y)
        sel = idx_r.ravel()
        Xr, Yr = X[sel], Y[sel]
        after = {"n": int(len(sel)), "n_pos": int(anchor_label(Yr, anchor).sum()),
                 "prevalence": float(anchor_label(Yr, anchor).mean())}
        return Xr, Yr, {**base_info, "after": after,
                        "n_unique_real_windows": int(len(np.unique(sel)))}

    # ---- synthetic: SMOTE / ADASYN ------------------------------------------
    if method not in ("smote", "adasyn", "smote_then_under"):
        raise ValueError(f"unknown method {method}")

    warnings.warn(
        f"'{method}' interpolates between glucose trajectories, producing "
        f"windows no patient generated. Report it alongside random_over / "
        f"class_weight rather than as the primary balancing method.",
        UserWarning, stacklevel=2)

    n, T, C = X.shape
    # Subsample before SMOTE if the split is large: k-NN over 500k points in
    # 240-D is impractical in Colab. Subsampling is stratified on the anchor
    # so the minority is fully retained.
    if n > max_smote_samples:
        # STRATIFIED subsample that PRESERVES the original prevalence.
        #
        # An earlier version kept all minority samples and trimmed only the
        # majority. That raised the subsample's prevalence close to (or past)
        # the target ratio, leaving SMOTE almost nothing to generate and
        # making ADASYN fail outright with "No samples will be generated".
        # Preserving prevalence keeps the imbalance being studied intact and
        # gives the oversampler real work to do.
        rng = np.random.default_rng(seed)
        pos = np.where(y == 1)[0]
        neg = np.where(y == 0)[0]
        frac = max_smote_samples / n
        keep_pos = rng.choice(pos, size=max(int(len(pos) * frac), 2), replace=False)
        keep_neg = rng.choice(neg, size=max(int(len(neg) * frac), 2), replace=False)
        sub = np.sort(np.concatenate([keep_pos, keep_neg]))
        Xs, Ys, ys = X[sub], Y[sub], y[sub]
        subsampled = {"from": int(n), "to": int(len(sub)),
                      "note": "stratified subsample preserving prevalence "
                              "(both classes reduced proportionally)"}
    else:
        Xs, Ys, ys = X, Y, y
        subsampled = None

    eff, cur = _over_strategy(ys, sampling_strategy)
    if eff is None:
        after = {"n": int(len(ys)), "n_pos": int(ys.sum()),
                 "prevalence": float(ys.mean())}
        return Xs, Ys, {**base_info, "after": after, "synthetic": False,
                        "subsampled": subsampled,
                        "skipped": f"minority already at ratio {cur:.3f} >= "
                                   f"target {sampling_strategy}; no synthesis"}

    flat = Xs[:, :, :LIVE].reshape(len(Xs), T * LIVE)
    if method == "smote_then_under":
        from imblearn.pipeline import Pipeline
        from imblearn.under_sampling import RandomUnderSampler
        from imblearn.over_sampling import SMOTE
        pipe = Pipeline([
            ("over", SMOTE(sampling_strategy=sampling_strategy,
                           random_state=seed, k_neighbors=k_neighbors)),
            ("under", RandomUnderSampler(sampling_strategy=1.0, random_state=seed)),
        ])
        flat_r, y_r = pipe.fit_resample(flat, ys)
    else:
        flat_r, y_r = _resampler(method, sampling_strategy, seed,
                                 k_neighbors).fit_resample(flat, ys)

    n_new = len(y_r)
    Xr = np.zeros((n_new, T, C), dtype=np.float32)
    Xr[:, :, :LIVE] = flat_r.reshape(n_new, T, LIVE).astype(np.float32)
    # dead channels stay zero, matching the real data

    # Multi-horizon labels for the resampled set.
    #
    # We do NOT assume the resampler returns the original rows first and
    # synthetic rows after: `smote_then_under` undersamples afterwards and
    # reorders, which silently scrambled labels in an earlier version (the
    # anchor prevalence came back 0.09 instead of 0.50). Instead every output
    # row is matched to its nearest row in the ORIGINAL feature space:
    #   - a retained real row matches itself at distance 0 -> exact labels
    #   - a synthetic row matches the closest real (minority) sample, so the
    #     other three horizons are copied from a real patient window rather
    #     than fabricated.
    # The anchor label is then overwritten with what the resampler produced,
    # so the balanced target is exact by construction.
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(flat)
    dist, near = nn.kneighbors(flat_r)
    Yr = Ys[near.ravel()].astype(np.float32).copy()
    if anchor != "any":
        Yr[:, HORIZONS.index(int(anchor))] = y_r.astype(np.float32)
    n_synth = int((dist.ravel() > 1e-6).sum())

    after = {"n": int(n_new), "n_pos": int(anchor_label(Yr, anchor).sum()),
             "prevalence": float(anchor_label(Yr, anchor).mean())}
    return Xr, Yr, {**base_info, "after": after, "synthetic": True,
                    "n_synthetic": n_synth, "subsampled": subsampled}


# ─── REPORTING ────────────────────────────────────────────────────────────────

def print_balance_info(info):
    b, a = info["before"], info["after"]
    print(f"  balancing: {info['method']}"
          + (f" (anchor h={info['anchor']})" if info.get("anchor") else ""))
    print(f"    before: n={b['n']:>9,}  pos={b['n_pos']:>8,}  "
          f"prevalence={b['prevalence']:.4f}")
    print(f"    after : n={a['n']:>9,}  pos={a['n_pos']:>8,}  "
          f"prevalence={a['prevalence']:.4f}")
    if info.get("synthetic"):
        print(f"    synthetic samples created: {info.get('n_synthetic', 0):,}")
    if info.get("skipped"):
        print(f"    SKIPPED: {info['skipped']}")
    if info.get("subsampled"):
        s = info["subsampled"]
        print(f"    NOTE: majority subsampled {s['from']:,} -> {s['to']:,} before SMOTE")
    if info.get("class_weights"):
        cw = info["class_weights"]
        print(f"    class weights: neg={cw[0]:.3f} pos={cw[1]:.3f}")
