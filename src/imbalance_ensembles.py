"""
imbalance_ensembles.py
========================
Imbalance-aware ensemble learners -- the model family that was missing
from the RQ1 sweep.

WHY THESE ARE DIFFERENT FROM WHAT WE ALREADY TESTED
---------------------------------------------------
Everything tested so far resamples the training set ONCE and then fits an
ordinary learner:

    SMOTE(train) -> RandomForest

These methods instead resample INSIDE ensemble construction, so each base
learner sees a different balanced view of the majority class:

    BalancedRandomForest  each tree gets its own undersampled bootstrap,
                          so majority information differs across trees
                          rather than being discarded once
    RUSBoost              undersampling is applied at each boosting round,
                          so hard and minority cases can receive
                          progressively more attention
    EasyEnsemble          several balanced majority subsets, each with its
                          own boosted learner, recovering majority
                          information that plain undersampling throws away
    BalancedBagging       bagging with per-bag resampling (completeness)

That mechanical difference is why the earlier null result on one-shot
resampling does not settle the question for this family.

DIRECT PRECEDENT FOR THIS TASK
------------------------------
RUSBoost has been applied to exactly this problem: Fleischer et al.
trained RUSBoost on 3.7M CGM readings from 225 people with T1D to predict
hypoglycaemia within 40 minutes (hypo defined as <70 mg/dL sustained
>= 15 min, patient-level split), reporting ROC-AUC 0.988, PR-AUC 0.767,
event sensitivity 90%, mean lead time 17.5 min, and event-level false
positives 38%. It is therefore a comparator a reviewer may expect to see,
and a null result against it is meaningful rather than merely untried.

Note that their 62% event-level precision comes from EVENT-level
matching, not per-window PPV. That difference, not the choice of
learner, is the main reason their precision looks so much better than
window-level numbers.

DO NOT STACK THESE WITH EXTERNAL RESAMPLING
-------------------------------------------
These models already contain an imbalance mechanism. Running
SMOTE -> BalancedRandomForest applies two corrections at once and makes
the result uninterpretable. `check_no_double_balancing()` enforces this:
use them with balance='none'.

WHAT COUNTS AS SUCCESS
----------------------
Not recall or F1 at a fixed threshold -- those move whenever the score
distribution shifts. The tests that matter are:

  A. AUPRC on the untouched natural-prevalence test set, with
     subject-level bootstrap CIs. Threshold-free, so a gain here is a
     genuine ranking improvement.
  B. The event-level recall vs false-alarms-per-day frontier, after
     independently tuning each model's threshold on validation.

A 2024 study of postoperative mortality (1.7% prevalence, untouched
validation) found ROS/SMOTE genuinely raised XGBoost AUPRC from 0.135 to
0.484 -- while AUROC fell from 0.888 to 0.570. So AUPRC gains from
imbalance handling do occur, but they can come with discrimination losses
elsewhere. Report both.

Requires: pip install imbalanced-learn
"""

import numpy as np

ENSEMBLE_MODELS = ["balanced_rf", "rusboost", "easy_ensemble", "balanced_bagging"]

# Matched NATURAL counterparts -- identical learner, no internal resampling.
#
# Comparing RUSBoost against XGBoost does not isolate balancing: it compares
# two different algorithms, so a RUSBoost loss could simply mean XGBoost is
# stronger. Each balanced ensemble therefore has a counterpart that differs
# ONLY in the resampling step:
#
#   balanced_rf       <-> rf              (already matched in stage1)
#   rusboost          <-> adaboost        same base tree, n_estimators, lr
#   easy_ensemble     <-> bagged_adaboost bagging of the same boosted learner
#   balanced_bagging  <-> bagging         same base tree, n_estimators
#
# Only with these pairs can the paper say "internal resampling did / did not
# improve the learner" rather than "one library's model beat another's".
NATURAL_COUNTERPARTS = ["adaboost", "bagging", "bagged_adaboost"]

COUNTERPART_OF = {
    "balanced_rf": "rf",
    "rusboost": "adaboost",
    "easy_ensemble": "bagged_adaboost",
    "balanced_bagging": "bagging",
}

# These carry their own internal resampling.
SELF_BALANCING = set(ENSEMBLE_MODELS)


def check_no_double_balancing(model_name, balance_method):
    """
    Refuse to combine an internally-balancing ensemble with external
    resampling: two corrections at once cannot be attributed to either.
    """
    if model_name in SELF_BALANCING and balance_method not in ("none", None):
        raise ValueError(
            f"'{model_name}' already resamples internally; combining it with "
            f"balance='{balance_method}' applies two corrections at once and "
            f"makes the result uninterpretable. Run it with --balance none.")


def build_ensemble(name, seed=42, n_jobs=-1, n_estimators=None,
                   sampling_strategy=None, rusboost_depth=1,
                   learning_rate=0.1):
    """
    Build one imbalance-aware ensemble.

    sampling_strategy=None uses each estimator's CORRECT default rather
    than a single shared value. This matters: BalancedRandomForest's
    library default is "all" (bootstrap every class, matching the
    original Balanced Random Forest algorithm), but its DOCSTRING still
    says "auto" (= "not minority", which skips the minority bootstrap).
    An earlier version of this file passed "auto" explicitly, silently
    overriding the intended behaviour and mis-configuring BRF. Passing
    None avoids re-introducing that bug; pass an explicit value only to
    deliberately override.
    """
    from imblearn.ensemble import (BalancedRandomForestClassifier,
                                   RUSBoostClassifier,
                                   EasyEnsembleClassifier,
                                   BalancedBaggingClassifier)

    if name == "balanced_rf":
        # Tree hyperparameters MATCH the plain RandomForest in
        # stage1_classical_ml.build_model (n_estimators=200,
        # min_samples_leaf=20, max_depth=None) so the only difference
        # between the two arms is the internal resampling. Otherwise the
        # comparison would confound balancing with tree configuration.
        kw = dict(n_estimators=n_estimators or 200, min_samples_leaf=20,
                  replacement=True, bootstrap=False,
                  n_jobs=n_jobs, random_state=seed)
        if sampling_strategy is not None:
            kw["sampling_strategy"] = sampling_strategy
        return BalancedRandomForestClassifier(**kw)

    if name == "rusboost":
        # Boosting is built around WEAK learners; the library default base
        # estimator is a depth-1 stump. An earlier version used depth 6,
        # which lets each round fit its heavily undersampled subset far
        # too closely. Depth is now a parameter (default 1) and should be
        # selected on VALIDATION data, never from the test tables.
        from sklearn.tree import DecisionTreeClassifier
        kw = dict(estimator=DecisionTreeClassifier(max_depth=rusboost_depth),
                  n_estimators=n_estimators or 200,
                  learning_rate=learning_rate, random_state=seed)
        if sampling_strategy is not None:
            kw["sampling_strategy"] = sampling_strategy
        return RUSBoostClassifier(**kw)

    if name == "easy_ensemble":
        # Fewer estimators by default: each one is itself a boosted
        # ensemble, so this is far more expensive per unit than the others.
        kw = dict(n_estimators=n_estimators or 20, n_jobs=n_jobs,
                  random_state=seed)
        if sampling_strategy is not None:
            kw["sampling_strategy"] = sampling_strategy
        return EasyEnsembleClassifier(**kw)

    if name == "balanced_bagging":
        from sklearn.tree import DecisionTreeClassifier
        kw = dict(estimator=DecisionTreeClassifier(max_depth=12,
                                                   min_samples_leaf=20),
                  n_estimators=n_estimators or 100, n_jobs=n_jobs,
                  random_state=seed)
        if sampling_strategy is not None:
            kw["sampling_strategy"] = sampling_strategy
        return BalancedBaggingClassifier(**kw)

    # ---- matched natural counterparts (no internal resampling) ----------
    from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier)
    from sklearn.tree import DecisionTreeClassifier

    if name == "adaboost":
        # Matched to rusboost: same base tree, n_estimators, learning rate.
        return AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=rusboost_depth),
            n_estimators=n_estimators or 200,
            learning_rate=learning_rate, random_state=seed)

    if name == "bagging":
        # Matched to balanced_bagging: same base tree and n_estimators.
        return BaggingClassifier(
            estimator=DecisionTreeClassifier(max_depth=12, min_samples_leaf=20),
            n_estimators=n_estimators or 100,
            n_jobs=n_jobs, random_state=seed)

    if name == "bagged_adaboost":
        # Matched to easy_ensemble: bagging of the same boosted learner,
        # but each bag keeps the natural class distribution.
        return BaggingClassifier(
            estimator=AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=rusboost_depth),
                n_estimators=10, learning_rate=learning_rate,
                random_state=seed),
            n_estimators=n_estimators or 20,
            n_jobs=n_jobs, random_state=seed)

    raise ValueError(
        f"unknown model '{name}'. Options: "
        f"{ENSEMBLE_MODELS + NATURAL_COUNTERPARTS}")


def episode_multiplicity(y, meta, times, gap_min=60):
    """
    How many positive WINDOWS come from each biological EPISODE.

    Consecutive positive windows separated by <= gap_min minutes within a
    subject are treated as one episode. This matters because every
    resampling method here treats rows as independent observations: a long
    episode that generates ten positive windows already carries ten times
    the training weight of a short one, before any oversampling multiplies
    it again. If a small number of episodes produces most positive
    windows, there is episode-level imbalance hiding inside the class
    imbalance, and that should be reported.

    Returns a summary dict.
    """
    y = np.asarray(y).astype(int)
    meta = np.asarray(meta)
    times = np.asarray(times)

    sizes = []
    for s in np.unique(meta):
        m = (meta == s) & (y == 1)
        if not m.any():
            continue
        t = np.sort(times[m])
        breaks = np.flatnonzero(np.diff(t) > gap_min)
        starts = np.concatenate([[0], breaks + 1])
        ends = np.concatenate([breaks + 1, [len(t)]])
        sizes.extend((ends - starts).tolist())

    if not sizes:
        return {"n_episodes": 0, "n_positive_windows": 0}

    sizes = np.asarray(sizes)
    order = np.sort(sizes)[::-1]
    top10 = order[:max(1, len(order) // 10)].sum() / sizes.sum()
    return {
        "n_episodes": int(len(sizes)),
        "n_positive_windows": int(sizes.sum()),
        "windows_per_episode_mean": float(sizes.mean()),
        "windows_per_episode_median": float(np.median(sizes)),
        "windows_per_episode_max": int(sizes.max()),
        "frac_positives_from_top10pct_episodes": float(top10),
        "note": ("If frac_positives_from_top10pct_episodes is high, a few long "
                 "episodes dominate the positive class and resampling "
                 "amplifies them further."),
    }
