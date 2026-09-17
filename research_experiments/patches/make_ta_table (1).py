"""
make_ta_table.py
==================
The representation table: 5-channel -> +TA -> +absolute state.

WHY THIS SCRIPT EXISTS
----------------------
T1_model_comparison is the PRE-TA table. Its parameter counts are 40,804 /
53,668 / 104,548 / 104,676 / 213,156 — all 5-channel. Nothing at 41,188
(GRU+TA) or 91,620 (TCN+TA). So the largest single effect in the entire
project, +0.092 AUPRC from threshold-aware features, has never appeared in
a table. It lives only in JSONs scattered across the architecture folders,
because each TA arm was run as the control arm of a different experiment.

This walks results/RQ2_models/, identifies every run by its channel count,
and assembles the comparison.

HOW RUNS ARE CLASSIFIED
-----------------------
By n_channels when the JSON records it, otherwise by parameter count
against the known values for each era. Anything unrecognised is listed
under "unclassified" rather than guessed at, so nothing is silently
mislabelled.

    5 channels   glucose, basal, bolus, carbs, carbs_observed
    7 channels   + ta_proximity_c, ta_downslope_vplus
    9 channels   + glucose_absolute, rate_absolute

WHAT IS COMPARED
----------------
Same architecture, same split, same loss, same protocol — only the input
representation differs. Where several seeds exist the mean and spread are
reported, because a single-seed delta is not interpretable against the
measured baseline variance of 0.00207.

Usage:
    python make_ta_table.py
    python make_ta_table.py --list        # just show what was found
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np

import config

HORIZONS = config.HORIZONS
OUT = config.RESULTS / "tables"

# parameter counts as a fallback when n_channels is absent
KNOWN = {
    40804: (5, "gru"), 53668: (5, "lstm"), 104548: (5, "tcn"),
    104676: (5, "transformer"), 213156: (5, "hybrid"),
    41188: (7, "gru"), 91620: (7, "tcn"), 76708: (7, "msd_tcn"),
    31988: (7, "dms_tcn"), 15974: (7, "tsl_gru"),
    41572: (9, "gru"),
}
FAMILY = {"gru": "GRU", "gru_ta": "GRU", "gru_abs": "GRU",
          "gru_ta_abs": "GRU", "lstm": "LSTM", "tcn": "TCN",
          "tcn_ta": "TCN", "transformer": "Transformer",
          "hybrid": "Hybrid", "tsl_gru": "TSL-GRU",
          "dms_tcn": "DMS-TCN", "msd_tcn": "MSD-TCN"}


def classify(rec):
    """Return (n_channels, family) or (None, None) if not identifiable."""
    ch = rec.get("n_channels")
    npar = rec.get("n_params")
    name = str(rec.get("model", "")).lower()

    if ch is None and npar in KNOWN:
        ch = KNOWN[npar][0]
    if ch is None:
        # name-based last resort
        if "abs" in name:
            ch = 9
        elif "_ta" in name:
            ch = 7
    fam = None
    for k, v in FAMILY.items():
        if name == k or name.startswith(k):
            fam = v
            break
    if fam is None and npar in KNOWN:
        fam = FAMILY.get(KNOWN[npar][1])
    return ch, fam


# ---------------------------------------------------------------------------
# CANONICAL SOURCES
#
# The first version of this script accepted every JSON under RQ2_models. That
# pooled lambda-frozen INTERVENTION runs into the GRU+TA baseline (26 runs,
# sd 0.074), mixed balancing VARIANTS into the 5-channel GRU as if they were
# seeds, and swept in a broken run at 0.4831. Each arm is therefore named
# explicitly here: folder, model, and the filename pattern that identifies a
# primary run rather than an ablation or intervention of it.
#
#   {seed} expands to the seed number. A file matches only if its stem equals
#   the pattern exactly, so gru_ta__s42__lamfrozen.json is excluded while
#   gru_ta__s42.json is kept.
# ---------------------------------------------------------------------------
# Filenames follow two conventions in this project:
#   stage2/stage345/tsl_gru:  {model}__{loss}__{balancing}__{labels}
#                             with __s43 / __s44 appended for the extra seeds
#                             and NO suffix on seed 42
#   dms_tcn:                  {model}__s{seed}
#
# tsl_gru also holds tune_t0..tune_t7 (the hyperparameter search) and
# __s43__best_t5. Those are excluded: mixing a tuned arm into an untuned
# baseline would measure optimisation effort rather than cell design.
_W = "weighted_bce__none__any"


def _seeds(model, folder, extra=("s43", "s44")):
    """Seed 42 carries no suffix; later seeds append __s43 / __s44."""
    return [(folder, f"{model}__{_W}", 42)] + \
           [(folder, f"{model}__{_W}__{e}", int(e[1:])) for e in extra]


CANON = {
    # These have __s43/__s44 files too. Taking seed 42 alone made the GRU
    # delta computed against its BEST seed while TCN used a 3-seed mean --
    # an asymmetry a reviewer would spot immediately.
    ("GRU", 5): _seeds("gru", "stage2_classical_dl"),
    ("LSTM", 5): _seeds("lstm", "stage2_classical_dl"),
    ("TCN", 5): _seeds("tcn", "stage345_advanced"),
    ("Transformer", 5): _seeds("transformer", "stage345_advanced"),
    ("Hybrid", 5): _seeds("hybrid", "stage345_advanced"),
    ("GRU", 7): _seeds("gru_ta", "tsl_gru"),
    ("TSL-GRU", 7): _seeds("tsl_gru", "tsl_gru"),
    ("TSL-static", 7): _seeds("tsl_static", "tsl_gru"),
    ("LiGRU-TA", 7): _seeds("ligru_ta", "tsl_gru", ()),
    ("TCN", 7): [("dms_tcn", "tcn_ta__s42", 42)],
    ("DMS-TCN", 7): [("dms_tcn", "dms_tcn__s42", 42)],
    ("MSD-TCN", 7): [("dms_tcn", "msd_tcn__s42", 42)],
    # 9-channel GRU: absolute_state/gru_ta_abs__s42 reports 0.8662, while
    # ar_rhu, ctf_ru, drs_gru and pac_gru each report 0.8640 from their own
    # 9-channel control arm -- four bit-identical nn.GRU runs against one.
    # The replicated value is the version of record; the single 0.8662 from
    # the original absolute-state pilot is treated as the outlier and is
    # reported in the discrepancy note below rather than used.
    ("GRU", 9): [("drs_gru", "gru_abs__s42", 42)],
}


def load_canonical(root):
    """Load exactly the named files. Report anything missing."""
    got, missing = defaultdict(list), []
    for (fam, ch), specs in CANON.items():
        for folder, stem, sd in specs:
            f = root / folder / f"{stem}.json"
            if not f.exists():
                missing.append(f"{folder}/{stem}.json")
                continue
            try:
                r = json.load(open(f))
                hz = r["horizons"]
                got[(fam, ch)].append({
                    "file": f"{folder}/{stem}", "seed": r.get("seed", sd),
                    "n_params": r.get("n_params"),
                    "auprc": {int(h): hz[str(h)]["per_subject_mean"]["auprc"]
                              for h in HORIZONS if str(h) in hz},
                    "ppv": {int(h): hz[str(h)]["per_subject_mean"]["ppv"]
                            for h in HORIZONS if str(h) in hz}})
            except Exception as e:
                missing.append(f"{folder}/{stem}.json ({type(e).__name__})")
    return got, missing


def scan(root):
    """Every result JSON that carries per-horizon AUPRC."""
    out = []
    for f in sorted(root.rglob("*.json")):
        if f.name.startswith("_"):
            continue
        try:
            r = json.load(open(f))
        except Exception:
            continue
        hz = r.get("horizons")
        if not isinstance(hz, dict):
            continue
        if not any(str(h) in hz for h in HORIZONS):
            continue
        try:
            aps = {int(h): hz[str(h)]["per_subject_mean"]["auprc"]
                   for h in HORIZONS if str(h) in hz}
        except (KeyError, TypeError):
            continue
        ch, fam = classify(r)
        out.append({"file": str(f.relative_to(root)), "folder": f.parent.name,
                    "model": r.get("model"), "seed": r.get("seed"),
                    "n_params": r.get("n_params"), "channels": ch,
                    "family": fam, "auprc": aps,
                    "ppv": {int(h): hz[str(h)]["per_subject_mean"]["ppv"]
                            for h in HORIZONS if str(h) in hz}})
    return out


def agg(runs, h):
    v = [r["auprc"][h] for r in runs if h in r["auprc"]]
    p = [r["ppv"][h] for r in runs if h in r["ppv"]]
    if not v:
        return None
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1))
            if len(v) > 1 else 0.0, "n": len(v),
            "ppv": float(np.mean(p)) if p else float("nan")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    root = Path(args.root) if args.root else config.RESULTS / "RQ2_models"
    if not root.exists():
        print(f"  {root} not found")
        return
    if args.list:
        runs = scan(root)
        print(f"Found {len(runs)} runs under {root}\n")
        print(f"  {'folder':>22} {'model':>16} {'seed':>5} {'params':>8} "
              f"{'ch':>3} {'family':>12} {'AUPRC h15':>10}")
        for r in runs:
            print(f"  {r['folder']:>22} {str(r['model']):>16} "
                  f"{str(r['seed']):>5} {str(r['n_params']):>8} "
                  f"{str(r['channels']):>3} {str(r['family']):>12} "
                  f"{r['auprc'].get(15, float('nan')):>10.4f}")
        return

    buckets, missing = load_canonical(root)
    if missing:
        print(f"  {len(missing)} canonical file(s) not found — those arms "
              f"will show as '—'.")
        # Print what IS in each affected folder, so the CANON patterns can be
        # corrected in one pass instead of guessing repeatedly.
        folders = sorted({m.split("/")[0] for m in missing})
        print(f"\n  Actual primary-looking JSONs in those folders "
              f"(interventions/ablations excluded by eye):")
        for fd in folders:
            d = root / fd
            if not d.exists():
                print(f"    {fd}/  — FOLDER DOES NOT EXIST")
                continue
            names = sorted(f.stem for f in d.glob("*.json")
                           if not f.name.startswith("_"))
            print(f"    {fd}/  ({len(names)} files)")
            for n in names:
                print(f"        {n}")
        print()
    print(f"Using {sum(len(v) for v in buckets.values())} explicitly named "
          f"runs:")
    for (fam, ch), rs in sorted(buckets.items()):
        print(f"  {fam:>12} {ch}ch  " +
              "  ".join(f"{r['file'].split('/')[-1]} "
                        f"{r['auprc'].get(15, float('nan')):.4f}" for r in rs))
    print()

    lines = []

    def emit(s=""):
        print(s)
        lines.append(s)

    emit("# Representation, not architecture")
    emit("")
    emit("Same architectures, same split, same loss, same protocol. Only the")
    emit("input representation differs.")
    emit("")
    emit("    5 channels   glucose, basal, bolus, carbs, carbs_observed")
    emit("    7 channels   + ta_proximity_c, ta_downslope_vplus")
    emit("    9 channels   + glucose_absolute, rate_absolute")
    emit("")

    for h in HORIZONS:
        emit(f"## h = {h} min")
        emit("")
        emit("| family | 5-ch AUPRC | +TA (7-ch) | Δ TA | +abs (9-ch) | "
             "Δ abs | Δ total |")
        emit("|---|---|---|---|---|---|---|")
        for fam in ["GRU", "TCN", "LSTM", "Transformer", "Hybrid"]:
            a5, a7, a9 = (agg(buckets.get((fam, c), []), h) for c in (5, 7, 9))
            if not a5 and not a7:
                continue

            def f(a):
                if not a:
                    return "—"
                return (f"{a['mean']:.4f}" +
                        (f" ± {a['sd']:.4f}" if a["n"] > 1 else "") +
                        (f" ({a['n']})" if a["n"] > 1 else ""))

            dta = f"**{a7['mean']-a5['mean']:+.4f}**" if (a5 and a7) else "—"
            dab = f"{a9['mean']-a7['mean']:+.4f}" if (a7 and a9) else "—"
            dto = f"**{a9['mean']-a5['mean']:+.4f}**" if (a5 and a9) else "—"
            emit(f"| {fam} | {f(a5)} | {f(a7)} | {dta} | {f(a9)} | {dab} "
                 f"| {dto} |")
        emit("")

    emit("Values are per-subject mean AUPRC; ± is the standard deviation and")
    emit("(n) the number of seeds where more than one was run.")
    emit("")
    emit("The 9-channel GRU is taken from drs_gru/gru_abs__s42 (0.8640), the")
    emit("value reproduced bit-identically by ar_rhu, ctf_ru, drs_gru and")
    emit("pac_gru. absolute_state/gru_ta_abs__s42 reports 0.8662 for the same")
    emit("nominal configuration; that single run is not used. Both are")
    emit("single-seed, so the absolute-state increment should be read as")
    emit("provisional.")
    emit("")
    emit("## Reading this")
    emit("")
    g5, g7 = agg(buckets.get(("GRU", 5), []), 15), agg(buckets.get(("GRU", 7), []), 15)
    t5, t7 = agg(buckets.get(("TCN", 5), []), 15), agg(buckets.get(("TCN", 7), []), 15)
    if g5 and g7:
        emit(f"Threshold-aware features improve the GRU by "
             f"{g7['mean']-g5['mean']:+.4f} AUPRC at h=15.")
    if t5 and t7:
        emit(f"They improve the TCN by {t7['mean']-t5['mean']:+.4f} — a "
             f"replication across a different")
        emit("computational family, which is what makes the finding a "
             "property of the")
        emit("representation rather than of one model.")
    emit("")
    emit("For comparison, the GRU's own run-to-run spread over five seeds is")
    emit("0.00207 mean AUPRC, and no architectural modification tested in "
        "this")
    emit("project exceeded it.")
    emit("")
    emit("Under causal per-patient z-scoring, 70 mg/dL maps to a different")
    emit("normalised value for every child and drifts as their running")
    emit("statistics update, so the network cannot locate the clinical")
    emit("threshold. TA restores it in units of that child's own variability.")
    emit("")
    emit("")
    emit("# Within the TA representation: proposed cells vs their baseline")
    emit("")
    emit("Once TA is supplied, the architecture comparison becomes fair —")
    emit("every arm below reads the same 7 channels. This is the table the")
    emit("TSL-GRU efficiency claim rests on: it must TIE on AUPRC for the")
    emit("parameter reduction to mean anything.")
    emit("")

    cells = {fam: rs for (fam, ch), rs in buckets.items() if ch == 7}
    order = ["GRU", "TSL-GRU", "TSL-static", "LiGRU-TA", "TCN", "DMS-TCN",
             "MSD-TCN"]
    ref = agg(cells.get("GRU", []), 15)

    emit("| model | params | " + " | ".join(f"AUPRC h={h}" for h in HORIZONS)
         + " | Δ vs GRU+TA (h=15) |")
    emit("|---|---|" + "---|" * (len(HORIZONS) + 1))
    for fam in order + [f for f in sorted(cells) if f not in order]:
        rs = cells.get(fam, [])
        if not rs:
            continue
        npar = next((r["n_params"] for r in rs if r["n_params"]), None)
        cols = []
        for h in HORIZONS:
            a = agg(rs, h)
            cols.append("—" if not a else
                        f"{a['mean']:.4f}" +
                        (f" ± {a['sd']:.4f}" if a["n"] > 1 else ""))
        a15 = agg(rs, 15)
        d = ("—" if (not ref or not a15 or fam == "GRU")
             else f"{a15['mean']-ref['mean']:+.4f}")
        emit(f"| {fam}{' + TA' if fam in ('GRU', 'TCN') else ''} | "
             f"{npar:,} | " + " | ".join(cols) + f" | {d} |")
    emit("")
    tsl = agg(cells.get("TSL-GRU", []), 15)
    if ref and tsl:
        dd = tsl["mean"] - ref["mean"]
        pr = next((r["n_params"] for r in cells["TSL-GRU"] if r["n_params"]),
                  None)
        pg = next((r["n_params"] for r in cells["GRU"] if r["n_params"]), None)
        NOISE = 0.00207
        verdict = ("inside the GRU's own 5-seed spread of "
                   f"{NOISE:.5f} — a tie, which is the claim"
                   if abs(dd) <= NOISE else
                   f"OUTSIDE the GRU's 5-seed spread of {NOISE:.5f}; check "
                   f"the source runs before reporting this as a tie")
        emit(f"TSL-GRU differs from GRU+TA by {dd:+.4f} AUPRC at h=15, "
             f"{verdict}.")
        if pr and pg:
            emit(f"It does so with {pr:,} parameters against {pg:,}, "
                 f"{100*(1-pr/pg):.0f}% fewer.")
        gs = agg(cells.get("GRU", []), 15)
        ts = agg(cells.get("TSL-GRU", []), 15)
        if gs and ts and gs["n"] > 1 and ts["n"] > 1:
            emit("")
            emit(f"TSL-GRU is the noisier of the two across seeds: sd "
                 f"{ts['sd']:.4f} against {gs['sd']:.4f} at h=15, and higher "
                 f"at every")
            emit(f"horizon. The tie holds against either spread, but it "
                 f"should be reported as")
            emit(f"\"ties, with higher seed variance\" rather than left for "
                 f"the reader to find in")
            emit(f"the ± column.")
        emit("")
        emit("All arms share hidden 64, 2 layers, dropout 0.2, Adam at 1e-3,")
        emit("batch 512 and ReduceLROnPlateau with patience 6. Only the")
        emit("recurrent cell differs, so the parameter reduction comes from")
        emit("cell design and not from a narrower model. No per-architecture")
        emit("tuning was performed; the shared settings were chosen on the")
        emit("GRU baseline, which is the conservative direction.")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "T13_representation.md").write_text("\n".join(lines) + "\n")
    try:
        import pandas as pd
        rows = []
        for (fam, ch), rs in sorted(buckets.items()):
            for h in HORIZONS:
                a = agg(rs, h)
                if a:
                    rows.append({"family": fam, "channels": ch, "horizon": h,
                                 "auprc_mean": a["mean"], "auprc_sd": a["sd"],
                                 "n_seeds": a["n"], "ppv_mean": a["ppv"]})
        pd.DataFrame(rows).to_csv(OUT / "T13_representation.csv", index=False)
        print(f"\n✓ Saved -> {OUT / 'T13_representation.md'} and .csv")
    except ImportError:
        print(f"\n✓ Saved -> {OUT / 'T13_representation.md'}")


if __name__ == "__main__":
    main()
