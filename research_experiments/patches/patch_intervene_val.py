"""
patch_intervene_val.py  --  report the intervention on VALIDATION too
======================================================================

tsl_intervene.py already computes val_probs for every condition, but uses
them only to pick thresholds. The reported table is test-only.

The test split in this project is development evidence -- 18 architectures
were selected after inspecting it -- so a headline result should not rest
on it. Every other decision in RQ2 was made on validation, and the
intervention should be reported on the same split as everything else.

WHAT CHANGES
------------
For each horizon and condition, the same evaluation and the same paired
bootstrap are run on validation as on test. Both appear in the printed
table and in the saved JSON:

    entry[cond]["val"]   evaluated on idx_va (36 subjects)
    entry[cond]          unchanged, test (38 subjects)

The threshold is still selected on validation per condition, which is
correct: the score distribution shifts under intervention, so a shared
threshold would confound calibration with discrimination. On the
validation rows this means threshold selection and scoring use the same
data -- that is optimistic in absolute terms, but it is identical across
the three conditions, so the DELTAS (which are the result) stay valid.
The test rows remain the clean-threshold version.

Nothing about the existing test numbers changes.

    python patch_intervene_val.py
    python patch_intervene_val.py --revert
"""

import argparse
import difflib
import os
import shutil
import sys

TARGET = "tsl_intervene.py"

OLD_HEADER = '''        for cond in CONDITIONS:
            # threshold re-selected on validation per condition: the score
            # distribution shifts under intervention, so reusing one
            # threshold would confound calibration with discrimination
            thr, _ = common.find_threshold(yv, val_probs[cond][:, j])
            ev = common.evaluate(yt, probs[cond][:, j], meta_te, thr)
            ev["threshold"] = float(thr)
            entry[cond] = ev
            m = ev["per_subject_mean"]
            print(f"  {cond:>10} {m['auroc']:>8.4f} {m['auprc']:>8.4f} "
                  f"{m['ppv']:>8.4f} {m['recall']:>8.4f} {m['f1']:>8.4f}")'''

NEW_HEADER = '''        for cond in CONDITIONS:
            # threshold re-selected on validation per condition: the score
            # distribution shifts under intervention, so reusing one
            # threshold would confound calibration with discrimination
            thr, _ = common.find_threshold(yv, val_probs[cond][:, j])
            ev = common.evaluate(yt, probs[cond][:, j], meta_te, thr)
            ev["threshold"] = float(thr)
            evv = common.evaluate(yv, val_probs[cond][:, j], meta_va, thr)
            evv["threshold"] = float(thr)
            entry[cond] = ev
            entry_val[cond] = evv
            m, mv = ev["per_subject_mean"], evv["per_subject_mean"]
            print(f"  {cond:>10} {mv['auprc']:>10.4f} {m['auprc']:>10.4f} "
                  f"{mv['recall']:>9.4f} {m['recall']:>9.4f}")'''

OLD_DELTA = '''        for other in ["constant", "shuffled"]:
            d = common.paired_delta(entry[other]["per_subject"],
                                    entry["normal"]["per_subject"],
                                    n_boot=args.n_boot)
            a = d["auprc"]
            sig = "*" if (a["lo"] > 0 or a["hi"] < 0) else " "
            verdict = ("temporal variation MATTERS" if a["lo"] > 0 else
                       "no evidence it matters" if a["hi"] > 0 else
                       "intervention IMPROVED the model")
            print(f"  normal - {other:<9}: dAUPRC {a['delta']:+.4f} "
                  f"[{a['lo']:+.4f}, {a['hi']:+.4f}]{sig}  -> {verdict}")
            entry[f"delta_normal_minus_{other}"] = d'''

NEW_DELTA = '''        for other in ["constant", "shuffled"]:
            d = common.paired_delta(entry[other]["per_subject"],
                                    entry["normal"]["per_subject"],
                                    n_boot=args.n_boot)
            dv = common.paired_delta(entry_val[other]["per_subject"],
                                     entry_val["normal"]["per_subject"],
                                     n_boot=args.n_boot)
            a, av = d["auprc"], dv["auprc"]
            sig = "*" if (a["lo"] > 0 or a["hi"] < 0) else " "
            sigv = "*" if (av["lo"] > 0 or av["hi"] < 0) else " "
            verdict = ("temporal variation MATTERS" if av["lo"] > 0 else
                       "no evidence it matters" if av["hi"] > 0 else
                       "intervention IMPROVED the model")
            print(f"  normal - {other:<9}:")
            print(f"      VAL  dAUPRC {av['delta']:+.4f} "
                  f"[{av['lo']:+.4f}, {av['hi']:+.4f}]{sigv}  -> {verdict}")
            print(f"      test dAUPRC {a['delta']:+.4f} "
                  f"[{a['lo']:+.4f}, {a['hi']:+.4f}]{sig}")
            entry[f"delta_normal_minus_{other}"] = d
            entry_val[f"delta_normal_minus_{other}"] = dv'''

EDITS = [
    ("        entry = {}\n", "        entry, entry_val = {}, {}\n"),
    ("    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)\n"
     "    meta_te = bw.meta[idx_te]",
     "    idx_va, idx_te = np.flatnonzero(va), np.flatnonzero(te)\n"
     "    meta_te = bw.meta[idx_te]\n"
     "    meta_va = bw.meta[idx_va]"),
    ('''        print(f"  {'condition':>10} {'AUROC':>8} {'AUPRC':>8} {'PPV':>8} "
              f"{'Recall':>8} {'F1':>8}")''',
     '''        print(f"  {'condition':>10} {'VAL AUPRC':>10} {'test AUPRC':>10} "
              f"{'VAL rec':>9} {'test rec':>9}")'''),
    (OLD_HEADER, NEW_HEADER),
    (OLD_DELTA, NEW_DELTA),
    ('''        res["horizons"][str(h)] = {
            k: ({kk: vv for kk, vv in v.items() if kk != "per_subject"}
                if isinstance(v, dict) and "per_subject" in v else v)
            for k, v in entry.items()}''',
     '''        def _strip(dd):
            return {k: ({kk: vv for kk, vv in v.items() if kk != "per_subject"}
                        if isinstance(v, dict) and "per_subject" in v else v)
                    for k, v in dd.items()}
        res["horizons"][str(h)] = _strip(entry)
        res["horizons"][str(h)]["_validation"] = _strip(entry_val)'''),
    ('    print("\\nScoring the test set under each condition "\n'
     '          f"({len(idx_te):,} windows each)...")',
     '    print(f"\\nScoring VALIDATION ({len(idx_va):,}) and test "\n'
     '          f"({len(idx_te):,}) under each condition...")'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    bak = a.file + ".val.bak"
    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy(bak, a.file)
        print(f"restored {a.file}")
        return

    if not os.path.exists(a.file):
        sys.exit(f"{a.file} not found -- run from src/")

    src = open(a.file).read()
    if "entry_val" in src:
        print("already patched")
        return

    out, missing = src, []
    for i, (old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        if c != 1:
            missing.append((i, c, old.splitlines()[0][:64]))
            continue
        out = out.replace(old, new, 1)

    if missing:
        print("COULD NOT APPLY -- anchors not found exactly once:\n")
        for i, c, frag in missing:
            print(f"  edit {i}: found {c}x  |  {frag}")
        print("\nNothing written.")
        return

    if not os.path.exists(bak):
        shutil.copy(a.file, bak)
        print(f"backup -> {bak}")
    open(a.file, "w").write(out)
    print("\n".join(difflib.unified_diff(
        src.splitlines(), out.splitlines(), "before", "after",
        lineterm="", n=2)))
    print("\napplied. The table now shows VAL and test side by side, and the "
          "verdict\nline is driven by the VALIDATION delta. Test numbers are "
          "unchanged.")


if __name__ == "__main__":
    main()
