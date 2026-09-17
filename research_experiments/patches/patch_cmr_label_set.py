"""
patch_cmr_label_set.py  --  add --label_set to cmr_pilot.py
============================================================

cmr_pilot hardcodes

    Y = bw._labels_any

which is the endpoint build_windows.py calls "legacy". The endpoint it
calls PRIMARY is bw._labels: hypoglycaemic EPISODE ONSET, defined as >= 3
consecutive readings < 70 with the episode ending only after >= 3
consecutive readings >= 70.

This patch makes the endpoint a flag. Default stays "any", so every
existing result and every existing command reproduces unchanged.

WHY THIS IS NOT A ROUTE TO A BIGGER NUMBER
------------------------------------------
Consensus prevalence is roughly a third of legacy (1.44% vs 4.93% at
h=15 on validation). AUPRC scales with prevalence, so consensus numbers
will be LOWER in absolute terms and are not comparable to anything
reported so far. This is a change of question, not a change of method.

WATCH THE POSITIVE WEIGHT
-------------------------
run_one computes pos_weight as (1-prev)/prev clipped to [1, 50]. At
consensus prevalence the raw value is ~68 at h=15, so the clip binds and
the effective weighting differs from the legacy runs. That is a real
difference between the two settings, not a bug, but it must be stated
when the two are discussed together.

    python patch_cmr_label_set.py
    python patch_cmr_label_set.py --revert
"""

import argparse
import difflib
import os
import shutil
import sys

TARGET = "cmr_pilot.py"

EDITS = [
    ('    ap.add_argument("--nadir_delta", type=float, default=1.0,\n'
     '                    help="Huber delta on the z-scored nadir target")',
     '    ap.add_argument("--nadir_delta", type=float, default=1.0,\n'
     '                    help="Huber delta on the z-scored nadir target")\n'
     '    ap.add_argument("--label_set", default="any",\n'
     '                    choices=["any", "consensus"],\n'
     '                    help="any: 1 if ANY future reading <70 (legacy). "\n'
     '                         "consensus: episode onset, >=3 consecutive "\n'
     '                         "readings <70 (called primary in "\n'
     '                         "build_windows.py). Prevalence differs ~3x, so "\n'
     '                         "the two are not comparable.")'),

    ("    Y = bw._labels_any",
     '    Y = bw._labels_any if args.label_set == "any" else bw._labels\n'
     '    print(f"  endpoint: {args.label_set}  "\n'
     '          f"prevalence {Y.mean(0).round(5).tolist()}")'),

    ('    if getattr(args, "lam", 0.0) > 0:\n'
     '        # keep auxiliary runs on their own checkpoint; TrainCheckpoint\n'
     '        # resumes by tag and would otherwise continue the baseline run\n'
     '        tag += f"__lam{args.lam:g}"',
     '    if getattr(args, "lam", 0.0) > 0:\n'
     '        # keep auxiliary runs on their own checkpoint; TrainCheckpoint\n'
     '        # resumes by tag and would otherwise continue the baseline run\n'
     '        tag += f"__lam{args.lam:g}"\n'
     '    if getattr(args, "label_set", "any") != "any":\n'
     '        # a consensus run is a different task, not a variant of the\n'
     '        # legacy run -- it must never share a checkpoint or a filename\n'
     '        tag += f"__{args.label_set}"'),

    ('            if args.lam > 0:\n'
     '                tag += f"__lam{args.lam:g}"',
     '            if args.lam > 0:\n'
     '                tag += f"__lam{args.lam:g}"\n'
     '            if args.label_set != "any":\n'
     '                tag += f"__{args.label_set}"'),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    bak = a.file + ".labelset.bak"
    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy(bak, a.file)
        print(f"restored {a.file}")
        return

    if not os.path.exists(a.file):
        sys.exit(f"{a.file} not found -- run from src/")

    src = open(a.file).read()
    if '--label_set' in src and 'bw._labels_any if args.label_set' in src:
        print("already patched")
        return

    out, missing = src, []
    for i, (old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        if c != 1:
            missing.append((i, c, old.splitlines()[0][:66]))
            continue
        out = out.replace(old, new, 1)

    if missing:
        print("COULD NOT APPLY -- anchors not found exactly once:\n")
        for i, c, frag in missing:
            print(f"  edit {i}: found {c}x  |  {frag}")
        print("\nNothing written. Edit 4 needs the skip-guard patch applied "
              "first; edits 1 and 3 need the nadir patch applied first.")
        return

    if not os.path.exists(bak):
        shutil.copy(a.file, bak)
        print(f"backup -> {bak}")
    open(a.file, "w").write(out)
    print("\n".join(difflib.unified_diff(
        src.splitlines(), out.splitlines(), "before", "after",
        lineterm="", n=2)))
    print(f"\napplied {len(EDITS)} edits")
    print("\nDefault is unchanged ('any'), so existing commands reproduce "
          "exactly. Consensus runs get their own __consensus tag.")


if __name__ == "__main__":
    main()
