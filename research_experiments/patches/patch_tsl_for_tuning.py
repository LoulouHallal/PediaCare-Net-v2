"""
patch_tsl_for_tuning.py  --  add the flags the tuning study needs
==================================================================

tsl_gru.py currently hardcodes dropout (via the TSLGRU default of 0.2)
and builds Adam with no weight decay, so two of the four searched
hyperparameters cannot be set from the command line.

This adds:
    --dropout        passed through to TSLGRU
    --weight_decay   passed to Adam
    --suffix         appended to the run name

WHY --suffix IS NOT OPTIONAL
----------------------------
Every trial of the same variant and seed would otherwise write to the
same name: tsl_gru__weighted_bce__none__any.json / .pt / _probs.npz.
Trial 5 would silently overwrite trial 2, and the search would report
whichever ran last rather than the best. This is the same failure the
lambda runs hit, where every arm collapsed into "gru" in the summary.

Defaults reproduce current behaviour exactly: dropout 0.2, weight decay
0.0, empty suffix. Existing commands are unaffected.

    python patch_tsl_for_tuning.py
    python patch_tsl_for_tuning.py --revert
"""

import argparse
import difflib
import os
import shutil
import sys

TARGET = "tsl_gru.py"

EDITS = [
    # ---- flags ----------------------------------------------------------
    ('    ap.add_argument("--force", action="store_true")',
     '    ap.add_argument("--force", action="store_true")\n'
     '    ap.add_argument("--dropout", type=float, default=0.2,\n'
     '                    help="dropout between recurrent layers")\n'
     '    ap.add_argument("--weight_decay", type=float, default=0.0,\n'
     '                    help="Adam weight decay")\n'
     '    ap.add_argument("--suffix", default="",\n'
     '                    help="appended to the run name. REQUIRED when "\n'
     '                         "sweeping, or trials overwrite each other.")'),

    # ---- optimiser ------------------------------------------------------
    ("    opt = torch.optim.Adam(model.parameters(), lr=args.lr)",
     "    opt = torch.optim.Adam(model.parameters(), lr=args.lr,\n"
     "                           weight_decay=getattr(args, 'weight_decay', 0.0))"),
]

# dropout is passed positionally or by keyword into TSLGRU(...); the
# constructor call is located at runtime rather than anchored on an exact
# string, because its formatting varies.
CTOR_HINT = "model = TSLGRU(variant, in_ch=bw.timeline.shape[1], hidden=args.hidden,"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    bak = a.file + ".tuning.bak"
    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy(bak, a.file)
        print(f"restored {a.file}")
        return

    if not os.path.exists(a.file):
        sys.exit(f"{a.file} not found -- run from src/")

    src = open(a.file).read()
    if "--weight_decay" in src:
        print("already patched")
        return

    out, missing = src, []
    for i, (old, new) in enumerate(EDITS, 1):
        c = out.count(old)
        if c != 1:
            missing.append((i, c, old.splitlines()[0][:64]))
            continue
        out = out.replace(old, new, 1)

    # dropout into the constructor
    if CTOR_HINT in out:
        if "dropout=args.dropout" not in out:
            out = out.replace(
                CTOR_HINT,
                CTOR_HINT + "\n                   dropout=args.dropout,", 1)
    else:
        missing.append((3, 0, "TSLGRU( constructor call"))

    if missing:
        print("COULD NOT APPLY -- anchors not found exactly once:\n")
        for i, c, frag in missing:
            print(f"  edit {i}: found {c}x  |  {frag}")
        print("\nNothing written. Paste the lines around the failed anchor "
              "and the patch can be re-anchored.")
        return

    if not os.path.exists(bak):
        shutil.copy(a.file, bak)
        print(f"backup -> {bak}")
    open(a.file, "w").write(out)
    print("\n".join(difflib.unified_diff(
        src.splitlines(), out.splitlines(), "before", "after",
        lineterm="", n=2)))
    print("\napplied. Defaults unchanged: dropout 0.2, weight_decay 0.0, "
          "suffix ''.")
    print("\nSTILL NEEDED: the run name must include --suffix. Check how "
          "`name` is\nbuilt near the np.savez_compressed call and confirm "
          "it picks up args.suffix;\nif it does not, trials will overwrite "
          "each other.")


if __name__ == "__main__":
    main()
