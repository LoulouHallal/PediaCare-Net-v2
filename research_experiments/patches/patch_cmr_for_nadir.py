"""
patch_cmr_for_nadir.py  --  apply the Phase 3 (E1) edits to cmr_pilot.py
=========================================================================

Idempotent: running it twice is a no-op. Writes cmr_pilot.py.bak first.

    python patch_cmr_for_nadir.py           # apply, show diff
    python patch_cmr_for_nadir.py --revert  # restore from .bak

WHAT IT CHANGES
---------------
1. imports nadir_head
2. --lam / --nadir_delta arguments
3. lambda in the run tag, so a lam>0 run cannot resume from the baseline
   checkpoint. TrainCheckpoint keys off tag alone; without this the
   auxiliary runs would silently continue the gru baseline's optimiser
   state and the comparison would be meaningless.
4. wraps the model in WithNadirHead when lam>0
5. wraps the training dataset so batches carry (x, y, g, m)
6. adds the masked Huber term to the loss
7. prints the two loss parts each epoch, so a scale mismatch between the
   BCE and nadir terms is visible rather than being mistaken for the
   mechanism failing

AT lam=0 EVERY PATH IS THE ORIGINAL ONE. That is the control run.
"""

import argparse
import difflib
import os
import shutil
import sys

TARGET = "cmr_pilot.py"

EDITS = [
    # ---- 1. import -------------------------------------------------------
    ("from interaction_screen import seed_everything, make_loader",
     "from interaction_screen import seed_everything, make_loader\n"
     "from nadir_head import NadirDataset, WithNadirHead, NadirLoss"),

    # ---- 2. cli ----------------------------------------------------------
    ('    ap.add_argument("--no_resume", action="store_true")',
     '    ap.add_argument("--no_resume", action="store_true")\n'
     '    ap.add_argument("--lam", type=float, default=0.0,\n'
     '                    help="weight on the future-nadir auxiliary loss. "\n'
     '                         "0 reproduces the baseline exactly.")\n'
     '    ap.add_argument("--nadir_delta", type=float, default=1.0,\n'
     '                    help="Huber delta on the z-scored nadir target")'),

    # ---- 3. tag ----------------------------------------------------------
    ('    tag = f"{name}__s{seed}" + (f"__{args.suffix}" if args.suffix else "")',
     '    tag = f"{name}__s{seed}" + (f"__{args.suffix}" if args.suffix else "")\n'
     '    if getattr(args, "lam", 0.0) > 0:\n'
     '        # keep auxiliary runs on their own checkpoint; TrainCheckpoint\n'
     '        # resumes by tag and would otherwise continue the baseline run\n'
     '        tag += f"__lam{args.lam:g}"'),

    # ---- 4. model wrapper ------------------------------------------------
    ("    model = CMRGRU(cfg).to(device)",
     "    model = CMRGRU(cfg).to(device)\n"
     "    if args.lam > 0:\n"
     "        model = WithNadirHead(model, n_out=len(HORIZONS)).to(device)\n"
     "        print(f'  + nadir head on {model.head_name} '\n"
     "              f'({model.in_features} -> {len(HORIZONS)}), lam {args.lam:g}')"),

    # ---- 5. loss + dataset ----------------------------------------------
    ("""    dl = make_loader(WindowDataset(bw, idx_tr, Y), args.batch, shuffle=True,
                     seed=seed, num_workers=args.workers, drop_last=True)""",
     """    crit_n = NadirLoss(crit, lam=args.lam, delta=args.nadir_delta)

    _train_ds = WindowDataset(bw, idx_tr, Y)
    if args.lam > 0:
        _nz = np.load(config.DATA_DERIVED / f"nadir_targets_{args.tag}.npz")
        _train_ds = NadirDataset(_train_ds, idx_tr, _nz["nadir_z"], _nz["valid"])
        print(f"  nadir targets: {_nz['valid'][idx_tr].mean():.4%} of train "
              f"slots valid, clip {float(_nz['clip_hi']):.0f} mg/dL")
    dl = make_loader(_train_ds, args.batch, shuffle=True,
                     seed=seed, num_workers=args.workers, drop_last=True)"""),

    # ---- 6. inner loop ---------------------------------------------------
    ("""        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)""",
     """        for _batch in dl:
            if args.lam > 0:
                xb, yb, gb, mb = _batch
                xb, yb = xb.to(device, non_blocking=True), yb.to(device)
                gb, mb = gb.to(device), mb.to(device)
                opt.zero_grad()
                _logits, _nad = model(xb, want_nadir=True)
                loss = crit_n(_logits, yb, _nad, gb, mb)
            else:
                xb, yb = _batch
                xb, yb = xb.to(device, non_blocking=True), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)"""),

    # ---- 7. epoch log ----------------------------------------------------
    ('''        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"VAL mean-AUPRC {v:.4f}{' *' if improved else ''}")''',
     '''        print(f"    epoch {ep:>3}  loss {tot/max(nb,1):.5f}  "
              f"VAL mean-AUPRC {v:.4f}{' *' if improved else ''}")
        if args.lam > 0:
            # if the nadir term dwarfs the bce term the classifier is being
            # drowned -- a scaling problem, not a failed mechanism
            print(f"      bce {crit_n.last['bce']:.5f}  "
                  f"nadir {crit_n.last['nadir']:.5f}  "
                  f"(lam*nadir {args.lam * crit_n.last['nadir']:.5f})")'''),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--revert", action="store_true")
    ap.add_argument("--file", default=TARGET)
    a = ap.parse_args()

    bak = a.file + ".bak"
    if a.revert:
        if not os.path.exists(bak):
            sys.exit(f"no backup at {bak}")
        shutil.copy(bak, a.file)
        print(f"restored {a.file} from {bak}")
        return

    if not os.path.exists(a.file):
        sys.exit(f"{a.file} not found -- run this from src/")

    src = open(a.file).read()
    if "from nadir_head import" in src:
        print("already patched; nothing to do "
              "(use --revert to undo, then re-run to re-apply)")
        return

    if not os.path.exists(bak):
        shutil.copy(a.file, bak)
        print(f"backup -> {bak}")

    out = src
    missing = []
    for i, (old, new) in enumerate(EDITS, 1):
        cnt = out.count(old)
        if cnt != 1:
            missing.append((i, cnt, old.splitlines()[0][:70]))
            continue
        out = out.replace(old, new, 1)

    if missing:
        print("\nCOULD NOT APPLY -- anchors not found exactly once:\n")
        for i, cnt, frag in missing:
            print(f"  edit {i}: found {cnt}x  |  {frag}")
        print("\nNothing was written. The file may differ from what was "
              "pasted; send the exact lines above and the patch can be "
              "re-anchored.")
        return

    open(a.file, "w").write(out)
    diff = difflib.unified_diff(src.splitlines(), out.splitlines(),
                                "before", "after", lineterm="", n=2)
    print("\n".join(diff))
    print(f"\napplied {len(EDITS)} edits to {a.file}")
    print("\nNEXT: run the lam=0 control first. It must reproduce the "
          "existing baseline exactly.")


if __name__ == "__main__":
    main()
