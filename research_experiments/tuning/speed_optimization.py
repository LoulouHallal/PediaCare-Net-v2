"""
speed_optimization.py
=======================
Which optimisations actually cut the runtime of the hand-written recurrent
cells, measured rather than assumed.

THE PROBLEM, FROM T10
---------------------
gru_ta_7ch is mathematically identical to gru_fused_9ch — same equations,
written as a Python loop instead of calling cuDNN. It costs 106.2 ms/batch
against 5.0. That 21x is pure overhead: 60 sequential steps, each issuing
a handful of tiny GPU operations, with the T4 idle between launches while
Python catches up. At hidden 64 and batch 512 a single step's matmul is
~2 MFLOPs, which a T4 finishes in microseconds.

So the cost is (number of kernel launches) x (launch overhead), not
(FLOPs) / (throughput). That is why TSL-GRU is SLOWER than the GRU despite
66% fewer MACs: it removes arithmetic but adds elementwise operations per
step.

WHAT IS TESTED
--------------
    eager              the current implementation, as a reference
    compile-default    torch.compile: fuses adjacent elementwise ops into
                       single kernels, cutting launch count
    compile-overhead   torch.compile(mode="reduce-overhead"): additionally
                       uses CUDA graphs, replaying the whole captured
                       sequence as one launch. This is the mode designed
                       for launch-bound workloads and should help most
    jit-script         TorchScript, the pre-compile path; sometimes still
                       wins on tight loops
    batch scaling      the loop runs once per BATCH, not per sample, so
                       larger batches amortise the fixed cost

Every variant's output is checked against eager before its timing is
reported. An optimisation that changes the numbers is not an optimisation.

CAVEATS
-------
torch.compile UNROLLS a 60-step Python loop, so first-call compilation can
take minutes and memory can spike; that cost is paid once per process, not
per batch, and is excluded from the timings. reduce-overhead requires
static shapes, so the final partial batch of an epoch must be dropped
(drop_last=True, which the pilots already use).

Usage:
    python speed_optimization.py
    python speed_optimization.py --only tsl_gru,gru_ta_7ch --batch 512
    python speed_optimization.py --skip_compile        # if torch < 2.0
"""

import gc
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

import config

T_LEN = 60
OUT_DIR = config.RESULTS / "tables"


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_zoo(want=None):
    zoo = []

    def add(name, fn, ch, note=""):
        if want and name not in want:
            return
        try:
            zoo.append((name, fn(), ch, note))
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")

    class Heads(nn.Module):
        def __init__(self, h, n=4):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(h, h // 2), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(h // 2, n))

        def forward(self, x):
            return self.net(x)

    class FusedGRU(nn.Module):
        def __init__(self, ch=9, h=64):
            super().__init__()
            self.g = nn.GRU(ch, h, num_layers=2, batch_first=True, dropout=0.2)
            self.head = Heads(h)

        def forward(self, x):
            return self.head(self.g(x)[0][:, -1])

    add("gru_fused", lambda: FusedGRU(9), 9, "cuDNN reference")
    try:
        from tsl_gru import TSLGRU
        add("gru_ta_7ch", lambda: TSLGRU("gru_ta", 7, 64), 7,
            "same maths as cuDNN GRU, hand-written")
        add("tsl_gru", lambda: TSLGRU("tsl_gru", 7, 64), 7,
            "the efficiency candidate")
    except Exception as e:
        print(f"  [skip] tsl_gru: {type(e).__name__}: {e}")
    try:
        from drs_gru import DRSGRU
        add("drs_gru", lambda: DRSGRU(9, 61), 9)
    except Exception:
        pass
    try:
        from cmr_gru import CMRGRU, CMRConfig, ARMS
        add("cmr", lambda: CMRGRU(CMRConfig(9, 84, **ARMS["cmr"])), 9)
    except Exception:
        pass
    return zoo


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def bench_infer(fn, x, repeats, dev, warmup=5):
    for _ in range(warmup):
        fn(x)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    sync(dev)
    return (time.perf_counter() - t0) / repeats


def bench_train(model, fn, x, y, repeats, dev, warmup=3):
    """
    Time a full forward+backward+step, then RESTORE the weights.

    Adam steps mutate the model, so without restoring, every variant after
    the first would be compared against a stale reference output and the
    correctness check would report spurious mismatches. That happened on
    the first run of this script.
    """
    snapshot = {k: v.detach().clone() for k, v in model.state_dict().items()}
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad(set_to_none=True)
        crit(fn(x), y).backward()
        opt.step()

    try:
        for _ in range(warmup):
            step()
        sync(dev)
        t0 = time.perf_counter()
        for _ in range(repeats):
            step()
        sync(dev)
        return (time.perf_counter() - t0) / repeats
    finally:
        model.load_state_dict(snapshot)
        model.zero_grad(set_to_none=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--skip_jit", action="store_true")
    ap.add_argument("--batch_scan", action="store_true",
                    help="also sweep batch size on the first model")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}   torch {torch.__version__}")
    has_compile = hasattr(torch, "compile") and not args.skip_compile
    if not has_compile:
        print("  torch.compile unavailable or skipped")
    print(f"\nbatch {args.batch}, sequence {T_LEN}, {args.repeats} repeats")
    print("  compile time is excluded from timings — it is paid once per "
          "process")
    if has_compile:
        print("  WARNING: torch.compile UNROLLS the 60-step loop across 2 "
              "layers into one\n           graph. Compilation can take "
              "several minutes per model and in this\n           sandbox it "
              "did not finish at all. Run --only <one model> first, and\n"
              "           use --skip_compile if it stalls; jit-trace is the "
              "cheap fallback\n           and gave 1.2-1.3x on CPU here.")
    print()

    want = set(args.only.split(",")) if args.only else None
    zoo = build_zoo(want)
    if not zoo:
        print("No models — run from src/.")
        return

    rows = []
    for name, model, ch, note in zoo:
        print(f"{'='*78}\n{name}  ({n_params(model):,} params)"
              + (f"  — {note}" if note else "") + f"\n{'='*78}")
        model = model.to(dev).eval()
        x = torch.randn(args.batch, T_LEN, ch, device=dev)
        y = (torch.rand(args.batch, 4, device=dev) < 0.2).float()

        with torch.no_grad():
            ref = model(x).clone()

        variants = [("eager", model)]
        if has_compile:
            for lab, mode in [("compile-default", None),
                              ("compile-overhead", "reduce-overhead")]:
                try:
                    t0 = time.perf_counter()
                    c = (torch.compile(model) if mode is None
                         else torch.compile(model, mode=mode))
                    with torch.no_grad():
                        c(x)                       # triggers compilation
                    sync(dev)
                    print(f"  {lab}: compiled in "
                          f"{time.perf_counter()-t0:.1f}s")
                    variants.append((lab, c))
                except Exception as e:
                    print(f"  {lab}: FAILED {type(e).__name__}: "
                          f"{str(e)[:70]}")
        if not args.skip_jit:
            try:
                j = torch.jit.trace(model, x, strict=False)
                variants.append(("jit-trace", j))
            except Exception as e:
                print(f"  jit-trace: FAILED {type(e).__name__}: "
                      f"{str(e)[:70]}")

        # verify every variant against the pristine reference FIRST, before
        # any optimiser step has touched the weights
        errs = {}
        for lab, fn in variants:
            try:
                with torch.no_grad():
                    errs[lab] = float((fn(x) - ref).abs().max())
            except Exception:
                errs[lab] = float("nan")

        base_i = base_t = None
        print(f"\n  {'variant':>18} {'infer us/win':>13} {'train ms/bat':>13} "
              f"{'speedup':>9} {'match':>7}")
        for lab, fn in variants:
            try:
                err = errs.get(lab, float("nan"))
                ok = err < args.tol
                ti = bench_infer(fn, x, args.repeats, dev)
                tt = bench_train(model, fn, x, y,
                                 max(args.repeats // 2, 3), dev)
                per = ti / args.batch * 1e6
                if base_i is None:
                    base_i, base_t = per, tt
                print(f"  {lab:>18} {per:>13.2f} {tt*1e3:>13.1f} "
                      f"{base_t/tt:>8.2f}x {('yes' if ok else f'{err:.1e}'):>7}")
                rows.append({"model": name, "variant": lab,
                             "infer_us_per_window": per,
                             "train_ms_per_batch": tt * 1e3,
                             "speedup_vs_eager": base_t / tt,
                             "max_abs_diff": err, "matches": ok})
            except Exception as e:
                print(f"  {lab:>18} FAILED {type(e).__name__}: {str(e)[:50]}")
        print()
        del model, variants
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    if args.batch_scan and zoo:
        name, _, ch, _ = zoo[0]
        print(f"{'='*78}\nBATCH SCALING — the loop runs once per BATCH, so\n"
              f"larger batches amortise the fixed per-step cost\n{'='*78}\n")
        m2 = build_zoo({name})[0][1].to(dev).eval()
        print(f"  {'batch':>7} {'infer us/win':>13} {'rel per-sample':>15}")
        ref_per = None
        for B in [128, 256, 512, 1024, 2048]:
            try:
                xb = torch.randn(B, T_LEN, ch, device=dev)
                ti = bench_infer(m2, xb, max(args.repeats // 2, 3), dev,
                                 warmup=3)
                per = ti / B * 1e6
                ref_per = ref_per or per
                print(f"  {B:>7} {per:>13.2f} {per/ref_per:>14.2f}x")
                del xb
                if dev == "cuda":
                    torch.cuda.empty_cache()
            except RuntimeError as e:
                print(f"  {B:>7} out of memory")
                break

    if not rows:
        return

    print(f"\n{'#'*78}\nREADING THIS\n{'#'*78}\n")
    best = {}
    for r in rows:
        if not r["matches"]:
            continue
        b = best.get(r["model"])
        if b is None or r["speedup_vs_eager"] > b["speedup_vs_eager"]:
            best[r["model"]] = r
    for mdl, r in best.items():
        if r["variant"] == "eager":
            print(f"  {mdl}: nothing beat eager. The loop is not the "
                  f"bottleneck here,\n     or compilation could not fuse it.")
        else:
            print(f"  {mdl}: {r['variant']} is {r['speedup_vs_eager']:.2f}x "
                  f"faster than eager\n     ({r['train_ms_per_batch']:.1f} "
                  f"ms/batch), outputs identical to {args.tol:.0e}")
    print("\n  Any variant marked with a number under 'match' changed the")
    print("  predictions and must not be used.")
    print("\n  If reduce-overhead wins, the cost was kernel launches, and the")
    print("  MAC reductions in T10 become realisable as wall-clock time.")

    try:
        import pandas as pd
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "T11_speed_optimization.csv",
                                  index=False)
        print(f"\n✓ Saved -> {OUT_DIR / 'T11_speed_optimization.csv'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
