"""
deployment_benchmark.py
=========================
Batch-1 latency, the full deployment curve, and peak memory measured
AFTER compilation warm-up.

WHY THIS AND NOT MORE PROFILING
-------------------------------
T11 established that torch.compile(mode="reduce-overhead") gives 6.48x on
TSL-GRU with identical outputs, which confirms the cost was kernel
launches rather than arithmetic. Two things are still missing, and both
matter more than closing the remaining ~1.3 us gap against GRU+TA:

  1. Batch-1 latency. Every timing so far used batch 512. A wearable or
     pump makes ONE prediction per CGM reading. At batch 1 there is no
     work to amortise launch overhead against, so this is where a
     step-wise recurrent loop is worst and where the numbers can look
     completely different from the batched ones.

  2. Peak memory after compilation. The T10 figures (TSL 142.3 MB,
     GRU+TA 157.8 MB) are EAGER. CUDA graphs in reduce-overhead mode
     cache workspace memory, so compiled peak can be higher. Claiming a
     memory advantage without remeasuring would be wrong.

Measurement order matters: compile, warm up until the graph is captured,
THEN reset peak stats, then measure. Resetting before warm-up would
attribute compilation's own allocations to inference.

Both allocated and reserved peaks are reported. Allocated is what tensors
hold; reserved is what the caching allocator took from the driver, which
is the number that determines whether a model fits on a device.

WHAT IS NOT DONE HERE, AND WHY
------------------------------
Precomputing input projections and fusing gate projections were
considered. torch.compile already fuses elementwise chains and can batch
GEMMs -- that is where the 6.48x came from. Measured separately, the
precompute rewrite gave 1.13x in eager mode. Hand-optimising on top risks
introducing graph breaks that cost more than the rewrite saves, and it
would mean editing equations that produced a validated result.

Use TORCH_LOGS="graph_breaks" to confirm the compiled graph is clean:

    TORCH_LOGS="graph_breaks" python deployment_benchmark.py --only tsl_gru

Usage:
    python deployment_benchmark.py
    python deployment_benchmark.py --only gru_ta_7ch,tsl_gru
    python deployment_benchmark.py --skip_compile
"""

import gc
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

import config

T_LEN = 60
BATCHES = [1, 8, 32, 128, 512]
OUT_DIR = config.RESULTS / "tables"


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def build_zoo(want=None):
    zoo = []

    def add(name, fn, ch, note=""):
        if want and name not in want:
            return
        try:
            zoo.append((name, fn, ch, note))
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
            "the arm TSL-GRU is matched against")
        add("tsl_gru", lambda: TSLGRU("tsl_gru", 7, 64), 7,
            "-61% params, -66% MACs")
    except Exception as e:
        print(f"  [skip] tsl_gru: {type(e).__name__}: {e}")
    return zoo


def sync(dev):
    if dev == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def latency(fn, x, dev, repeats, warmup):
    for _ in range(warmup):
        fn(x)
    sync(dev)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn(x)
    sync(dev)
    return (time.perf_counter() - t0) / repeats


@torch.no_grad()
def peak_memory(fn, x, dev, warmup=20, iters=50):
    """
    Warm up FIRST so compilation and CUDA-graph capture are complete, then
    reset the counters, then measure. Resetting before warm-up would blame
    inference for compilation's allocations.
    """
    if dev != "cuda":
        return float("nan"), float("nan")
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(iters):
        fn(x)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() / 1024 ** 2,
            torch.cuda.max_memory_reserved() / 1024 ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--mem_batch", type=int, default=512)
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {dev}   torch {torch.__version__}")
    has_compile = hasattr(torch, "compile") and not args.skip_compile
    print("\n  reduce-overhead uses CUDA graphs, which require STATIC shapes:")
    print("  each batch size is compiled separately, so expect a pause at "
          "every\n  new size. That is also why training needs drop_last=True."
          "\n")

    want = set(args.only.split(",")) if args.only else None
    zoo = build_zoo(want)
    if not zoo:
        print("No models — run from src/.")
        return

    rows = []
    for name, ctor, ch, note in zoo:
        print(f"{'='*80}\n{name}" + (f"  — {note}" if note else "")
              + f"\n{'='*80}")
        model = ctor().to(dev).eval()
        print(f"  {n_params(model):,} params\n")

        # ---- deployment latency curve ---------------------------------
        print(f"  DEPLOYMENT LATENCY AND PEAK MEMORY (per-window, "
              f"lower is better)")
        print(f"  {'batch':>6} {'eager us':>10} {'comp us':>10} {'x':>6} "
              f"{'e-alloc':>9} {'c-alloc':>9} {'c-resv':>9} {'match':>7}")
        for B in BATCHES:
            x = torch.randn(B, T_LEN, ch, device=dev)
            reps = 200 if B <= 8 else (50 if B <= 128 else 20)
            warm = 20 if B <= 8 else 5
            with torch.no_grad():
                ref = model(x).clone()

            te = latency(model, x, dev, reps, warm) / B * 1e6
            ea, er = peak_memory(model, x, dev, warmup=warm, iters=reps)

            tc, err, ca, cr = (float("nan"),) * 4
            if has_compile:
                try:
                    cm = torch.compile(model, mode="reduce-overhead")
                    with torch.no_grad():
                        out = cm(x)
                    err = float((out - ref).abs().max())
                    tc = latency(cm, x, dev, reps, warm) / B * 1e6
                    # warm up again so CUDA-graph capture is complete before
                    # the counters are reset
                    ca, cr = peak_memory(cm, x, dev, warmup=warm, iters=reps)
                except Exception as e:
                    print(f"  {B:>6} compile failed: {type(e).__name__}")

            sp = te / tc if (tc == tc and tc > 0) else float("nan")
            ok = "yes" if (err == err and err < args.tol) else (
                f"{err:.1e}" if err == err else "-")
            print(f"  {B:>6} {te:>10.1f} {tc:>10.1f} {sp:>5.2f}x "
                  f"{ea:>9.1f} {ca:>9.1f} {cr:>9.1f} {ok:>7}")
            rows.append({"model": name, "params": n_params(model), "batch": B,
                         "eager_us": te, "compiled_us": tc, "speedup": sp,
                         "eager_alloc_MB": ea, "eager_reserved_MB": er,
                         "compiled_alloc_MB": ca, "compiled_reserved_MB": cr,
                         "max_abs_diff": err})
            del x
            gc.collect()
            if dev == "cuda":
                torch.cuda.empty_cache()

        if dev == "cuda":
            infl = [r for r in rows if r["model"] == name
                    and r["compiled_reserved_MB"] == r["compiled_reserved_MB"]
                    and r["eager_reserved_MB"] > 0
                    and r["compiled_reserved_MB"] > 1.15 * r["eager_reserved_MB"]]
            if infl:
                print(f"\n    NOTE: reserved memory rose >15% under "
                      f"reduce-overhead at batch "
                      f"{[r['batch'] for r in infl]} — CUDA-graph workspace.")
                print(f"    Any memory claim must cite the COMPILED numbers.")

        print()
        del model
        gc.collect()
        if dev == "cuda":
            torch.cuda.empty_cache()

    if not rows:
        return

    print(f"{'#'*80}\nREADING THIS\n{'#'*80}\n")
    b1 = [r for r in rows if r["batch"] == 1]
    if b1:
        print("  BATCH 1 — the deployment case, one prediction per reading:")
        for r in sorted(b1, key=lambda z: z["compiled_us"]
                        if z["compiled_us"] == z["compiled_us"] else 1e9):
            c = r["compiled_us"]
            print(f"    {r['model']:>12}  eager {r['eager_us']:>9.1f}  "
                  f"compiled {c:>9.1f} us/window")
        print("\n  At batch 1 there is nothing to amortise launch overhead")
        print("  against, so this is the honest single-sample latency. A CGM")
        print("  reading arrives every 5 minutes = 300,000,000 us, so even")
        print("  the slowest figure here is many orders of magnitude inside")
        print("  the real-time budget. Latency is not the binding constraint")
        print("  for this application; parameters and memory are, on an")
        print("  embedded device.")
    if dev == "cuda":
        print("\n  PEAK MEMORY, compiled (allocated MB):")
        for B in [1, 512]:
            sel = [r for r in rows if r["batch"] == B
                   and r["compiled_alloc_MB"] == r["compiled_alloc_MB"]]
            if sel:
                print(f"    batch {B}: " + "   ".join(
                    f"{r['model']} {r['compiled_alloc_MB']:.1f}" for r in sel))
        print("  On an embedded device this, not latency, is the binding")
        print("  constraint — a pump has megabytes, not gigabytes.")

    g = next((r for r in rows if r["model"] == "gru_ta_7ch"
              and r["batch"] == 512), None)
    t = next((r for r in rows if r["model"] == "tsl_gru"
              and r["batch"] == 512), None)
    if g and t and t["compiled_us"] == t["compiled_us"]:
        d = t["compiled_us"] - g["compiled_us"]
        print(f"\n  TSL-GRU vs GRU+TA, compiled, batch 512: "
              f"{d:+.2f} us/window")
        if d > 0:
            print("  TSL-GRU remains slower despite 66% fewer MACs. State")
            print("  that plainly: the parameter and MAC reductions are real")
            print("  and hardware-independent; they do not convert to lower")
            print("  latency at this scale because the cell adds elementwise")
            print("  work per step while removing arithmetic.")

    try:
        import pandas as pd
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(OUT_DIR / "T12_deployment.csv", index=False)
        print(f"\n✓ Saved -> {OUT_DIR / 'T12_deployment.csv'}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
