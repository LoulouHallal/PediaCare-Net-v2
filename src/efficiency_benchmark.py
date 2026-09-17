"""
efficiency_benchmark.py
=========================
Parameters, MACs, inference latency, training step time and peak memory for
every architecture built in this project, measured under identical
conditions in one process on one device.

WHY THE EXISTING TIMINGS CANNOT BE USED
---------------------------------------
Each pilot records res["minutes"], but that number is wall-clock for a
whole run and is not comparable across architectures. The same GRU config
gave 125.9, 112.6, 93.2, 36.8 and 25.6 minutes across five seeds — the
spread is early-stopping epoch count, resumed runs, and CPU-vs-GPU
scheduling, not architecture. Two recent runs silently landed on CPU.
It also bundles training, per-epoch validation over 454k windows, test
prediction and bootstrapping into a single figure.

This measures each cost separately, same batch, same sequence length,
same device, back to back.

WHAT IS REPORTED
----------------
    params            trainable parameter count
    MACs/window       multiply-accumulates for one forward pass, counted
                      analytically from module shapes (torch has no
                      built-in counter and thop may not be installed)
    infer us/window   batched inference, the clinically relevant number:
                      this model would run on a pump or phone
    train ms/batch    forward + backward + step
    peak MB           peak allocated memory during a training step
    epoch estimate    train ms/batch scaled to 400k windows, which is why
                      the custom cells took hours

CAVEAT ON MACs
--------------
Counted from Linear, GRU, LSTM, Conv1d and MultiheadAttention shapes,
multiplied by sequence length for recurrent cells. Elementwise ops,
softmax, entmax and einsum-based block-diagonal recurrence are NOT
counted, so custom cells are slightly understated. It is a comparable
proxy, not an exact FLOP count — the measured latency is the number to
trust.

Usage:
    python efficiency_benchmark.py
    python efficiency_benchmark.py --batch 256 --repeats 20
    python efficiency_benchmark.py --only gru,tsl_gru,cmr
"""

import gc
import time
import argparse
import numpy as np

import torch
import torch.nn as nn

import config

T_LEN = 60
N_CH_5, N_CH_7, N_CH_9 = 5, 7, 9
OUT_DIR = config.RESULTS / "tables"


# ─── analytic MAC counting ────────────────────────────────────────────────────

def count_macs(model, in_ch, T=T_LEN):
    """
    Analytic MACs for one window. Recurrent cells written as Python loops
    have their per-step Linear cost multiplied by T; fused nn.GRU/nn.LSTM
    are handled by their gate structure.
    """
    macs = 0
    # A model that contains NO nn.GRU / nn.LSTM / nn.Conv1d but consumes a
    # sequence must be stepping a hand-written cell T times in Python, so
    # its Linear layers run once per timestep. Some custom cells hold their
    # layers flat on the model rather than in a ModuleList (AR-RHU does),
    # so detect by what is ABSENT rather than by structure.
    recurrent_loop = not any(
        isinstance(m, (nn.GRU, nn.LSTM, nn.Conv1d)) for m in model.modules())

    for m in model.modules():
        cn = m.__class__.__name__
        if isinstance(m, nn.Linear):
            per = m.in_features * m.out_features
            macs += per * (T if recurrent_loop else 1)
        elif isinstance(m, nn.GRU):
            for L in range(m.num_layers):
                i = m.input_size if L == 0 else m.hidden_size
                macs += 3 * (i * m.hidden_size + m.hidden_size ** 2) * T
        elif isinstance(m, nn.LSTM):
            for L in range(m.num_layers):
                i = m.input_size if L == 0 else m.hidden_size
                macs += 4 * (i * m.hidden_size + m.hidden_size ** 2) * T
        elif isinstance(m, nn.Conv1d):
            macs += (m.in_channels * m.out_channels * m.kernel_size[0]
                     * T // max(m.stride[0], 1))
        elif isinstance(m, nn.MultiheadAttention):
            d = m.embed_dim
            macs += 4 * d * d * T + 2 * d * T * T      # projections + scores
    if recurrent_loop:
        # the output head runs ONCE per window, not once per timestep
        head = getattr(model, "head", None) or getattr(model, "classifier", None)
        if head is not None:
            hm = sum(l.in_features * l.out_features
                     for l in head.modules() if isinstance(l, nn.Linear))
            macs -= hm * (T - 1)
    return macs


def n_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


# ─── model zoo ────────────────────────────────────────────────────────────────

def build_zoo(want=None):
    """(name, module, in_channels, note). Missing modules are skipped."""
    zoo = []

    def add(name, fn, ch, note):
        if want and name not in want:
            return
        try:
            zoo.append((name, fn(), ch, note))
        except Exception as e:
            print(f"  [skip] {name}: {type(e).__name__}: {e}")

    class Heads(nn.Module):
        def __init__(self, h, n=4):
            super().__init__()
            h2 = h // 2
            self.net = nn.Sequential(nn.Linear(h, h2), nn.ReLU(),
                                     nn.Dropout(0.1), nn.Linear(h2, n))

        def forward(self, x):
            return self.net(x)

    class FusedGRU(nn.Module):
        def __init__(self, ch, h=64, L=2):
            super().__init__()
            self.g = nn.GRU(ch, h, num_layers=L, batch_first=True, dropout=0.2)
            self.head = Heads(h)

        def forward(self, x):
            return self.head(self.g(x)[0][:, -1])

    class FusedLSTM(nn.Module):
        def __init__(self, ch, h=64, L=2):
            super().__init__()
            self.g = nn.LSTM(ch, h, num_layers=L, batch_first=True, dropout=0.2)
            self.head = Heads(h)

        def forward(self, x):
            return self.head(self.g(x)[0][:, -1])

    add("gru_fused_5ch", lambda: FusedGRU(N_CH_5), N_CH_5,
        "nn.GRU, the 5-channel stage-2 baseline")
    add("lstm_fused_5ch", lambda: FusedLSTM(N_CH_5), N_CH_5, "nn.LSTM")
    add("gru_fused_9ch", lambda: FusedGRU(N_CH_9), N_CH_9,
        "nn.GRU + TA + absolute — the project's best model")

    # --- custom recurrent cells --------------------------------------------
    try:
        from tsl_gru import TSLGRU
        add("gru_ta_7ch", lambda: TSLGRU("gru_ta", N_CH_7, 64), N_CH_7,
            "GRU + TA, the arm TSL-GRU was matched against")
        add("tsl_gru", lambda: TSLGRU("tsl_gru", N_CH_7, 64), N_CH_7,
            "trajectory-scaled retention, 61% fewer params")
    except Exception as e:
        print(f"  [skip] tsl_gru: {type(e).__name__}: {e}")
    try:
        from ar_rhu import ARRHU
        add("ar_rhu", lambda: ARRHU(64), N_CH_9, "anticipatory/reactive split")
        add("ar_rhu_h88", lambda: ARRHU(88), N_CH_9, "capacity-matched")
    except ImportError:
        pass
    try:
        from ctf_ru import ClinicalThresholdFluxRU

        class CTFWrap(nn.Module):
            """CTF-RU takes physical glucose and rate as separate args."""

            def __init__(self):
                super().__init__()
                self.m = ClinicalThresholdFluxRU(
                    input_size=N_CH_5, hidden_per_bin=55, head_hidden=114,
                    grid_min_mgdl=20., grid_max_mgdl=400.)

            def forward(self, x):
                B, T, _ = x.shape
                g = 120.0 + 30.0 * x[:, :, 0]        # plausible mg/dL
                r = torch.zeros_like(g)
                r[:, 1:] = (g[:, 1:] - g[:, :-1]) / 5.0
                return self.m(x, g, r)

        add("ctf_ru", CTFWrap, N_CH_5,
            "transport over a 20-state clinical grid")
    except Exception as e:
        print(f"  [skip] ctf_ru: {type(e).__name__}: {e}")
    try:
        from drs_gru import DRSGRU
        add("drs_gru", lambda: DRSGRU(N_CH_9, 61), N_CH_9,
            "signed recurrent history")
    except ImportError:
        pass
    try:
        from pac_gru import PACGRU, PACConfig, ARMS as PARMS
        add("pac_gru", lambda: PACGRU(PACConfig(N_CH_9, hidden_size=58,
                                                **PARMS["pac_gru"])), N_CH_9,
            "prediction-anchored scaling + coupled gates")
    except ImportError:
        pass
    try:
        from kew_gru import KEWGRU
        add("kew_entmax", lambda: KEWGRU(N_CH_9, 56, "kew_entmax"), N_CH_9,
            "keep/erase/write simplex, entmax")
        add("kew_softmax", lambda: KEWGRU(N_CH_9, 56, "kew_softmax"), N_CH_9,
            "same, dense normalisation")
    except ImportError:
        pass
    try:
        from adew_gru import ADEWGRU, ADEWConfig, ARMS as AARMS
        add("adew", lambda: ADEWGRU(ADEWConfig(N_CH_9, hidden_size=64,
                                               **AARMS["adew"])), N_CH_9,
            "anchored factorised erase")
    except ImportError:
        pass
    try:
        from xgru import XGRU, XGRUConfig
        add("xgru", lambda: XGRU(XGRUConfig(N_CH_9, 51, rank=8)), N_CH_9,
            "exponential gating + matrix memory")
    except ImportError:
        pass
    try:
        from cmr_gru import CMRGRU, CMRConfig, ARMS as CARMS
        add("cmr", lambda: CMRGRU(CMRConfig(N_CH_9, 84, **CARMS["cmr"])),
            N_CH_9, "FiLM + modular + routing")
    except ImportError:
        pass
    try:
        from dms_tcn import TCNModel
        add("tcn_ta", lambda: TCNModel("tcn_ta", N_CH_7), N_CH_7, "TCN + TA")
        add("msd_tcn", lambda: TCNModel("msd_tcn", N_CH_7), N_CH_7,
            "multi-scale dilated TCN")
        add("dms_tcn", lambda: TCNModel("dms_tcn", N_CH_7), N_CH_7,
            "dynamic multi-scale TCN, 65% fewer params")
    except Exception as e:
        print(f"  [skip] dms_tcn: {type(e).__name__}: {e}")
    return zoo


# ─── measurement ──────────────────────────────────────────────────────────────

@torch.no_grad()
def time_inference(model, x, repeats, device):
    model.eval()
    for _ in range(3):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeats):
        model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / repeats


def time_training(model, x, y, repeats, device):
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.BCEWithLogitsLoss()

    def step():
        opt.zero_grad()
        out = model(x)
        crit(out, y).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(3):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(repeats):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / repeats
    peak = (torch.cuda.max_memory_allocated() / 1e6 if device == "cuda"
            else float("nan"))
    return dt, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--train_windows", type=int, default=400_000)
    ap.add_argument("--only", default=None,
                    help="comma-separated subset of model names")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: CPU timings are not comparable to the GPU runs "
              "these models were trained under.\n")
    torch.manual_seed(0)

    want = set(args.only.split(",")) if args.only else None
    zoo = build_zoo(want)
    if not zoo:
        print("No models available — run from src/ so the modules import.")
        return

    print(f"\nbatch {args.batch}, sequence {T_LEN}, "
          f"{args.repeats} timed repeats per measurement\n")
    print(f"{'model':>16} {'ch':>3} {'params':>8} {'MMACs':>8} "
          f"{'infer us/win':>13} {'train ms/bat':>13} {'peak MB':>9} "
          f"{'epoch min':>10}")
    print("-" * 96)

    rows = []
    for name, model, ch, note in zoo:
        try:
            model = model.to(device)
            x = torch.randn(args.batch, T_LEN, ch, device=device)
            y = (torch.rand(args.batch, 4, device=device) < 0.2).float()
            p = n_params(model)
            mac = count_macs(model, ch)
            ti = time_inference(model, x, args.repeats, device)
            tt, peak = time_training(model, x, y, args.repeats, device)
            per_win = ti / args.batch * 1e6
            epoch_min = tt * (args.train_windows / args.batch) / 60
            print(f"{name:>16} {ch:>3} {p:>8,} {mac/1e6:>8.1f} "
                  f"{per_win:>13.2f} {tt*1e3:>13.1f} {peak:>9.1f} "
                  f"{epoch_min:>10.2f}")
            rows.append({"model": name, "channels": ch, "params": p,
                         "MMACs_per_window": mac / 1e6,
                         "infer_us_per_window": per_win,
                         "train_ms_per_batch": tt * 1e3,
                         "peak_MB": peak, "epoch_min_est": epoch_min,
                         "note": note})
        except Exception as e:
            print(f"{name:>16} FAILED: {type(e).__name__}: {e}")
        finally:
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    if not rows:
        return

    fused = next((r for r in rows if r["model"] == "gru_fused_9ch"), None)
    if fused:
        print(f"\n\n{'#'*96}\n# RELATIVE TO nn.GRU (9 channels)\n{'#'*96}\n")
        print(f"{'model':>16} {'params':>9} {'infer':>9} {'train':>9}")
        for r in rows:
            print(f"{r['model']:>16} "
                  f"{r['params']/fused['params']:>8.2f}x "
                  f"{r['infer_us_per_window']/fused['infer_us_per_window']:>8.2f}x "
                  f"{r['train_ms_per_batch']/fused['train_ms_per_batch']:>8.2f}x")
        slowest = max(rows, key=lambda r: r["train_ms_per_batch"])
        print(f"\n  The custom cells are Python loops over {T_LEN} steps; "
              f"nn.GRU is a fused kernel.")
        print(f"  Slowest here is {slowest['model']} at "
              f"{slowest['train_ms_per_batch']/fused['train_ms_per_batch']:.1f}x "
              f"the training cost of nn.GRU,")
        print(f"  which is why those pilots took hours rather than minutes.")

    print(f"\n\n{'#'*96}\n# EFFICIENCY FINDINGS WORTH REPORTING\n{'#'*96}\n")
    for a, b in [("gru_ta_7ch", "tsl_gru"), ("tcn_ta", "dms_tcn"),
                 ("ar_rhu_h88", "ar_rhu")]:
        ra = next((r for r in rows if r["model"] == a), None)
        rb = next((r for r in rows if r["model"] == b), None)
        if ra and rb:
            print(f"  {b} vs {a}:")
            print(f"    params {rb['params']:,} vs {ra['params']:,}  "
                  f"({100*(rb['params']/ra['params']-1):+.0f}%)")
            print(f"    MACs   {rb['MMACs_per_window']:.1f}M vs "
                  f"{ra['MMACs_per_window']:.1f}M  "
                  f"({100*(rb['MMACs_per_window']/ra['MMACs_per_window']-1):+.0f}%)")
            print(f"    infer  {rb['infer_us_per_window']:.2f} vs "
                  f"{ra['infer_us_per_window']:.2f} us/window\n")
    print("  Note: parameter and MAC reductions are architectural and will "
          "hold on any\n  device. The latency numbers here reflect an "
          "unoptimised Python loop, so a\n  fused or compiled implementation "
          "would narrow the inference gap. Report\n  params and MACs as the "
          "efficiency claim, latency as measured on this setup.")

    try:
        import pandas as pd
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        df.to_csv(OUT_DIR / "T10_efficiency.csv", index=False)
        with open(OUT_DIR / "T10_efficiency.md", "w") as f:
            f.write(f"# Computational cost\n\nDevice: {device}, batch "
                    f"{args.batch}, sequence {T_LEN}.\n\n")
            f.write(df.drop(columns=["note"]).to_markdown(
                index=False, floatfmt=".2f"))
            f.write("\n\n## Notes\n\n")
            for r in rows:
                f.write(f"- **{r['model']}** — {r['note']}\n")
        print(f"\n✓ Saved -> {OUT_DIR / 'T10_efficiency.csv'} and .md")
    except ImportError:
        print("\n  (pandas unavailable; table not written)")


if __name__ == "__main__":
    main()
