"""
stage345_advanced.py
======================
RQ2 stages 3, 4 and 5 — the architectural progression:

    stage 3   tcn            temporal convolutions alone
    stage 4   transformer    self-attention alone
    stage 5   hybrid         both pathways with learned gating

Plus two ablations that isolate what the hybrid's extra machinery buys:

    hybrid_concat  same two pathways, concatenated instead of gated
                   -> isolates the LEARNED GATE
    hybrid_meanpool same gating, mean pooling instead of attention pooling
                   -> isolates ATTENTION POOLING

Without those two, a hybrid win could come from the gate, the pooling, or
simply from having two pathways, and there would be no way to tell which.

EVERYTHING ELSE IS UNCHANGED FROM STAGE 2
-----------------------------------------
Same windows, same subject split (170/36/38, seed 42), same "any" labels,
same losses, same lazy WindowDataset, same optimiser and early stopping on
validation mean-AUPRC, same threshold policy, same metrics and
subject-level bootstrap. Only the architecture changes, so stage-2 and
stage-345 rows are directly comparable in one table.

The loss and balancing options are identical too, so the RQ1 finding
(weighted BCE is the only mechanism that improves AUPRC) can be tested on
these architectures as well.

CAUSALITY
---------
Every architecture here is strictly causal: the TCN uses left-padded
dilated convolutions with the right padding trimmed, and the Transformer
uses an upper-triangular attention mask. A model that could see future
timesteps would leak the label.

WHAT TO EXPECT
--------------
Fourteen configurations from logistic regression to LSTM already fall
between AUPRC 0.66 and 0.77 at h=15, with the deep models clustered at
0.75-0.77. If TCN, Transformer and the hybrid land in that same band, the
conclusion is that architecture is a second-order factor for this task --
which is a legitimate and reportable result, not a failure. A prior
version of this project found a plain GRU matching a dual-pathway
TCN-Transformer, so this outcome has precedent here.

Usage:
    python stage345_advanced.py --model tcn --loss weighted_bce
    python stage345_advanced.py --model all --loss weighted_bce
    python stage345_advanced.py --collect
"""

import math
import argparse

import torch
import torch.nn as nn

import config
import stage2_deep as S2
from stage2_deep import Heads, MultiHorizonLoss, WindowDataset  # noqa: F401

HORIZONS = config.HORIZONS
MODELS = ["tcn", "transformer", "hybrid", "hybrid_concat", "hybrid_meanpool"]
OUT_DIR = config.RESULTS / "RQ2_models" / "stage345_advanced"


# ─── TCN (stage 3) ────────────────────────────────────────────────────────────

class CausalConv1d(nn.Module):
    """Left-padded dilated convolution; right padding trimmed so no
    timestep can attend to the future."""

    def __init__(self, c_in, c_out, k, dilation):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation, padding=self.pad)

    def forward(self, x):
        return self.conv(x)[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1, dropout=0.2):
        super().__init__()
        self.c1 = CausalConv1d(c_in, c_out, k, dilation)
        self.c2 = CausalConv1d(c_out, c_out, k, dilation)
        self.n1, self.n2 = nn.BatchNorm1d(c_out), nn.BatchNorm1d(c_out)
        self.relu, self.drop = nn.ReLU(), nn.Dropout(dropout)
        self.res = (nn.Conv1d(c_in, c_out, 1) if c_in != c_out else nn.Identity())

    def forward(self, x):
        r = self.res(x)
        o = self.drop(self.relu(self.n1(self.c1(x))))
        o = self.drop(self.relu(self.n2(self.c2(o))))
        return self.relu(o + r)


class TCNPathway(nn.Module):
    """
    Four residual blocks, dilations 1/2/4/8, kernel 3.
    Receptive field = 1 + 2*(k-1)*(2^4 - 1) = 61 timesteps, which covers
    the full 60-step window exactly.
    """

    def __init__(self, in_ch, hidden=64, k=3, dropout=0.2):
        super().__init__()
        self.proj = nn.Conv1d(in_ch, hidden, 1)
        self.blocks = nn.ModuleList([TCNBlock(hidden, hidden, k, d, dropout)
                                     for d in (1, 2, 4, 8)])

    def forward(self, x):                      # (B,T,F) -> (B,T,D)
        o = self.proj(x.permute(0, 2, 1))
        for b in self.blocks:
            o = b(o)
        return o.permute(0, 2, 1)


# ─── Transformer (stage 4) ────────────────────────────────────────────────────

class TransformerPathway(nn.Module):
    """Two encoder layers, 4 heads, sinusoidal positions, causal mask."""

    def __init__(self, in_ch, hidden=64, heads=4, layers=2, dropout=0.1,
                 max_len=60):
        super().__init__()
        self.proj = nn.Linear(in_ch, hidden)
        self.register_buffer("pe", self._pe(max_len, hidden))
        layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=heads,
                                           dim_feedforward=hidden * 4,
                                           dropout=dropout, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)

    @staticmethod
    def _pe(n, d):
        pe = torch.zeros(n, d)
        pos = torch.arange(n, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        return pe.unsqueeze(0)

    def forward(self, x):
        T = x.size(1)
        o = self.proj(x) + self.pe[:, :T, :]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        return self.enc(o, mask=mask)


# ─── Fusion and pooling (stage 5) ─────────────────────────────────────────────

class GatingFusion(nn.Module):
    """gate = sigmoid(W[tcn; tf]); fused = gate*tcn + (1-gate)*tf."""

    def __init__(self, hidden=64):
        super().__init__()
        self.g = nn.Linear(hidden * 2, hidden)

    def forward(self, a, b):
        gate = torch.sigmoid(self.g(torch.cat([a, b], dim=-1)))
        return gate * a + (1 - gate) * b


class AttentionPooling(nn.Module):
    """Learned temporal weighting, replacing mean pooling."""

    def __init__(self, hidden=64):
        super().__init__()
        self.a = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(),
                               nn.Linear(hidden // 2, 1, bias=False))

    def forward(self, x):
        return (torch.softmax(self.a(x), dim=1) * x).sum(dim=1)


# ─── Models ───────────────────────────────────────────────────────────────────

class AdvancedModel(nn.Module):
    """
    One class covering stages 3-5 plus the two hybrid ablations, so that
    every variant shares identical heads, pooling code and forward logic.
    Only the components named by `variant` differ.
    """

    def __init__(self, variant, in_ch, hidden=64):
        super().__init__()
        self.variant = variant
        self.use_tcn = variant != "transformer"
        self.use_tf = variant != "tcn"

        if self.use_tcn:
            self.tcn = TCNPathway(in_ch, hidden)
        if self.use_tf:
            self.tf = TransformerPathway(in_ch, hidden)

        if self.use_tcn and self.use_tf:
            if variant == "hybrid_concat":
                self.merge = nn.Linear(hidden * 2, hidden)   # no learned gate
            else:
                self.gate = GatingFusion(hidden)

        self.pool = (None if variant == "hybrid_meanpool"
                     else AttentionPooling(hidden))
        self.head = Heads(hidden)
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if self.use_tcn and self.use_tf:
            a, b = self.tcn(x), self.tf(x)
            z = (self.merge(torch.cat([a, b], dim=-1))
                 if self.variant == "hybrid_concat" else self.gate(a, b))
        else:
            z = self.tcn(x) if self.use_tcn else self.tf(x)
        z = z.mean(dim=1) if self.pool is None else self.pool(z)
        return self.head(z)


def build(name, in_ch, hidden):
    """Extends stage 2's builder so both stages share one training loop."""
    if name in MODELS:
        return AdvancedModel(name, in_ch, hidden)
    return S2.build(name, in_ch, hidden)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="tcn", choices=MODELS + ["all", "core"])
    ap.add_argument("--loss", default="weighted_bce",
                    choices=S2.LOSSES + ["all"])
    ap.add_argument("--balance", default="none", choices=S2.SEQ_BALANCE)
    ap.add_argument("--labels", default="any", choices=["any", "consensus"])
    ap.add_argument("--tag", default="stride6")
    ap.add_argument("--alpha", type=float, default=0.75)
    ap.add_argument("--gamma", type=float, default=2.0)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=80,
                    help="raised from 30 so the limit is never binding; "
                         "early stopping decides when to halt, and all "
                         "architectures then share one protocol")
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--lr", type=float, default=None,
                    help="default 1e-3, but 3e-4 for transformer-containing "
                         "variants: self-attention is unstable at 1e-3 "
                         "without warmup, especially with the large "
                         "pos_weight used by weighted BCE")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--train_cap", type=int, default=400_000)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--n_boot", type=int, default=config.N_BOOT)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Point stage 2's machinery at this stage's folder and model builder, so
    # training, evaluation and collection are literally the same code.
    S2.OUT_DIR = OUT_DIR
    S2.build = build

    if args.collect:
        S2.collect(args)
        return

    import numpy as np
    import common
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cpu":
        print("  WARNING: no GPU detected. These models are ~40x slower on "
              "CPU. Check Runtime > Change runtime type > GPU.")

    bw = common.load_built(args.tag)
    common.verify_split(bw.meta)
    masks = common.get_split(bw.meta)
    Y = bw._labels if args.labels == "consensus" else bw._labels_any
    print(f"windows {len(bw):,} | channels {bw.n_channels} | "
          f"train {masks[0].sum():,} val {masks[1].sum():,} "
          f"test {masks[2].sum():,}")

    models = {"all": MODELS,
              "core": ["tcn", "transformer", "hybrid"]}.get(args.model,
                                                            [args.model])
    losses = S2.LOSSES if args.loss == "all" else [args.loss]

    for mo in models:
        for lo in losses:
            args.lr = (args.lr if args.lr is not None
                       else (3e-4 if mo != "tcn" else 1e-3))
            name = S2.run_name(mo, lo, args)     # includes the seed
            if (OUT_DIR / f"{name}.json").exists() and not args.force:
                print(f"\n[skip] {name} already done")
                continue
            try:
                S2.run_one(mo, lo, bw, Y, masks, device, args, name)
            except Exception as e:
                print(f"  !! {name} FAILED: {type(e).__name__}: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()

    S2.collect(args)


if __name__ == "__main__":
    main()
