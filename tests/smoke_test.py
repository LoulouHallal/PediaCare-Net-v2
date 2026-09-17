"""Dataset-free smoke test for the final PediaCare-Net v2 model path.

Run from the repository root:
    python tests/smoke_test.py
"""

from __future__ import annotations

import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import config  # noqa: E402
from splits import subject_level_split  # noqa: E402
from tsl_gru import TSLGRU  # noqa: E402


def main() -> None:
    # 1) Portability: the default project root must be this clone, not Drive.
    if "PEDIACARE_PROJECT_ROOT" not in os.environ:
        assert config.PROJECT_ROOT == ROOT.resolve(), (config.PROJECT_ROOT, ROOT)
    assert "/content/drive/" not in str(config.PROJECT_ROOT)

    # 2) Historical split logic: 244 subjects -> 170/36/38.
    meta = np.array([f"S{i:03d}" for i in range(244)])
    tr, va, te = subject_level_split(meta, seed=42)
    counts = tuple(int(mask.sum()) for mask in (tr, va, te))
    assert counts == (170, 36, 38), counts
    assert not np.any(tr & va) and not np.any(tr & te) and not np.any(va & te)
    assert np.all(tr | va | te)

    # 3) Final 7-channel TSL-GRU: real implementation, no mock model.
    torch.manual_seed(42)
    model = TSLGRU(
        "tsl_gru", in_ch=7, hidden=64, layers=2, dropout=0.2, rho=0.9
    ).eval()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 15_974, n_params

    x = torch.randn(4, 60, 7, dtype=torch.float32)
    with torch.no_grad():
        y = model(x)

    assert tuple(y.shape) == (4, 4), tuple(y.shape)
    assert bool(torch.isfinite(y).all()), "non-finite TSL-GRU output"
    assert bool(((y >= 0.0) & (y <= 1.0)).all()), "heads must return probabilities"

    print("PediaCare-Net smoke test: PASSED")
    print(f"  project_root: {config.PROJECT_ROOT}")
    print(f"  split: train/val/test = {counts}")
    print(f"  TSL-GRU params: {n_params:,}")
    print(f"  output shape: {tuple(y.shape)}")


if __name__ == "__main__":
    main()
