"""Static public-repository audit for PediaCare-Net v2."""
from __future__ import annotations

import compileall
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

FORBIDDEN_SOURCE_STRINGS = (
    "/content/drive/MyDrive",
    "LEGACY_CODE_DIR",
    "train_metabonet_pediatric",
)
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".ckpt", ".npy", ".npz", ".parquet")
FORBIDDEN_PUBLIC_FILES = (
    ROOT / "data" / "split_reference.json",
    ROOT / "data" / "private_split_reference.json",
)


def main() -> None:
    problems = []

    # Final thesis-facing source must be portable. Archived exploratory code is
    # intentionally excluded from this check because it preserves historical
    # working-directory assumptions for provenance only.
    for p in SRC.glob("*.py"):
        text = p.read_text(errors="replace")
        for needle in FORBIDDEN_SOURCE_STRINGS:
            if needle in text:
                problems.append(
                    f"hardcoded/legacy dependency in {p.relative_to(ROOT)}: {needle}"
                )

    # Restricted/large artifacts must never be in the public tree.
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in {".git", ".venv", "__pycache__"} for part in p.parts):
            continue
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"large/private artifact present: {p.relative_to(ROOT)}")

    # Subject-ID split references are private by design.
    for p in FORBIDDEN_PUBLIC_FILES:
        if p.exists():
            problems.append(f"private split-reference file present: {p.relative_to(ROOT)}")

    # Public results should contain only aggregate tables plus the aggregate
    # tuning protocol. This prevents accidental publication of per-subject JSON.
    results = ROOT / "results"
    if results.exists():
        for p in results.rglob("*.json"):
            if p.resolve() != (results / "tuning" / "protocol.json").resolve():
                problems.append(f"per-run JSON should not be public: {p.relative_to(ROOT)}")

    for d in (SRC, ROOT / "tests", ROOT / "scripts"):
        if not compileall.compile_dir(str(d), quiet=1):
            problems.append(f"compileall failed for {d.relative_to(ROOT)}/")

    if problems:
        print("PediaCare-Net repository audit: FAILED")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)

    print("PediaCare-Net repository audit: PASSED")
    print("  no legacy Google Drive dependency in final src/")
    print("  no raw/derived/checkpoint artifacts in the repository")
    print("  no public split-reference or per-subject JSON outputs")
    print("  src/, tests/, and scripts/ compile successfully")


if __name__ == "__main__":
    main()
