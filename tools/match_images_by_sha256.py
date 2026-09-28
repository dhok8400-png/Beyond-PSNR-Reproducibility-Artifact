#!/usr/bin/env python3
import csv
import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "supplementary" / "Supplementary_Table_S1_Image_Identity_Manifest.csv"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def main():
    if len(sys.argv) != 2:
        print("Usage: python tools/match_images_by_sha256.py /path/to/downloaded/dataset")
        raise SystemExit(2)

    source = Path(sys.argv[1]).expanduser().resolve()
    if not source.exists():
        raise SystemExit(f"Not found: {source}")

    expected = {}
    with MANIFEST.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            expected[row["source_sha256"].lower()] = row["candidate_id"]

    matches = {}
    scanned = 0
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        scanned += 1
        try:
            digest = sha256(path).lower()
        except (OSError, PermissionError):
            continue
        if digest in expected:
            matches[expected[digest]] = path
            print(f"{expected[digest]}  {path}")

    missing = sorted(set(expected.values()) - set(matches))
    print(f"\nScanned files: {scanned}")
    print(f"Matched study images: {len(matches)}/{len(expected)}")
    if missing:
        print("Missing IDs:", ", ".join(missing))
        raise SystemExit(1)

if __name__ == "__main__":
    main()
