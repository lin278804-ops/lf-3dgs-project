"""SHA-256 index of the experiment records copied from the server (source_data/gap, source_data/gapA)."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REMOTE = "$LFPIV_WORK"

rows = []
for base, origin in ((ROOT / "source_data/gap", REMOTE), (ROOT / "source_data/gapA", "experiments/gapA_*.py (local analysis of the recordings)")):
    for f in sorted(base.rglob("*")):
        if f.is_file() and not f.name.startswith(".") and f.name != "PROVENANCE.csv":
            rel = f.relative_to(ROOT / "source_data")
            rows.append([str(rel), origin + ("/" + str(f.relative_to(base)) if origin == REMOTE else ""),
                         hashlib.sha256(f.read_bytes()).hexdigest(), f.stat().st_size])
with (ROOT / "source_data/gap/PROVENANCE.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["file", "origin", "sha256", "bytes"])
    w.writerows(rows)
print(len(rows), "files indexed")
