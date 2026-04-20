from __future__ import annotations

import csv
from pathlib import Path


def load_rows_by_split(path: Path) -> dict[str, list[dict[str, str]]]:
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    out = {"train": [], "valid": [], "test": []}
    for row in rows:
        out[row["split"]].append(row)
    return out
