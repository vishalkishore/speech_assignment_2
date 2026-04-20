from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # stage1_lid/
DEFAULT_UTTERANCES = PROJECT_ROOT / "manifests/utterances.csv"
DEFAULT_SPANS = PROJECT_ROOT / "manifests/gemini_lid_spans_original.csv"
DEFAULT_OUTDIR = PROJECT_ROOT / "manifests"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Create a clean held-out split for the Modi-Lex clip.")
    p.add_argument("--utterances-csv", type=Path, default=DEFAULT_UTTERANCES)
    p.add_argument("--spans-csv", type=Path, default=DEFAULT_SPANS)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--heldout-start-idx", type=int, default=30)
    p.add_argument("--heldout-count", type=int, default=8)
    p.add_argument("--valid-count", type=int, default=5)
    p.add_argument("--heldout-ids", type=str, default="", help="Comma-separated explicit utterance_ids for held-out test.")
    p.add_argument("--valid-ids", type=str, default="", help="Comma-separated explicit utterance_ids for adapt_valid.")
    return p.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    args = parse_args()
    utterances = read_csv(args.utterances_csv)
    utterances = sorted(utterances, key=lambda r: float(r["start_sec"]))
    by_id = {r["utterance_id"]: r for r in utterances}

    explicit_heldout_ids = {x.strip() for x in args.heldout_ids.split(",") if x.strip()}
    explicit_valid_ids = {x.strip() for x in args.valid_ids.split(",") if x.strip()}
    unknown = sorted((explicit_heldout_ids | explicit_valid_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown utterance ids in explicit split config: {unknown}")

    n = len(utterances)
    hs = max(0, min(args.heldout_start_idx, n))
    he = max(hs, min(hs + args.heldout_count, n))
    if explicit_heldout_ids:
        heldout_ids = explicit_heldout_ids
    else:
        heldout_ids = {r["utterance_id"] for r in utterances[hs:he]}

    remaining = [r for r in utterances if r["utterance_id"] not in heldout_ids]
    if explicit_valid_ids:
        if explicit_valid_ids & heldout_ids:
            raise ValueError("valid_ids and heldout_ids overlap")
        valid_ids = explicit_valid_ids
    else:
        valid_rows = remaining[-args.valid_count :] if args.valid_count > 0 else []
        valid_ids = {r["utterance_id"] for r in valid_rows}

    split_rows = []
    for i, row in enumerate(utterances):
        new_row = dict(row)
        uid = row["utterance_id"]
        if uid in heldout_ids:
            new_row["clean_split"] = "heldout_test"
        elif uid in valid_ids:
            new_row["clean_split"] = "adapt_valid"
        else:
            new_row["clean_split"] = "adapt_train"
        split_rows.append(new_row)

    spans = read_csv(args.spans_csv)
    spans_rows = []
    for row in spans:
        uid = row["utterance_id"]
        if uid in heldout_ids:
            split = "heldout_test"
        elif uid in valid_ids:
            split = "adapt_valid"
        else:
            split = "adapt_train"
        new_row = dict(row)
        new_row["clean_split"] = split
        spans_rows.append(new_row)

    args.outdir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.outdir / "utterances_clean_split.csv",
        split_rows,
        fieldnames=list(split_rows[0].keys()),
    )
    write_csv(
        args.outdir / "spans_clean_split.csv",
        spans_rows,
        fieldnames=list(spans_rows[0].keys()),
    )

    summary = {
        "utterances_total": n,
        "heldout_start_idx": hs,
        "heldout_end_idx_exclusive": he,
        "heldout_count": len(heldout_ids),
        "valid_count": len(valid_ids),
        "train_count": n - len(heldout_ids) - len(valid_ids),
        "heldout_utterance_ids": sorted(heldout_ids),
    }
    (args.outdir / "heldout_split_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
