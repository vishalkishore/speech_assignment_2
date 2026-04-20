from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a chunk-level Maithili TTS manifest from cleaned Lex-Modi spans.")
    p.add_argument("--spans-csv", type=Path, required=True)
    p.add_argument("--dictionary-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    return p.parse_args()


def load_dictionary(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    token_map: dict[str, str] = {}
    phrase_map: list[tuple[str, str]] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = row["source"].strip().lower()
            target = row["target_maithili"].strip()
            if not source or not target:
                continue
            if " " in source:
                phrase_map.append((source, target))
            else:
                token_map[source] = target
    phrase_map.sort(key=lambda x: len(x[0]), reverse=True)
    return token_map, phrase_map


def translate_text(text: str, token_map: dict[str, str], phrase_map: list[tuple[str, str]]) -> tuple[str, float]:
    working = text
    for src, tgt in phrase_map:
        working = re.sub(rf"\b{re.escape(src)}\b", tgt, working, flags=re.IGNORECASE)

    tokens = TOKEN_RE.findall(working)
    out: list[str] = []
    mapped = 0
    lexical = 0
    for tok in tokens:
        key = tok.lower()
        if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+", tok):
            lexical += 1
            if key in token_map:
                out.append(token_map[key])
                mapped += 1
            else:
                out.append(tok)
        else:
            out.append(tok)

    coverage = mapped / lexical if lexical else 0.0
    pieces: list[str] = []
    for i, tok in enumerate(out):
        if i > 0 and tok not in ",.!?:;)" and out[i - 1] not in "([":  # light detokenization
            pieces.append(" ")
        pieces.append(tok)
    return "".join(pieces).strip(), coverage


def main() -> None:
    args = parse_args()
    token_map, phrase_map = load_dictionary(args.dictionary_csv)

    grouped: dict[str, dict[str, object]] = {}
    with args.spans_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    rows.sort(key=lambda r: (r["utterance_id"], int(r["span_index"])))
    for row in rows:
        utt_id = row["utterance_id"]
        entry = grouped.setdefault(
            utt_id,
            {
                "utterance_id": utt_id,
                "audio_path": row["audio_path"],
                "source": row["source"],
                "split": row["split"],
                "chunk_start_sec": float(row["chunk_start_sec"]),
                "chunk_end_sec": float(row["chunk_end_sec"]),
                "texts": [],
            },
        )
        text = (row.get("text") or "").strip()
        if text:
            entry["texts"].append(text)

    out_rows: list[dict[str, object]] = []
    for utt_id, entry in grouped.items():
        source_text = " ".join(entry["texts"]).strip()
        target_maithili, coverage = translate_text(source_text, token_map, phrase_map)
        duration = float(entry["chunk_end_sec"]) - float(entry["chunk_start_sec"])
        out_rows.append(
            {
                "utterance_id": utt_id,
                "audio_path": str(entry["audio_path"]),
                "source": str(entry["source"]),
                "split": str(entry["split"]),
                "chunk_start_sec": f"{float(entry['chunk_start_sec']):.3f}",
                "chunk_end_sec": f"{float(entry['chunk_end_sec']):.3f}",
                "chunk_duration_sec": f"{duration:.3f}",
                "source_text": source_text,
                "target_maithili": target_maithili,
                "dictionary_coverage": f"{coverage:.4f}",
            }
        )

    out_rows.sort(key=lambda r: (r["source"], float(r["chunk_start_sec"]), r["utterance_id"]))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "utterance_id",
                "audio_path",
                "source",
                "split",
                "chunk_start_sec",
                "chunk_end_sec",
                "chunk_duration_sec",
                "source_text",
                "target_maithili",
                "dictionary_coverage",
            ],
        )
        writer.writeheader()
        writer.writerows(out_rows)

    total_sec = sum(float(r["chunk_duration_sec"]) for r in out_rows)
    print(f"rows={len(out_rows)}")
    print(f"duration_sec={total_sec:.2f}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
