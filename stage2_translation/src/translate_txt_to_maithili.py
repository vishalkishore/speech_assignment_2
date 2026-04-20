from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # assignment2-final/
PART_ROOT = PROJECT_ROOT / "stage2_translation"
DEFAULT_INPUT_TXT = PROJECT_ROOT / "stage1_stt/stt/whisper_large_v3_oracle_spans_clean_lexmodi_10m.hyp.txt"
DEFAULT_DICT_CSV = PART_ROOT / "dictionary/maithili_parallel_dictionary_500.csv"
DEFAULT_OUTPUT_TXT = PART_ROOT / "output/whisper_large_v3_oracle_spans_clean_lexmodi_10m.maithili.txt"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\w\s]", re.UNICODE)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dictionary-first translation of a transcript .txt into Maithili.")
    p.add_argument("--input-txt", type=Path, default=DEFAULT_INPUT_TXT)
    p.add_argument("--dictionary-csv", type=Path, default=DEFAULT_DICT_CSV)
    p.add_argument("--output-txt", type=Path, default=DEFAULT_OUTPUT_TXT)
    return p.parse_args()


def load_dictionary(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    token_map: dict[str, str] = {}
    phrase_map: list[tuple[str, str]] = []
    with path.open("r", newline="") as f:
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
    out = []
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
    pieces = []
    for i, tok in enumerate(out):
        if i > 0 and tok not in ",.!?:;)" and out[i - 1] not in "([":  # light detokenization
            pieces.append(" ")
        pieces.append(tok)
    return "".join(pieces).strip(), coverage


def main() -> None:
    args = parse_args()
    text = args.input_txt.read_text(encoding="utf-8", errors="ignore").strip()
    token_map, phrase_map = load_dictionary(args.dictionary_csv)
    out, coverage = translate_text(text, token_map, phrase_map)

    args.output_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_txt.write_text(out + "\n", encoding="utf-8")
    print(f"coverage={coverage:.3f}")
    print(f"output_txt={args.output_txt}")


if __name__ == "__main__":
    main()
