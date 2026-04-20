from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path("/home/vishal/assignment2-final")
PART_ROOT = PROJECT_ROOT / "stage2_translation"
DEFAULT_INPUT = PROJECT_ROOT / "stage1_stt/stt/whisper_large_v3_oracle_spans_clean_lexmodi_10m.hyp.txt"
DEFAULT_OUT = PART_ROOT / "dictionary/maithili_parallel_candidates_500.csv"


WORD_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+")

# Small, pragmatic stoplist to avoid wasting the 500-word budget on glue words.
EN_STOP = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "for", "from",
    "had", "has", "have", "having", "he", "her", "hers", "him", "his",
    "i", "if", "in", "into", "is", "it", "its", "just", "me", "my", "no",
    "not", "now", "of", "on", "or", "our", "ours", "out", "she", "so", "some",
    "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "too", "up", "us", "was", "we", "were", "what", "when",
    "where", "which", "who", "why", "will", "with", "would", "you", "your",
    "yours", "yes",
}

HI_STOP_DEV = {
    "और", "या", "है", "हैं", "था", "थे", "थी", "में", "पर", "से", "को", "की", "का", "के",
    "यह", "वह", "ये", "वे", "मैं", "हम", "आप", "तुम", "तो", "भी", "नहीं", "हाँ",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a 500-word parallel-dictionary template from a transcript .txt.")
    p.add_argument("--input-txt", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--min-freq", type=int, default=1)
    p.add_argument(
        "--include-bigrams",
        action="store_true",
        help="Also include frequent 2-word phrases (after stopword filtering) to help reach 500 entries.",
    )
    return p.parse_args()


def iter_words(text: str) -> list[str]:
    return [m.group(0) for m in WORD_RE.finditer(text)]


def is_devanagari(tok: str) -> bool:
    return bool(re.fullmatch(r"[\u0900-\u097F]+", tok))


def is_noise(tok: str) -> bool:
    t = tok.lower()
    if len(t) <= 1:
        return True
    if t.isdigit():
        return True
    # common filler
    if t in {"um", "uh", "hmm", "mm", "erm"}:
        return True
    return False


def main() -> None:
    args = parse_args()
    text = args.input_txt.read_text(encoding="utf-8", errors="ignore")
    counts: Counter[str] = Counter()
    kept_tokens: list[str] = []
    for w in iter_words(text):
        if is_noise(w):
            continue
        if is_devanagari(w):
            if w in HI_STOP_DEV:
                continue
            counts[w] += 1
            kept_tokens.append(w)
        else:
            wl = w.lower()
            if wl in EN_STOP:
                continue
            counts[wl] += 1
            kept_tokens.append(wl)

    if args.include_bigrams and len(kept_tokens) >= 2:
        bi = Counter(" ".join(kept_tokens[i : i + 2]) for i in range(len(kept_tokens) - 1))
        # Add bigrams with a lower weight so unigrams dominate.
        for k, v in bi.items():
            counts[k] += max(1, v // 2)

    rows = []
    for w, freq in counts.most_common():
        if freq < args.min_freq:
            continue
        rows.append(
            {
                "source": w,
                "source_type": "candidate",
                "target_maithili": "",
                "target_script": "Devanagari",
                "notes": f"auto_from_transcript freq={freq}",
            }
        )
        if len(rows) >= args.limit:
            break

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["source", "source_type", "target_maithili", "target_script", "notes"],
        )
        w.writeheader()
        w.writerows(rows)

    print(f"unique_terms={len(counts)}")
    print(f"rows_written={len(rows)}")
    print(f"out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
