from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


TOKEN_RE = re.compile(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\w\s]", re.UNICODE)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.?!।])\s+")

REPLACEMENTS = {
    "basic": "मूल",
    "follows": "पालन करैत अछि",
    "identically": "समान रूप सँ",
    "drawing": "खींचबाक",
    "arrange": "व्यवस्थित करू",
    "alphabetical": "वर्णक्रमानुसार",
    "per": "अनुसार",
    "picking": "चुनि रहल",
    "trained": "प्रशिक्षित",
    "comes": "अबैत अछि",
    "come": "अबि",
    "still": "अखन धरि",
    "again": "फेर",
    "my": "हमर",
    "we": "हम",
    "i": "हम",
    "it": "ई",
}

ACRONYM_MAP = {
    "IID": "आईआईडी",
    "OOD": "ओओडी",
    "ML": "एमएल",
    "MFCC": "एमएफसीसी",
    "DTW": "डीटीडब्ल्यू",
    "HMM": "एचएमएम",
    "LSTM": "एलएसटीएम",
    "CNN": "सीएनएन",
    "RNN": "आरएनएन",
    "FFT": "एफएफटी",
    "DFT": "डीएफटी",
}

FALLBACK_WORD_MAP = {
    "motivation": "मोटिवेशन",
    "head": "हेड",
    "government": "गवर्नमेंट",
    "inspiration": "इंस्पिरेशन",
    "one": "वन",
    "billion": "बिलियन",
    "mood": "मूड",
    "science": "साइंस",
    "spirituality": "स्पिरिचुअलिटी",
    "district": "डिस्ट्रिक्ट",
    "expert": "एक्सपर्ट",
    "opinion": "ओपिनियन",
}

DEVANAGARI_DIGITS = str.maketrans("0123456789", "०१२३४५६७८९")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare duration-aware Maithili TTS text.")
    p.add_argument("--translation-csv", type=Path, required=True)
    p.add_argument("--utterances-csv", type=Path, required=True)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--text-column", default="target_maithili")
    p.add_argument("--source-text-column", default="source_text")
    p.add_argument("--seconds-per-word", type=float, default=0.52)
    p.add_argument("--max-word-multiplier", type=float, default=1.0)
    p.add_argument("--min-words", type=int, default=6)
    return p.parse_args()


def normalize_text(text: str) -> str:
    text = re.sub(r"\bN/?A\b", " ", text, flags=re.IGNORECASE)
    for src, tgt in ACRONYM_MAP.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text)
    for src, tgt in FALLBACK_WORD_MAP.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text, flags=re.IGNORECASE)
    for src, tgt in REPLACEMENTS.items():
        text = re.sub(rf"\b{re.escape(src)}\b", tgt, text, flags=re.IGNORECASE)
    text = text.replace("%", " प्रतिशत ")
    text = text.replace("'", " ")
    text = re.sub(r"\s*/\s*", " ", text)
    text = re.sub(r"\b[ms]\b", " ", text)
    text = re.sub(r"[A-Za-z]+(?:-[A-Za-z]+)?", " ", text)
    text = text.translate(DEVANAGARI_DIGITS)
    text = re.sub(r"\s+([,?.!;:।])", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def lexical_tokens(text: str) -> list[str]:
    return [tok for tok in TOKEN_RE.findall(text) if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+|\d+", tok)]


def trim_to_word_budget(text: str, max_words: int) -> str:
    if len(lexical_tokens(text)) <= max_words:
        return text.strip()

    sentences = [s.strip() for s in SENTENCE_SPLIT_RE.split(text.strip()) if s.strip()]
    kept: list[str] = []
    used = 0
    for sent in sentences:
        sent_words = len(lexical_tokens(sent))
        if not kept and sent_words >= max_words:
            return trim_tokenwise(sent, max_words)
        if used + sent_words <= max_words:
            kept.append(sent)
            used += sent_words
        else:
            break
    if kept:
        return " ".join(kept).strip()
    return trim_tokenwise(text, max_words)


def trim_tokenwise(text: str, max_words: int) -> str:
    tokens = TOKEN_RE.findall(text)
    out: list[str] = []
    lexical = 0
    for tok in tokens:
        if re.fullmatch(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u0900-\u097F]+|\d+", tok):
            if lexical >= max_words:
                break
            lexical += 1
        out.append(tok)
    pieces: list[str] = []
    for i, tok in enumerate(out):
        if i > 0 and tok not in ",.!?:;)" and out[i - 1] not in "([": 
            pieces.append(" ")
        pieces.append(tok)
    return "".join(pieces).strip(" ,")


def main() -> None:
    args = parse_args()
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    durations: dict[str, float] = {}
    with args.utterances_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utt_id = row["utterance_id"]
            durations[utt_id] = float(row["end_sec"]) - float(row["start_sec"])

    rows_out = []
    with args.translation_csv.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            utt_id = row["utterance_id"]
            duration = durations.get(utt_id, 15.0)
            source_text = row.get(args.source_text_column, "") or ""
            target_text = normalize_text(row.get(args.text_column, "") or "")

            source_words = len(lexical_tokens(source_text))
            duration_budget = int(duration / args.seconds_per_word)
            source_budget = int(max(source_words * args.max_word_multiplier, 1))
            max_words = max(args.min_words, min(duration_budget, source_budget))
            trimmed = trim_to_word_budget(target_text, max_words)

            new_row = dict(row)
            new_row["tts_text_maithili"] = trimmed
            new_row["tts_word_budget"] = str(max_words)
            new_row["chunk_duration_sec"] = f"{duration:.3f}"
            rows_out.append(new_row)

    with args.output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"rows={len(rows_out)}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
