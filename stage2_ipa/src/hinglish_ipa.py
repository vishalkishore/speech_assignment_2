from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path("/home/vishal/assignment2-final")
PART_ROOT = PROJECT_ROOT / "stage2_ipa"
DEFAULT_LEXICON = PART_ROOT / "data/technical_lexicon_ipa.json"
DEFAULT_ENGLISH_OVERRIDES = PART_ROOT / "data/english_overrides_ipa.json"
DEFAULT_ROMAN_HINDI_OVERRIDES = PART_ROOT / "data/roman_hindi_overrides_ipa.json"
DEFAULT_INPUT = PROJECT_ROOT / "stage1_stt/stt/whisper_large_v3_oracle_spans_clean_lexmodi_10m.hyp.txt"
TOKEN_RE = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[\u0900-\u097F]+|\d+|[^\w\s]", re.UNICODE)


DEV_TO_IPA = {
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː", "उ": "ʊ", "ऊ": "uː",
    "ए": "eː", "ऐ": "ɛː", "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ",
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ", "ू": "uː", "े": "eː",
    "ै": "ɛː", "ो": "oː", "ौ": "ɔː", "ं": "ŋ", "ँ": "̃", "ः": "h",
    "क": "k", "ख": "kʰ", "ग": "g", "घ": "gʱ", "च": "tʃ", "छ": "tʃʰ",
    "ज": "dʒ", "झ": "dʒʱ", "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʱ",
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʱ", "प": "p", "फ": "pʰ",
    "ब": "b", "भ": "bʱ", "म": "m", "न": "n", "ण": "ɳ", "ङ": "ŋ",
    "य": "j", "र": "r", "ल": "l", "व": "ʋ", "श": "ʃ", "ष": "ʂ",
    "स": "s", "ह": "ɦ", "ळ": "ɭ"
}

ROMAN_HINDI_PATTERNS = [
    ("aa", "aː"), ("ii", "iː"), ("ee", "iː"), ("oo", "uː"), ("uu", "uː"),
    ("ai", "ɛː"), ("au", "ɔː"), ("kh", "kʰ"), ("gh", "gʱ"), ("chh", "tʃʰ"),
    ("ch", "tʃ"), ("jh", "dʒʱ"), ("th", "tʰ"), ("dh", "dʱ"), ("ph", "pʰ"),
    ("bh", "bʱ"), ("sh", "ʃ"), ("ng", "ŋ"), ("ny", "ɲ"), ("a", "ə"),
    ("i", "ɪ"), ("u", "ʊ"), ("e", "eː"), ("o", "oː"), ("k", "k"), ("g", "g"),
    ("j", "dʒ"), ("t", "t̪"), ("d", "d̪"), ("p", "p"), ("b", "b"), ("m", "m"),
    ("n", "n"), ("r", "r"), ("l", "l"), ("v", "ʋ"), ("w", "ʋ"), ("s", "s"),
    ("h", "ɦ"), ("y", "j")
]

ENGLISH_DIGRAPHS = [
    ("tion", "ʃən"), ("sion", "ʒən"), ("ture", "tʃər"), ("ph", "f"), ("sh", "ʃ"),
    ("ch", "tʃ"), ("th", "θ"), ("gh", "g"), ("qu", "kw"), ("ck", "k"), ("ng", "ŋ")
]
ENGLISH_CHARS = {
    "a": "æ", "b": "b", "c": "k", "d": "d", "e": "ɛ", "f": "f", "g": "g",
    "h": "h", "i": "ɪ", "j": "dʒ", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "ɑ", "p": "p", "q": "k", "r": "r", "s": "s", "t": "t", "u": "ʌ",
    "v": "v", "w": "w", "x": "ks", "y": "j", "z": "z"
}
ACRONYM_IPA = {
    "a": "eɪ", "b": "biː", "c": "siː", "d": "diː", "e": "iː", "f": "ɛf",
    "g": "dʒiː", "h": "eɪtʃ", "i": "aɪ", "j": "dʒeɪ", "k": "keɪ", "l": "ɛl",
    "m": "ɛm", "n": "ɛn", "o": "oʊ", "p": "piː", "q": "kjuː", "r": "ɑr",
    "s": "ɛs", "t": "tiː", "u": "juː", "v": "viː", "w": "dʌbəljuː", "x": "ɛks",
    "y": "waɪ", "z": "zɛd"
}
ROMAN_HINDI_HINTS = {
    "hai", "haan", "haanji", "nahi", "nahin", "kya", "ka", "ki", "ke", "mein",
    "main", "hum", "aap", "yeh", "woh", "jo", "kyunki", "phir", "agar", "wala",
    "wali", "wale", "samajh", "accha", "acha", "theek", "thik", "sirf", "sab"
}
@dataclass
class TokenIpa:
    token: str
    token_type: str
    ipa: str


def load_lexicon(path: Path) -> dict[str, str]:
    return {k.lower(): v for k, v in json.loads(path.read_text(encoding="utf-8")).items()}


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def is_devanagari(token: str) -> bool:
    return bool(re.fullmatch(r"[\u0900-\u097F]+", token))


def is_acronym(token: str) -> bool:
    return token.isalpha() and token.isupper() and 1 < len(token) <= 6


def looks_roman_hindi(token: str) -> bool:
    t = token.lower()
    if t in ROMAN_HINDI_HINTS:
        return True
    if len(t) <= 3:
        return False
    if t.endswith(("wala", "wali", "wale", "ji")):
        return True
    return any(x in t for x in ("bh", "dh", "kh", "gh", "ph", "nahi", "samaj", "thik", "acha"))


def devanagari_to_ipa(token: str) -> str:
    out: list[str] = []
    for ch in token:
        out.append(DEV_TO_IPA.get(ch, ch))
    return "".join(out)


def roman_hindi_to_ipa(token: str, roman_hindi_overrides: dict[str, str]) -> str:
    text = token.lower()
    if text in roman_hindi_overrides:
        return roman_hindi_overrides[text]
    out: list[str] = []
    i = 0
    patterns = sorted(ROMAN_HINDI_PATTERNS, key=lambda x: len(x[0]), reverse=True)
    while i < len(text):
        matched = False
        for pat, ipa in patterns:
            if text.startswith(pat, i):
                out.append(ipa)
                i += len(pat)
                matched = True
                break
        if not matched:
            out.append(text[i])
            i += 1
    return "".join(out)


def english_fallback_to_ipa(token: str, english_overrides: dict[str, str]) -> str:
    text = token.lower()
    if text in english_overrides:
        return english_overrides[text]
    for pat, rep in ENGLISH_DIGRAPHS:
        text = text.replace(pat, rep)
    out: list[str] = []
    for ch in text:
        out.append(ENGLISH_CHARS.get(ch, ch))
    return "".join(out)


def acronym_to_ipa(token: str) -> str:
    return " ".join(ACRONYM_IPA.get(ch.lower(), ch.lower()) for ch in token)


def classify_and_map(
    token: str,
    lexicon: dict[str, str],
    english_overrides: dict[str, str],
    roman_hindi_overrides: dict[str, str],
) -> TokenIpa:
    if re.fullmatch(r"\d+", token):
        return TokenIpa(token=token, token_type="number", ipa=token)
    if re.fullmatch(r"[^\w\s]", token, re.UNICODE):
        return TokenIpa(token=token, token_type="punct", ipa=token)
    if is_devanagari(token):
        return TokenIpa(token=token, token_type="devanagari", ipa=devanagari_to_ipa(token))
    token_l = token.lower()
    if token_l in lexicon:
        return TokenIpa(token=token, token_type="technical", ipa=lexicon[token_l])
    if token_l in english_overrides:
        return TokenIpa(token=token, token_type="english", ipa=english_overrides[token_l])
    if is_acronym(token):
        return TokenIpa(token=token, token_type="acronym", ipa=acronym_to_ipa(token))
    if looks_roman_hindi(token):
        return TokenIpa(token=token, token_type="roman_hindi", ipa=roman_hindi_to_ipa(token, roman_hindi_overrides))
    return TokenIpa(token=token, token_type="english", ipa=english_fallback_to_ipa(token, english_overrides))


def convert_text_to_ipa(
    text: str,
    lexicon: dict[str, str],
    english_overrides: dict[str, str],
    roman_hindi_overrides: dict[str, str],
) -> list[TokenIpa]:
    return [classify_and_map(tok, lexicon, english_overrides, roman_hindi_overrides) for tok in tokenize(text)]


def build_ipa_string(items: list[TokenIpa]) -> str:
    return " ".join(x.ipa for x in items if x.token_type != "punct").strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage-1 Hinglish -> IPA mapper.")
    p.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--text-column", default="transcript")
    p.add_argument("--lexicon", type=Path, default=DEFAULT_LEXICON)
    p.add_argument("--english-overrides", type=Path, default=DEFAULT_ENGLISH_OVERRIDES)
    p.add_argument("--roman-hindi-overrides", type=Path, default=DEFAULT_ROMAN_HINDI_OVERRIDES)
    p.add_argument("--output-csv", type=Path, required=True)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--split", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    lexicon = load_lexicon(args.lexicon)
    english_overrides = load_lexicon(args.english_overrides)
    roman_hindi_overrides = load_lexicon(args.roman_hindi_overrides)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []
    with args.input_csv.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if args.split and row.get("split") != args.split:
                continue
            text = (row.get(args.text_column) or "").strip()
            if not text:
                continue
            items = convert_text_to_ipa(text, lexicon, english_overrides, roman_hindi_overrides)
            rows_out.append(
                {
                    "utterance_id": row.get("utterance_id", ""),
                    "source": row.get("source", ""),
                    "split": row.get("split", ""),
                    "text": text,
                    "ipa": build_ipa_string(items),
                    "token_debug_json": json.dumps([x.__dict__ for x in items], ensure_ascii=False),
                }
            )
            if args.limit is not None and len(rows_out) >= args.limit:
                break

    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["utterance_id", "source", "split", "text", "ipa", "token_debug_json"],
        )
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"rows={len(rows_out)}")
    print(f"output_csv={args.output_csv}")


if __name__ == "__main__":
    main()
