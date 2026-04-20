from __future__ import annotations

import argparse
import json
from pathlib import Path

import sys

# Allow running without installing a package: import directly from stage2_ipa/src.
PART_SRC = Path("/home/vishal/assignment2-final/stage2_ipa/src")
if str(PART_SRC) not in sys.path:
    sys.path.insert(0, str(PART_SRC))

from hinglish_ipa import build_ipa_string, convert_text_to_ipa, load_lexicon  # type: ignore


PROJECT_ROOT = Path("/home/vishal/assignment2-final")
PART_ROOT = PROJECT_ROOT / "stage2_ipa"
DEFAULT_TECH_LEXICON = PART_ROOT / "data/technical_lexicon_ipa.json"
DEFAULT_EN_OVERRIDES = PART_ROOT / "data/english_overrides_ipa.json"
DEFAULT_ROM_HI_OVERRIDES = PART_ROOT / "data/roman_hindi_overrides_ipa.json"
DEFAULT_INPUT_TXT = PROJECT_ROOT / "stage1_stt/stt/whisper_large_v3_oracle_spans_clean_lexmodi_10m.hyp.txt"
DEFAULT_OUTPUT_IPA = PART_ROOT / "ipa/whisper_large_v3_oracle_spans_clean_lexmodi_10m.ipa.txt"
DEFAULT_OUTPUT_DEBUG = PART_ROOT / "ipa/whisper_large_v3_oracle_spans_clean_lexmodi_10m.ipa.debug.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert a transcript .txt into a unified Hinglish IPA string.")
    p.add_argument("--input-txt", type=Path, default=DEFAULT_INPUT_TXT)
    p.add_argument("--output-ipa-txt", type=Path, default=DEFAULT_OUTPUT_IPA)
    p.add_argument("--output-debug-json", type=Path, default=DEFAULT_OUTPUT_DEBUG)
    p.add_argument("--lexicon", type=Path, default=DEFAULT_TECH_LEXICON)
    p.add_argument("--english-overrides", type=Path, default=DEFAULT_EN_OVERRIDES)
    p.add_argument("--roman-hindi-overrides", type=Path, default=DEFAULT_ROM_HI_OVERRIDES)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    text = args.input_txt.read_text(encoding="utf-8", errors="ignore").strip()

    lexicon = load_lexicon(args.lexicon) if args.lexicon.exists() else {}
    english_overrides = load_lexicon(args.english_overrides) if args.english_overrides.exists() else {}
    roman_hindi_overrides = load_lexicon(args.roman_hindi_overrides) if args.roman_hindi_overrides.exists() else {}

    items = convert_text_to_ipa(text, lexicon, english_overrides, roman_hindi_overrides)
    ipa = build_ipa_string(items)

    args.output_ipa_txt.parent.mkdir(parents=True, exist_ok=True)
    args.output_ipa_txt.write_text(ipa + "\n", encoding="utf-8")
    print(f"output_ipa_txt={args.output_ipa_txt}")

    if args.output_debug_json is not None:
        args.output_debug_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_debug_json.write_text(
            json.dumps([x.__dict__ for x in items], ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"output_debug_json={args.output_debug_json}")


if __name__ == "__main__":
    main()
