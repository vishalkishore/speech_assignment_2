from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9']+|[\u0900-\u097F]+")


def normalize(text: str) -> list[str]:
    text = text.lower()
    return [m.group(0) for m in WORD_RE.finditer(text)]


def wer(ref_tokens: list[str], hyp_tokens: list[str]) -> float:
    # Levenshtein distance / |ref|
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0:
        return 0.0 if m == 0 else 1.0
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[m] / n


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute WER between reference text and hypothesis text.")
    p.add_argument("--ref", type=Path, required=True, help="Reference transcript .txt")
    p.add_argument("--hyp", type=Path, required=True, help="Hypothesis transcript .txt")
    p.add_argument("--out-json", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ref_text = args.ref.read_text(encoding="utf-8", errors="ignore")
    hyp_text = args.hyp.read_text(encoding="utf-8", errors="ignore")
    ref_toks = normalize(ref_text)
    hyp_toks = normalize(hyp_text)
    out = {
        "ref_tokens": len(ref_toks),
        "hyp_tokens": len(hyp_toks),
        "wer": wer(ref_toks, hyp_toks),
        "ref_path": str(args.ref),
        "hyp_path": str(args.hyp),
    }
    print(json.dumps(out, indent=2))
    if args.out_json is not None:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

