from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torchaudio
from transformers import LogitsProcessor, LogitsProcessorList, WhisperForConditionalGeneration, WhisperProcessor


PROJECT_ROOT = Path("/home/vishal/assignment2-final")
PART_ROOT = PROJECT_ROOT / "stage1_stt"
# Keep filename for backwards-compat, but we now default to Whisper Large v3.
DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_UTTERANCES = PROJECT_ROOT / "data/manifests/shared/utterances.csv"
DEFAULT_LM_JSON = PART_ROOT / "domain/ngram_lm_3gram.json"
DEFAULT_LEXICON_TXT = PART_ROOT / "domain/custom_terms.txt"
DEFAULT_OUT_CSV = PART_ROOT / "stt/constrained_decode_outputs.csv"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+|[\u0900-\u097F]+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline + constrained decoding with Whisper + custom n-gram logit bias.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--utterances-manifest", type=Path, default=DEFAULT_UTTERANCES)
    parser.add_argument("--audio", type=Path, default=None, help="Optional: transcribe a single audio file instead of utterances.csv")
    parser.add_argument("--lm-json", type=Path, default=DEFAULT_LM_JSON)
    parser.add_argument("--lexicon-txt", type=Path, default=DEFAULT_LEXICON_TXT)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    parser.add_argument("--split", default=None, help="Optional split filter from utterances.csv")
    parser.add_argument("--source", default=None, help="Optional source filter from utterances.csv")
    parser.add_argument("--start-sec", type=float, default=None, help="Optional start time filter (inclusive).")
    parser.add_argument("--end-sec", type=float, default=None, help="Optional end time filter (exclusive overlap).")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-new-tokens", type=int, default=180)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--language-hint", default=None, help="Optional language hint for Whisper (e.g. en, hi).")
    parser.add_argument("--use-forced-decoder-ids", action="store_true", help="Use forced_decoder_ids instead of task=transcribe.")
    parser.add_argument("--context-topk", type=int, default=10)
    parser.add_argument("--lm-bias", type=float, default=2.0)
    parser.add_argument("--term-bias", type=float, default=1.0)
    parser.add_argument("--cache-dir", type=Path, default=Path("/home/vishal/.cache/huggingface/hub"))
    parser.add_argument("--allow-remote", action="store_true", help="Allow remote model fetch if cache is missing.")
    return parser.parse_args()


def load_utterances(
    path: Path,
    split: str | None,
    source: str | None,
    limit: int | None,
    start_sec: float | None,
    end_sec: float | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", newline="") as handle:
        for row in csv.DictReader(handle):
            if split and row["split"] != split:
                continue
            if source and row["source"] != source:
                continue
            if start_sec is not None or end_sec is not None:
                utt_start = float(row["start_sec"])
                utt_end = float(row["end_sec"])
                lo = start_sec if start_sec is not None else float("-inf")
                hi = end_sec if end_sec is not None else float("inf")
                if not (utt_end > lo and utt_start < hi):
                    continue
            rows.append(row)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def tokenize_words(text: str) -> list[str]:
    return [m.group(0).lower() for m in WORD_RE.finditer(text)]


@dataclass
class NgramLM:
    order: int
    log_probs: dict[str, float]

    @classmethod
    def from_json(cls, path: Path) -> "NgramLM":
        obj = json.loads(path.read_text(encoding="utf-8"))
        return cls(order=int(obj["order"]), log_probs=obj["log_probs"])

    def next_word_candidates(self, history_words: list[str], topk: int) -> list[tuple[str, float]]:
        if self.order <= 1:
            return []
        need = self.order - 1
        if len(history_words) < need:
            return []
        ctx = " ".join(history_words[-need:])
        prefix = ctx + " "
        out: list[tuple[str, float]] = []
        for key, lp in self.log_probs.items():
            if not key.startswith(prefix):
                continue
            next_word = key[len(prefix) :].split(" ")[0].strip()
            if not next_word:
                continue
            out.append((next_word, float(lp)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:topk]


class NgramLexiconBiasProcessor(LogitsProcessor):
    def __init__(
        self,
        tokenizer,
        lm: NgramLM,
        lexicon_terms: set[str],
        lm_bias: float,
        term_bias: float,
        context_topk: int,
    ) -> None:
        self.tokenizer = tokenizer
        self.lm = lm
        self.lexicon_terms = lexicon_terms
        self.lm_bias = lm_bias
        self.term_bias = term_bias
        self.context_topk = context_topk
        self.lexicon_token_ids = self._build_lexicon_token_ids()

    def _build_lexicon_token_ids(self) -> set[int]:
        ids: set[int] = set()
        for term in self.lexicon_terms:
            pieces = self.tokenizer.encode(" " + term, add_special_tokens=False)
            if pieces:
                ids.add(int(pieces[0]))
        return ids

    def _candidate_token_ids_from_lm(self, history_text: str) -> list[int]:
        words = tokenize_words(history_text)
        if not words:
            return []
        cands = self.lm.next_word_candidates(words, self.context_topk)
        token_ids: list[int] = []
        for word, _ in cands:
            pieces = self.tokenizer.encode(" " + word, add_special_tokens=False)
            if pieces:
                token_ids.append(int(pieces[0]))
        return token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch = input_ids.shape[0]
        for b in range(batch):
            history_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            lm_token_ids = self._candidate_token_ids_from_lm(history_text)
            if lm_token_ids:
                scores[b, lm_token_ids] += self.lm_bias
            if self.lexicon_token_ids:
                scores[b, list(self.lexicon_token_ids)] += self.term_bias
        return scores


def read_audio_16k(path: Path) -> torch.Tensor:
    waveform, sr = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    return waveform.squeeze(0)


def generate_text(
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_16k: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    beam_size: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    logits_processor: LogitsProcessorList | None,
    forced_decoder_ids: list[list[int]] | None,
) -> str:
    inputs = processor.feature_extractor(
        audio_16k.numpy(), sampling_rate=16000, return_tensors="pt"
    )
    input_features = inputs.input_features.to(device)

    gen_kwargs = {
        "input_features": input_features,
        "max_new_tokens": max_new_tokens,
        "num_beams": beam_size,
        "do_sample": False,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }
    if forced_decoder_ids is None:
        gen_kwargs["task"] = "transcribe"
    else:
        gen_kwargs["forced_decoder_ids"] = forced_decoder_ids
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor

    with torch.no_grad():
        pred_ids = model.generate(**gen_kwargs)
    text = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
    return text.strip()


def text_quality_score(text: str) -> float:
    tokens = tokenize_words(text)
    if not tokens:
        return -1e9
    uniq_ratio = len(set(tokens)) / max(1, len(tokens))
    trigram_repeat = 0
    for i in range(len(tokens) - 2):
        if tokens[i] == tokens[i + 1] == tokens[i + 2]:
            trigram_repeat += 1
    # Higher is better: reward diversity/coverage, penalize heavy repetition.
    return (uniq_ratio * 2.0) + (len(tokens) / 120.0) - (trigram_repeat * 0.25)


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)

    if args.audio is not None:
        rows = [
            {
                "utterance_id": args.audio.stem,
                "audio_path": str(args.audio),
                "source": args.audio.name,
                "split": args.split or "",
                "start_sec": args.start_sec if args.start_sec is not None else "",
                "end_sec": args.end_sec if args.end_sec is not None else "",
            }
        ]
    else:
        rows = load_utterances(
            args.utterances_manifest,
            args.split,
            args.source,
            args.limit,
            args.start_sec,
            args.end_sec,
        )
        if not rows:
            raise SystemExit("No utterances matched filters.")

    lm = NgramLM.from_json(args.lm_json)
    lexicon_terms = {
        ln.strip().lower()
        for ln in args.lexicon_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
        if ln.strip()
    }

    device = torch.device(args.device)
    local_files_only = not args.allow_remote
    processor = WhisperProcessor.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir),
        local_files_only=local_files_only,
    )
    model = WhisperForConditionalGeneration.from_pretrained(
        args.model_id,
        cache_dir=str(args.cache_dir),
        local_files_only=local_files_only,
    ).to(device)
    model.eval()

    bias_processor = NgramLexiconBiasProcessor(
        tokenizer=processor.tokenizer,
        lm=lm,
        lexicon_terms=lexicon_terms,
        lm_bias=args.lm_bias,
        term_bias=args.term_bias,
        context_topk=args.context_topk,
    )
    bias_list = LogitsProcessorList([bias_processor])
    forced_decoder_ids = None
    if args.use_forced_decoder_ids:
        forced_decoder_ids = processor.get_decoder_prompt_ids(
            language=args.language_hint, task="transcribe"
        )

    fieldnames = [
        "utterance_id",
        "audio_path",
        "source",
        "split",
        "start_sec",
        "end_sec",
        "baseline_text",
        "constrained_text",
        "final_text",
    ]
    with args.out_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for i, row in enumerate(rows, start=1):
            audio_path = Path(row["audio_path"])
            audio_16k = read_audio_16k(audio_path)
            baseline = generate_text(
                model=model,
                processor=processor,
                audio_16k=audio_16k,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                logits_processor=None,
                forced_decoder_ids=forced_decoder_ids,
            )
            constrained = generate_text(
                model=model,
                processor=processor,
                audio_16k=audio_16k,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                logits_processor=bias_list,
                forced_decoder_ids=forced_decoder_ids,
            )
            baseline_score = text_quality_score(baseline)
            constrained_score = text_quality_score(constrained)
            final_text = constrained if constrained_score >= baseline_score else baseline
            writer.writerow(
                {
                    "utterance_id": row["utterance_id"],
                    "audio_path": row["audio_path"],
                    "source": row["source"],
                    "split": row["split"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "baseline_text": baseline,
                    "constrained_text": constrained,
                    "final_text": final_text,
                }
            )
            handle.flush()
            print(f"processed={i}/{len(rows)} utterance_id={row['utterance_id']}")

    print(f"out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
