from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from transformers import LogitsProcessor, LogitsProcessorList, WhisperForConditionalGeneration, WhisperProcessor


PROJECT_ROOT = Path("/home/vishal/assignment2-final")
PART_ROOT = PROJECT_ROOT / "stage1_stt"

DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_LM_JSON = PART_ROOT / "domain/ngram_lm_3gram.json"
DEFAULT_LEXICON_TXT = PART_ROOT / "domain/custom_terms.txt"

WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-']+|[\u0900-\u097F]+")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chunked Whisper transcription with custom n-gram / lexicon logit bias.")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--lm-json", type=Path, default=DEFAULT_LM_JSON)
    p.add_argument("--lexicon-txt", type=Path, default=DEFAULT_LEXICON_TXT)
    p.add_argument(
        "--disable-ngram",
        action="store_true",
        help="Disable n-gram next-word biasing (lexicon bias can still be used).",
    )
    p.add_argument(
        "--disable-lexicon",
        action="store_true",
        help="Disable lexicon term biasing (n-gram bias can still be used).",
    )
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--out-hyp-txt", type=Path, required=True)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", type=Path, default=Path("/home/vishal/.cache/huggingface/hub"))
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--chunk-sec", type=float, default=30.0)
    p.add_argument("--stride-sec", type=float, default=0.0, help="Overlap between chunks (seconds).")
    p.add_argument("--max-new-tokens", type=int, default=224)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--context-topk", type=int, default=10)
    p.add_argument("--lm-bias", type=float, default=2.0)
    p.add_argument("--term-bias", type=float, default=1.0)
    return p.parse_args()


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
            if next_word:
                out.append((next_word, float(lp)))
        out.sort(key=lambda x: x[1], reverse=True)
        return out[:topk]


class NgramLexiconBiasProcessor(LogitsProcessor):
    def __init__(
        self,
        *,
        tokenizer,
        lm: NgramLM | None,
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
        if self.lm is None:
            return []
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
        for b in range(input_ids.shape[0]):
            history_text = self.tokenizer.decode(input_ids[b], skip_special_tokens=True)
            lm_token_ids = self._candidate_token_ids_from_lm(history_text)
            if lm_token_ids:
                scores[b, lm_token_ids] += self.lm_bias
            if self.lexicon_token_ids:
                scores[b, list(self.lexicon_token_ids)] += self.term_bias
        return scores


def read_audio_16k(path: Path) -> tuple[torch.Tensor, int]:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != 16000:
        wav = torchaudio.functional.resample(wav, sr, 16000)
        sr = 16000
    return wav.squeeze(0), sr


def generate_text(
    *,
    model: WhisperForConditionalGeneration,
    processor: WhisperProcessor,
    audio_16k: torch.Tensor,
    device: torch.device,
    max_new_tokens: int,
    beam_size: int,
    no_repeat_ngram_size: int,
    repetition_penalty: float,
    logits_processor: LogitsProcessorList | None,
) -> str:
    inputs = processor.feature_extractor(audio_16k.numpy(), sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device)
    gen_kwargs = {
        "input_features": input_features,
        "task": "transcribe",
        "max_new_tokens": max_new_tokens,
        "num_beams": beam_size,
        "do_sample": False,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
    }
    if logits_processor is not None:
        gen_kwargs["logits_processor"] = logits_processor
    with torch.no_grad():
        pred_ids = model.generate(**gen_kwargs)
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_hyp_txt.parent.mkdir(parents=True, exist_ok=True)

    wav, sr = read_audio_16k(args.audio)
    total_sec = wav.numel() / sr

    local_files_only = not args.allow_remote
    processor = WhisperProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir), local_files_only=local_files_only)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id, cache_dir=str(args.cache_dir), local_files_only=local_files_only)
    model.to(torch.device(args.device))
    model.eval()

    lm = None
    if not args.disable_ngram:
        lm = NgramLM.from_json(args.lm_json)

    lexicon_terms: set[str] = set()
    if not args.disable_lexicon:
        lexicon_terms = {
            ln.strip().lower()
            for ln in args.lexicon_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if ln.strip()
        }

    bias_list = None
    if lm is not None or lexicon_terms:
        bias_proc = NgramLexiconBiasProcessor(
            tokenizer=processor.tokenizer,
            lm=lm,
            lexicon_terms=lexicon_terms,
            lm_bias=args.lm_bias if lm is not None else 0.0,
            term_bias=args.term_bias if lexicon_terms else 0.0,
            context_topk=args.context_topk,
        )
        bias_list = LogitsProcessorList([bias_proc])

    stride = max(0.0, float(args.stride_sec))
    chunk = max(1e-3, float(args.chunk_sec))
    step = max(1e-3, chunk - stride)

    fieldnames = [
        "chunk_index",
        "start_sec",
        "end_sec",
        "baseline_text",
        "constrained_text",
        "final_text",
    ]

    final_lines: list[str] = []
    with args.out_csv.open("w", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=fieldnames)
        w.writeheader()
        idx = 0
        t = 0.0
        device = torch.device(args.device)
        while t < total_sec - 1e-6:
            start = t
            end = min(total_sec, t + chunk)
            s0 = int(round(start * sr))
            s1 = int(round(end * sr))
            audio_chunk = wav[s0:s1]

            baseline = generate_text(
                model=model,
                processor=processor,
                audio_16k=audio_chunk,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                logits_processor=None,
            )
            constrained = generate_text(
                model=model,
                processor=processor,
                audio_16k=audio_chunk,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                logits_processor=bias_list,
            )
            final_text = constrained if len(constrained) >= len(baseline) else baseline

            w.writerow(
                {
                    "chunk_index": idx,
                    "start_sec": f"{start:.3f}",
                    "end_sec": f"{end:.3f}",
                    "baseline_text": baseline,
                    "constrained_text": constrained,
                    "final_text": final_text,
                }
            )
            handle.flush()
            if final_text:
                final_lines.append(final_text)

            idx += 1
            t += step

    args.out_hyp_txt.write_text("\n".join(final_lines).strip() + "\n", encoding="utf-8")
    print(f"out_csv={args.out_csv}")
    print(f"out_hyp_txt={args.out_hyp_txt}")


if __name__ == "__main__":
    main()
