from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
import torchaudio
from transformers import LogitsProcessorList, WhisperForConditionalGeneration, WhisperProcessor

from transcribe_whisper_chunked import NgramLM, NgramLexiconBiasProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # assignment2-final/
PART12_ROOT = PROJECT_ROOT / "stage1_stt"

DEFAULT_AUDIO = PROJECT_ROOT / "source_clip_10min.wav"
DEFAULT_SPANS = PART12_ROOT / "reference/gemini_pro_spans_clean.csv"
DEFAULT_MODEL_ID = "openai/whisper-large-v3"
DEFAULT_LM_JSON = PART12_ROOT / "domain/ngram_lm_3gram.json"
DEFAULT_LEXICON_TXT = PART12_ROOT / "domain/custom_terms.txt"
DEFAULT_OUT_CSV = PART12_ROOT / "stt/whisper_large_v3_oracle_spans_lexmodi_10m.csv"
DEFAULT_OUT_HYP = PART12_ROOT / "stt/whisper_large_v3_oracle_spans_lexmodi_10m.hyp.txt"

LANG_TO_WHISPER = {"english": "en", "hindi": "hi"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Transcribe using oracle reference language spans.")
    p.add_argument("--audio", type=Path, default=DEFAULT_AUDIO)
    p.add_argument("--spans-csv", type=Path, default=DEFAULT_SPANS)
    p.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    p.add_argument("--lm-json", type=Path, default=DEFAULT_LM_JSON)
    p.add_argument("--lexicon-txt", type=Path, default=DEFAULT_LEXICON_TXT)
    p.add_argument("--disable-ngram", action="store_true")
    p.add_argument("--disable-lexicon", action="store_true")
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT_CSV)
    p.add_argument("--out-hyp-txt", type=Path, default=DEFAULT_OUT_HYP)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache/huggingface/hub")
    p.add_argument("--allow-remote", action="store_true")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--segment-pad-sec", type=float, default=0.10)
    p.add_argument("--min-segment-sec", type=float, default=0.35)
    p.add_argument("--max-segment-sec", type=float, default=20.0)
    p.add_argument("--merge-gap-sec", type=float, default=0.35, help="Merge adjacent same-language oracle spans when the silence/gap is below this.")
    p.add_argument("--min-turn-sec", type=float, default=1.2, help="Absorb tiny opposite-language islands shorter than this.")
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--beam-size", type=int, default=5)
    p.add_argument("--no-repeat-ngram-size", type=int, default=3)
    p.add_argument("--repetition-penalty", type=float, default=1.1)
    p.add_argument("--context-topk", type=int, default=10)
    p.add_argument("--lm-bias", type=float, default=2.0)
    p.add_argument("--term-bias", type=float, default=1.25)
    return p.parse_args()


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


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
    forced_decoder_ids: list[list[int]],
    logits_processor: LogitsProcessorList | None,
) -> str:
    inputs = processor.feature_extractor(audio_16k.numpy(), sampling_rate=16000, return_tensors="pt")
    kwargs = {
        "input_features": inputs.input_features.to(device),
        "max_new_tokens": max_new_tokens,
        "num_beams": beam_size,
        "do_sample": False,
        "no_repeat_ngram_size": no_repeat_ngram_size,
        "repetition_penalty": repetition_penalty,
        "forced_decoder_ids": forced_decoder_ids,
    }
    if logits_processor is not None:
        kwargs["logits_processor"] = logits_processor
    with torch.no_grad():
        pred_ids = model.generate(**kwargs)
    return processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0].strip()


def main() -> None:
    args = parse_args()
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_hyp_txt.parent.mkdir(parents=True, exist_ok=True)
    audio_16k = load_audio_mono(args.audio, args.sample_rate)
    total_sec = audio_16k.numel() / args.sample_rate

    rows = list(csv.DictReader(args.spans_csv.open("r", newline="", encoding="utf-8")))
    base_segments: list[dict[str, float | str]] = []
    for row in rows:
        lang = row["language"].strip().lower()
        if lang not in LANG_TO_WHISPER:
            continue
        start_sec = float(row["chunk_start_sec"]) + float(row["start_sec"])
        end_sec = float(row["chunk_start_sec"]) + float(row["end_sec"])
        base_segments.append({"lang": lang, "start_sec": start_sec, "end_sec": end_sec})

    base_segments.sort(key=lambda x: (float(x["start_sec"]), float(x["end_sec"])))
    merged_segments: list[dict[str, float | str]] = []
    for seg in base_segments:
        if not merged_segments:
            merged_segments.append(dict(seg))
            continue
        prev = merged_segments[-1]
        if (
            prev["lang"] == seg["lang"]
            and float(seg["start_sec"]) - float(prev["end_sec"]) <= args.merge_gap_sec
        ):
            prev["end_sec"] = max(float(prev["end_sec"]), float(seg["end_sec"]))
        else:
            merged_segments.append(dict(seg))

    changed = True
    while changed and len(merged_segments) >= 3:
        changed = False
        new_segments: list[dict[str, float | str]] = []
        i = 0
        while i < len(merged_segments):
            if 0 < i < len(merged_segments) - 1:
                prev = merged_segments[i - 1]
                cur = merged_segments[i]
                nxt = merged_segments[i + 1]
                cur_dur = float(cur["end_sec"]) - float(cur["start_sec"])
                if prev["lang"] == nxt["lang"] and cur_dur < args.min_turn_sec:
                    if new_segments:
                        new_segments.pop()
                    new_segments.append(
                        {
                            "lang": prev["lang"],
                            "start_sec": float(prev["start_sec"]),
                            "end_sec": float(nxt["end_sec"]),
                        }
                    )
                    i += 2
                    changed = True
                    continue
            new_segments.append(dict(merged_segments[i]))
            i += 1
        merged_segments = new_segments

    segments: list[dict[str, float | str]] = []
    for seg in merged_segments:
        start_sec = float(seg["start_sec"])
        end_sec = float(seg["end_sec"])
        cur = start_sec
        while cur < end_sec - 1e-6:
            nxt = min(end_sec, cur + args.max_segment_sec)
            if (nxt - cur) >= args.min_segment_sec:
                segments.append({"lang": seg["lang"], "start_sec": cur, "end_sec": nxt})
            cur = nxt

    local_files_only = not args.allow_remote
    processor = WhisperProcessor.from_pretrained(args.model_id, cache_dir=str(args.cache_dir), local_files_only=local_files_only)
    model = WhisperForConditionalGeneration.from_pretrained(args.model_id, cache_dir=str(args.cache_dir), local_files_only=local_files_only)
    device = torch.device(args.device)
    model.to(device)
    model.eval()

    lm = None
    if not args.disable_ngram and args.lm_json.exists():
        lm = NgramLM.from_json(args.lm_json)

    lexicon_terms: set[str] = set()
    if not args.disable_lexicon and args.lexicon_txt.exists():
        lexicon_terms = {
            x.strip().lower()
            for x in args.lexicon_txt.read_text(encoding="utf-8", errors="ignore").splitlines()
            if x.strip()
        }

    bias_list = None
    if lm is not None or lexicon_terms:
        bias_list = LogitsProcessorList(
            [
                NgramLexiconBiasProcessor(
                    tokenizer=processor.tokenizer,
                    lm=lm,
                    lexicon_terms=lexicon_terms,
                    lm_bias=args.lm_bias if lm is not None else 0.0,
                    term_bias=args.term_bias if lexicon_terms else 0.0,
                    context_topk=args.context_topk,
                )
            ]
        )

    fieldnames = ["segment_index", "lang", "start_sec", "end_sec", "duration_sec", "final_text"]
    final_lines: list[str] = []
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, seg in enumerate(segments):
            start_sec = float(seg["start_sec"])
            end_sec = float(seg["end_sec"])
            audio_start = max(0.0, start_sec - args.segment_pad_sec)
            audio_end = min(total_sec, end_sec + args.segment_pad_sec)
            s0 = int(round(audio_start * args.sample_rate))
            s1 = int(round(audio_end * args.sample_rate))
            chunk = audio_16k[s0:s1]
            forced_decoder_ids = processor.get_decoder_prompt_ids(language=LANG_TO_WHISPER[str(seg["lang"])], task="transcribe")
            text = generate_text(
                model=model,
                processor=processor,
                audio_16k=chunk,
                device=device,
                max_new_tokens=args.max_new_tokens,
                beam_size=args.beam_size,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                repetition_penalty=args.repetition_penalty,
                forced_decoder_ids=forced_decoder_ids,
                logits_processor=bias_list,
            )
            writer.writerow(
                {
                    "segment_index": idx,
                    "lang": seg["lang"],
                    "start_sec": f"{start_sec:.3f}",
                    "end_sec": f"{end_sec:.3f}",
                    "duration_sec": f"{(end_sec-start_sec):.3f}",
                    "final_text": text,
                }
            )
            if text.strip():
                final_lines.append(text.strip())
            print(f"segment={idx+1}/{len(segments)} lang={seg['lang']} start={start_sec:.2f} end={end_sec:.2f}", flush=True)
    args.out_hyp_txt.write_text("\n".join(final_lines).strip() + "\n", encoding="utf-8")
    print(f"out_csv={args.out_csv}")
    print(f"out_hyp_txt={args.out_hyp_txt}")


if __name__ == "__main__":
    main()
