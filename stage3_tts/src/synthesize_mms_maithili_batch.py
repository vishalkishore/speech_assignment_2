from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
from transformers import AutoTokenizer, VitsModel


DEFAULT_MODEL = "facebook/mms-tts-mai"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch synthesize Maithili chunks with MMS TTS.")
    p.add_argument("--input-csv", type=Path, required=True)
    p.add_argument("--text-column", default="tts_text_maithili")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--log-every", type=int, default=10)
    return p.parse_args()


def synthesize_text(model, tokenizer, text: str, device: str) -> tuple[torch.Tensor, int]:
    inputs = tokenizer(text=text, return_tensors="pt")
    moved_inputs = {}
    for k, v in inputs.items():
        if k in {"input_ids", "attention_mask"}:
            moved_inputs[k] = v.to(device=device, dtype=torch.long)
        else:
            moved_inputs[k] = v.to(device=device)
    with torch.no_grad():
        output = model(**moved_inputs).waveform
    waveform = output.squeeze(0).detach().cpu()
    sr = int(model.config.sampling_rate)
    return waveform, sr


def write_manifest(path: Path, rows_out: list[dict[str, object]]) -> None:
    if not rows_out:
        return
    fieldnames = list(rows_out[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows_out)


def format_minutes(seconds: float) -> str:
    return f"{seconds / 60.0:.1f}m"


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.manifest_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    if args.split is not None:
        df = df[df["split"] == args.split]
    if args.limit is not None:
        df = df.head(args.limit)

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = VitsModel.from_pretrained(args.model_id).to(args.device)
    model.eval()

    rows_out = []
    total = len(df)
    start_time = time.time()
    reused = 0
    for idx, row in enumerate(df.itertuples(index=False), start=1):
        text = str(getattr(row, args.text_column, "")).strip()
        if not text:
            continue
        utt_id = str(row.utterance_id)
        out_path = args.output_dir / f"{utt_id}.wav"
        if args.skip_existing and out_path.exists():
            info = sf.info(out_path)
            rows_out.append(
                {
                    "utterance_id": utt_id,
                    "source": str(row.source),
                    "split": str(row.split),
                    "tts_text_maithili": text,
                    "chunk_duration_sec": float(getattr(row, "chunk_duration_sec", 0.0) or 0.0),
                    "audio_path": str(out_path),
                    "sample_rate": int(info.samplerate),
                    "num_samples": int(info.frames),
                }
            )
            reused += 1
            continue

        try:
            wav, sr = synthesize_text(model, tokenizer, text, args.device)
            sf.write(out_path, wav.numpy(), sr)
            rows_out.append(
                {
                    "utterance_id": utt_id,
                    "source": str(row.source),
                    "split": str(row.split),
                    "tts_text_maithili": text,
                    "chunk_duration_sec": float(getattr(row, "chunk_duration_sec", 0.0) or 0.0),
                    "audio_path": str(out_path),
                    "sample_rate": sr,
                    "num_samples": int(wav.numel()),
                }
            )
        except Exception as exc:  # noqa: BLE001
            print(f"error_utterance_id={utt_id} error={exc}", flush=True)
            if torch.cuda.is_available() and str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
            continue
        if idx % args.log_every == 0 or idx == total:
            elapsed = time.time() - start_time
            rate = idx / elapsed if elapsed > 0 else 0.0
            remaining = max(total - idx, 0)
            eta_seconds = remaining / rate if rate > 0 else 0.0
            write_manifest(args.manifest_csv, rows_out)
            print(
                f"processed={idx}/{total} reused={reused} "
                f"elapsed={format_minutes(elapsed)} eta={format_minutes(eta_seconds)} "
                f"last_utterance_id={utt_id}",
                flush=True,
            )

    write_manifest(args.manifest_csv, rows_out)

    print(f"rows={len(rows_out)}")
    print(f"manifest_csv={args.manifest_csv}")


if __name__ == "__main__":
    main()
