from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf
import torch
import torchaudio.functional as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stitch synthesized chunk wavs into a single lecture waveform.")
    p.add_argument("--manifest-csv", type=Path, required=True)
    p.add_argument("--output-wav", type=Path, required=True)
    p.add_argument("--target-sr", type=int, default=22050)
    p.add_argument("--gap-ms", type=float, default=120.0)
    p.add_argument("--match-target-durations", action="store_true")
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def stretch_to_duration(wav: torch.Tensor, base_sr: int, target_sec: float) -> torch.Tensor:
    target_samples = max(int(round(target_sec * base_sr)), 1)
    if wav.numel() == 0:
        return torch.zeros(target_samples, dtype=torch.float32)
    if wav.numel() == target_samples:
        return wav
    wav2 = F.resample(wav.unsqueeze(0), wav.numel(), target_samples).squeeze(0)
    return wav2


def main() -> None:
    args = parse_args()
    rows = load_rows(args.manifest_csv)
    if not rows:
        raise SystemExit("No rows found in manifest.")

    chunks = []
    base_sr = None
    matched = 0
    for row in rows:
        wav_path = Path(row["audio_path"])
        waveform, sr = sf.read(wav_path, dtype="float32")
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        wav = torch.from_numpy(waveform).to(torch.float32)
        if base_sr is None:
            base_sr = sr
        elif sr != base_sr:
            raise ValueError(f"Mismatched sample rates in chunk set: {sr} vs {base_sr}")
        if args.match_target_durations:
            try:
                target_sec = float(row.get("chunk_duration_sec", "") or 0.0)
            except ValueError:
                target_sec = 0.0
            if target_sec > 0.0:
                wav = stretch_to_duration(wav, int(base_sr), target_sec)
                matched += 1
        chunks.append(wav)

    gap_samples = int(base_sr * (args.gap_ms / 1000.0))
    gap = torch.zeros(gap_samples, dtype=torch.float32)
    stitched = []
    for idx, chunk in enumerate(chunks):
        stitched.append(chunk)
        if idx != len(chunks) - 1 and gap_samples > 0:
            stitched.append(gap)

    full = torch.cat(stitched, dim=0)
    peak = full.abs().max().item() if full.numel() else 0.0
    if peak > 0.98:
        full = full * (0.98 / peak)

    if int(base_sr) != int(args.target_sr):
        full = F.resample(full.unsqueeze(0), int(base_sr), int(args.target_sr)).squeeze(0)
        out_sr = int(args.target_sr)
    else:
        out_sr = int(base_sr)

    args.output_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output_wav, full.numpy(), out_sr)

    print(f"rows={len(rows)}")
    if args.match_target_durations:
        print(f"matched_target_durations={matched}")
    print(f"input_sr={base_sr}")
    print(f"output_sr={out_sr}")
    print(f"duration_sec={full.numel()/out_sr:.2f}")
    print(f"output_wav={args.output_wav}")


if __name__ == "__main__":
    main()
