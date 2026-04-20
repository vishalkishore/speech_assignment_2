from __future__ import annotations

import argparse
from pathlib import Path

from prosody import extract_prosody


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract F0 and energy contours from an audio file.")
    p.add_argument("input_audio", type=Path)
    p.add_argument("--out-npz", type=Path, required=True)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--fmin", type=float, default=75.0)
    p.add_argument("--fmax", type=float, default=400.0)
    p.add_argument("--pitch-method", choices=["yin", "pyin"], default="yin")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    feats = extract_prosody(
        audio_path=args.input_audio,
        sample_rate=args.sample_rate,
        hop_length=args.hop_length,
        win_length=args.win_length,
        fmin=args.fmin,
        fmax=args.fmax,
        pitch_method=args.pitch_method,
    )
    feats.to_npz(args.out_npz)
    feats.to_json_summary(args.out_json)
    print(f"out_npz={args.out_npz}")
    print(f"out_json={args.out_json}")
    print(f"num_frames={len(feats.times_sec)}")


if __name__ == "__main__":
    main()
