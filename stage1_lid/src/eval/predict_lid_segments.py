from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio

from src.models.lid_bilstm import FrameLIDBiLSTM


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # stage1_lid/
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/lid_clip_adapt_clean_v2_best_clip.pt"


ID_TO_LANG = {0: "english", 1: "hindi"}


@dataclass(frozen=True)
class Segment:
    lang_id: int
    start_frame: int
    end_frame: int  # exclusive

    def to_dict(self, hop_sec: float) -> dict[str, object]:
        return {
            "lang": ID_TO_LANG.get(self.lang_id, str(self.lang_id)),
            "start_sec": self.start_frame * hop_sec,
            "end_sec": self.end_frame * hop_sec,
            "duration_sec": (self.end_frame - self.start_frame) * hop_sec,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
        }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run LID on a single file and export smoothed segments + switches.")
    p.add_argument("--audio", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--out-json", type=Path, default=None)
    p.add_argument("--out-csv", type=Path, default=None)
    p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-fft", type=int, default=400)
    p.add_argument("--win-length", type=int, default=400)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--n-mels", type=int, default=80)
    p.add_argument("--smooth-window-frames", type=int, default=11, help="Odd window size for median smoothing.")
    p.add_argument("--min-seg-ms", type=float, default=120.0, help="Merge segments shorter than this.")
    return p.parse_args()


def load_audio_mono(path: Path, target_sr: int) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    return wav.squeeze(0)


def median_smooth_1d(x: torch.Tensor, window: int) -> torch.Tensor:
    if window <= 1:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    # Replicate padding then sliding median.
    x_pad = torch.cat([x[:1].repeat(pad), x, x[-1:].repeat(pad)], dim=0)
    cols = []
    for i in range(window):
        cols.append(x_pad[i : i + x.numel()])
    stacked = torch.stack(cols, dim=1)
    return stacked.median(dim=1).values.to(dtype=x.dtype)


def to_segments(labels: torch.Tensor) -> list[Segment]:
    if labels.numel() == 0:
        return []
    segs: list[Segment] = []
    cur = int(labels[0].item())
    start = 0
    for i in range(1, labels.numel()):
        v = int(labels[i].item())
        if v != cur:
            segs.append(Segment(cur, start, i))
            cur = v
            start = i
    segs.append(Segment(cur, start, labels.numel()))
    return segs


def merge_short_segments(segs: list[Segment], *, min_frames: int) -> list[Segment]:
    if not segs:
        return []
    out = [segs[0]]
    for s in segs[1:]:
        prev = out[-1]
        prev_len = prev.end_frame - prev.start_frame
        if prev_len < min_frames:
            # Merge prev into current (prefer extending current backwards).
            out[-1] = Segment(s.lang_id, prev.start_frame, s.end_frame)
            continue
        if (s.end_frame - s.start_frame) < min_frames and out:
            # Merge current into prev.
            out[-1] = Segment(prev.lang_id, prev.start_frame, s.end_frame)
            continue
        out.append(s)
    # One more pass: if final is short, merge into previous.
    if len(out) >= 2 and (out[-1].end_frame - out[-1].start_frame) < min_frames:
        last = out.pop()
        prev = out.pop()
        out.append(Segment(prev.lang_id, prev.start_frame, last.end_frame))
    # Collapse any accidental same-label adjacency.
    collapsed: list[Segment] = []
    for s in out:
        if collapsed and collapsed[-1].lang_id == s.lang_id:
            collapsed[-1] = Segment(s.lang_id, collapsed[-1].start_frame, s.end_frame)
        else:
            collapsed.append(s)
    return collapsed


def main() -> None:
    args = parse_args()
    hop_sec = args.hop_length / args.sample_rate
    min_frames = max(1, int(round((args.min_seg_ms / 1000.0) / hop_sec)))

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = FrameLIDBiLSTM()
    model.load_state_dict(ckpt["model_state"])
    model.to(torch.device(args.device))
    model.eval()

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        win_length=args.win_length,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
        center=False,
        power=2.0,
    )

    wav = load_audio_mono(args.audio, args.sample_rate)
    feats = mel(wav.unsqueeze(0)).squeeze(0).transpose(0, 1)
    feats = torch.log(feats.clamp_min(1e-5))

    with torch.no_grad():
        logits = model(feats.unsqueeze(0).to(torch.device(args.device))).squeeze(0).cpu()
        raw = logits.argmax(dim=-1).to(dtype=torch.int64)

    smooth = median_smooth_1d(raw, args.smooth_window_frames)
    segs = to_segments(smooth)
    segs = merge_short_segments(segs, min_frames=min_frames)

    switches = []
    for a, b in zip(segs, segs[1:]):
        switches.append(
            {
                "from": ID_TO_LANG.get(a.lang_id, str(a.lang_id)),
                "to": ID_TO_LANG.get(b.lang_id, str(b.lang_id)),
                "at_sec": b.start_frame * hop_sec,
                "at_frame": b.start_frame,
            }
        )

    total = int(smooth.numel())
    english = int((smooth == 0).sum().item())
    hindi = int((smooth == 1).sum().item())
    result = {
        "audio": str(args.audio),
        "frames": total,
        "hop_sec": hop_sec,
        "smooth_window_frames": args.smooth_window_frames,
        "min_seg_ms": args.min_seg_ms,
        "english_ratio": english / total if total else 0.0,
        "hindi_ratio": hindi / total if total else 0.0,
        "segments": [s.to_dict(hop_sec) for s in segs],
        "switches": switches,
    }

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=["lang", "start_sec", "end_sec", "duration_sec", "start_frame", "end_frame"],
            )
            w.writeheader()
            for s in segs:
                w.writerow(s.to_dict(hop_sec))

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
