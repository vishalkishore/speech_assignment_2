from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torchaudio


PROJECT_ROOT = Path(__file__).resolve().parents[3]  # stage1_lid/
DEFAULT_UTTERANCES = PROJECT_ROOT / "manifests/utterances.csv"
DEFAULT_SPANS = PROJECT_ROOT / "manifests/gemini_lid_spans_original.csv"
DEFAULT_OUT = PROJECT_ROOT / "manifests/clip_pseudo_frame_supervision.csv"
DEFAULT_SPLIT_UTTERANCES = PROJECT_ROOT / "manifests/utterances_clean_split.csv"

LABEL_TO_ID = {"english": 0, "hindi": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build frame-level pseudo-label manifest from Gemini clip spans.")
    p.add_argument("--utterances-csv", type=Path, default=DEFAULT_UTTERANCES)
    p.add_argument("--spans-csv", type=Path, default=DEFAULT_SPANS)
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT)
    p.add_argument("--split-utterances-csv", type=Path, default=None)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--hop-ms", type=float, default=10.0)
    p.add_argument("--valid-every", type=int, default=5, help="Every Nth utterance goes to valid split.")
    p.add_argument("--confidence", type=float, default=0.95)
    return p.parse_args()


def load_utterances(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_spans(path: Path) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out.setdefault(row["utterance_id"], []).append(row)
    return out


def main() -> None:
    args = parse_args()
    hop_sec = args.hop_ms / 1000.0
    utterances = load_utterances(args.utterances_csv)
    spans_by_utt = load_spans(args.spans_csv)
    clean_splits: dict[str, str] = {}
    if args.split_utterances_csv is not None:
        for row in load_utterances(args.split_utterances_csv):
            clean_splits[row["utterance_id"]] = row.get("clean_split", "")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "audio_path",
        "utterance_id",
        "split",
        "source",
        "frame_index",
        "frame_start_sec",
        "frame_end_sec",
        "label_id",
        "label",
        "confidence",
    ]
    rows_out: list[dict[str, object]] = []
    stats = {"utterances": 0, "frames": 0, "labeled_frames": 0}

    for idx, utt in enumerate(utterances):
        utt_id = utt["utterance_id"]
        audio_path = utt["audio_path"]
        spans = spans_by_utt.get(utt_id, [])
        info = torchaudio.info(audio_path)
        num_frames = info.num_frames
        sr = info.sample_rate
        duration_sec = num_frames / float(sr)
        frame_count = int(duration_sec / hop_sec)
        if clean_splits:
            mapped = clean_splits.get(utt_id, "")
            if mapped == "heldout_test":
                continue
            if mapped == "adapt_valid":
                split = "valid"
            else:
                split = "train"
        else:
            split = "valid" if (idx % max(1, args.valid_every) == 0) else "train"
        labels = [-100] * frame_count
        conf = [0.0] * frame_count

        for span in spans:
            lang = span["language"].strip().lower()
            if lang not in LABEL_TO_ID:
                continue
            label_id = LABEL_TO_ID[lang]
            start = max(0, int(float(span["start_sec"]) / hop_sec))
            end = min(frame_count, int(float(span["end_sec"]) / hop_sec))
            if end <= start:
                end = min(frame_count, start + 1)
            for fi in range(start, end):
                labels[fi] = label_id
                conf[fi] = args.confidence

        for fi in range(frame_count):
            rows_out.append(
                {
                    "audio_path": audio_path,
                    "utterance_id": utt_id,
                    "split": split,
                    "source": utt["source"],
                    "frame_index": fi,
                    "frame_start_sec": f"{fi * hop_sec:.3f}",
                    "frame_end_sec": f"{(fi + 1) * hop_sec:.3f}",
                    "label_id": labels[fi],
                    "label": "" if labels[fi] == -100 else ("english" if labels[fi] == 0 else "hindi"),
                    "confidence": f"{conf[fi]:.4f}",
                }
            )
        stats["utterances"] += 1
        stats["frames"] += frame_count
        stats["labeled_frames"] += sum(1 for x in labels if x != -100)

    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows_out)

    print(f"out_csv={args.out_csv}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
