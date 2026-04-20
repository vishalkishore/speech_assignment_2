from __future__ import annotations

import argparse
import csv
from pathlib import Path

import soundfile as sf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build bona fide/spoof window manifests for Part 4.1.")
    p.add_argument("--bona-wav", type=Path, required=True)
    p.add_argument("--spoof-wav", type=Path, required=True)
    p.add_argument("--out-csv", type=Path, required=True)
    p.add_argument("--window-sec", type=float, default=2.0)
    p.add_argument("--stride-sec", type=float, default=2.0)
    return p.parse_args()


def build_rows(path: Path, label_name: str, label: int, split_plan: tuple[float, float, float], window_sec: float, stride_sec: float) -> list[dict[str, str]]:
    info = sf.info(str(path))
    dur = info.frames / float(info.samplerate)
    starts = []
    t = 0.0
    while t + window_sec <= dur + 1e-6:
        starts.append(round(t, 3))
        t += stride_sec
    n = len(starts)
    n_train = int(n * split_plan[0])
    n_valid = int(n * split_plan[1])
    rows = []
    for idx, s in enumerate(starts):
        if idx < n_train:
            split = "train"
        elif idx < n_train + n_valid:
            split = "valid"
        else:
            split = "test"
        rows.append(
            {
                "sample_id": f"{label_name}_{idx:04d}",
                "audio_path": str(path),
                "label_name": label_name,
                "label": str(label),
                "split": split,
                "start_sec": f"{s:.3f}",
                "end_sec": f"{s + window_sec:.3f}",
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    rows = []
    rows.extend(build_rows(args.bona_wav, "bona_fide", 0, (0.6, 0.2, 0.2), args.window_sec, args.stride_sec))
    rows.extend(build_rows(args.spoof_wav, "spoof", 1, (0.6, 0.2, 0.2), args.window_sec, args.stride_sec))
    # balance by downsampling spoof to bona fide count per split
    by_split = {"train": [], "valid": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(row)
    balanced = []
    for split, split_rows in by_split.items():
        bona = [r for r in split_rows if r["label"] == "0"]
        spoof = [r for r in split_rows if r["label"] == "1"][: len(bona)]
        balanced.extend(bona + spoof)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "audio_path", "label_name", "label", "split", "start_sec", "end_sec"],
        )
        writer.writeheader()
        writer.writerows(sorted(balanced, key=lambda r: (r["split"], r["label"], r["sample_id"])))
    print(f"rows={len(balanced)}")
    print(f"out_csv={args.out_csv}")


if __name__ == "__main__":
    main()
