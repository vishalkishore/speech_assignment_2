from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_PROJECT = Path(__file__).resolve().parents[2]  # assignment2-final/
DEFAULT_MANIFEST = DEFAULT_PROJECT / "stage3_voice/manifests/voice_reference_manifest.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Register the student's 60s voice reference and extracted embedding.")
    parser.add_argument("--audio-path", type=Path, required=True)
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--meta-json", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--speaker-id", default="student_voice_ref")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    meta = json.loads(args.meta_json.read_text(encoding="utf-8"))
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if args.manifest.exists():
        with args.manifest.open("r", newline="") as f:
            rows = list(csv.DictReader(f))

    new_row = {
        "speaker_id": args.speaker_id,
        "audio_path": str(args.audio_path),
        "embedding_path": str(args.embedding_path),
        "meta_json": str(args.meta_json),
        "duration_sec": f"{meta.get('duration_sec', 0.0):.2f}",
        "embedding_dim": str(meta.get("embedding_dim", "")),
        "backend": str(meta.get("backend", "")),
    }

    rows = [r for r in rows if r.get("speaker_id") != args.speaker_id]
    rows.append(new_row)

    with args.manifest.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "speaker_id",
                "audio_path",
                "embedding_path",
                "meta_json",
                "duration_sec",
                "embedding_dim",
                "backend",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"manifest={args.manifest}")
    print(f"speaker_id={args.speaker_id}")


if __name__ == "__main__":
    main()
