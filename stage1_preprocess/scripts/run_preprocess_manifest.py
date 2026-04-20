from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]  # assignment2-final/
DEFAULT_INPUT_MANIFEST = PROJECT_ROOT / "data/manifests/shared/utterances.csv"
DEFAULT_OUT_MANIFEST = PROJECT_ROOT / "stage1_preprocess/manifests/utterances_preprocessed.csv"
DEFAULT_OUT_AUDIO_DIR = PROJECT_ROOT / "stage1_preprocess/processed_audio"
DEFAULT_OUT_META_DIR = PROJECT_ROOT / "stage1_preprocess/meta"
DEFAULT_PREPROCESS_SCRIPT = PROJECT_ROOT / "stage1_preprocess/src/preprocess_audio.py"
DEFAULT_PYTHON = Path.home() / ".venv/bin/python"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch denoise+normalize utterances manifest (resumable).")
    p.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    p.add_argument("--output-manifest", type=Path, default=DEFAULT_OUT_MANIFEST)
    p.add_argument("--out-audio-dir", type=Path, default=DEFAULT_OUT_AUDIO_DIR)
    p.add_argument("--out-meta-dir", type=Path, default=DEFAULT_OUT_META_DIR)
    p.add_argument("--preprocess-script", type=Path, default=DEFAULT_PREPROCESS_SCRIPT)
    p.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    p.add_argument("--split", default=None, help="Optional split filter")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", type=int, default=50)
    return p.parse_args()


def run_one(python_bin: Path, preprocess_script: Path, in_audio: Path, out_audio: Path, out_meta: Path) -> None:
    cmd = [
        str(python_bin),
        "-u",
        str(preprocess_script),
        str(in_audio),
        str(out_audio),
        "--meta-json",
        str(out_meta),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"preprocess failed for {in_audio}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )


def main() -> None:
    args = parse_args()
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.out_audio_dir.mkdir(parents=True, exist_ok=True)
    args.out_meta_dir.mkdir(parents=True, exist_ok=True)

    with args.input_manifest.open("r", newline="") as f:
        in_rows = list(csv.DictReader(f))
    if args.split is not None:
        in_rows = [r for r in in_rows if r.get("split") == args.split]
    if args.limit is not None:
        in_rows = in_rows[: args.limit]

    out_fields = list(in_rows[0].keys()) + [
        "orig_audio_path",
        "preprocessed_audio_path",
        "preprocess_meta_path",
        "preprocess_status",
    ]

    out_rows = []
    start = time.time()
    for i, row in enumerate(in_rows, start=1):
        in_audio = Path(row["audio_path"])
        stem = row["utterance_id"]
        out_audio = args.out_audio_dir / f"{stem}.wav"
        out_meta = args.out_meta_dir / f"{stem}.json"

        status = "done"
        if args.overwrite or (not out_audio.exists()) or (not out_meta.exists()):
            try:
                run_one(args.python_bin, args.preprocess_script, in_audio, out_audio, out_meta)
            except Exception as exc:  # noqa: BLE001
                status = f"error:{exc}".replace("\n", " ")[:1000]

        new_row = dict(row)
        new_row["orig_audio_path"] = row["audio_path"]
        new_row["preprocessed_audio_path"] = str(out_audio)
        new_row["preprocess_meta_path"] = str(out_meta)
        new_row["preprocess_status"] = status
        new_row["audio_path"] = str(out_audio) if status == "done" else row["audio_path"]
        out_rows.append(new_row)

        if i % max(args.log_every, 1) == 0 or i == len(in_rows):
            elapsed = time.time() - start
            rate = i / max(elapsed, 1e-9)
            remain = (len(in_rows) - i) / max(rate, 1e-9)
            print(
                f"processed={i}/{len(in_rows)} elapsed={elapsed/60.0:.1f}m eta={remain/60.0:.1f}m",
                flush=True,
            )

    with args.output_manifest.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        w.writerows(out_rows)
    done = sum(1 for r in out_rows if r["preprocess_status"] == "done")
    print(f"done={done}/{len(out_rows)}")
    print(f"output_manifest={args.output_manifest}")

    if done != len(out_rows):
        print("Some files failed. Re-run with same command; completed files are skipped.", file=sys.stderr)


if __name__ == "__main__":
    main()
