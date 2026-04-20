#!/bin/sh
set -eu

VENV_PY="${VENV_PY:-$HOME/.venv/bin/python}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PART="$ROOT/stage4_spoof"

MANIFEST="$PART/manifests/lfcc_cm_manifest.csv"
OUTDIR="$PART/checkpoints/lfcc_cm_v1"

mkdir -p "$PART/manifests" "$PART/checkpoints"

env PYTHONUNBUFFERED=1 "$VENV_PY" -u "$PART/scripts/build_lfcc_cm_manifest.py" \
  --bona-wav "$ROOT/stage3_voice/student_voice_ref_60s.wav" \
  --spoof-wav "$ROOT/stage3_tts/final_audio/output_lrl_cloned_22050.wav" \
  --out-csv "$MANIFEST" \
  --window-sec 2.0 \
  --stride-sec 2.0

env PYTHONUNBUFFERED=1 PYTHONPATH="$PART/src" "$VENV_PY" -u "$PART/scripts/train_lfcc_cm.py" \
  --manifest-csv "$MANIFEST" \
  --outdir "$OUTDIR" \
  --epochs 20 \
  --batch-size 8 \
  --lr 1e-3 \
  --device "${DEVICE:-cpu}"
