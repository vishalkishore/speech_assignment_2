#!/bin/sh
set -eu

VENV_PY="${VENV_PY:-$HOME/.venv/bin/python}"
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PART="$ROOT/stage3_tts"

FLUENT_CSV="$PART/manifests/lexmodi_maithili_chunk_translation_fluent.csv"
TTS_CSV="$PART/manifests/lexmodi_maithili_tts_duration_aware_fluent.csv"
CHUNK_DIR="$PART/synth_chunks/lexmodi_mms_fluent"
MANIFEST_CSV="$PART/manifests/lexmodi_mms_fluent_manifest.csv"
OUTPUT_WAV="$PART/final_audio/output_lrl_cloned_22050.wav"

mkdir -p "$PART/final_audio" "$CHUNK_DIR" "$PART/manifests"

if [ ! -f "$FLUENT_CSV" ]; then
  echo "Missing fluent translation manifest: $FLUENT_CSV" >&2
  echo "This cleaned repo keeps the finalized fluent Maithili translations, not the Gemini generation step." >&2
  exit 1
fi

env PYTHONUNBUFFERED=1 "$VENV_PY" -u "$PART/src/prepare_duration_aware_tts_input.py" \
  --translation-csv "$FLUENT_CSV" \
  --utterances-csv "$ROOT/stage1_lid/manifests/utterances.csv" \
  --output-csv "$TTS_CSV" \
  --text-column target_maithili_fluent \
  --source-text-column source_text

DEVICE="${DEVICE:-cpu}"
env PYTHONUNBUFFERED=1 "$VENV_PY" -u "$PART/src/synthesize_mms_maithili_batch.py" \
  --input-csv "$TTS_CSV" \
  --output-dir "$CHUNK_DIR" \
  --manifest-csv "$MANIFEST_CSV" \
  --device "$DEVICE" \
  --skip-existing

env PYTHONUNBUFFERED=1 "$VENV_PY" -u "$PART/src/stitch_synth_chunks.py" \
  --manifest-csv "$MANIFEST_CSV" \
  --output-wav "$OUTPUT_WAV" \
  --target-sr 22050 \
  --gap-ms 0 \
  --match-target-durations
