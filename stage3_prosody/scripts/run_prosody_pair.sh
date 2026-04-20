#!/bin/sh
set -eu

if [ $# -lt 2 ]; then
  echo "usage: $0 /abs/path/reference_professor.wav /abs/path/query_synth.wav"
  exit 1
fi

REF_AUDIO="$1"
QRY_AUDIO="$2"
PYTHON_BIN="${PYTHON_BIN:-$HOME/.venv/bin/python}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

REF_STEM="$(basename "$REF_AUDIO")"
REF_STEM="${REF_STEM%.*}"
QRY_STEM="$(basename "$QRY_AUDIO")"
QRY_STEM="${QRY_STEM%.*}"
PAIR_TAG="${REF_STEM}__${QRY_STEM}"

REF_NPZ="$ROOT/prosody/${REF_STEM}.npz"
REF_JSON="$ROOT/prosody/${REF_STEM}.json"
QRY_NPZ="$ROOT/prosody/${QRY_STEM}.npz"
QRY_JSON="$ROOT/prosody/${QRY_STEM}.json"
DTW_NPZ="$ROOT/dtw/${PAIR_TAG}.npz"
DTW_JSON="$ROOT/dtw/${PAIR_TAG}.json"
OUT_WAV="$ROOT/warped/${PAIR_TAG}.wav"
OUT_JSON="$ROOT/warped/${PAIR_TAG}.json"

"$PYTHON_BIN" -u "$ROOT/src/extract_prosody.py" "$REF_AUDIO" --out-npz "$REF_NPZ" --out-json "$REF_JSON"
"$PYTHON_BIN" -u "$ROOT/src/extract_prosody.py" "$QRY_AUDIO" --out-npz "$QRY_NPZ" --out-json "$QRY_JSON"
"$PYTHON_BIN" -u "$ROOT/src/dtw_align_prosody.py" --reference-npz "$REF_NPZ" --query-npz "$QRY_NPZ" --out-npz "$DTW_NPZ" --out-json "$DTW_JSON"
"$PYTHON_BIN" -u "$ROOT/src/warp_prosody.py" --reference-npz "$REF_NPZ" --query-npz "$QRY_NPZ" --dtw-npz "$DTW_NPZ" --query-audio "$QRY_AUDIO" --out-audio "$OUT_WAV" --out-json "$OUT_JSON"

echo "reference_npz=$REF_NPZ"
echo "query_npz=$QRY_NPZ"
echo "dtw_npz=$DTW_NPZ"
echo "warped_audio=$OUT_WAV"
