#!/bin/sh
set -eu

ROOT="/home/vishal/assignment2-final"
PYTHON_BIN="${PYTHON_BIN:-/home/vishal/.venv/bin/python}"

ORIGINAL_URL="${ORIGINAL_URL:-https://www.youtube.com/watch?v=xyLNpi35iwA}"
START_SEC="${START_SEC:-8400}"
DURATION_SEC="${DURATION_SEC:-600}"
OUT_WAV="${OUT_WAV:-$ROOT/stage1_lid/audio/source_clip_10min.wav}"
TEMP_DIR="${TEMP_DIR:-/tmp/lexmodi_original_download}"

# Examples:
#   COOKIES_FROM_BROWSER=chrome sh download_lexmodi_original_clip.sh
#   COOKIES_FILE=/path/to/cookies.txt sh download_lexmodi_original_clip.sh

ARGS=""
if [ -n "${COOKIES_FROM_BROWSER:-}" ]; then
  ARGS="$ARGS --cookies-from-browser ${COOKIES_FROM_BROWSER}"
fi
if [ -n "${COOKIES_FILE:-}" ]; then
  ARGS="$ARGS --cookies-file ${COOKIES_FILE}"
fi

# shellcheck disable=SC2086
exec "$PYTHON_BIN" "$ROOT/scripts/download_youtube_clip.py" \
  --url "$ORIGINAL_URL" \
  --start-sec "$START_SEC" \
  --duration-sec "$DURATION_SEC" \
  --out-wav "$OUT_WAV" \
  --temp-dir "$TEMP_DIR" \
  $ARGS
