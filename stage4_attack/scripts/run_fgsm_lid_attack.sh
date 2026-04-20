#!/bin/sh
set -eu

ROOT=/home/vishal/assignment2-final
PYTHON=/home/vishal/.venv/bin/python

PYTHONPATH="$ROOT/stage4_attack:$ROOT/stage1_lid" \
  "$PYTHON" "$ROOT/stage4_attack/src/run_fgsm_lid_attack.py" \
  --device "${DEVICE:-cuda:0}" \
  --duration-sec "${DURATION_SEC:-5.0}" \
  --snr-min-db "${SNR_MIN_DB:-40.0}" \
  --eps-min "${EPS_MIN:-0.00005}" \
  --eps-max "${EPS_MAX:-0.0008}" \
  --eps-steps "${EPS_STEPS:-16}" \
  --attack-steps "${ATTACK_STEPS:-12}" \
  --save-prefix "${SAVE_PREFIX:-attack_eval}"
