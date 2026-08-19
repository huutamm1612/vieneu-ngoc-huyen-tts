#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: $0 MODEL INPUT_TXT REF_WAV REF_TEXT OUTPUT_WAV [NUM_GPUS]" >&2
  exit 2
fi

MODEL="$1"
INPUT_PATH="$2"
REFERENCE_AUDIO="$3"
REFERENCE_TEXT="$4"
OUTPUT_PATH="$5"
NUM_GPUS="${6:-1}"

python -m inference \
  --config configs/inference.yaml \
  --model "$MODEL" \
  --input "$INPUT_PATH" \
  --reference-audio "$REFERENCE_AUDIO" \
  --reference-text "$REFERENCE_TEXT" \
  --output "$OUTPUT_PATH" \
  --num-gpus "$NUM_GPUS"
