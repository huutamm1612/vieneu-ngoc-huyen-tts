#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${1:-ngoc-a10-v1}"
CONFIG="${2:-pipeline_3phase.yaml}"
MODAL_ENVIRONMENT="${MODAL_ENVIRONMENT:-main}"

cd "$PROJECT_ROOT"
echo "Stage 1: NeuCodec on Modal T4 (4 CPU cores, 6 GiB RAM)"
echo "Stage 2: all three training phases on A10 (4 CPU cores, 6 GiB RAM)"
modal run --detach --env "$MODAL_ENVIRONMENT" cloud/modal_app.py \
  --config "$CONFIG" \
  --run-name "$RUN_NAME"
