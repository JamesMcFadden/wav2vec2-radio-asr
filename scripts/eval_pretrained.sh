#!/usr/bin/env bash
# Reproduce the WER numbers in the README.
#
#   ./scripts/eval_pretrained.sh              # smoke corpus, seconds
#   ./scripts/eval_pretrained.sh --real       # real test-clean, ~350 MB download
set -euo pipefail

cd "$(dirname "$0")/.."

MODEL="${MODEL:-facebook/wav2vec2-base-960h}"
BATCH_SIZE="${BATCH_SIZE:-8}"

if [[ "${1:-}" == "--real" ]]; then
    shift
    DATA_ARGS=(--config clean --split test)
    echo "Evaluating on real LibriSpeech test-clean (published reference: 3.4% WER)"
else
    DATA_ARGS=(--dummy)
    echo "Evaluating on the 73-utterance smoke corpus (expect ~5.6% WER)"
    echo "Pass --real for the full test-clean split."
fi

exec uv run python -m asr.evaluate \
    --model "$MODEL" \
    --batch-size "$BATCH_SIZE" \
    "${DATA_ARGS[@]}" \
    "$@"
