#!/bin/bash
# Offline DFlash training on a compressed hidden-state cache.
#
# Runs the offline pipeline end to end: data preparation, vLLM hidden-state
# extraction with the int4 backend (~4x smaller files than bf16), offline cache
# generation, then training. The int4 cache is unpacked transparently by the
# trainer. To simulate other schemes without regenerating the cache, keep a bf16
# cache (HS_BACKEND=file) and pass --cache-qdq-aux / --cache-qdq-teacher.
#
# Usage: Copy this script, modify the configuration variables below, then run:
#   bash examples/train/dflash_qwen3_8b_offline_int4_cache.sh
#
# Walkthrough: https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/

set -euo pipefail

# ============ Configuration ============
MODEL="Qwen/Qwen3-8B"
DATASET="hf:inference-optimization/speculators-ci-datasets:tutorial_regen"
OUTPUT_DIR="./output/dflash_qwen3_8b_int4_cache"
VLLM_PORT=8000
MAX_SAMPLES=5000
SEQ_LENGTH=8192
EPOCHS=5
LR=3e-4

# Hidden-state storage: "int4" packs on write and unpacks on read (~4x smaller);
# "fp8" halves the size; "file" keeps bf16.
HS_BACKEND="int4"
HS_PATH="$OUTPUT_DIR/hidden_states"

# Load-time simulation of a compressed cache, applied to the training split only.
# Leave empty to train on the cache as stored. Example for a bf16 cache:
  QDQ_ARGS="--cache-qdq-aux int4 --cache-qdq-teacher fp8 --cache-qdq-transform-variant hadamard"
# QDQ_ARGS=""

# DFlash-specific parameters
SPECULATOR_TYPE="dflash"
BLOCK_SIZE=8
MAX_ANCHORS=3072
NUM_LAYERS=5
DRAFT_VOCAB_SIZE=32000
TARGET_LAYER_IDS="2 18 33"  # Must match vLLM's eagle_aux_hidden_state_layer_ids

VLLM_GPUS="0"
TRAIN_GPUS="1"
NUM_TRAIN_GPUS=1
# =======================================

echo "=== Step 1: Preparing data ==="
speculators prepare-data \
    --model "$MODEL" \
    --data "$DATASET" \
    --output "$OUTPUT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --seq-length "$SEQ_LENGTH" \
    --no-skip-token-freq

echo "=== Step 2: Launching vLLM hidden-state server ($HS_BACKEND backend) ==="
mkdir -p "$HS_PATH"
CUDA_VISIBLE_DEVICES="$VLLM_GPUS" python scripts/launch_vllm.py "$MODEL" \
    --target-layer-ids $TARGET_LAYER_IDS \
    --hidden-states-backend "$HS_BACKEND" \
    --hidden-states-path "$HS_PATH" \
    --provenance-dir "$OUTPUT_DIR" \
    -- --port "$VLLM_PORT" &
VLLM_PID=$!

cleanup() {
    echo "Stopping vLLM server..."
    kill "$VLLM_PID" 2>/dev/null || true
    wait "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT

echo "Waiting for vLLM server to be ready..."
until curl -sf "http://localhost:${VLLM_PORT}/health" > /dev/null 2>&1; do
    sleep 2
done
echo "vLLM server ready."

echo "=== Step 3: Generating the hidden-state cache ==="
speculators generate-offline-data \
    --model "$MODEL" \
    --preprocessed-data "$OUTPUT_DIR" \
    --endpoint "http://localhost:${VLLM_PORT}/v1" \
    --output "$HS_PATH" \
    --max-samples "$MAX_SAMPLES" \
    --concurrency 32 \
    --validate-outputs

cleanup
trap - EXIT
du -sh "$HS_PATH"

echo "=== Step 4: Training ==="
CUDA_VISIBLE_DEVICES="$TRAIN_GPUS" torchrun \
    --standalone --nproc_per_node "$NUM_TRAIN_GPUS" \
    -m speculators.train \
    --verifier-name-or-path "$MODEL" \
    --data-path "$OUTPUT_DIR" \
    --hidden-states-backend "$HS_BACKEND" \
    --hidden-states-path "$HS_PATH" \
    --on-missing raise \
    --save-path "$OUTPUT_DIR/checkpoints" \
    --draft-vocab-size "$DRAFT_VOCAB_SIZE" \
    --epochs "$EPOCHS" \
    --lr "$LR" \
    --total-seq-len "$SEQ_LENGTH" \
    --speculator-type "$SPECULATOR_TYPE" \
    --block-size "$BLOCK_SIZE" \
    --max-anchors "$MAX_ANCHORS" \
    --num-layers "$NUM_LAYERS" \
    --target-layer-ids $TARGET_LAYER_IDS \
    $QDQ_ARGS

echo "Done. Checkpoints saved to $OUTPUT_DIR/checkpoints/"
