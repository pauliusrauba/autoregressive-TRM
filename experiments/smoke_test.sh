#!/usr/bin/env bash
set -euo pipefail

# experiments/smoke_test.sh
# Quick smoke test to verify all models run without errors.
# Total runtime: ~5 minutes
#
# Usage: ./experiments/smoke_test.sh
# Or with specific GPU: GPU=1 ./experiments/smoke_test.sh

# =============================================================================
# Configuration - Minimal settings for fast execution
# =============================================================================

# Use GPU 0 by default, can override with GPU=X environment variable
GPU="${GPU:-0}"

# Minimal model architecture (to run fast)
N_HEAD=2
N_EMBD=64
N_LAYER=2
BLOCK_SIZE=64
DROPOUT=0.1

# Minimal training settings
MAX_STEPS=20
BATCH_SIZE=32
EVAL_INTERVAL=10
EVAL_ITERS=5
SEED=42

# Minimal data for algorithmic tasks
ALGO_TRAIN_LEN=10

# Loop settings for TRM/ut_level2
N_INNER_LOOPS=2
N_OUTER_LOOPS=2

# All models to test
MODELS=(
    "gpt"
    "gpt_level1"
    "gpt_level2"
    "ut"
    "ut_level1"
    "ut_level2"
    "trm"
)

# Dataset for testing (fast algorithmic task)
DATASET="copy_char"

# W&B tag
TAG="smoke_test_$(date +%Y%m%d_%H%M%S)"

# =============================================================================
# Logging
# =============================================================================

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# =============================================================================
# Main Execution
# =============================================================================

echo "=========================================="
echo "  Smoke Test - All Models"
echo "=========================================="
echo "Models: ${MODELS[*]}"
echo "Dataset: ${DATASET}"
echo "GPU: ${GPU}"
echo "Max Steps: ${MAX_STEPS}"
echo "Tag: ${TAG}"
echo "=========================================="
echo ""

FAILED_MODELS=()
PASSED_MODELS=()

for model in "${MODELS[@]}"; do
    run_name="test_${model}_${TAG}"
    
    log_info "Testing model: ${model}"
    echo "  Run name: ${run_name}"
    
    # Build command
    cmd="uv run python train.py \
        --model ${model} \
        --dataset ${DATASET} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer ${N_LAYER} \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --seed ${SEED} \
        --algo-train-len ${ALGO_TRAIN_LEN} \
        --gpu ${GPU} \
        --run-name ${run_name}"
    
    # Add loop parameters for models that use them
    if [[ "$model" == "ut_level2" || "$model" == "trm" ]]; then
        cmd="${cmd} --n-inner-loops ${N_INNER_LOOPS} --n-outer-loops ${N_OUTER_LOOPS}"
    fi
    
    # Run and capture result
    echo "  Running..."
    start_time=$(date +%s)
    
    if eval "${cmd}"; then
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        log_info "✓ ${model} PASSED (${duration}s)"
        PASSED_MODELS+=("$model")
    else
        log_error "✗ ${model} FAILED"
        FAILED_MODELS+=("$model")
    fi
    
    echo ""
done

# =============================================================================
# Summary
# =============================================================================

echo "=========================================="
echo "  Smoke Test Summary"
echo "=========================================="
echo ""

if [ ${#PASSED_MODELS[@]} -gt 0 ]; then
    log_info "Passed (${#PASSED_MODELS[@]}/${#MODELS[@]}):"
    for model in "${PASSED_MODELS[@]}"; do
        echo "    ✓ ${model}"
    done
fi

echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    log_error "Failed (${#FAILED_MODELS[@]}/${#MODELS[@]}):"
    for model in "${FAILED_MODELS[@]}"; do
        echo "    ✗ ${model}"
    done
    echo ""
    echo "=========================================="
    exit 1
else
    log_info "All models passed!"
    echo ""
    echo "W&B runs tagged with: ${TAG}"
    echo "View at: https://wandb.ai/YOUR_ENTITY/icml-recursive-llms"
    echo "=========================================="
    exit 0
fi
