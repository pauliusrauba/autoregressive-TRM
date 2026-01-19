#!/usr/bin/env bash
set -euo pipefail

# exp v5
# experiments/exp_compute_optimal.sh
# Compute-Optimal Training Experiment (Chinchilla-style)
#
# Goal: Given a fixed total training FLOPs budget, what's the optimal allocation
# between model size (n_embd) and training duration (steps)?
#
# Key constraints:
#   - TRAINING FLOPs: What we vary (n_embd × steps)
#   - TESTING FLOPs: Fixed at 12 block passes for ALL models
#   - FULL COMPUTE: All models run all iterations (no early halting)
#
# To ensure equal testing FLOPs, we set n_layer differently per model:
#   - GPT, GPT-L1, GPT-L2, UT: n_layer=12 (12 passes)
#   - UT-L1: n_layer=6 (2×6=12 passes)
#   - UT-L2, TRM: n_layer=4, H=1, L=2 (4×1×3=12 passes)
#
# Training FLOPs formula:
#   FLOPs ∝ steps × batch_size × block_passes × n_embd²
#        = steps × 64 × 12 × n_embd²
#
# Since block_passes=12 for all, the trade-off is purely:
#   - Bigger model (n_embd) for fewer steps
#   - Smaller model for more steps
#
# Runtime: ~5-6 hours on 2 GPUs (63 experiments)
# Output: Results logged to W&B with tag "compute_optimal"

# =============================================================================
# Configuration
# =============================================================================

# Task: Addition only (most discriminative)
TASK="addition_char"

# Training length (fixed)
ALGO_TRAIN_LEN=10
ALGO_EVAL_LENGTHS="10 15 20 30"  # 1x, 1.5x, 2x, 3x

# Model width scaling (n_embd)
# Note: n_head must divide n_embd
N_EMBD_SIZES=(96 128 192)

# Training duration scaling (steps)
TRAINING_STEPS=(1000 2000 4000)

# Fixed testing FLOPs: 12 block passes for all models
TARGET_BLOCK_PASSES=12

# Fixed H-L structure for UT-Level2/TRM
HL_OUTER=1
HL_INNER=2

# Other fixed settings
BLOCK_SIZE=120        # Max sequence for addition L=30 = ~93 tokens + buffer
DROPOUT=0.1
BATCH_SIZE=64
EVAL_INTERVAL=200
EVAL_ITERS=50
ALGO_EVAL_N=100
SEED=1337

# Data directory
DATA_DIR="/mnt/pdata/pr501/icml2025"

# All models to evaluate
MODELS=(
    "gpt"
    "gpt_level1"
    "gpt_level2"
    "ut"
    "ut_level1"
    "ut_level2"
    "trm"
)

# Tag for W&B grouping
BASE_TAG="compute_optimal"

# =============================================================================
# Model-Specific Configuration
# =============================================================================

# Get n_layer for a model to achieve TARGET_BLOCK_PASSES
get_n_layer() {
    local model=$1
    
    case "$model" in
        gpt|gpt_level1|gpt_level2|ut)
            # block_passes = n_layer
            echo $TARGET_BLOCK_PASSES
            ;;
        ut_level1)
            # block_passes = 2 * n_layer
            echo $((TARGET_BLOCK_PASSES / 2))
            ;;
        ut_level2|trm)
            # block_passes = n_layer * H * (L+1) = n_layer * 1 * 3
            echo $((TARGET_BLOCK_PASSES / 3))
            ;;
        *)
            echo $TARGET_BLOCK_PASSES
            ;;
    esac
}

# Get n_head that divides n_embd
get_n_head() {
    local n_embd=$1
    
    # Use 4 heads for all - divides 96, 128, 192
    echo 4
}

# Compute training FLOP proxy (in millions)
# FLOPs ∝ steps × batch_size × block_passes × n_embd²
compute_training_flops() {
    local n_embd=$1
    local steps=$2
    
    # All models have same block_passes, so:
    # FLOP_proxy = steps × BATCH_SIZE × TARGET_BLOCK_PASSES × n_embd²
    local flops=$((steps * BATCH_SIZE * TARGET_BLOCK_PASSES * n_embd * n_embd))
    echo $((flops / 1000000))  # In millions
}

# =============================================================================
# Job Queue Implementation
# =============================================================================

declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

LOGDIR="${DATA_DIR}/experiment_logs"
mkdir -p "${LOGDIR}"
MASTER_LOG="${LOGDIR}/run_${BASE_TAG}_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

wait_for_gpu() {
    while true; do
        for gpu in 0 1; do
            pid="${GPU_PIDS[$gpu]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                echo "$gpu"
                return
            fi
        done
        sleep 10
    done
}

run_experiment() {
    local model="$1"
    local n_embd="$2"
    local max_steps="$3"
    local gpu="$4"
    
    local n_layer=$(get_n_layer $model)
    local n_head=$(get_n_head $n_embd)
    local flop_proxy=$(compute_training_flops $n_embd $max_steps)
    local tag="${BASE_TAG}_embd${n_embd}_steps${max_steps}"
    local run_name="${model}_${TASK}_${tag}"
    local log_file="${LOGDIR}/${run_name}.log"
    
    log "Starting: ${run_name} on GPU ${gpu}"
    log "  n_embd=${n_embd}, n_layer=${n_layer}, steps=${max_steps}, training_FLOPs=${flop_proxy}M"
    
    # Build base command
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${TASK} \
        --data-dir ${DATA_DIR} \
        --n-head ${n_head} \
        --n-embd ${n_embd} \
        --n-layer ${n_layer} \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --algo-train-len ${ALGO_TRAIN_LEN} \
        --algo-eval-lengths ${ALGO_EVAL_LENGTHS} \
        --algo-eval-n ${ALGO_EVAL_N} \
        --max-steps ${max_steps} \
        --batch-size ${BATCH_SIZE} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --seed ${SEED} \
        --gpu ${gpu} \
        --run-name ${run_name}"
    
    # Add model-specific parameters
    case "$model" in
        ut|ut_level1)
            cmd="${cmd} --ponder-cost-weight 0"
            ;;
        ut_level2|trm)
            cmd="${cmd} --ponder-cost-weight 0"
            cmd="${cmd} --n-outer-loops ${HL_OUTER}"
            cmd="${cmd} --n-inner-loops ${HL_INNER}"
            ;;
    esac
    
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    log "  PID: ${pid}, Log: ${log_file}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Compute-Optimal Training Experiment"
log "(Chinchilla-style: Training FLOPs allocation)"
log "=========================================="
log ""
log "Key Design:"
log "  - TRAINING FLOPs: Varies with n_embd and steps"
log "  - TESTING FLOPs: Fixed at ${TARGET_BLOCK_PASSES} block passes for ALL models"
log "  - FULL COMPUTE: No early halting (ponder_cost_weight=0)"
log ""
log "Configuration:"
log "  Task: ${TASK}"
log "  Train Length: ${ALGO_TRAIN_LEN}"
log "  Eval Lengths: ${ALGO_EVAL_LENGTHS}"
log "  n_embd sizes: ${N_EMBD_SIZES[*]}"
log "  Training steps: ${TRAINING_STEPS[*]}"
log "  Target block passes: ${TARGET_BLOCK_PASSES}"
log "  H-L for UT-L2/TRM: H=${HL_OUTER}, L=${HL_INNER}"
log ""
log "n_layer per model (to achieve ${TARGET_BLOCK_PASSES} block passes):"
for model in "${MODELS[@]}"; do
    n_layer=$(get_n_layer $model)
    log "  ${model}: n_layer=${n_layer}"
done
log ""
log "Training FLOP proxy (millions) for each (n_embd, steps):"
for n_embd in "${N_EMBD_SIZES[@]}"; do
    for steps in "${TRAINING_STEPS[@]}"; do
        flops=$(compute_training_flops $n_embd $steps)
        log "  n_embd=${n_embd}, steps=${steps}: ${flops}M FLOPs"
    done
done
log "=========================================="

# Calculate total experiments
TOTAL=$((${#MODELS[@]} * ${#N_EMBD_SIZES[@]} * ${#TRAINING_STEPS[@]}))
log "Total experiments: ${TOTAL}"
log "Estimated runtime: ~$((TOTAL * 5 / 2 / 60)) hours on 2 GPUs"
log "=========================================="

# Run experiments
count=0
for n_embd in "${N_EMBD_SIZES[@]}"; do
    for max_steps in "${TRAINING_STEPS[@]}"; do
        for model in "${MODELS[@]}"; do
            count=$((count + 1))
            log "Queuing ${count}/${TOTAL}: ${model} n_embd=${n_embd} steps=${max_steps}"
            
            gpu=$(wait_for_gpu)
            run_experiment "$model" "$n_embd" "$max_steps" "$gpu"
            
            sleep 3
        done
    done
done

# Wait for completion
log "All experiments queued. Waiting for completion..."
wait

log "=========================================="
log "All experiments completed!"
log "=========================================="
log "Results logged to W&B project: icml-recursive-llms"
log "Tag filter: ${BASE_TAG}"
log ""
log "Analysis: Iso-Training-FLOP curves"
log "  - X-axis: Training FLOPs (steps × n_embd²)"
log "  - Y-axis: Performance (seq_acc at 1x, 1.5x, 2x, 3x)"
log "  - Color: Model size (n_embd)"
log "  - Facets: Model architecture"
log ""
log "Key questions this answers:"
log "  1. For a fixed training FLOP budget, what model size is optimal?"
log "  2. Do different architectures have different compute-optimal points?"
log "  3. Does weight-tying affect the optimal width/training trade-off?"
log "  4. How does the optimal allocation change for in-dist vs OOD?"
