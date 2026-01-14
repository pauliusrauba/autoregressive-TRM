#!/usr/bin/env bash
set -euo pipefail

# Exp v3
# experiments/exp_train_length_scaling.sh
# Training Length Scaling Experiments for Addition Task
#
# This experiment investigates how training length affects generalization by:
# 1. Training at multiple lengths: L = 2, 5, 10, 20, 40
# 2. Evaluating at relative lengths: 1x, 1.1x, 1.3x, 1.5x, 2x, 2.5x, 3x
# 3. Testing across 4 compute budgets
#
# Expected Outputs:
#   Plot 1: Performance vs task difficulty (absolute input length)
#   Plot 2: Generalization consistency across equivalent relative lengths
#   Plot 3: Pareto frontier (training length vs generalization)
#
# Runtime: ~10 hours on 2 GPUs
# Storage: ~12-14 GB (140 experiments × 10 checkpoints × ~8-10 MB each)
# Output: Results logged to W&B, checkpoints saved for inference experiments

# =============================================================================
# Configuration
# =============================================================================
# Notes: trainlen finished at trm L20-C18. (killed early).

# Training lengths to test
TRAIN_LENGTHS=(2 5 10 20 40)

# Compute budgets (4 levels from low to high)
COMPUTE_BUDGETS=(6 12 18 24)

# Task: Addition only (to keep experiment tractable)
TASK="addition_char"

# SMALLER Model architecture for faster training
N_HEAD=4              # Reduced from 6
N_EMBD=128            # Reduced from 384  
N_LAYER=4             # Reduced from 6 (base, adjusted by compute)
DROPOUT=0.1

# Maximum block_size to accommodate largest eval (L=40 @ 3x = 120 digits)
# Addition: num1 + num2 = result → 120+1+120+1+121 = 363 tokens max
BLOCK_SIZE=400

# Loop settings for UTLevel2/TRM
# NOTE: These are now computed dynamically by the compute normalization
# to achieve the exact target compute budget. Only used if --compute-budget is not set.

# Training settings - reduced for faster experiments
MAX_STEPS=2000        # Reduced from 5000
BATCH_SIZE=64
EVAL_INTERVAL=200     # 10 eval points
EVAL_ITERS=50         # Reduced from 100
ALGO_EVAL_N=100
SEED=1337
CHECKPOINT_INTERVAL=200  # Save 10 checkpoints per experiment (for inference experiments)

# Data directory
DATA_DIR="/mnt/pdata/pr501/icml2025"

# Models to evaluate (all 7)
MODELS=(
    "gpt"
    "gpt_level1"
    "gpt_level2"
    "ut"
    "ut_level1"
    "ut_level2"
    "trm"
)

# Base tag for experiment identification
BASE_TAG="trainlen_scaling"

# =============================================================================
# Helper Functions
# =============================================================================

# Calculate eval lengths for a given training length
# Returns: 1x, 1.1x, 1.3x, 1.5x, 2x, 2.5x, 3x (rounded to integers)
calculate_eval_lengths() {
    local train_len=$1
    local ratios=(1.0 1.1 1.3 1.5 2.0 2.5 3.0)
    local eval_lengths=""
    
    for ratio in "${ratios[@]}"; do
        # Use bc for floating point, round to nearest integer
        local eval_len=$(echo "$train_len * $ratio" | bc | awk '{printf "%.0f\n", $1}')
        # Ensure at least 1
        if [ "$eval_len" -lt 1 ]; then
            eval_len=1
        fi
        eval_lengths="${eval_lengths} ${eval_len}"
    done
    
    # Trim leading space and return unique sorted values
    echo $eval_lengths | tr ' ' '\n' | sort -n | uniq | tr '\n' ' '
}

# Calculate required block size for a training length
calculate_block_size() {
    local max_eval_len=$1
    # Addition format: N1 + N2 = R where N1,N2 have max_eval_len digits, R has max_eval_len+1
    local seq_len=$((max_eval_len + 1 + max_eval_len + 1 + max_eval_len + 1 + 10))
    echo $seq_len
}

# =============================================================================
# Job Queue Implementation (uses both GPUs)
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
    local train_len="$2"
    local compute_budget="$3"
    local eval_lengths="$4"
    local block_size="$5"
    local gpu="$6"
    
    local tag="${BASE_TAG}_L${train_len}_C${compute_budget}"
    local run_name="${model}_${TASK}_${tag}"
    local log_file="${LOGDIR}/${run_name}.log"
    
    log "Starting: ${run_name} on GPU ${gpu} (eval_lengths: ${eval_lengths})"
    
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${TASK} \
        --data-dir ${DATA_DIR} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer ${N_LAYER} \
        --block-size ${block_size} \
        --dropout ${DROPOUT} \
        --compute-budget ${compute_budget} \
        --algo-train-len ${train_len} \
        --algo-eval-lengths ${eval_lengths} \
        --algo-eval-n ${ALGO_EVAL_N} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --checkpoint-interval ${CHECKPOINT_INTERVAL} \
        --seed ${SEED} \
        --gpu ${gpu} \
        --run-name ${run_name}"
    
    # Add ACT-specific params for recurrent models
    case "$model" in
        ut|ut_level1|ut_level2|trm)
            cmd="${cmd} --ponder-cost-weight 0"
            ;;
    esac
    
    # NOTE: Loop parameters for ut_level2/trm are computed dynamically by
    # the compute normalization to achieve the exact target compute budget.
    
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    log "  PID: ${pid}, Log: ${log_file}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Training Length Scaling Experiment"
log "=========================================="
log "Training Lengths: ${TRAIN_LENGTHS[*]}"
log "Compute Budgets: ${COMPUTE_BUDGETS[*]}"
log "Task: ${TASK}"
log "Models: ${MODELS[*]}"
log "Max Steps: ${MAX_STEPS}"
log "Checkpoint Interval: ${CHECKPOINT_INTERVAL} (=$((MAX_STEPS / CHECKPOINT_INTERVAL)) checkpoints per run)"
log "Model Size: n_embd=${N_EMBD}, n_head=${N_HEAD}, n_layer=${N_LAYER}"
log "=========================================="

# Pre-calculate all eval lengths
declare -A EVAL_LENGTHS_MAP
declare -A BLOCK_SIZE_MAP

for train_len in "${TRAIN_LENGTHS[@]}"; do
    eval_lengths=$(calculate_eval_lengths $train_len)
    EVAL_LENGTHS_MAP[$train_len]="$eval_lengths"
    
    # Get max eval length for block size calculation
    max_eval=$(echo $eval_lengths | tr ' ' '\n' | sort -n | tail -1)
    block_size=$(calculate_block_size $max_eval)
    # Use at least BLOCK_SIZE
    if [ "$block_size" -lt "$BLOCK_SIZE" ]; then
        block_size=$BLOCK_SIZE
    fi
    BLOCK_SIZE_MAP[$train_len]=$block_size
    
    log "Train L=${train_len}: eval_lengths=[${eval_lengths}], block_size=${block_size}"
done

# Calculate total experiments
TOTAL=$((${#MODELS[@]} * ${#TRAIN_LENGTHS[@]} * ${#COMPUTE_BUDGETS[@]}))
log "Total experiments: ${TOTAL}"
log "Estimated runtime: ~$((TOTAL * 8 / 2 / 60)) hours on 2 GPUs (assuming ~8 min/experiment)"
log "=========================================="

# Run experiments
count=0
for train_len in "${TRAIN_LENGTHS[@]}"; do
    eval_lengths="${EVAL_LENGTHS_MAP[$train_len]}"
    block_size="${BLOCK_SIZE_MAP[$train_len]}"
    
    for compute_budget in "${COMPUTE_BUDGETS[@]}"; do
        for model in "${MODELS[@]}"; do
            count=$((count + 1))
            log "Queuing ${count}/${TOTAL}: ${model} L=${train_len} C=${compute_budget}"
            
            gpu=$(wait_for_gpu)
            run_experiment "$model" "$train_len" "$compute_budget" "$eval_lengths" "$block_size" "$gpu"
            
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
log "Next step: Generate analysis tables and plots:"
log "  uv run python scripts/plot_train_length_scaling.py --tag ${BASE_TAG}"
