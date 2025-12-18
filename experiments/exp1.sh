#!/usr/bin/env bash
set -euo pipefail

# experiments/run_compute_normalized.sh
# Compute-normalized experiments across all models and algorithmic tasks.
#
# Compute is measured in "block passes" - the number of times a transformer block
# is applied during a forward pass. This normalizes compute across architectures.
#
# Block passes per model (with --compute-budget flag):
# - GPT, GPT_Level1, GPT_Level2, UT: n_layer = compute_budget
# - UT_Level1: n_layer = compute_budget / 2
# - UT_Level2, TRM: n_layer = compute_budget / (n_outer * (n_inner + 1))
#
# Strategy: Keep loops fixed (2 inner, 2 outer) for TRM/ut_level2, adjust n_layer.

# =============================================================================
# Configuration
# =============================================================================

COMPUTE_BUDGET=24          # Target block passes (adjust as needed)
ALGO_TRAIN_LEN=20          # L_train
# Evaluation lengths: L_train to 5x L_train
ALGO_EVAL_LENGTHS="20 40 60 80 100"

# Model architecture settings
N_HEAD=6
N_EMBD=384
BLOCK_SIZE=256
DROPOUT=0.1

# Loop settings for TRM/ut_level2 (kept fixed for compute normalization)
N_INNER_LOOPS=2
N_OUTER_LOOPS=2

# Training settings
MAX_STEPS=6500
BATCH_SIZE=64
SEED=1337

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

# All tasks to evaluate
TASKS=(
    "addition_char"
    "copy_char"
    "reverse_char"
)

# Tag for W&B grouping
TAG="compute${COMPUTE_BUDGET}_train${ALGO_TRAIN_LEN}"

# =============================================================================
# Job Queue Implementation (uses both GPUs)
# =============================================================================

# Track background jobs per GPU
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

# Log file for tracking experiments
LOGDIR="experiments/logs"
mkdir -p "${LOGDIR}"
MASTER_LOG="${LOGDIR}/run_${TAG}_$(date +%Y%m%d_%H%M%S).log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "${MASTER_LOG}"
}

# Wait for a GPU to become available and return its ID
wait_for_gpu() {
    while true; do
        for gpu in 0 1; do
            pid="${GPU_PIDS[$gpu]}"
            if [[ -z "$pid" ]] || ! kill -0 "$pid" 2>/dev/null; then
                # GPU is free
                echo "$gpu"
                return
            fi
        done
        # Both GPUs busy, wait for any job to finish
        sleep 5
    done
}

# Run a single experiment
run_experiment() {
    local model="$1"
    local task="$2"
    local gpu="$3"
    
    local run_name="${model}_${task}_${TAG}"
    local log_file="${LOGDIR}/${run_name}.log"
    
    log "Starting: ${run_name} on GPU ${gpu}"
    
    # Build the command
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${task} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer 6 \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --algo-train-len ${ALGO_TRAIN_LEN} \
        --algo-eval-lengths ${ALGO_EVAL_LENGTHS} \
        --compute-budget ${COMPUTE_BUDGET} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --seed ${SEED} \
        --gpu ${gpu} \
        --run-name ${run_name}"
    
    # Add loop parameters for models that use them
    if [[ "$model" == "ut_level2" || "$model" == "trm" ]]; then
        cmd="${cmd} --n-inner-loops ${N_INNER_LOOPS} --n-outer-loops ${N_OUTER_LOOPS}"
    fi
    
    # Run in background, redirect output to log file
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    log "  PID: ${pid}, Log: ${log_file}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Compute-Normalized Experiment Suite"
log "=========================================="
log "Compute Budget: ${COMPUTE_BUDGET} block passes"
log "Training Length: ${ALGO_TRAIN_LEN}"
log "Eval Lengths: ${ALGO_EVAL_LENGTHS}"
log "Models: ${MODELS[*]}"
log "Tasks: ${TASKS[*]}"
log "=========================================="

# Queue all experiments
for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        # Wait for an available GPU
        gpu=$(wait_for_gpu)
        
        # Run the experiment on that GPU
        run_experiment "$model" "$task" "$gpu"
        
        # Small delay to avoid race conditions
        sleep 2
    done
done

# Wait for all remaining jobs to complete
log "All experiments queued. Waiting for completion..."
wait

log "=========================================="
log "All experiments completed!"
log "=========================================="
log "Results are logged to W&B project: icml-recursive-llms"
log "Individual logs: ${LOGDIR}/"