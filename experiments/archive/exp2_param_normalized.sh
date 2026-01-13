#!/usr/bin/env bash
set -euo pipefail

# experiments/exp2_param_normalized.sh
# Parameter-normalized experiments with unlimited inference compute.
#
# Key differences from exp1 (compute-normalized):
# - All shared-block models have ~same parameter count (~1.9M)
# - ACT models can use many more steps (max_act_steps=64)
# - No ponder cost penalty (ponder_cost_weight=0) to allow free thinking
# - Longer training (15000 steps) for better convergence
# - GPT included as parameter-rich baseline (42.7M params)
#
# Goal: Test what recurrent architectures can achieve when not compute-constrained.

# =============================================================================
# Configuration
# =============================================================================

# Model architecture settings (same for all shared-block models)
N_HEAD=6
N_EMBD=192
BLOCK_SIZE=256
DROPOUT=0.1
N_LAYER=6                  # Base architectural depth

# ACT-specific settings - key difference from exp1
MAX_ACT_STEPS=64           # Allow up to 64 ACT steps (vs 4-24 in exp1)
PONDER_COST_WEIGHT=0.0     # No penalty for thinking (vs 0.01 default)

# Loop settings for UTLevel2/TRM
N_INNER_LOOPS=4
N_OUTER_LOOPS=2

# Training settings - longer than exp1 for better convergence
MAX_STEPS=15000            # vs 6500 in exp1
BATCH_SIZE=64
SEED=1337

# Algorithmic task settings
ALGO_TRAIN_LEN=20          # L_train
ALGO_EVAL_LENGTHS="20 40 60 80 100"

# All models to evaluate
# Shared-block models (~1.9M params) + GPT baseline (~42.7M params)
MODELS=(
    "gpt"           # Baseline: 42.7M params, no recurrence
    "gpt_level1"    # Shared block, no ACT
    "gpt_level2"    # Shared block + step embeddings, no ACT
    "ut"            # ACT with single shared block
    "ut_level1"     # ACT with reasoning/solution separation
    "ut_level2"     # ACT with nested loops
    "trm"           # ACT with TBPTT + exploration
)

# All tasks to evaluate
TASKS=(
    "addition_char"
    "copy_char"
    "reverse_char"
)

# Tag for W&B grouping
TAG="param_normalized_act${MAX_ACT_STEPS}_ponder${PONDER_COST_WEIGHT}"

# =============================================================================
# Job Queue Implementation (uses both GPUs)
# =============================================================================

# Track background jobs per GPU
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

# Log file for tracking experiments
LOGDIR="/mnt/pdata/pr501/icml2025/experiment_logs"
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
                echo "$gpu"
                return
            fi
        done
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
    
    # Build the base command
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${task} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer ${N_LAYER} \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --algo-train-len ${ALGO_TRAIN_LEN} \
        --algo-eval-lengths ${ALGO_EVAL_LENGTHS} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --seed ${SEED} \
        --gpu ${gpu} \
        --run-name ${run_name}"
    
    # Add ACT-specific params for recurrent models
    case "$model" in
        ut|ut_level1|ut_level2|trm)
            cmd="${cmd} --max-act-steps ${MAX_ACT_STEPS}"
            cmd="${cmd} --ponder-cost-weight ${PONDER_COST_WEIGHT}"
            ;;
    esac
    
    # Add loop parameters for models that use them
    case "$model" in
        ut_level2|trm)
            cmd="${cmd} --n-inner-loops ${N_INNER_LOOPS} --n-outer-loops ${N_OUTER_LOOPS}"
            ;;
    esac
    
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
log "Parameter-Normalized Experiment Suite"
log "=========================================="
log "Architecture: n_embd=${N_EMBD}, n_head=${N_HEAD}, n_layer=${N_LAYER}"
log "ACT Settings: max_act_steps=${MAX_ACT_STEPS}, ponder_cost_weight=${PONDER_COST_WEIGHT}"
log "Training: max_steps=${MAX_STEPS}"
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
