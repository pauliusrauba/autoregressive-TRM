#!/usr/bin/env bash
set -euo pipefail

# experiments/exp_length_generalization.sh
# Length generalization experiments for the four plots:
#   Plot 1: Absolute-length generalization at fixed inference compute
#   Plot 2: Relative length ratio (L_eval / L_train) vs performance
#   Plot 3: Generalization gap vs relative length
#   Plot 4: Training FLOPs vs performance at different lengths
#
# Runtime: ~8 hours on 2 GPUs
# Output: Results logged to W&B, checkpoints to /mnt/pdata/pr501/icml2025/

# =============================================================================
# Configuration
# =============================================================================

# Compute budget (block passes) - all models normalized to this
COMPUTE_BUDGET=24

# Length settings
ALGO_TRAIN_LEN=10
# Eval lengths: 1x, 2x, 3x, 4x, 5x training length
ALGO_EVAL_LENGTHS="10 20 30 40 50"

# Model architecture settings
N_HEAD=6
N_EMBD=384
BLOCK_SIZE=180  # Max sequence: addition L=50 = 153 tokens, +27 buffer
DROPOUT=0.1
N_LAYER=6  # Base n_layer (adjusted by compute normalization)

# Loop settings for UTLevel2/TRM (kept fixed for compute normalization)
N_INNER_LOOPS=2
N_OUTER_LOOPS=2

# Training settings
MAX_STEPS=5000
BATCH_SIZE=64
EVAL_INTERVAL=250    # 20 eval points for FLOPs curve
EVAL_ITERS=100
ALGO_EVAL_N=100      # Samples per evaluation
SEED=1337

# Data directory (all outputs go here)
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

# All tasks to evaluate
TASKS=(
    "addition_char"
    "copy_char"
    "reverse_char"
)

# Tag for W&B grouping and results identification
TAG="lengthgen_train${ALGO_TRAIN_LEN}_compute${COMPUTE_BUDGET}"

# =============================================================================
# Job Queue Implementation (uses both GPUs)
# =============================================================================

# Track background jobs per GPU
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

# Log directory on data mount
LOGDIR="${DATA_DIR}/experiment_logs"
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
        sleep 10
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
        --data-dir ${DATA_DIR} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer ${N_LAYER} \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --compute-budget ${COMPUTE_BUDGET} \
        --algo-train-len ${ALGO_TRAIN_LEN} \
        --algo-eval-lengths ${ALGO_EVAL_LENGTHS} \
        --algo-eval-n ${ALGO_EVAL_N} \
        --max-steps ${MAX_STEPS} \
        --batch-size ${BATCH_SIZE} \
        --eval-interval ${EVAL_INTERVAL} \
        --eval-iters ${EVAL_ITERS} \
        --seed ${SEED} \
        --gpu ${gpu} \
        --run-name ${run_name}"
    
    # Add ACT-specific params for recurrent models (full compute, no halting penalty)
    case "$model" in
        ut|ut_level1|ut_level2|trm)
            cmd="${cmd} --ponder-cost-weight 0"
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
log "Length Generalization Experiment Suite"
log "=========================================="
log "Compute Budget: ${COMPUTE_BUDGET} block passes"
log "Training Length: ${ALGO_TRAIN_LEN}"
log "Eval Lengths: ${ALGO_EVAL_LENGTHS}"
log "Max Steps: ${MAX_STEPS}"
log "Eval Interval: ${EVAL_INTERVAL} (${MAX_STEPS}/${EVAL_INTERVAL} = $((MAX_STEPS/EVAL_INTERVAL)) data points)"
log "Models: ${MODELS[*]}"
log "Tasks: ${TASKS[*]}"
log "Data Dir: ${DATA_DIR}"
log "=========================================="

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#TASKS[@]}))
log "Total experiments: ${TOTAL_EXPERIMENTS}"
log "Estimated runtime: ~$((TOTAL_EXPERIMENTS * 25 / 2)) minutes (~$((TOTAL_EXPERIMENTS * 25 / 120)) hours) on 2 GPUs"
log "=========================================="

# Queue all experiments
experiment_count=0
for task in "${TASKS[@]}"; do
    for model in "${MODELS[@]}"; do
        experiment_count=$((experiment_count + 1))
        log "Queuing experiment ${experiment_count}/${TOTAL_EXPERIMENTS}: ${model} on ${task}"
        
        # Wait for an available GPU
        gpu=$(wait_for_gpu)
        
        # Run the experiment on that GPU
        run_experiment "$model" "$task" "$gpu"
        
        # Small delay to avoid race conditions
        sleep 3
    done
done

# Wait for all remaining jobs to complete
log "All experiments queued. Waiting for completion..."
wait

log "=========================================="
log "All experiments completed!"
log "=========================================="
log "Results are logged to W&B project: icml-recursive-llms"
log "Tag filter: ${TAG}"
log "Individual logs: ${LOGDIR}/"
log ""
log "Next step: Run the plotting script to generate figures:"
log "  uv run python scripts/plot_length_generalization.py --tag ${TAG}"
