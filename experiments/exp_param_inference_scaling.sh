#!/usr/bin/env bash
set -euo pipefail

# exp v2
# experiments/exp_param_inference_scaling.sh
# Parameter and inference compute scaling experiments for four plots:
#   Plot 1: Parameter-performance frontier (best over inference compute)
#   Plot 2: Performance at different inference compute budgets
#   Plot 3: Training budget (FLOPs) vs testing budget (block passes) heatmap
#   Plot 4: Training budget vs performance per parameter
#
# Runtime: ~5 hours on 2 GPUs (28 experiments × 20 min / 2 GPUs)
# Output: Results logged to W&B, checkpoints to /mnt/pdata/pr501/icml2025/

# =============================================================================
# Configuration
# =============================================================================

# Task (use addition as the most discriminative algorithmic task)
TASK="addition_char"

# Training length for algorithmic task
ALGO_TRAIN_LEN=10
ALGO_EVAL_LENGTHS="10 20 30"

# Parameter scaling: different embedding dimensions
# These give roughly: 192→~1.5M, 256→~2.5M, 384→~5.5M params (varies by model)
# Using 2 sizes to fit within 6-hour budget
N_EMBD_SIZES=(192 384)

# Inference compute scaling: different compute budgets (block passes)
COMPUTE_BUDGETS=(12 24)

# Fixed architecture settings
N_HEAD=6              # Heads (must divide n_embd)
BLOCK_SIZE=150        # Max sequence for addition L=30 = ~93 tokens + buffer
DROPOUT=0.1
N_LAYER=6             # Base n_layer (adjusted by compute normalization)

# Loop settings for UTLevel2/TRM
# NOTE: These are now computed dynamically by the compute normalization
# to achieve the exact target compute budget.

# Training settings (reduced for 6-hour budget)
MAX_STEPS=4000
BATCH_SIZE=64
EVAL_INTERVAL=200     # 20 eval points for smooth curves
EVAL_ITERS=100
ALGO_EVAL_N=100       # Samples per evaluation
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
TAG="paraminf_scaling"

# =============================================================================
# Job Queue Implementation (uses both GPUs)
# =============================================================================

# Track background jobs per GPU
declare -A GPU_PIDS
GPU_PIDS[0]=""
GPU_PIDS[1]=""

# Log directory
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
    local n_embd="$2"
    local compute_budget="$3"
    local gpu="$4"
    
    local run_name="${model}_${TASK}_embd${n_embd}_compute${compute_budget}_${TAG}"
    local log_file="${LOGDIR}/${run_name}.log"
    
    log "Starting: ${run_name} on GPU ${gpu}"
    
    # Determine n_head: must divide n_embd evenly
    # For n_embd in {192, 256, 384}, 6 heads works for all
    local n_head=${N_HEAD}
    if (( n_embd % n_head != 0 )); then
        # Fallback: use 4 heads for 256, 8 for others
        if (( n_embd == 256 )); then
            n_head=4
        else
            n_head=8
        fi
    fi
    
    # Build the base command
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${TASK} \
        --data-dir ${DATA_DIR} \
        --n-head ${n_head} \
        --n-embd ${n_embd} \
        --n-layer ${N_LAYER} \
        --block-size ${BLOCK_SIZE} \
        --dropout ${DROPOUT} \
        --compute-budget ${compute_budget} \
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
    
    # Add ACT-specific params for recurrent models (no halting penalty for max compute)
    case "$model" in
        ut|ut_level1|ut_level2|trm)
            cmd="${cmd} --ponder-cost-weight 0"
            ;;
    esac
    
    # NOTE: Loop parameters for ut_level2/trm are computed dynamically by
    # the compute normalization to achieve the exact target compute budget.
    
    # Run in background
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    log "  PID: ${pid}, Log: ${log_file}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "Parameter & Inference Scaling Experiment"
log "=========================================="
log "Task: ${TASK}"
log "Training Length: ${ALGO_TRAIN_LEN}"
log "Eval Lengths: ${ALGO_EVAL_LENGTHS}"
log "Embedding Sizes: ${N_EMBD_SIZES[*]}"
log "Compute Budgets: ${COMPUTE_BUDGETS[*]}"
log "Max Steps: ${MAX_STEPS}"
log "Eval Interval: ${EVAL_INTERVAL}"
log "Models: ${MODELS[*]}"
log "Data Dir: ${DATA_DIR}"
log "=========================================="

# Calculate total experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#N_EMBD_SIZES[@]} * ${#COMPUTE_BUDGETS[@]}))
log "Total experiments: ${TOTAL_EXPERIMENTS}"
log "Estimated runtime: ~$((TOTAL_EXPERIMENTS * 20 / 2)) minutes (~$((TOTAL_EXPERIMENTS * 20 / 120)) hours) on 2 GPUs"
log "=========================================="

# Queue all experiments
experiment_count=0
for n_embd in "${N_EMBD_SIZES[@]}"; do
    for compute_budget in "${COMPUTE_BUDGETS[@]}"; do
        for model in "${MODELS[@]}"; do
            experiment_count=$((experiment_count + 1))
            log "Queuing experiment ${experiment_count}/${TOTAL_EXPERIMENTS}: ${model} embd=${n_embd} compute=${compute_budget}"
            
            # Wait for an available GPU
            gpu=$(wait_for_gpu)
            
            # Run the experiment
            run_experiment "$model" "$n_embd" "$compute_budget" "$gpu"
            
            # Small delay to avoid race conditions
            sleep 3
        done
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
log "  uv run python scripts/plot_param_inference_scaling.py --tag ${TAG}"
