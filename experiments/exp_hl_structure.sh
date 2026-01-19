#!/usr/bin/env bash
set -euo pipefail

# exp v4: goal is to understaand H-L behavior
# experiments/exp_hl_structure.sh
# H-L Loop Structure Experiment for TRM and UT-Level2
#
# This experiment investigates how the H-L loop decomposition affects performance:
#   - H (n_outer_loops): solution update cycles per ACT step
#   - L (n_inner_loops): reasoning refinements per H-cycle
#
# Note: Compute varies with H-L structure. This is intentional - we want to see
# if deeper H-L structure is worth the extra compute.
#
# Total block passes = n_layer × H × (L+1):
#   (1,1) →  8 passes,  (2,2) → 24 passes,  (4,2) → 48 passes
#   (6,2) → 72 passes,  (8,2) → 96 passes,  (8,4) → 160 passes
#
# Runtime: ~3-4 hours on 2 GPUs (48 experiments)
# Output: Results logged to W&B

# =============================================================================
# Configuration
# =============================================================================

# Training lengths to test
TRAIN_LENGTHS=(2 5 10 20)

# H-L configurations: (n_outer_loops, n_inner_loops)
# Format: "H:L" pairs
HL_CONFIGS=(
    "1:1"
    "2:2"
    "2:1"
    "4:2"
    "6:2"
    "8:2"
    "8:4"
)

# Task: Addition only
TASK="addition_char"

# Models to compare
MODELS=(
    "ut_level2"
    "trm"
)

# Fixed model architecture (same as exp_train_length_scaling.sh)
N_HEAD=4
N_EMBD=128
N_LAYER=4              # Fixed - compute varies via H-L
DROPOUT=0.1

# Maximum block_size to accommodate largest eval (L=20 @ 3x = 60 digits)
# Addition: num1 + num2 = result → 60+1+60+1+61 = 183 tokens max
BLOCK_SIZE=200

# Training settings
MAX_STEPS=2000
BATCH_SIZE=64
EVAL_INTERVAL=200     # 10 eval points
EVAL_ITERS=50
ALGO_EVAL_N=100
SEED=1337
CHECKPOINT_INTERVAL=200

# Data directory
DATA_DIR="/mnt/pdata/pr501/icml2025"

# Base tag for experiment identification
BASE_TAG="hl_structure"

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
        local eval_len=$(echo "$train_len * $ratio" | bc | awk '{printf "%.0f\n", $1}')
        if [ "$eval_len" -lt 1 ]; then
            eval_len=1
        fi
        eval_lengths="${eval_lengths} ${eval_len}"
    done
    
    echo $eval_lengths | tr ' ' '\n' | sort -n | uniq | tr '\n' ' '
}

# Calculate block size for a training length
calculate_block_size() {
    local max_eval_len=$1
    local seq_len=$((max_eval_len + 1 + max_eval_len + 1 + max_eval_len + 1 + 10))
    echo $seq_len
}

# Calculate total block passes for logging
calculate_block_passes() {
    local n_layer=$1
    local h=$2
    local l=$3
    echo $((n_layer * h * (l + 1)))
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
    local h="$3"
    local l="$4"
    local eval_lengths="$5"
    local block_size="$6"
    local gpu="$7"
    
    local block_passes=$(calculate_block_passes $N_LAYER $h $l)
    local tag="${BASE_TAG}_L${train_len}_H${h}_L${l}"
    local run_name="${model}_${TASK}_${tag}"
    local log_file="${LOGDIR}/${run_name}.log"
    
    log "Starting: ${run_name} on GPU ${gpu} (H=${h}, L=${l}, ${block_passes} block passes)"
    
    local cmd="uv run python train.py \
        --model ${model} \
        --dataset ${TASK} \
        --data-dir ${DATA_DIR} \
        --n-head ${N_HEAD} \
        --n-embd ${N_EMBD} \
        --n-layer ${N_LAYER} \
        --block-size ${block_size} \
        --dropout ${DROPOUT} \
        --n-inner-loops ${l} \
        --n-outer-loops ${h} \
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
        --ponder-cost-weight 0 \
        --run-name ${run_name}"
    
    eval "${cmd}" > "${log_file}" 2>&1 &
    local pid=$!
    GPU_PIDS[$gpu]=$pid
    
    log "  PID: ${pid}, Log: ${log_file}"
}

# =============================================================================
# Main Execution
# =============================================================================

log "=========================================="
log "H-L Loop Structure Experiment"
log "=========================================="
log "Training Lengths: ${TRAIN_LENGTHS[*]}"
log "H-L Configs: ${HL_CONFIGS[*]}"
log "Task: ${TASK}"
log "Models: ${MODELS[*]}"
log "Max Steps: ${MAX_STEPS}"
log "Model Size: n_embd=${N_EMBD}, n_head=${N_HEAD}, n_layer=${N_LAYER}"
log ""
log "Block passes by H-L config:"
for hl in "${HL_CONFIGS[@]}"; do
    IFS=':' read -r h l <<< "$hl"
    passes=$(calculate_block_passes $N_LAYER $h $l)
    log "  H=${h}, L=${l} → ${passes} block passes"
done
log "=========================================="

# Pre-calculate all eval lengths
declare -A EVAL_LENGTHS_MAP
declare -A BLOCK_SIZE_MAP

for train_len in "${TRAIN_LENGTHS[@]}"; do
    eval_lengths=$(calculate_eval_lengths $train_len)
    EVAL_LENGTHS_MAP[$train_len]="$eval_lengths"
    
    max_eval=$(echo $eval_lengths | tr ' ' '\n' | sort -n | tail -1)
    block_size=$(calculate_block_size $max_eval)
    if [ "$block_size" -lt "$BLOCK_SIZE" ]; then
        block_size=$BLOCK_SIZE
    fi
    BLOCK_SIZE_MAP[$train_len]=$block_size
    
    log "Train L=${train_len}: eval_lengths=[${eval_lengths}], block_size=${block_size}"
done

# Calculate total experiments
TOTAL=$((${#MODELS[@]} * ${#TRAIN_LENGTHS[@]} * ${#HL_CONFIGS[@]}))
log "Total experiments: ${TOTAL}"
log "Estimated runtime: ~$((TOTAL * 5 / 2 / 60)) hours on 2 GPUs (assuming ~5 min/experiment)"
log "=========================================="

# Run experiments
count=0
for train_len in "${TRAIN_LENGTHS[@]}"; do
    eval_lengths="${EVAL_LENGTHS_MAP[$train_len]}"
    block_size="${BLOCK_SIZE_MAP[$train_len]}"
    
    for hl in "${HL_CONFIGS[@]}"; do
        IFS=':' read -r h l <<< "$hl"
        
        for model in "${MODELS[@]}"; do
            count=$((count + 1))
            log "Queuing ${count}/${TOTAL}: ${model} L=${train_len} H=${h} L=${l}"
            
            gpu=$(wait_for_gpu)
            run_experiment "$model" "$train_len" "$h" "$l" "$eval_lengths" "$block_size" "$gpu"
            
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
log "Analysis questions this answers:"
log "  1. Does deeper H-L structure improve generalization?"
log "  2. Is more H (solution updates) or more L (reasoning refinements) better?"
log "  3. How does TRM compare to UT-Level2 at matched H-L configs?"
log ""
log "Next step: Analyze results in W&B or create a plotting script"
