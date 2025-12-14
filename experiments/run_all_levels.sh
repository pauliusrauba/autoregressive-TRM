#!/usr/bin/env bash
set -euo pipefail

# experiments/run_all_levels.sh
# Runs UT-GPT levels 0..6 with consistent settings and clear run names.

DATASET="copy_char"
BLOCK_SIZE=256
MODEL="ut_gpt"

# Common UT/ACT settings (ACT only used at levels 5-6; ponder only at 6)
UT_MAX_STEPS=16
ACT_THRESHOLD=0.99
ACT_LOSS_WEIGHT=0.01

# Recurrent fixed steps for levels 1-4 (adjust to match your baseline depth)
NUM_STEPS=6

# Optional: tag runs so they're easy to group in W&B / logs
TAG="copy_bs${BLOCK_SIZE}_ns${NUM_STEPS}_ms${UT_MAX_STEPS}"

run_level () {
  local lvl="$1"
  local run_name="utgpt_L${lvl}_${TAG}"

  echo "=============================="
  echo "Running level ${lvl}: ${run_name}"
  echo "=============================="

  if [[ "$lvl" -le 0 ]]; then
    # Level 0: baseline behavior inside unified model
    python train.py \
      --model "${MODEL}" \
      --level "${lvl}" \
      --dataset "${DATASET}" \
      --block-size "${BLOCK_SIZE}" \
      --run-name "${run_name}"

  elif [[ "$lvl" -ge 1 && "$lvl" -le 4 ]]; then
    # Levels 1-4: fixed recurrent steps
    python train.py \
      --model "${MODEL}" \
      --level "${lvl}" \
      --num-steps "${NUM_STEPS}" \
      --ut-max-steps "${UT_MAX_STEPS}" \
      --dataset "${DATASET}" \
      --block-size "${BLOCK_SIZE}" \
      --run-name "${run_name}"

  elif [[ "$lvl" -eq 5 ]]; then
    # Level 5: ACT (no ponder penalty)
    python train.py \
      --model "${MODEL}" \
      --level "${lvl}" \
      --ut-max-steps "${UT_MAX_STEPS}" \
      --act-threshold "${ACT_THRESHOLD}" \
      --dataset "${DATASET}" \
      --block-size "${BLOCK_SIZE}" \
      --run-name "${run_name}"

  elif [[ "$lvl" -eq 6 ]]; then
    # Level 6: ACT + ponder penalty
    python train.py \
      --model "${MODEL}" \
      --level "${lvl}" \
      --ut-max-steps "${UT_MAX_STEPS}" \
      --act-threshold "${ACT_THRESHOLD}" \
      --act-loss-weight "${ACT_LOSS_WEIGHT}" \
      --dataset "${DATASET}" \
      --block-size "${BLOCK_SIZE}" \
      --run-name "${run_name}"

  else
    echo "Unknown level: ${lvl}"
    exit 1
  fi
}

for lvl in 4 5 6; do
  run_level "${lvl}"
done
