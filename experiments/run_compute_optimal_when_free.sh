#!/usr/bin/env bash
# Waits for GPUs to be free, then runs the compute-optimal experiment

echo "Waiting for GPUs to be free..."
while true; do
    # Check if any Python processes are using GPUs
    gpu_procs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$gpu_procs" -eq 0 ]; then
        echo "GPUs are free! Starting experiment..."
        break
    fi
    echo "$(date): $gpu_procs GPU processes running. Waiting 60s..."
    sleep 60
done

# Run the experiment
cd "$(dirname "$0")/.."
./experiments/exp_compute_optimal.sh
