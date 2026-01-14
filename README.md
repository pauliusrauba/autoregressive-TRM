# TRM-LLM

Codebase for TRM (Transformer with Recursive Mechanism) LLMs research project.

## Setup

1. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Verify installation:
   ```bash
   uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

## Quick Start

```bash
# Train GPT baseline on addition task
uv run python train.py --model gpt --dataset addition_char --gpu 0

# Train TRM on copy task
uv run python train.py --model trm --dataset copy_char --algo-train-len 20 --gpu 0
```

## Project Structure

```
trm-llm/
├── train.py                    # Main training script
├── eval_algorithmic.py         # Evaluate on algorithmic tasks
├── eval_inference_scaling.py   # Test inference compute scaling
├── models/                     # Model implementations
│   ├── gpt.py                  # GPT-2 baseline
│   ├── gpt_level1.py           # + Weight sharing
│   ├── gpt_level2.py           # + Step embeddings
│   ├── ut.py                   # Universal Transformer (ACT)
│   ├── ut_level1.py            # + Reasoning/solution separation
│   ├── ut_level2.py            # + Nested loops
│   ├── trm.py                  # Full TRM model
│   └── common/                 # Shared components
├── data_modules/               # Dataset implementations
├── callbacks/                  # Training callbacks
├── experiments/                # Experiment scripts
└── notebooks/                  # Analysis notebooks
```

## Models

| Model | Description | Block Passes |
|-------|-------------|--------------|
| `gpt` | GPT-2 baseline (unique blocks) | `n_layer` |
| `gpt_level1` | Weight sharing (same block reused) | `n_layer` |
| `gpt_level2` | + Learnable step embeddings | `n_layer` |
| `ut` | Universal Transformer with ACT | `n_layer` (max) |
| `ut_level1` | + Reasoning/solution decoupling | `2 × n_layer` |
| `ut_level2` | + Nested inner/outer loops | `n_layer × n_outer × (n_inner + 1)` |
| `trm` | Full TRM (TBPTT + exploration) | `n_layer × n_outer × (n_inner + 1)` |

**Block passes** = number of transformer block applications per forward pass.

## Datasets

| Dataset | Task | Example |
|---------|------|---------|
| `addition_char` | Integer addition | `123+456=` → `579` |
| `copy_char` | Copy sequence | `abc>` → `abc` |
| `reverse_char` | Reverse sequence | `abc>` → `cba` |
| `shakespeare_char` | Text generation | Shakespeare corpus |
| `gsm8k_char` | Math word problems | GSM8K dataset |

## Usage Examples

### Basic Training

```bash
uv run python train.py \
  --model ut \
  --dataset addition_char \
  --n-layer 6 \
  --n-embd 384 \
  --n-head 6 \
  --block-size 256 \
  --dropout 0.1 \
  --algo-train-len 20 \
  --max-steps 5000 \
  --gpu 0
```

### Compute-Normalized Experiments

Compare models with equal compute (block passes):

```bash
# GPT with 24 block passes → n_layer=24
uv run python train.py --model gpt --dataset addition_char --compute-budget 24 --gpu 0

# TRM with 24 block passes → n_layer=4 (with 2×2 loops: 4 × 2 × 3 = 24)
uv run python train.py --model trm --dataset addition_char --compute-budget 24 \
  --n-inner-loops 2 --n-outer-loops 2 --gpu 0
```

| Model | Compute Budget 24 | Calculation |
|-------|-------------------|-------------|
| `gpt`, `gpt_level1`, `gpt_level2`, `ut` | n_layer=24 | 24 passes |
| `ut_level1` | n_layer=12 | 12 × 2 = 24 passes |
| `ut_level2`, `trm` | n_layer=4 | 4 × 2 × 3 = 24 passes |

### Parameter-Normalized Experiments

Compare models with equal parameters but unlimited inference compute:

```bash
uv run python train.py \
  --model ut \
  --dataset addition_char \
  --n-layer 6 \
  --n-embd 192 \
  --max-act-steps 64 \
  --ponder-cost-weight 0.0 \
  --max-steps 15000 \
  --gpu 0
```

Key parameters:
- `--max-act-steps`: Maximum ACT iterations at inference (default: `n_layer`)
- `--ponder-cost-weight`: Penalty for extra thinking (0 = no penalty)

### Evaluating at Different Lengths

Test length generalization during training:

```bash
uv run python train.py \
  --model trm \
  --dataset addition_char \
  --algo-train-len 20 \
  --algo-eval-lengths 20 40 60 80 100 \
  --gpu 0
```

Logs metrics to W&B: `TaskEvaluation/{task}/L{length}/seq_acc`

### Inference Scaling Evaluation

Test trained checkpoints with varying inference compute:

```bash
uv run python eval_inference_scaling.py \
  --ckpt-dir /mnt/pdata/pr501/icml2025/checkpoints \
  --max-act-steps 8 16 32 64 128 \
  --tasks copy reverse addition \
  --lengths 20 40 60 80 100 \
  --output results_inference_scaling.csv
```

## Checkpointing

Checkpoints saved to `{data_dir}/checkpoints/{run_name}/`:

| File | Description |
|------|-------------|
| `last.ckpt` | Latest checkpoint |
| `step=XXXXX.ckpt` | Periodic (every 2000 steps) |
| `best_seq_acc.ckpt` | Best sequence accuracy |

Custom location:
```bash
uv run python train.py --data-dir /path/to/data --run-name my_experiment ...
```

## Running Experiment Suites

### Smoke Test (~5 min)

```bash
./experiments/smoke_test.sh
# Or: GPU=1 ./experiments/smoke_test.sh
```

### Full Compute-Normalized Suite

```bash
# Run in tmux
tmux new-session -d -s exp1 './experiments/exp1.sh'
tmux attach -t exp1  # Ctrl+B, D to detach

# Check logs
tail -f /mnt/pdata/pr501/icml2025/experiment_logs/run_*.log
```

Configuration in `experiments/exp1.sh`:
```bash
COMPUTE_BUDGET=24
ALGO_TRAIN_LEN=20
ALGO_EVAL_LENGTHS="20 40 60 80 100"
MAX_STEPS=5000
```

### Parameter-Normalized Suite

```bash
./experiments/exp2_param_normalized.sh
```

## Dependencies

```bash
uv add package-name          # Add dependency
uv lock --upgrade && uv sync # Update all
```

### Limited Disk Space (HPC)

```bash
export DATA_DIR="/mnt/pdata/YOUR_USERNAME/icml2025"
mkdir -p $DATA_DIR/.uv-cache $DATA_DIR/.venv
ln -sf $DATA_DIR/.venv .venv
export UV_CACHE_DIR="$DATA_DIR/.uv-cache"
export UV_LINK_MODE=copy
uv sync
```

## Results

Results logged to W&B project: `icml-recursive-llms`

Analysis notebooks in `notebooks/`:
- `1. Exploration.ipynb`
- `2. Results.ipynb`


---

ADded step embedding flag

# Mode A: With step embeddings (current behavior)
python train.py --model ut_level2 --compute-budget 12 --use-step-embeddings

# Mode B: Without step embeddings (new option)  
python train.py --model ut_level2 --compute-budget 12 --no-step-embeddings