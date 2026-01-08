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

## Running Code

Use `uv run` to execute Python scripts:

```bash
uv run python train.py --model gpt --dataset addition_char
```

Or activate the environment manually:

```bash
source .venv/bin/activate
python train.py --model gpt --dataset addition_char
```

## Project Structure

```
icml/
├── models/                 # Model implementations
│   ├── gpt.py             # Vanilla GPT-2
│   ├── gpt_level1.py      # Weight sharing (reuses same block)
│   ├── gpt_level2.py      # + Step/time embedding
│   ├── ut.py              # Universal Transformer
│   ├── ut_level1.py       # Decoupling reasoning from solution
│   ├── ut_level2.py       # Additional modifications
│   ├── trm.py             # Final TRM model
│   └── common/            # Shared layers and trainer
├── data_modules/          # Dataset implementations
├── callbacks/             # Training callbacks
├── experiments/           # Experiment scripts
├── train.py               # Main training script
├── pyproject.toml         # Dependency specifications
└── uv.lock                # Locked versions (for reproducibility)
```

## Example Training Commands

```bash
# Train Universal Transformer on addition task
uv run python train.py \
  --model ut \
# trm-llm

Codebase for TRM LLMs

Codebase structure so far:
- models/common contains two items: layers.py has some basic layers implemented (not optimized for computation) and trainer.py has the pytorch lightning pytorch trainer
- gpt.py is a gpt-2 vanilla model.
Then the levels are added based on how they're described in the paper
- gpt_level1.py reuses the same block instead of two
- gpt_level2.py adding step/time embedding
- UT is the universal transformer

Then the changes from UT toward TRM are also implemented in 2 sub-models and resulting in a TRM.
- ut_level1. Decoupling reasoning from solution.
- ut_level2. Some other stuff I Can't remember now.
- trm

## Installation

```bash
pip install -e .
# or with uv:
uv pip install -e .
```

## Basic Usage

```bash
python train.py \
  --model ut \
  --dataset addition_char \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
  --algo-train-len 20 \
  --dropout 0.1 \
  --gpu 0
```

# Train TRM on addition task
uv run python train.py \
  --model trm \
  --gpu 0
```

## Experiments

### Available Models

| Model | Description | Block Passes |
|-------|-------------|--------------|
| `gpt` | GPT-2 baseline | `n_layer` |
| `gpt_level1` | Shared block (reused n_layer times) | `n_layer` |
| `gpt_level2` | + step embeddings | `n_layer` |
| `ut` | Universal Transformer with ACT | `n_layer` |
| `ut_level1` | + reasoning/solution decoupling | `2 * n_layer` |
| `ut_level2` | + inner/outer loops | `n_layer * n_outer * (n_inner + 1)` |
| `trm` | Full TRM | `n_layer * n_outer * (n_inner + 1)` |

### Available Datasets

- `addition_char` - Addition task (e.g., "123+456=")
- `copy_char` - Copy task
- `reverse_char` - Reverse task
- `shakespeare_char` - Shakespeare text generation
- `gsm8k_char` - GSM8K math problems

### Multiple Evaluation Lengths

Evaluate at multiple sequence lengths to measure extrapolation:

```bash
python train.py \
  --model trm \
  --dataset addition_char \
  --algo-train-len 20 \
  --dropout 0.1 \
  --gpu 0 \
  --algo-eval-extrap-len 40
```

## Managing Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a dev dependency
uv add --dev package-name

# Update all dependencies
uv lock --upgrade
uv sync
```

After adding/updating dependencies, commit both `pyproject.toml` and `uv.lock`.

## For Collaborators

1. Clone the repo
2. Run `uv sync` — this installs the exact versions from `uv.lock`
3. You now have an identical environment!

The key files for reproducibility:
- `pyproject.toml` — dependency specifications
- `uv.lock` — exact pinned versions (commit this!)

## Advanced: Limited Disk Space Environments

If you're on a shared HPC cluster with limited home directory quota, you may need to store the virtual environment and cache on a data mount:

```bash
# Set your data directory
export DATA_DIR="/mnt/pdata/YOUR_USERNAME/icml2025"

# Create directories
mkdir -p $DATA_DIR/.uv-cache
mkdir -p $DATA_DIR/.venv

# Symlink .venv to data mount (from project directory)
ln -sf $DATA_DIR/.venv .venv

# Configure UV cache (add to ~/.bashrc)
export UV_CACHE_DIR="$DATA_DIR/.uv-cache"
export UV_LINK_MODE=copy

# Then run uv sync as usual
uv sync
```
  --algo-eval-lengths 20 40 60 80 100 \
  --gpu 0
```

This logs metrics to W&B at each length:
- `TaskEvaluation/addition/L20/seq_acc`
- `TaskEvaluation/addition/L40/seq_acc`
- etc.

If `--algo-eval-lengths` is not specified, defaults to `[algo_train_len, 5 * algo_train_len]`.

### Compute-Normalized Experiments

For fair comparison across architectures, use `--compute-budget` to normalize by **block passes** (number of times a transformer block is applied per forward pass):

```bash
# GPT with 24 block passes -> n_layer=24
python train.py --model gpt --compute-budget 24

# TRM with 24 block passes -> n_layer=4 (with 2 inner, 2 outer loops: 4 * 2 * 3 = 24)
python train.py --model trm --compute-budget 24 --n-inner-loops 2 --n-outer-loops 2
```

The strategy keeps loop structure fixed for TRM/ut_level2 and adjusts `n_layer`:

| Model | Compute Budget 24 | Effective Config |
|-------|-------------------|------------------|
| GPT/UT variants | n_layer=24 | 24 passes |
| UT_Level1 | n_layer=12 | 12 × 2 = 24 passes |
| TRM/UT_Level2 | n_layer=4 | 4 × 2 × 3 = 24 passes |

### Running Full Experiment Suite

Run all models × all tasks with compute normalization using both GPUs:

```bash
cd /home/azureuser/icml
chmod +x ./experiments/exp1.sh

# Run in tmux (detached)
tmux new-session -d -s experiments './experiments/exp1.sh'

# Monitor progress
tmux attach -t experiments  # Ctrl+B, D to detach

# Check logs
tail -f experiments/logs/run_*.log
```

The script runs 7 models × 3 tasks = 21 experiments, using both GPUs in parallel.

**Configuration (edit `experiments/exp1.sh`):**
```bash
COMPUTE_BUDGET=24           # Target block passes
ALGO_TRAIN_LEN=20           # Training length
ALGO_EVAL_LENGTHS="20 40 60 80 100"  # Evaluation lengths
MAX_STEPS=6500              # Training steps
```

Results are logged to W&B project: `icml-recursive-llms`

---
Summary of Changes
1. Model Changes (4 files)
Added max_act_steps parameter to all ACT models, decoupling inference compute from n_layer:
models/ut.py - UT now accepts max_act_steps (defaults to n_layer for backward compatibility)
models/ut_level1.py - Same change
models/ut_level2.py - Same change
models/trm.py - Same change
Key changes in each model:
# New parametermax_act_steps: int = None# Step embedding table now sized to max_act_stepsself.step_embedding_table = nn.Embedding(self.max_act_steps, n_embd)# Forward loop uses max_act_stepsfor step in range(self.max_act_steps):
2. Training Script (train.py)
Added two new command-line arguments:
--max-act-steps - Override max ACT steps for recurrent models
--ponder-cost-weight - Override ponder cost weight (set to 0 to disable penalty)
3. New Experiment Script (experiments/exp2_param_normalized.sh)
Parameter-normalized experiments with:
MAX_ACT_STEPS=64 (vs 4-24 in exp1)
PONDER_COST_WEIGHT=0.0 (no penalty for thinking)
MAX_STEPS=15000 (longer training)
Same architecture for all shared-block models (~1.9M params)
4. Inference Scaling Eval Script (eval_inference_scaling.py)
New script to test trained checkpoints with varying inference budgets:
python eval_inference_scaling.py \    --ckpt-dir /mnt/pdata/pr501/icml2025/checkpoints \    --max-act-steps 8 16 32 64 128 \    --output results_inference_scaling.csv
How to Run
Run the new parameter-normalized experiments:
cd /home/azureuser/icml./experiments/exp2_param_normalized.sh
Evaluate existing checkpoints with inference scaling:
python eval_inference_scaling.py \    --ckpt-dir /mnt/pdata/pr501/icml2025/checkpoints \    --max-act-steps 8 16 32 64 128
The output CSV can be used to create compute-efficiency frontier plots (inference compute on x-axis, accuracy on y-axis).
