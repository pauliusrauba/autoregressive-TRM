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
  --dataset addition_char \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
  --algo-train-len 20 \
  --dropout 0.1 \
  --gpu 1 \
  --algo-eval-extrap-len 40

# Train TRM on addition task
uv run python train.py \
  --model trm \
  --dataset addition_char \
  --n-head 6 \
  --n-layer 6 \
  --block-size 256 \
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
