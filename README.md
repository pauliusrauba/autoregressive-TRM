# TRM-LLM

Codebase for TRM (Transformer with Recursive Mechanism) LLMs research project.

## Quick Start

```bash
# 1. Install UV (one-time)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# 2. Setup (see detailed instructions below)
./setup.sh  # or follow manual steps

# 3. Run training
uv run python train.py --model gpt --dataset addition_char
```

## Setup Instructions

This project uses [UV](https://docs.astral.sh/uv/) for fast, reproducible dependency management.

### Prerequisites

- Linux (tested on Ubuntu)
- ~5GB disk space for dependencies (can be on a data mount)
- CUDA-capable GPU (optional, for training)

### Step 1: Install UV

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc  # or restart your terminal
```

### Step 2: Configure Data Directory

Since dependencies are large (~5GB with PyTorch/CUDA), we store them on the data mount rather than the home directory.

**Set your data directory** (adjust the path for your setup):

```bash
# Example: /mnt/pdata/YOUR_USERNAME/icml2025
export DATA_DIR="/mnt/pdata/pr501/icml2025"
```

**Create the cache and venv directories:**

```bash
mkdir -p $DATA_DIR/.uv-cache
mkdir -p $DATA_DIR/.venv
```

**Create a symlink for the virtual environment** (from the project directory):

```bash
cd /path/to/icml
ln -sf $DATA_DIR/.venv .venv
```

**Add UV cache to your shell config** (`~/.bashrc`):

```bash
echo 'export UV_CACHE_DIR="/mnt/pdata/YOUR_USERNAME/icml2025/.uv-cache"' >> ~/.bashrc
echo 'export UV_LINK_MODE=copy' >> ~/.bashrc
source ~/.bashrc
```

### Step 3: Install Dependencies

```bash
cd /path/to/icml
uv sync
```

This reads `uv.lock` and installs the exact same package versions for reproducibility.

### Step 4: Verify Installation

```bash
uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Running Code

Always use `uv run` to execute Python scripts:

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

## Troubleshooting

### "No space left on device"

The UV cache and virtual environment need ~5GB. Make sure they're on a disk with enough space:

```bash
# Check where your .venv points
ls -la .venv

# If needed, move to data mount
rm -rf .venv
mkdir -p /your/data/mount/.venv
ln -s /your/data/mount/.venv .venv
uv sync
```

### UV command not found

Add UV to your PATH:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

## For Collaborators

1. Clone the repo
2. Follow the setup instructions above, using YOUR data directory path
3. Run `uv sync` — this installs the exact versions from `uv.lock`
4. You now have an identical environment!

The key files for reproducibility:
- `pyproject.toml` — dependency specifications
- `uv.lock` — exact pinned versions (commit this!)
