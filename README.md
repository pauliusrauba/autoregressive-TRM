# Tiny Autoregressive Recursive Models

Official code for the paper:

> **Tiny Autoregressive Recursive Models**
> Paulius Rauba, Claudio Fanconi, Mihaela van der Schaar
> *ICLR 2026 Workshop on AI with Recursive Self-Improvement* (Spotlight)
>
> [[Paper]](https://openreview.net/forum?id=aY5kmaNrwB)

We extend the Tiny Recursive Model (TRM) to the autoregressive setting and perform a controlled, compute-matched comparison against a suite of Transformer variants on character-level algorithmic tasks. We find that two-level refinement baselines show strong performance, but the full Autoregressive TRM architecture does not yield reliable gains — offering promise for refinement mechanisms broadly while cautioning against the autoregressive TRM specifically.

## Models

The paper compares seven architectures that gradually transform a standard Transformer into an Autoregressive TRM. All share the same Transformer block; they differ only in how blocks are applied:

| `--model`    | Architecture                       | Block Passes                             |
|--------------|------------------------------------|------------------------------------------|
| `gpt`        | Dense Transformer (unique blocks)  | `n_layer`                                |
| `gpt_level1` | Iterative (weight-shared blocks)  | `n_layer`                                |
| `gpt_level2` | + learnable step embeddings       | `n_layer`                                |
| `ut`         | Universal Transformer (ACT)        | up to `n_layer`                          |
| `ut_level1`  | + dual-stream reasoning/solution  | `2 × n_layer`                            |
| `ut_level2`  | + nested inner/outer loops        | `n_layer × n_outer × (n_inner + 1)`     |
| `trm`        | Full Autoregressive TRM           | `n_layer × n_outer × (n_inner + 1)`     |

**Block passes** = total transformer block applications per forward pass — the compute measure used to compare models fairly.

## Setup

1. Install [uv](https://docs.astral.sh/uv/):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Install dependencies:
   ```bash
   uv sync
   ```

3. Verify:
   ```bash
   uv run python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
   ```

On HPC with limited home disk, use `./setup.sh /path/to/data` to symlink `.venv` and caches to a data mount.

## Quick Start

```bash
uv run python train.py --model gpt --dataset addition_char --compute-budget 24 --gpu 0
```

This trains a GPT baseline on 10-digit addition with 24 block passes, logging metrics to [Weights & Biases](https://wandb.ai).

## Project Structure

```
trm-llm/
├── train.py                       # Training entry point (all models, all tasks)
├── eval_algorithmic.py            # Evaluate a single checkpoint on one task/length
├── eval_inference_scaling.py      # Batch-evaluate checkpoints at varying ACT steps → CSV
│
├── models/                        # Model implementations
│   ├── gpt.py                     # Dense Transformer (GPT-2 baseline)
│   ├── gpt_level1.py              # + weight sharing (iterative)
│   ├── gpt_level2.py              # + step embeddings
│   ├── ut.py                      # Universal Transformer (ACT halting)
│   ├── ut_level1.py               # + dual-stream (reasoning/solution)
│   ├── ut_level2.py               # + nested inner/outer loops
│   ├── trm.py                     # Full Autoregressive TRM
│   └── common/                    # Shared: layers, ACT, recurrence, compute utils
│
├── data_modules/                  # Dataset loaders
│   ├── addition_char.py           # Integer addition
│   ├── copy_char.py               # Sequence copy
│   ├── reverse_char.py            # Sequence reversal
│   ├── shakespeare_char.py        # Character-level text generation
│   └── gsm8k_char.py             # Math word problems (GSM8K)
│
├── callbacks/                     # PyTorch Lightning callbacks
│   └── algorithmic_eval.py        # Periodic eval at multiple lengths → W&B
│
├── experiments/                   # Shell scripts that launch full experiment suites
│   ├── exp_length_generalization.sh
│   ├── exp_compute_optimal.sh
│   ├── exp_train_length_scaling.sh
│   ├── exp_param_inference_scaling.sh
│   └── exp_hl_structure.sh
│
└── notebooks/                     # Data export & visualization
    ├── export_wandb_logs.py       # Export one experiment from W&B → CSV
    ├── export_all_experiments.py  # Export all experiments from W&B → CSV
    ├── data/                      # Exported CSVs (generated, not checked in)
    └── *.ipynb                    # Analysis & figure notebooks
```

## Pipeline Overview

The workflow has three stages:

```
1. Train          2. Export              3. Visualize
experiments/*.sh  export_wandb_logs.py   notebooks/*.ipynb
    │                  │                      │
    ▼                  ▼                      ▼
 train.py ──► W&B ──► notebooks/data/*.csv ──► figures
```

**Stage 1 — Train.** Experiment scripts launch `train.py` for each model × task combination. All metrics go to W&B (project `icml-recursive-llms`). Checkpoints are saved locally.

**Stage 2 — Export.** `notebooks/export_all_experiments.py` pulls finished runs from W&B and writes `{name}_summary.csv` (one row per run) and `{name}_history.csv` (one row per eval step) into `notebooks/data/`.

**Stage 3 — Visualize.** Jupyter notebooks in `notebooks/` load the CSVs and produce paper figures.

## Reproducing Paper Results

### 1. Run the experiments

Each script launches a full model × task grid and manages GPU scheduling:

| Script | Paper Section | What it tests |
|--------|---------------|---------------|
| `exp_length_generalization.sh` | Length generalization | Train on length 10, eval on 10–50 |
| `exp_compute_optimal.sh` | Compute-optimal comparison | Sweep compute budgets across models |
| `exp_train_length_scaling.sh` | Training length scaling | Vary training length (2, 5, 10, 20, 40) |
| `exp_param_inference_scaling.sh` | Inference scaling | Scale inference compute (ACT steps) |
| `exp_hl_structure.sh` | Architecture ablation | Vary hidden-layer structure |

Run in tmux for long experiments:

```bash
tmux new-session -d -s lengthgen './experiments/exp_length_generalization.sh'
tmux attach -t lengthgen
```

### 2. Export the results

```bash
uv run python notebooks/export_all_experiments.py
```

This produces pairs of files in `notebooks/data/`:
- `{name}_summary.csv` — one row per finished run (final metrics)
- `{name}_history.csv` — one row per eval step per run (full training curves)

### 3. Generate figures

Open the Jupyter notebooks in `notebooks/` and run them against the exported CSVs.

## Training Details

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--model` | Model architecture (see table above) | `gpt` |
| `--dataset` | Task: `addition_char`, `copy_char`, `reverse_char` | `shakespeare_char` |
| `--compute-budget` | Target block passes; auto-adjusts `n_layer` and loop counts per model | None |
| `--algo-train-len` | Training sequence length for algorithmic tasks | `40` |
| `--algo-eval-lengths` | Lengths to evaluate during training (space-separated) | `[train_len, 5×train_len]` |
| `--max-act-steps` | Max ACT iterations at inference (recurrent models only) | `n_layer` |
| `--ponder-cost-weight` | ACT ponder penalty (0 = no penalty) | `0.01` |
| `--no-step-embeddings` | Disable step embeddings (enables clean compute extrapolation) | off |
| `--gpu` | GPU device ID | auto |

### Compute-Normalized Comparison

The `--compute-budget` flag ensures all models use the same number of block passes:

```bash
uv run python train.py --model gpt  --dataset addition_char --compute-budget 24 --gpu 0  # n_layer=24
uv run python train.py --model ut   --dataset addition_char --compute-budget 24 --gpu 0  # n_layer=24
uv run python train.py --model trm  --dataset addition_char --compute-budget 24 \
  --n-inner-loops 2 --n-outer-loops 2 --gpu 0                                            # n_layer=4 (4×2×3=24)
```

### Length Generalization

Train on short sequences, evaluate on longer ones:

```bash
uv run python train.py --model trm --dataset addition_char \
  --algo-train-len 10 --algo-eval-lengths 10 20 30 40 50 \
  --compute-budget 24 --gpu 0
```

### Checkpoints

Saved to `{data-dir}/checkpoints/{run-name}/`:

| File | Description |
|------|-------------|
| `last.ckpt` | Most recent checkpoint |
| `step=XXXXX.ckpt` | Periodic (every `max_steps/10` steps) |
| `best_seq_acc.ckpt` | Best sequence accuracy at training length |

## Evaluation Scripts

For post-hoc evaluation of saved checkpoints:

```bash
# Single checkpoint, single task/length (prints to stdout)
uv run python eval_algorithmic.py --ckpt path/to/last.ckpt --task addition --length 100

# Batch evaluation with inference scaling (writes CSV)
uv run python eval_inference_scaling.py \
  --ckpt-dir /path/to/checkpoints \
  --max-act-steps 8 16 32 64 128 \
  --tasks copy reverse addition \
  --lengths 20 40 60 80 100 \
  --output results_inference_scaling.csv
```

## Citation

```bibtex
@inproceedings{rauba2026tiny,
  title={Tiny Autoregressive Recursive Models},
  author={Rauba, Paulius and Fanconi, Claudio and van der Schaar, Mihaela},
  booktitle={ICLR 2026 Workshop on AI with Recursive Self-Improvement},
  year={2026},
  url={https://openreview.net/forum?id=aY5kmaNrwB}
}
```
