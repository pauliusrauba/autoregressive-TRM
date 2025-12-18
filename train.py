# train.py
import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from callbacks.algorithmic_eval import AlgorithmicEvalCallback, AlgoEvalSpec

from models import build_model, normalize_model_kwargs_for_compute, calculate_block_passes
from data_modules import load_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--dataset", type=str, default="shakespeare_char")
    parser.add_argument("--data-dir", type=str, default="/mnt/pdata/pr501/icml2025")
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional W&B run name (defaults to '<model>-<dataset>')",
    )

    # Model hyperparameters
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)

    # Unified GPT/UT levels (only used if args.model == "unified_gpt_ut")
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help="Unified GPT/UT level: 0=GPT, 1=recurrent, 2=+learned step, 3=+sinusoidal 2D, 4=UT step, 5=+ACT, 6=+ponder",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Number of recurrent refinement steps for levels 1-4. Defaults to n_layer if omitted.",
    )
    parser.add_argument(
        "--ut-max-steps",
        type=int,
        default=16,
        help="Max steps for learned step embeddings (level 2) and ACT loop (levels 5-6).",
    )
    parser.add_argument(
        "--act-threshold",
        type=float,
        default=0.99,
        help="ACT halting threshold (levels 5-6).",
    )
    parser.add_argument(
        "--act-loss-weight",
        type=float,
        default=0.01,
        help="Ponder cost coefficient (level 6).",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., 0 or 1). If not specified, uses first available GPU.",
    )
    
    # Loop parameters for UTLevel2/TRM models
    parser.add_argument(
        "--n-inner-loops",
        type=int,
        default=None,
        help="Number of inner loops for UTLevel2/TRM (default: 4 for UTLevel2, 2 for TRM)",
    )
    parser.add_argument(
        "--n-outer-loops",
        type=int,
        default=None,
        help="Number of outer loops for UTLevel2/TRM (default: 4 for UTLevel2, 2 for TRM)",
    )
    
    # Compute budget for fair comparison experiments
    parser.add_argument(
        "--compute-budget",
        type=int,
        default=None,
        help="Target compute budget in block passes. When set, adjusts n_layer (and loops for UTLevel2/TRM) "
             "to match this budget across different models. Leave unset to use original parameters.",
    )


    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=6500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)

    # Algorithmic dataset args
    parser.add_argument("--algo-train-len", type=int, default=40)
    parser.add_argument("--algo-val-len", type=int, default=40)
    parser.add_argument("--algo-train-examples", type=int, default=50000)
    parser.add_argument("--algo-val-examples", type=int, default=5000)

    # Algorithmic eval callback args
    parser.add_argument(
        "--algo-eval-extrap-len",
        type=int,
        default=400,
        help="Sequence length for extrapolation evaluation (default: 400)",
    )
    parser.add_argument(
        "--algo-eval-n",
        type=int,
        default=100,
        help="Number of samples per evaluation spec (default: 100)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(args.seed)

    # 1) Dataset
    train_loader, val_loader, tokenizer, vocab_size = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        seed=args.seed,
        algo_train_len=args.algo_train_len,
        algo_val_len=args.algo_val_len,
        algo_train_examples=args.algo_train_examples,
        algo_val_examples=args.algo_val_examples,
    )

    # 2) Model
    model_kwargs = dict(
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        lr=args.lr,
    )
    
    # Add loop parameters for UTLevel2/TRM if specified
    if args.n_inner_loops is not None:
        model_kwargs['n_inner_loops'] = args.n_inner_loops
    if args.n_outer_loops is not None:
        model_kwargs['n_outer_loops'] = args.n_outer_loops

    # Apply compute budget normalization if specified
    model_kwargs, compute_summary = normalize_model_kwargs_for_compute(
        args.model,
        model_kwargs,
        compute_budget=args.compute_budget, # This becomes the effective n_layers
    )

    model = build_model(args.model, **model_kwargs)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Parameters: {total_params:.2f}M")
    print(f"Compute: {compute_summary}")

# Weights & Biases logger
    wandb_logger = WandbLogger(
        project="icml-recursive-llms",
        name = args.run_name if args.run_name is not None else f"{args.model}-{args.dataset}-train-{args.algo_train_len}-eval-{args.algo_eval_extrap_len}",
        config=vars(args),
    )

    # 3) Checkpointing
    data_dir = "/mnt/pdata/pr501/icml2025"
    ckpt_dir = os.path.join(data_dir, "checkpoints", f"{args.dataset}_{args.model}")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # Get callbacks
    callbacks = [checkpoint_cb]
    if args.dataset in ("copy_char", "reverse_char", "addition_char"):
        # Determine task name from dataset
        task_map = {
            "copy_char": "copy",
            "reverse_char": "reverse", 
            "addition_char": "addition"
        }
        task = task_map[args.dataset]
        
        # Evaluate at training length (in-distribution) and longer (extrapolation)
        algo_specs = [
            AlgoEvalSpec(task=task, length=args.algo_train_len, n=args.algo_eval_n),  # in-distribution
            AlgoEvalSpec(task=task, length=args.algo_eval_extrap_len, n=args.algo_eval_n),  # extrapolation
        ]
        algo_callback = AlgorithmicEvalCallback(specs=algo_specs, seed=args.seed)
        callbacks.append(algo_callback)

    # Determine devices based on --gpu argument
    if torch.cuda.is_available():
        devices = [args.gpu] if args.gpu is not None else 1
        accelerator = "gpu"
    else:
        devices = -1
        accelerator = "cpu"

    # 4) Trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        val_check_interval=args.eval_interval,
        log_every_n_steps=50,
        accelerator=accelerator,
        devices=devices,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    # 5) Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Best checkpoint saved to: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
