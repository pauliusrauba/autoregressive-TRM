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
    
    # ACT-specific parameters for recurrent models (UT, UTLevel1, UTLevel2, TRM)
    parser.add_argument(
        "--max-act-steps",
        type=int,
        default=None,
        help="Maximum ACT steps for recurrent models. Decouples inference compute from n_layer. "
             "If not specified, defaults to n_layer.",
    )
    parser.add_argument(
        "--ponder-cost-weight",
        type=float,
        default=None,
        help="Weight for ponder cost in ACT loss. Set to 0 to disable ponder penalty. "
             "If not specified, uses model default (0.01).",
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
        "--algo-eval-lengths",
        type=int,
        nargs="+",
        default=None,
        help="Evaluation lengths (e.g., --algo-eval-lengths 20 40 60 80 100). "
             "If not specified, defaults to [algo_train_len, 5*algo_train_len].",
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
    
    # Add ACT-specific parameters for recurrent models
    if args.max_act_steps is not None:
        model_kwargs['max_act_steps'] = args.max_act_steps
    if args.ponder_cost_weight is not None:
        model_kwargs['ponder_cost_weight'] = args.ponder_cost_weight

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
    if args.max_act_steps is not None:
        print(f"Max ACT Steps: {args.max_act_steps}")
    if args.ponder_cost_weight is not None:
        print(f"Ponder Cost Weight: {args.ponder_cost_weight}")

# Weights & Biases logger
    wandb_logger = WandbLogger(
        project="icml-recursive-llms",
        name = args.run_name if args.run_name is not None else f"{args.model}-{args.dataset}-train-{args.algo_train_len}-compute-{args.compute_budget}",
        config=vars(args),
    )

    # 3) Checkpointing
    data_dir = "/mnt/pdata/pr501/icml2025"
    run_name = args.run_name if args.run_name else f"{args.model}_{args.dataset}"
    ckpt_dir = os.path.join(data_dir, "checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Checkpoint every 2000 steps
    periodic_checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="step={step:05d}",
        every_n_train_steps=2000,
        save_top_k=-1,  # Keep all periodic checkpoints
    )

    # Also save last checkpoint
    last_checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="last",
        save_last=True,
    )

    # Get callbacks
    callbacks = [periodic_checkpoint_cb, last_checkpoint_cb]
    if args.dataset in ("copy_char", "reverse_char", "addition_char"):
        # Determine task name from dataset
        task_map = {
            "copy_char": "copy",
            "reverse_char": "reverse", 
            "addition_char": "addition"
        }
        task = task_map[args.dataset]
        
        # Determine evaluation lengths
        if args.algo_eval_lengths is not None:
            eval_lengths = args.algo_eval_lengths
        else:
            # Default: training length and 5x training length
            eval_lengths = [args.algo_train_len, 5 * args.algo_train_len]
        
        # Create eval specs for each length
        algo_specs = [
            AlgoEvalSpec(task=task, length=length, n=args.algo_eval_n)
            for length in eval_lengths
        ]
        algo_callback = AlgorithmicEvalCallback(specs=algo_specs, seed=args.seed)
        callbacks.append(algo_callback)
        
        # Checkpoint based on best task performance (seq_acc at training length)
        perf_metric = f"TaskEvaluation/{task}/L{args.algo_train_len}/seq_acc"
        best_perf_checkpoint_cb = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best_seq_acc",
            monitor=perf_metric,
            mode="max",  # Maximize accuracy
            save_top_k=1,
            save_on_train_epoch_end=False,  # Save after validation, not training epoch
        )
        callbacks.append(best_perf_checkpoint_cb)

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

    print(f"Checkpoints saved to: {ckpt_dir}")


if __name__ == "__main__":
    main()
