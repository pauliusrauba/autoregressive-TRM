# train.py
import os
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from models import build_model
from data_modules import load_dataset



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--dataset", type=str, default="shakespeare_char")
    parser.add_argument("--data-dir", type=str, default="data")
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

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=6500)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    
    # train.py parse_args()
    parser.add_argument("--algo-train-len", type=int, default=40)
    parser.add_argument("--algo-val-len", type=int, default=40)
    parser.add_argument("--algo-train-examples", type=int, default=50000)
    parser.add_argument("--algo-val-examples", type=int, default=5000)

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
    model = build_model(
        args.model,
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        lr=args.lr,
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model: {args.model}, Dataset: {args.dataset}")
    print(f"Parameters: {total_params:.2f}M")

    # Weights & Biases logger
    wandb_logger = WandbLogger(
        project="icml-recursive-llms",
        name=args.run_name or f"{args.model}-{args.dataset}",
        config=vars(args),
    )

    # 3) Checkpointing
    ckpt_dir = os.path.join("checkpoints", f"{args.dataset}_{args.model}")
    os.makedirs(ckpt_dir, exist_ok=True)

    checkpoint_cb = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="{epoch:02d}-step={step}-val={val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    # 4) Trainer
    trainer = pl.Trainer(
        max_steps=args.max_steps,
        val_check_interval=args.eval_interval,
        log_every_n_steps=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
        logger=wandb_logger,
        callbacks=[checkpoint_cb],
    )

    # 5) Train
    trainer.fit(model, train_loader, val_loader)

    print(f"Best checkpoint saved to: {checkpoint_cb.best_model_path}")


if __name__ == "__main__":
    main()
