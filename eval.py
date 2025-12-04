# eval.py
import argparse
import torch
import pytorch_lightning as pl

from models import build_model
from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt")
    parser.add_argument("--dataset", type=str, default="shakespeare_char")
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--ckpt-path", type=str, required=True)

    # Must match training hyperparameters (or be stored in ckpt; here we re-specify)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--n-embd", type=int, default=384)
    parser.add_argument("--n-head", type=int, default=6)
    parser.add_argument("--n-layer", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-iters", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)

    parser.add_argument("--gen-tokens", type=int, default=500)

    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Dataset (for val loader + tokenizer)
    _, val_loader, tokenizer, vocab_size = load_dataset(
        args.dataset,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )

    # Model
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

    # Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )

    # Validate
    print(f"Evaluating {args.model} on {args.dataset} from {args.ckpt_path}")
    results = trainer.validate(model, val_loader, ckpt_path=args.ckpt_path)
    print("Validation results:", results)

    # Load checkpoint weights into the model object for generation
    model = type(model).load_from_checkpoint(
        args.ckpt_path,
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        lr=args.lr,
    )
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Simple generation from an all-zero context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.gen_tokens)[0].tolist()
    text = tokenizer.decode(out)
    print("\n--- Sample generation ---")
    print(text)


if __name__ == "__main__":
    main()
