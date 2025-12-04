# eval.py
import argparse
import re

import torch
import pytorch_lightning as pl
from datasets import load_dataset as hf_load_dataset  # HuggingFace datasets

from models import build_model
from data_modules import load_dataset as load_lm_dataset  # local loaders


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

    # GSM8K-specific eval knobs
    parser.add_argument("--gsm8k-max-examples", type=int, default=200)
    parser.add_argument("--gsm8k-max-new-tokens", type=int, default=64)

    return parser.parse_args()


# ---------------------------
# Core LM evaluation (loss)
# ---------------------------
def evaluate_lm_loss(base_model, val_loader, ckpt_path):
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
    )
    results = trainer.validate(base_model, val_loader, ckpt_path=ckpt_path)
    # Lightning returns a list; take first dict
    return results[0] if results else {}


# ---------------------------
# Task-specific: GSM8K
# ---------------------------

def extract_numeric_answer(text: str):
    """
    Extract the final integer in the string; returns None if none found.
    GSM8K answers often end with '#### 42'; this is robust to format noise.
    """
    matches = re.findall(r"-?\d+", text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def evaluate_gsm8k_accuracy(
    model,
    tokenizer,
    data_dir: str,
    block_size: int,
    device,
    max_examples: int,
    max_new_tokens: int,
):
    """
    Evaluate GSM8K correctness on the HF 'openai/gsm8k' test split.

    - Prompt: 'Question: {q}\\nAnswer:'
    - Generate up to max_new_tokens characters.
    - Compare final numeric answer to gold.
    """
    ds = hf_load_dataset("openai/gsm8k", "main", cache_dir=data_dir)
    test_split = ds["test"]

    n = min(max_examples, len(test_split))
    correct = 0
    total = 0

    model.eval()

    for i in range(n):
        ex = test_split[i]
        q = ex["question"]
        gold_answer_text = ex["answer"]

        prompt = f"Question: {q}\nAnswer:"
        prompt_ids = tokenizer.encode(prompt)
        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        # Truncate if prompt is longer than block_size
        if idx.shape[1] > block_size:
            idx = idx[:, -block_size:]

        with torch.no_grad():
            out_ids = model.generate(idx, max_new_tokens=max_new_tokens)[0].tolist()

        # Slice off the prompt part to get only the generated answer text.
        # Note: if you truncated the prompt, you should re-encode to recompute length.
        prompt_len = len(prompt_ids)
        gen_answer_ids = out_ids[prompt_len:]
        gen_answer_text = tokenizer.decode(gen_answer_ids)

        gold_num = extract_numeric_answer(gold_answer_text)
        pred_num = extract_numeric_answer(gen_answer_text)

        if gold_num is not None and pred_num is not None and gold_num == pred_num:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    return {
        "gsm8k_accuracy": accuracy,
        "gsm8k_correct": correct,
        "gsm8k_total": total,
    }


# ---------------------------
# Main
# ---------------------------

def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # 1) LM dataset (for val loader + tokenizer)
    _, val_loader, tokenizer, vocab_size = load_lm_dataset(
        args.dataset,
        data_dir=args.data_dir,
        block_size=args.block_size,
        batch_size=args.batch_size,
        eval_iters=args.eval_iters,
        seed=args.seed,
    )

    # 2) Base model skeleton for LM loss evaluation
    base_model = build_model(
        args.model,
        vocab_size=vocab_size,
        block_size=args.block_size,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_layer=args.n_layer,
        dropout=args.dropout,
        lr=args.lr,
    )

    print(f"Evaluating {args.model} on {args.dataset} from {args.ckpt_path}")

    # 3) LM loss evaluation via Lightning
    lm_results = evaluate_lm_loss(base_model, val_loader, args.ckpt_path)
    print("Validation (LM) results:", lm_results)

    # 4) Load full model for generation and task-level evaluation
    model = type(base_model).load_from_checkpoint(
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

    # 5) Simple generation from an all-zero context (for qualitative inspection)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=args.gen_tokens)[0].tolist()
    text = tokenizer.decode(out)
    print("\n--- Sample generation ---")
    print(text)

    # 6) Task-specific evaluation: GSM8K accuracy if applicable
    if args.dataset.lower() == "gsm8k_char":
        gsm8k_metrics = evaluate_gsm8k_accuracy(
            model=model,
            tokenizer=tokenizer,
            data_dir=args.data_dir,
            block_size=args.block_size,
            device=device,
            max_examples=args.gsm8k_max_examples,
            max_new_tokens=args.gsm8k_max_new_tokens,
        )
        print(
            f"\n--- GSM8K numeric accuracy on {gsm8k_metrics['gsm8k_total']} examples ---"
        )
        print(
            f"Correct: {gsm8k_metrics['gsm8k_correct']}/"
            f"{gsm8k_metrics['gsm8k_total']}  "
            f"Accuracy: {gsm8k_metrics['gsm8k_accuracy']:.4f}"
        )


if __name__ == "__main__":
    main()
