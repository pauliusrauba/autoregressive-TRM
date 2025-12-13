# eval_algorithmic.py
import argparse
import torch

from datasets.algorithmic_char import _build_tokenizer, _make_example
from models.gpt import LitGPT  # adjust if your import path differs


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--task", type=str, choices=["copy", "reverse", "addition"], required=True)
    p.add_argument("--length", type=int, default=400)       # L in the UT paper
    p.add_argument("--n", type=int, default=200)            # number of test samples
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    tok = _build_tokenizer()

    model = LitGPT.load_from_checkpoint(args.ckpt)
    model.eval().to(args.device)

    gen = torch.Generator(device="cpu").manual_seed(args.seed)

    total_chars = 0
    correct_chars = 0
    correct_seqs = 0

    for _ in range(args.n):
        prompt, target, _full = _make_example(args.task, args.length, gen)

        # Encode prompt and generate target length
        x = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=args.device)

        out_len = args.length if args.task in {"copy", "reverse"} else (args.length + 1)
        y_ids = model.generate(x, max_new_tokens=out_len)[0].tolist()

        # Decode only the newly generated portion
        gen_ids = y_ids[len(prompt): len(prompt) + out_len]
        pred = tok.decode(gen_ids)

        # Accuracy on output only
        assert len(pred) == len(target), (len(pred), len(target))
        char_matches = sum(int(a == b) for a, b in zip(pred, target))
        correct_chars += char_matches
        total_chars += len(target)
        correct_seqs += int(pred == target)

    char_acc = correct_chars / max(1, total_chars)
    seq_acc = correct_seqs / max(1, args.n)

    print(f"Task={args.task} L={args.length} N={args.n}")
    print(f"char-acc: {char_acc:.4f}")
    print(f"seq-acc : {seq_acc:.4f}")


if __name__ == "__main__":
    main()
