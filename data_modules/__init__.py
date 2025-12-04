# data_modules/__init__.py
from .shakespeare_char import load_shakespeare_char
from .gsm8k_char import load_gsm8k_char


def load_dataset(
    name: str,
    *,
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    seed: int,
):
    name = name.lower()
    if name == "shakespeare_char":
        train_loader, val_loader, tokenizer = load_shakespeare_char(
            data_dir=data_dir,
            block_size=block_size,
            batch_size=batch_size,
            eval_iters=eval_iters,
            seed=seed,
        )
        vocab_size = tokenizer.vocab_size
        return train_loader, val_loader, tokenizer, vocab_size

    elif name == "gsm8k_char":
        train_loader, val_loader, tokenizer = load_gsm8k_char(
            data_dir=data_dir,
            block_size=block_size,
            batch_size=batch_size,
            eval_iters=eval_iters,
            seed=seed,
        )
        vocab_size = tokenizer.vocab_size
        return train_loader, val_loader, tokenizer, vocab_size

    else:
        raise ValueError(f"Unknown dataset name: {name}")
