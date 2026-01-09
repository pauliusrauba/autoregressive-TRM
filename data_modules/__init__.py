# data_modules/__init__.py
from .shakespeare_char import load_shakespeare_char
from .gsm8k_char import load_gsm8k_char
from .copy_char import load_copy_char
from .reverse_char import load_reverse_char
from .addition_char import load_addition_char
from typing import Optional

def load_dataset(
    name: str,
    *,
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    seed: int,
    algo_train_len: Optional[int] = None,
    algo_val_len: Optional[int] = None,
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

    elif name == "copy_char":
            train_loader, val_loader, tokenizer = load_copy_char(
                data_dir=data_dir,
                block_size=block_size,
                batch_size=batch_size,
                eval_iters=eval_iters,
                seed=seed,
                train_seq_len=algo_train_len if algo_train_len is not None else 40,
                val_seq_len=algo_val_len if algo_val_len is not None else 40,
            )
            return train_loader, val_loader, tokenizer, tokenizer.vocab_size

    elif name == "reverse_char":
        train_loader, val_loader, tokenizer = load_reverse_char(
            data_dir=data_dir,
            block_size=block_size,
            batch_size=batch_size,
            eval_iters=eval_iters,
            seed=seed,
            train_seq_len=algo_train_len if algo_train_len is not None else 40,
            val_seq_len=algo_val_len if algo_val_len is not None else 40,
        )
        return train_loader, val_loader, tokenizer, tokenizer.vocab_size

    elif name == "addition_char":
        train_loader, val_loader, tokenizer = load_addition_char(
            data_dir=data_dir,
            block_size=block_size,
            batch_size=batch_size,
            eval_iters=eval_iters,
            seed=seed,
            train_seq_len=algo_train_len if algo_train_len is not None else 40,
            val_seq_len=algo_val_len if algo_val_len is not None else 40,
        )
        return train_loader, val_loader, tokenizer, tokenizer.vocab_size

    else:
        raise ValueError(f"Unknown dataset name: {name}")
