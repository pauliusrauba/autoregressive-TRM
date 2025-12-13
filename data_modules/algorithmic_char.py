# datasets/algorithmic_char.py
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


@dataclass
class CharTokenizer:
    stoi: dict
    itos: dict

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, s: str):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join([self.itos[int(i)] for i in ids])


class RandomChunkTrain(IterableDataset):
    """Infinite iterator for training; each sample is a random (x, y) chunk."""
    def __init__(self, data_tensor: torch.Tensor, block_size: int, seed: int):
        super().__init__()
        self.data = data_tensor
        self.block_size = block_size
        self.seed = seed

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        gen = torch.Generator()
        gen.manual_seed(self.seed + worker_info.id if worker_info else self.seed)

        data_len = len(self.data)
        while True:
            i = torch.randint(
                low=0,
                high=data_len - self.block_size - 1,
                size=(1,),
                generator=gen,
            ).item()
            x = self.data[i : i + self.block_size]
            y = self.data[i + 1 : i + self.block_size + 1]
            yield x, y


class RandomChunkEval(Dataset):
    """Finite-length dataset for validation to ensure eval finishes."""
    def __init__(self, data_tensor: torch.Tensor, block_size: int, length: int):
        super().__init__()
        self.data = data_tensor
        self.block_size = block_size
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        i = torch.randint(
            low=0,
            high=len(self.data) - self.block_size - 1,
            size=(1,),
        ).item()
        x = self.data[i : i + self.block_size]
        y = self.data[i + 1 : i + self.block_size + 1]
        return x, y


def _build_tokenizer() -> CharTokenizer:
    # Minimal vocabulary: digits plus separators used by tasks
    chars = list("0123456789|+=\n")
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return CharTokenizer(stoi=stoi, itos=itos)


def _rand_digits(gen: torch.Generator, n: int) -> str:
    # fixed-length digits with leading zeros allowed
    d = torch.randint(0, 10, (n,), generator=gen).tolist()
    return "".join(str(x) for x in d)


def _make_example(task: str, L: int, gen: torch.Generator) -> Tuple[str, str, str]:
    """
    Returns (prompt, target, full_example_text).

    Formats:
      copy:    <x>|<y>\n where y == x
      reverse: <x>|<y>\n where y == reversed(x)
      add:     <a>+<b>=<c>\n where a,b are L digits and c is (L+1) digits (zero-padded)
    """
    task = task.lower()
    if task not in {"copy", "reverse", "addition", "add"}:
        raise ValueError(f"Unknown algorithmic task: {task}")

    if task in {"copy", "reverse"}:
        x = _rand_digits(gen, L)
        y = x if task == "copy" else x[::-1]
        prompt = f"{x}|"
        target = y
        full = f"{x}|{y}\n"
        return prompt, target, full

    # addition
    a = _rand_digits(gen, L)
    b = _rand_digits(gen, L)
    c_int = int(a) + int(b)
    c = f"{c_int:0{L+1}d}"  # fixed length L+1 (pads with leading zeros)
    prompt = f"{a}+{b}="
    target = c
    full = f"{a}+{b}={c}\n"
    return prompt, target, full


def _build_corpus(task: str, L: int, n_examples: int, seed: int) -> str:
    gen = torch.Generator().manual_seed(seed)
    parts = []
    for _ in range(n_examples):
        _, _, full = _make_example(task, L, gen)
        parts.append(full)
    return "".join(parts)


def load_algorithmic_char(
    task: str,
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    seed: int,
    train_seq_len: int = 40,
    val_seq_len: int = 40,
    train_examples: int = 50000,
    val_examples: int = 5000,
):
    """
    Produces random-chunk LM training data from synthetic algorithmic examples.

    Note:
      - train_seq_len/val_seq_len are the *digit lengths* L used in examples.
      - To evaluate extrapolation (e.g. L=400) you should use a separate eval script.
    """
    os.makedirs(data_dir, exist_ok=True)
    tokenizer = _build_tokenizer()

    train_text = _build_corpus(task, train_seq_len, train_examples, seed=seed)
    val_text = _build_corpus(task, val_seq_len, val_examples, seed=seed + 1)

    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    train_ds = RandomChunkTrain(train_data, block_size=block_size, seed=seed)
    val_len = eval_iters * batch_size
    val_ds = RandomChunkEval(val_data, block_size=block_size, length=val_len)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, tokenizer
