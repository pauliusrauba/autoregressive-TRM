# datasets/gsm8k_char.py
import os
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader

from datasets import load_dataset  # pip install datasets


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
        return "".join([self.itos[i] for i in ids])


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
        if worker_info is not None:
            gen.manual_seed(self.seed + worker_info.id)
        else:
            gen.manual_seed(self.seed)

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


def _format_example(question: str, answer: str) -> str:
    # You can tweak this template later if you want to force "#### final_answer" style.
    return f"Question: {question}\nAnswer: {answer}\n\n"


def load_gsm8k_char(
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    seed: int,
):
    """
    Load GSM8K as a single long char sequence for LM training.

    - Uses Hugging Face 'openai/gsm8k', config 'main'.
    - Train split -> training text; test split -> validation text.
    """
    os.makedirs(data_dir, exist_ok=True)

    ds = load_dataset("openai/gsm8k", "main", cache_dir=data_dir)
    train_split = ds["train"]
    test_split = ds["test"]

    # Build text blocks
    train_blocks = [
        _format_example(ex["question"], ex["answer"]) for ex in train_split
    ]
    val_blocks = [
        _format_example(ex["question"], ex["answer"]) for ex in test_split
    ]

    train_text = "".join(train_blocks)
    val_text = "".join(val_blocks)

    # Build char vocab from both splits to avoid OOV at test time
    full_text = train_text + val_text
    chars = sorted(list(set(full_text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer = CharTokenizer(stoi=stoi, itos=itos)

    # Encode to tensors
    train_data = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
    val_data = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

    # Datasets / loaders
    train_ds = RandomChunkTrain(train_data, block_size=block_size, seed=seed)
    val_len = eval_iters * batch_size
    val_ds = RandomChunkEval(val_data, block_size=block_size, length=val_len)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer
