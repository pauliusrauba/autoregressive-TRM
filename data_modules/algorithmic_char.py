# data_modules/algorithmic_char.py
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
    def __init__(self, task: str, L: int, tokenizer: CharTokenizer, block_size: int, seed: int):
        super().__init__()
        self.task = task
        self.L = L
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.seed = seed
        self.pad_id = tokenizer.stoi['\n']
        self.ignore_index = -100 # Standard PyTorch ignore index

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        gen = torch.Generator()
        gen.manual_seed(self.seed + worker_info.id if worker_info else self.seed)

        while True:
            # 1. Generate text
            prompt, _, text = _make_example(self.task, self.L, gen)
            
            # 2. Encode full text and prompt separately to find the split point
            token_ids = self.tokenizer.encode(text)
            prompt_len = len(self.tokenizer.encode(prompt))
            
            # 3. Handle Padding / Truncation
            needed_len = self.block_size + 1
            original_len = len(token_ids) # Store length before padding

            if len(token_ids) > needed_len:
                token_ids = token_ids[:needed_len]
            
            if len(token_ids) < needed_len:
                padding = [self.pad_id] * (needed_len - len(token_ids))
                token_ids = token_ids + padding

            data = torch.tensor(token_ids, dtype=torch.long)
            
            # 4. Create x and y
            x = data[:-1]
            y = data[1:].clone() # Clone to modify safely

            # 5. Apply Masking to y
            y[:prompt_len - 1] = self.ignore_index

            if original_len < needed_len:
                y[original_len:] = self.ignore_index

            yield x, y


class RandomChunkEval(Dataset):
    def __init__(self, task: str, L: int, tokenizer: CharTokenizer, block_size: int, length: int, seed: int):
        super().__init__()
        self.task = task
        self.L = L
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.length = length
        self.seed = seed
        self.pad_id = tokenizer.stoi['\n']
        self.ignore_index = -100

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        gen = torch.Generator()
        gen.manual_seed(self.seed + idx)
        
        prompt, _, text = _make_example(self.task, self.L, gen)
        token_ids = self.tokenizer.encode(text)
        prompt_len = len(self.tokenizer.encode(prompt))

        needed_len = self.block_size + 1
        original_len = len(token_ids)
        
        if len(token_ids) > needed_len:
            token_ids = token_ids[:needed_len]
            
        if len(token_ids) < needed_len:
            padding = [self.pad_id] * (needed_len - len(token_ids))
            token_ids = token_ids + padding

        data = torch.tensor(token_ids, dtype=torch.long)
        x = data[:-1]
        y = data[1:].clone()

        # Mask Prompt
        y[:prompt_len - 1] = self.ignore_index
        
        # Mask Padding (keep first \n)
        if original_len < needed_len:
            y[original_len:] = self.ignore_index

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
        full = f"{x}|{y}"
        return prompt, target, full

    # addition
    a = _rand_digits(gen, L)
    b = _rand_digits(gen, L)
    c_int = int(a) + int(b)
    c = f"{c_int:0{L+1}d}"  # fixed length L+1 (pads with leading zeros)
    prompt = f"{a}+{b}="
    target = c
    full = f"{a}+{b}={c}"
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
):
    os.makedirs(data_dir, exist_ok=True)
    tokenizer = _build_tokenizer()

    train_ds = RandomChunkTrain(
        task=task, 
        L=train_seq_len, 
        tokenizer=tokenizer, 
        block_size=block_size, 
        seed=seed
    )
    
    val_len = eval_iters * batch_size
    val_ds = RandomChunkEval(
        task=task, 
        L=val_seq_len, 
        tokenizer=tokenizer, 
        block_size=block_size, 
        length=val_len,
        seed=seed + 1
    )

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, tokenizer
