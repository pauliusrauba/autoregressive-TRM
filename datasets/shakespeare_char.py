# datasets/shakespeare_char.py
import os
import urllib.request
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader


SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


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


def download_shakespeare(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    path = os.path.join(data_dir, "tinyshakespeare_input.txt")
    if not os.path.exists(path):
        print(f"Downloading tiny Shakespeare to {path}...")
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path


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


def load_shakespeare_char(
    data_dir: str,
    block_size: int,
    batch_size: int,
    eval_iters: int,
    seed: int,
):
    # 1) Download / load raw text
    path = download_shakespeare(data_dir)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    # 2) Build tokenizer
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    tokenizer = CharTokenizer(stoi=stoi, itos=itos)

    # 3) Encode full text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]

    # 4) Datasets / loaders
    train_ds = RandomChunkTrain(train_data, block_size=block_size, seed=seed)
    val_len = eval_iters * batch_size
    val_ds = RandomChunkEval(val_data, block_size=block_size, length=val_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
    )

    return train_loader, val_loader, tokenizer
