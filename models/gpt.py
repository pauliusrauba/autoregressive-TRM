# lit_gpt.py
import os
import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# Hyperparameters 
# --------------------
batch_size = 64
block_size = 256
max_iters = 6500
eval_interval = 500
learning_rate = 3e-4
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
seed = 1337
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------------------
# Data: load + tokenize
# --------------------
torch.manual_seed(seed)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(s: str):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

checkpoint_cb = ModelCheckpoint(
    dirpath="checkpoints/shakespeare-gpt",
    filename="{epoch:02d}-step={step}-val={val_loss:.4f}",
    monitor="val_loss",
    mode="min",
    save_top_k=1,      # keep only the best
    save_last=True,    # also keep last.ckpt
)

# --------------------
# Datasets
# --------------------
class RandomChunkTrain(IterableDataset):
    """Infinite iterator for training; each sample is a random (x, y) chunk."""
    def __init__(self, data_tensor: torch.Tensor, block_size: int):
        super().__init__()
        self.data = data_tensor
        self.block_size = block_size

    def __iter__(self):
        # worker-safe RNG
        worker_info = torch.utils.data.get_worker_info()
        gen = torch.Generator()
        if worker_info is not None:
            gen.manual_seed(seed + worker_info.id)
        else:
            gen.manual_seed(seed)

        data_len = len(self.data)
        while True:
            i = torch.randint(low=0, high=data_len - self.block_size - 1, size=(1,), generator=gen).item()
            x = self.data[i:i+self.block_size]
            y = self.data[i+1:i+self.block_size+1]
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
        # idx unused; sample randomly each time
        i = torch.randint(low=0, high=len(self.data) - self.block_size - 1, size=(1,)).item()
        x = self.data[i:i+self.block_size]
        y = self.data[i+1:i+self.block_size+1]
        return x, y

# --------------------
# Model components 
# --------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        out = wei @ v      # (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

# --------------------
# LightningModule
# --------------------
class LitGPT(pl.LightningModule):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, lr):
        super().__init__()
        self.save_hyperparameters()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)                       # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)                    # (T, C)
        x = tok_emb + pos_emb                                           # (B, T, C)
        x = self.blocks(x)                                              # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                                        # (B, T, V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.block_size:]
            logits, _ = self(idx_cond) # (B, T, V)
            logits = logits[:, -1, :]          # (B, V)
            probs = F.softmax(logits, dim=-1)  # (B, V)
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)             # (B, T+1)
        return idx

    # Lightning hooks
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits, loss = self(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.size(0))

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

# --------------------
# DataLoaders
# --------------------
def make_dataloaders():
    train_ds = RandomChunkTrain(train_data, block_size)
    val_len = eval_iters * batch_size
    val_ds = RandomChunkEval(val_data, block_size, length=val_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    return train_loader, val_loader

# --------------------
# Main
# --------------------
def main():
    model = LitGPT(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=n_embd,
        n_head=n_head,
        n_layer=n_layer,
        dropout=dropout,
        lr=learning_rate,
    )

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"{total_params:.2f} M parameters")

    train_loader, val_loader = make_dataloaders()

    # Match your step-based training and eval cadence
    trainer = pl.Trainer(
        max_steps=max_iters,
        val_check_interval=eval_interval,  # validate every N training steps
        log_every_n_steps=50,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        enable_progress_bar=True,
        callbacks=[checkpoint_cb]
    )

    trainer.fit(model, train_loader, val_loader)

    # --------------------
    # Generation demo
    # --------------------
    model = model.to(device)
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, max_new_tokens=500)[0].tolist()
    print(decode(out))

if __name__ == "__main__":
    main()
