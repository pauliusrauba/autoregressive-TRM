# models/unified_gpt_ut.py
# A single LightningModule that can behave like:
#   level 0: original GPT (stacked distinct blocks, learned pos emb)
#   level 1: recurrent depth only (one shared block repeated), learned pos emb
#   level 2: level 1 + learned step/time embedding
#   level 3: level 1 but with UT-style 2D sinusoidal coordinates (pos + step)
#   level 4: explicit UT recurrent step update (u=h+p; LN(u+SA(u)); LN(a+FF(a)))
#   level 5: level 4 + ACT halting (dynamic steps) WITHOUT ponder cost
#   level 6: level 5 + ponder cost added to loss
#
# Keeps the external API compatible with your LitGPT:
#   forward(idx, targets=None) -> (logits, loss)
#   generate(idx, max_new_tokens) -> token ids

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


# -----------------------------
# Baseline GPT components (as-is)
# -----------------------------
class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, _ = x.shape
        k = self.key(x)  # (B, T, hs)
        q = self.query(x)  # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)  # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  # (B, T, hs)
        out = wei @ v  # (B, T, hs)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, num_heads, head_size, block_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(n_embd, head_size, block_size, dropout) for _ in range(num_heads)]
        )
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
    """Your original Block (pre-LN style)."""

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


# -----------------------------
# UT-specific utilities/modules
# -----------------------------
def sinusoidal_table(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """(length, dim) sinusoidal encoding table."""
    pe = torch.zeros(length, dim, device=device)
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class UTRecurrentBlock(nn.Module):
    """
    Explicit UT recurrent step:
        u = h + p_t
        a = LN(u + SA(u))
        h = LN(a + FF(a))
    Uses the same attention/FFN primitives as the baseline.
    """

    def __init__(self, n_embd, n_head, block_size, dropout):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_embd, n_head, head_size, block_size, dropout)
        self.ff = FeedForward(n_embd, dropout)
        self.ln_attn = nn.LayerNorm(n_embd)
        self.ln_ff = nn.LayerNorm(n_embd)

    def forward(self, h: torch.Tensor, p_t: torch.Tensor) -> torch.Tensor:
        # h: (B, T, C), p_t: (T, C) broadcastable to (B, T, C)
        u = h + p_t
        a = self.ln_attn(u + self.sa(u))
        h_next = self.ln_ff(a + self.ff(a))
        return h_next


# -----------------------------
# Unified model with levels 0..6
# -----------------------------
class LitUnifiedGPTUT(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        lr: float,
        # new:
        level: int = 0,
        num_steps: int | None = None,   # used for levels 1-4; default = n_layer
        max_steps: int = 16,            # used for learned step table (lvl2) and ACT (lvl5-6)
        act_threshold: float = 0.99,    # lvl5-6
        act_loss_weight: float = 0.01,  # lvl6
    ):
        super().__init__()
        if level < 0 or level > 6:
            raise ValueError("level must be in [0..6]")
        if num_steps is None:
            num_steps = n_layer
        if num_steps < 1:
            raise ValueError("num_steps must be >= 1")
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        if not (0.0 < act_threshold <= 1.0):
            raise ValueError("act_threshold must be in (0, 1]")
        if act_loss_weight < 0.0:
            raise ValueError("act_loss_weight must be >= 0")

        self.save_hyperparameters()

        # embeddings (token always)
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # learned position embedding used for levels 0-2
        self.position_embedding_table = nn.Embedding(block_size, n_embd) if level <= 2 else None

        # learned step embedding used for level 2
        self.step_embedding_table = nn.Embedding(max_steps, n_embd) if level == 2 else None

        # core compute modules
        if level == 0:
            self.blocks = nn.Sequential(*[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)])
            self.shared_block = None
            self.ut_block = None
        elif level in (1, 2, 3):
            self.blocks = None
            self.shared_block = Block(n_embd, n_head, block_size, dropout)
            self.ut_block = None
        else:  # level 4,5,6
            self.blocks = None
            self.shared_block = None
            self.ut_block = UTRecurrentBlock(n_embd, n_head, block_size, dropout)

        # ACT halting head for levels 5-6
        self.halt_proj = nn.Linear(n_embd, 1) if level >= 5 else None

        # output head
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        # sinusoidal coordinate tables lazily allocated (levels 3+)
        self.register_buffer("_pos_table", torch.empty(0), persistent=False)
        self.register_buffer("_step_table", torch.empty(0), persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # ---------- coordinate helpers ----------
    def _ensure_sinusoidal_tables(self, device: torch.device):
        """Ensure sinusoidal tables exist on the right device (levels 3+)."""
        if self._pos_table.numel() == 0 or self._pos_table.device != device:
            self._pos_table = sinusoidal_table(self.hparams.block_size, self.hparams.n_embd, device)
        if self._step_table.numel() == 0 or self._step_table.device != device:
            self._step_table = sinusoidal_table(self.hparams.max_steps, self.hparams.n_embd, device)

    def _sinusoidal_p_t(self, T: int, step_idx: int, device: torch.device) -> torch.Tensor:
        """p_t = pos_enc[:T] + step_enc[step_idx], returns (T, C)."""
        self._ensure_sinusoidal_tables(device)
        pos = self._pos_table[:T]  # (T, C)
        if step_idx < self._step_table.size(0):
            step = self._step_table[step_idx]  # (C,)
        else:
            # allow inference with more steps than max_steps (compute one-off)
            step = sinusoidal_table(step_idx + 1, self.hparams.n_embd, device)[-1]
        return pos + step.unsqueeze(0)

    # ---------- forward variants ----------
    def _forward_level_0_to_3(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Returns hidden states x of shape (B, T, C) after either:
          level0: stacked blocks
          level1: recurrent shared block (no step enc)
          level2: recurrent + learned step enc
          level3: recurrent + sinusoidal (pos+step)
        """
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)

        level = self.hparams.level

        if level <= 2:
            # learned positions
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.position_embedding_table(pos)  # (T, C)
            x = tok_emb + pos_emb  # (B, T, C)
        else:
            # level 3: start from token embedding, inject p_t per step
            x = tok_emb

        if level == 0:
            x = self.blocks(x)
            return x

        # levels 1-3: recurrent shared block
        steps = int(self.hparams.num_steps)
        for s in range(steps):
            if level == 2:
                if s >= self.hparams.max_steps:
                    raise ValueError("Level 2 uses learned step_embedding_table; increase max_steps.")
                step_emb = self.step_embedding_table(torch.tensor(s, device=idx.device))  # (C,)
                x = x + step_emb.view(1, 1, -1)
                x = self.shared_block(x)
            else:  # level 1
                x = self.shared_block(x)

        return x

    def _forward_level_4_fixed_steps(self, idx: torch.Tensor) -> torch.Tensor:
        """Level 4: explicit UTRecurrentBlock, fixed steps, sinusoidal p_t."""
        _, T = idx.shape
        x = self.token_embedding_table(idx)
        steps = int(self.hparams.num_steps)
        for s in range(steps):
            if s >= self.hparams.max_steps:
                raise ValueError("Level 4 uses learned step_embedding_table; increase max_steps.")
            step_emb = self.step_embedding_table(torch.tensor(s, device=idx.device))  # (C,)
            p_t = x + step_emb.view(1, 1, -1)
            x = self.ut_block(x, p_t)
        return x

    def _forward_level_5_6_act(self, idx: torch.Tensor):
        """
        Levels 5-6: UTRecurrentBlock + ACT.
        Returns: (x, ponder_time_mean)
        """
        B, T = idx.shape
        state = self.token_embedding_table(idx)

        max_steps = int(self.hparams.max_steps)
        threshold = float(self.hparams.act_threshold)

        halting_prob = torch.zeros(B, T, 1, device=idx.device)
        remainders = torch.zeros(B, T, 1, device=idx.device)
        n_updates = torch.zeros(B, T, 1, device=idx.device)

        acc_state = torch.zeros_like(state)

        for s in range(max_steps):
            still_running = (halting_prob < 1.0).float()  # (B,T,1)

            p = torch.sigmoid(self.halt_proj(state))  # (B,T,1)

            # halt decision this step
            new_halted = ((halting_prob + p * still_running) > threshold).float() * still_running
            still_running = ((halting_prob + p * still_running) <= threshold).float() * still_running

            halting_prob = halting_prob + p * still_running
            remainders = remainders + new_halted * (1.0 - halting_prob)
            halting_prob = halting_prob + new_halted * remainders

            n_updates = n_updates + still_running + new_halted

            update_weights = p * still_running + new_halted * remainders  # (B,T,1)

            # Use step embedding table instead of sinusoidal
            step_emb = self.step_embedding_table(torch.tensor(s, device=idx.device))  # (C,)
            p_t = step_emb.view(1, 1, -1)  # (1, 1, C), broadcastable to (B, T, C)
            transformed = self.ut_block(state, p_t)

            acc_state = transformed * update_weights + acc_state * (1.0 - update_weights)
            state = transformed

            if torch.all(halting_prob >= 1.0):
                break

        ponder_time = (n_updates + remainders).mean()  # scalar
        return acc_state, ponder_time

    # ---------- public forward ----------
    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.hparams.block_size

        level = int(self.hparams.level)
        ponder_time = None

        if level <= 3:
            x = self._forward_level_0_to_3(idx)
        elif level == 4:
            x = self._forward_level_4_fixed_steps(idx)
        else:  # 5-6
            x, ponder_time = self._forward_level_5_6_act(idx)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
            if level == 6:
                # add ponder cost
                loss = loss + float(self.hparams.act_loss_weight) * ponder_time

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        """Sampling wrapper (returns token ids)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # Lightning hooks
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=x.size(0),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=x.size(0),
        )

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
