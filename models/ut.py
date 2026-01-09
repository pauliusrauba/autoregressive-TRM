# models/ut.py
"""
Universal Transformer with Adaptive Computation Time.

Delta from GPT-Level2:
  - Adds ACT (adaptive halting) via ACTController
  - Adds ponder_cost to loss
  
Everything else (weight tying, step embedding) is inherited from GPT-Level2 structure.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block
from models.common.act import ACTController, ACTState


class UT(BaseLitGPT):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        lr: float,
        ponder_cost_weight: float = 0.01,
        max_act_steps: int = None,
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()

        self.max_act_steps = max_act_steps if max_act_steps is not None else n_layer

        # === Same as GPT-Level2 ===
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(self.max_act_steps, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer

        # === NEW: ACT Controller ===
        self.act = ACTController(n_embd)

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.hparams.block_size

        # Embeddings (same as GPT-Level2)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # === NEW: ACT loop (replaces fixed-depth loop) ===
        act_state = ACTState.init(B, T, self.hparams.n_embd, device)
        
        for step in range(self.max_act_steps):
            step_emb = self.step_embedding_table(torch.tensor(step, device=device))
            x = self.shared_block(x + step_emb)
            
            act_state, all_halted = self.act.step(x, act_state)
            if all_halted:
                break

        x, ponder_cost = self.act.finalize(x, act_state)
        self._last_ponder_cost = ponder_cost.item()
        # === END ACT ===

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            # NEW: Add ponder cost to loss
            loss = ce_loss + self.hparams.ponder_cost_weight * ponder_cost

        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        self.log('ponder_cost', self._last_ponder_cost, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        return loss
