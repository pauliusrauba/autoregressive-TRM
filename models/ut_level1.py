# models/ut_level1.py
"""
Universal Transformer Level 1: Dual-Stream Reasoning.

Delta from UT:
  - Adds dual-stream state: y (solution) and z (reasoning)
  - Two block calls per step instead of one
  - Separate learned initializations for y and z
  
The key insight: separating "thinking" (z) from "answering" (y)
with distinct update rules:
  - z' = block(x + y + z + step)   # reasoning uses input
  - y' = block(y + z' + step)      # answer does NOT use input
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block
from models.common.act import ACTController, ACTState
from models.common.dual_stream import DualStreamState, DualStreamStep


class UTLevel1(BaseLitGPT):
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

        # === Same as UT ===
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(self.max_act_steps, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer
        self.act = ACTController(n_embd)

        # === NEW: Separate y_init and z_init (TRM paper requirement) ===
        self.solution_param = nn.Parameter(torch.randn(n_embd) * 0.02)   # y_init
        self.reasoning_param = nn.Parameter(torch.randn(n_embd) * 0.02)  # z_init
        
        self.dual_step = DualStreamStep(self.shared_block)

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.hparams.block_size

        # Embeddings: x = tok + pos (static input)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x_input = tok_emb + pos_emb

        # === NEW: Initialize with SEPARATE y and z ===
        stream = DualStreamState.init(x_input, self.solution_param, self.reasoning_param)
        
        act_state = ACTState.init(B, T, self.hparams.n_embd, device)

        for step in range(self.max_act_steps):
            step_emb = self.step_embedding_table(torch.tensor(step, device=device))
            
            # Dual-stream update: z' = f(x,y,z), then y' = f(y,z')
            stream = self.dual_step(stream, step_emb)
            
            # ACT uses solution (y) state for halting decision
            act_state, all_halted = self.act.step(stream.solution, act_state, step)
            if all_halted and not self._force_full_compute:
                break

        x, ponder_cost = self.act.finalize(stream.solution, act_state)
        self._last_ponder_cost = ponder_cost.item()

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            loss = ce_loss + self.hparams.ponder_cost_weight * ponder_cost

        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        self.log('ponder_cost', self._last_ponder_cost, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        return loss
