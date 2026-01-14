# models/gpt_level2.py
"""
GPT Level 2: Weight-Tied Transformer with Step Embeddings.

Delta from GPT-Level1:
  - Adds step embeddings to distinguish iteration steps

Supports:
  - use_step_embeddings: Set to False to disable step embeddings (enables clean extrapolation)
  - set_inference_steps(): Run more/fewer steps at inference than training
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block


class GPTLevel2(BaseLitGPT):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        lr: float,
        use_step_embeddings: bool = True,
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        
        # Step embeddings (optional - disable for clean compute extrapolation)
        self.use_step_embeddings = use_step_embeddings
        if use_step_embeddings:
            self.step_embedding_table = nn.Embedding(n_layer, n_embd)
        self._trained_max_steps = n_layer  # Remember training steps for clamping

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.hparams.block_size

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        x = tok_emb + pos_emb
        
        # Get number of steps (allows inference-time extrapolation)
        num_steps = self.get_inference_steps(self.n_layer)
        
        for step in range(num_steps):
            if self.use_step_embeddings and hasattr(self, 'step_embedding_table'):
                # Clamp step index for extrapolation (reuse last embedding for extra steps)
                step_idx = min(step, self._trained_max_steps - 1)
                step_emb = self.step_embedding_table(torch.tensor(step_idx, device=device))
                x = self.shared_block(x + step_emb)
            else:
                # No step embeddings - pure iteration (like GPT-Level1)
                x = self.shared_block(x)
                
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        return logits, loss
