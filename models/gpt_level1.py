# models/gpt_level1.py
"""
GPT Level 1: Weight-Tied Transformer.

Delta from GPT:
  - Uses a single shared block repeated n_layer times (weight tying)
  - No step embeddings (pure fixed-point iteration)

This model naturally supports compute extrapolation since there are no
step-specific embeddings - you can run more iterations at inference time
using set_inference_steps().
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block


class GPTLevel1(BaseLitGPT):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, lr):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Level 1 Change: Single shared block
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.n_layer = n_layer
        
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        # Get number of steps (allows inference-time extrapolation)
        num_steps = self.get_inference_steps(self.n_layer)
        
        # Level 1: Reusing the same block (pure iteration, no step embeddings)
        # This naturally supports compute extrapolation
        for _ in range(num_steps):
            x = self.shared_block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss