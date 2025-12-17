# models/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block


class GPTBase(BaseLitGPT):
    """Standard GPT model."""
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, lr):
        # Pass args to Base so they are saved
        super().__init__(vocab_size, block_size, n_embd, lr)
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Standard: Sequential blocks
        self.blocks = nn.Sequential(
            *[Block(n_embd, n_head, block_size, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)  # Standard sequential pass
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
