# models/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block



class GPTLevel1(BaseLitGPT):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, lr):
        super().__init__(vocab_size, block_size, n_embd, lr)
        
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
        
        # Level 1 Change: Reusing the same block
        for _ in range(self.n_layer):
            x = self.shared_block(x)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss