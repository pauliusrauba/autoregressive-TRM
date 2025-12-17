# models/gpt.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trainer import BaseLitGPT
from models.layers import Block

class GPTLevel2(BaseLitGPT):
    def __init__(self, vocab_size, block_size, n_embd, n_head, n_layer, dropout, lr):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(n_layer, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer

        self.apply(self._init_weights)


    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.hparams.block_size

        tok_emb = self.token_embedding_table(idx)  # (B, T, C)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, C)
        x = tok_emb + pos_emb  
        for step in range(self.n_layer):
            step_emb = self.step_embedding_table(
                torch.tensor(step, device=idx.device)
            ) # (C, )
            u = x + step_emb # (B, T, C)
            x = self.shared_block(u)                       # (B, T, C)
        x = self.ln_f(x)
        logits = self.lm_head(x)                      # (B, T, V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )
        return logits, loss
