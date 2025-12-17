import torch
import torch.nn as nn
import torch.nn.functional as F
from models.trainer import BaseLitGPT
from models.layers import Block

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
        ponder_cost_weight: float = 0.01
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(n_layer, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.halt_head = nn.Linear(n_embd, 1)

        # Store ACT hyperparams
        self.max_steps = n_layer
        self.halt_threshold = 1.0 - 1e-6 # For numerical stabilitt
        self.act_epsilon = 0.01 # for bias initialization

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
        x = tok_emb + pos_emb  # (B, T, C)

        # ===== ACT Tensors =====
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)  # Which positions have halted
        halt_probs_accum = torch.zeros(B, T, device=device)  # Cumulative halting probability
        remainders = torch.zeros(B, T, device=device)  # Remainder for final step
        n_updates = torch.zeros(B, T, device=device)  # Number of updates per position
        output_accum = torch.zeros(B, T, self.hparams.n_embd, device=device)  # Weighted state accumulator

        for step in range(self.hparams.n_layer):
            # Get step embedding
            step_emb = self.step_embedding_table(
                torch.tensor(step, device=device)
            )  # (C,)
            
            # Apply shared block
            u = x + step_emb  # (B, T, C)
            x_new = self.shared_block(u)  # (B, T, C)

            # Compute halting probability for each position
            halt_logits = self.halt_head(x_new).squeeze(-1)  # (B, T)
            halt_prob = torch.sigmoid(halt_logits)  # (B, T)

            # For positions that haven't halted yet
            still_running = ~halted  # (B, T)
            
            # Check which positions will halt at this step
            new_halted = (halt_probs_accum + halt_prob >= self.halt_threshold) & still_running
            
            # For newly halted positions, remainder = 1 - accumulated so far
            remainders = torch.where(
                new_halted,
                1.0 - halt_probs_accum,
                remainders
            )
            
            # For still running (but not newly halted), use the halt_prob
            # For newly halted, use the remainder
            p = torch.where(
                new_halted,
                remainders,
                torch.where(still_running, halt_prob, torch.zeros_like(halt_prob))
            )
            
            # Accumulate weighted outputs
            output_accum = output_accum + p.unsqueeze(-1) * x_new
            
            # Update cumulative halt probability (only for running positions)
            halt_probs_accum = torch.where(
                still_running & ~new_halted,
                halt_probs_accum + halt_prob,
                halt_probs_accum
            )
            
            # Track number of updates
            n_updates = n_updates + still_running.float()
            
            # Update halted mask
            halted = halted | new_halted
            
            # Update state for next iteration (only matters for non-halted)
            x = x_new
            
            # Early exit if all positions have halted
            if halted.all():
                break

        # For any positions that never halted by max_steps, assign remainder
        still_running = ~halted
        remainders = torch.where(still_running, 1.0 - halt_probs_accum, remainders)
        output_accum = output_accum + (still_running.float() * remainders).unsqueeze(-1) * x

        # Compute ponder cost: sum of (n_updates + remainder) across positions
        ponder_cost = (n_updates + remainders).mean()  # Mean over batch and positions
        self._last_ponder_cost = ponder_cost.item() # Store for logging

        x = output_accum
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, V)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            ce_loss = F.cross_entropy(
                logits.view(B * T, C),
                targets.view(B * T),
            )
            # Add ponder cost to loss
            loss = ce_loss + self.hparams.ponder_cost_weight * ponder_cost

        return logits, loss

   
    # Lightning hooks
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            batch_size=x.size(0),
        )
        self.log('ponder_cost', self._last_ponder_cost, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))

        return loss