# models/ut_level2.py
"""
Universal Transformer Level 2: Hierarchical Recurrence.

Delta from UT-Level1:
  - Adds hierarchical HL loop structure (nested fixed-point iteration)
  - Partial gradient detachment: only last H-cycle gets gradients
  
The key insight: running multiple refinement cycles per ACT step
allows the model to "settle" into better fixed points, while
detaching H-1 cycles saves memory without losing expressiveness.

Update pattern (TRM paper):
  - z' = block(x + y + z + step)   # reasoning uses input
  - y' = block(y + z' + step)      # answer does NOT use input

Supports:
  - use_step_embeddings: Set to False to disable step embeddings (enables clean extrapolation)
  - set_inference_steps(): Run more/fewer steps at inference than training
  - set_full_compute(): Force all steps without early halting
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block
from models.common.act import ACTController, ACTState
from models.common.dual_stream import DualStreamState
from models.common.recurrence import RecurrenceEngine, RecurrenceEngineWithTBPTT


class UTLevel2(BaseLitGPT):
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
        n_inner_loops: int = 4,   # L: reasoning refinements per H-cycle
        n_outer_loops: int = 4,   # H: solution update cycles per ACT step
        max_act_steps: int = None,
        use_step_embeddings: bool = True,
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()

        self.max_act_steps = max_act_steps if max_act_steps is not None else n_layer
        self._trained_max_steps = self.max_act_steps  # Remember for clamping during extrapolation

        # === Same as UT-Level1 ===
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer
        self.act = ACTController(n_embd)
        
        # Step embeddings (optional - disable for clean compute extrapolation)
        self.use_step_embeddings = use_step_embeddings
        if use_step_embeddings:
            self.step_embedding_table = nn.Embedding(self.max_act_steps, n_embd)
        
        # Separate y_init and z_init (TRM paper requirement)
        self.solution_param = nn.Parameter(torch.randn(n_embd) * 0.02)   # y_init
        self.reasoning_param = nn.Parameter(torch.randn(n_embd) * 0.02)  # z_init

        # === Hierarchical recurrence engine ===
        self.recurrence = RecurrenceEngineWithTBPTT(
            self.shared_block,
            n_inner_loops=n_inner_loops,
            n_outer_loops=n_outer_loops,
        )

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.hparams.block_size

        # Embeddings: x = tok + pos (static input)
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x_input = tok_emb + pos_emb

        # Initialize with SEPARATE y and z
        stream = DualStreamState.init(x_input, self.solution_param, self.reasoning_param)
        
        act_state = ACTState.init(B, T, self.hparams.n_embd, device)

        # Get number of steps (allows inference-time extrapolation)
        num_steps = self.get_inference_steps(self.max_act_steps)

        for step in range(num_steps):
            if self.use_step_embeddings and hasattr(self, 'step_embedding_table'):
                # Clamp step index for extrapolation (reuse last embedding for extra steps)
                step_idx = min(step, self._trained_max_steps - 1)
                step_emb = self.step_embedding_table(torch.tensor(step_idx, device=device))
            else:
                # No step embeddings - enables clean compute extrapolation
                step_emb = None
            
            # H×L recurrence with partial gradient detachment
            stream = self.recurrence(stream, step_emb)
            
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
