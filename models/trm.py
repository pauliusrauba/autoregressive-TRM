# models/trm.py
"""
Tiny Recursive Model (TRM): Original paper's Q-based halting.

Delta from UT-Level2:
  - Replaces Graves' ACT with binary Q-based halting
  - No weighted accumulation, no ponder cost
  - Halts at both train and eval time (unlike original paper which used fixed eval)
  - Use set_full_compute(True) to force max_act_steps (no halting)

Supports:
  - use_step_embeddings: Set to False to disable step embeddings (enables clean extrapolation)
  - set_inference_steps(): Run more/fewer steps at inference than training
  - set_full_compute(): Force all steps without early halting
"""
import torch
import torch.nn.functional as F
from models.ut_level2 import UTLevel2
from models.common.qhalt import QHaltController, create_exploration_min_steps_batch


class TRM(UTLevel2):
    """TRM with Q-based halting at both train and eval time."""
    
    def __init__(self, *args, halt_exploration_prob: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.save_hyperparameters()
        self.halt_exploration_prob = halt_exploration_prob
        
        # Replace ACT with Q-based halting
        del self.act
        self.halt = QHaltController(self.hparams.n_embd)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x_input = tok_emb + pos_emb
        
        from models.common.dual_stream import DualStreamState
        stream = DualStreamState.init(x_input, self.solution_param, self.reasoning_param)
        
        # Get number of steps (allows inference-time extrapolation)
        num_steps = self.get_inference_steps(self.max_act_steps)
        
        # Exploration only during training
        min_steps = create_exploration_min_steps_batch(
            B, num_steps, self.halt_exploration_prob, device, self.training
        )
        halted = torch.zeros(B, dtype=torch.bool, device=device)
        steps_taken = num_steps
        
        for step in range(num_steps):
            if self.use_step_embeddings and hasattr(self, 'step_embedding_table'):
                # Clamp step index for extrapolation (reuse last embedding for extra steps)
                step_idx = min(step, self._trained_max_steps - 1)
                step_emb = self.step_embedding_table(torch.tensor(step_idx, device=device))
            else:
                # No step embeddings - enables clean compute extrapolation
                step_emb = None
            
            stream = self.recurrence(stream, step_emb, detach_input=True)
            
            # Check halting (unless force_full_compute is set)
            if not self._force_full_compute:
                halted = self.halt.step(stream.solution, step, halted, min_steps, is_last=(step == num_steps - 1))
                if halted.all():
                    steps_taken = step + 1
                    break

        self._last_steps = steps_taken
        x = self.ln_f(stream.solution)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(-1))

        return logits, loss

    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('train_loss', loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        self.log('steps', float(self._last_steps), on_step=True, on_epoch=False, prog_bar=True, batch_size=x.size(0))
        return loss
