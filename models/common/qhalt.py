# models/common/qhalt.py
"""
Q-based halting controller (original TRM paper).

Unlike Graves' ACT which accumulates weighted outputs, this uses:
  - Binary halt decision: halt when q_halt > 0 (sigmoid > 0.5)
  - Per-batch-element halting (not per-position)
  - No ponder cost - just early stopping
  - Eval mode: always runs max steps (no adaptive halting)
"""
import torch
import torch.nn as nn
from typing import Optional


class QHaltController(nn.Module):
    """
    Q-based halting for TRM (original paper).
    
    Usage:
        halt_ctrl = QHaltController(n_embd)
        
        for step in range(max_steps):
            x = block(x)
            if training:
                halted = halt_ctrl.step(x, step, halted, min_steps, is_last=(step==max_steps-1))
                if halted.all():
                    break
        # Use x directly (no weighted accumulation)
    """
    
    def __init__(self, n_embd: int):
        super().__init__()
        self.q_head = nn.Linear(n_embd, 1)
        
        # Initialize to encourage "continue" initially
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)
    
    def step(
        self,
        x: torch.Tensor,              # (B, T, C)
        step: int,
        halted: torch.Tensor,         # (B,) bool
        min_steps: torch.Tensor,      # (B,) exploration minimum
        is_last: bool,
    ) -> torch.Tensor:
        """Returns updated halted mask (B,)."""
        # Use first position for halt decision (like original TRM)
        q_halt = self.q_head(x[:, 0, :]).squeeze(-1)  # (B,)
        
        should_halt = (q_halt > 0) | is_last
        should_halt = should_halt & (step + 1 >= min_steps)  # step is 0-indexed
        
        return halted | should_halt


def create_exploration_min_steps_batch(
    B: int,
    max_steps: int,
    exploration_prob: float,
    device: torch.device,
    training: bool,
) -> torch.Tensor:
    """
    Create per-batch-element minimum steps for exploration.
    
    Returns (B,) tensor of minimum steps before halting is allowed.
    """
    if not training or exploration_prob <= 0:
        return torch.zeros(B, device=device, dtype=torch.long)
    
    # Handle edge case where max_steps is too small for exploration
    if max_steps < 2:
        return torch.zeros(B, device=device, dtype=torch.long)
    
    do_explore = torch.rand(B, device=device) < exploration_prob
    min_steps = torch.randint(1, max_steps + 1, (B,), device=device)
    return torch.where(do_explore, min_steps, torch.zeros_like(min_steps))
