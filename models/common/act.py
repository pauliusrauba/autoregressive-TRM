# models/common/act.py
"""
Adaptive Computation Time (ACT) controller.

This module isolates the ACT halting mechanism so it can be cleanly
composed with any recurrent transformer. The controller manages:
  - Halting probability accumulation
  - Weighted output accumulation  
  - Ponder cost computation
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class ACTState:
    """Tracks ACT state across steps."""
    halted: torch.Tensor          # (B, T) bool - which positions have halted
    halt_probs_accum: torch.Tensor  # (B, T) - cumulative halting probability
    remainders: torch.Tensor      # (B, T) - remainder for final step
    n_updates: torch.Tensor       # (B, T) - number of updates per position
    output_accum: torch.Tensor    # (B, T, C) - weighted state accumulator

    @classmethod
    def init(cls, B: int, T: int, C: int, device: torch.device) -> "ACTState":
        return cls(
            halted=torch.zeros(B, T, dtype=torch.bool, device=device),
            halt_probs_accum=torch.zeros(B, T, device=device),
            remainders=torch.zeros(B, T, device=device),
            n_updates=torch.zeros(B, T, device=device),
            output_accum=torch.zeros(B, T, C, device=device),
        )


class ACTController(nn.Module):
    """
    Manages Adaptive Computation Time halting logic.
    
    Usage:
        controller = ACTController(n_embd)
        state = ACTState.init(B, T, C, device)
        
        for step in range(max_steps):
            x = block(x + step_emb)
            state, should_stop = controller.step(x, state)
            if should_stop:
                break
        
        output, ponder_cost = controller.finalize(x, state)
    """
    
    def __init__(self, n_embd: int, halt_threshold: float = 1.0 - 1e-6):
        super().__init__()
        self.halt_head = nn.Linear(n_embd, 1)
        self.halt_threshold = halt_threshold
    
    def step(
        self, 
        x: torch.Tensor,  # (B, T, C) - current state
        state: ACTState,
    ) -> Tuple[ACTState, bool]:
        """
        Process one ACT step: compute halting, update accumulators.
        
        Returns:
            Updated ACTState and bool indicating if all positions halted.
        """
        # Compute halting probability
        halt_logits = self.halt_head(x).squeeze(-1)  # (B, T)
        halt_prob = torch.sigmoid(halt_logits)
        
        still_running = ~state.halted
        
        # Check which positions will halt at this step
        new_halted = (state.halt_probs_accum + halt_prob >= self.halt_threshold) & still_running
        
        # For newly halted positions, remainder = 1 - accumulated so far
        remainders = torch.where(
            new_halted,
            1.0 - state.halt_probs_accum,
            state.remainders
        )
        
        # Compute weight for this step's contribution
        p = torch.where(
            new_halted,
            remainders,
            torch.where(still_running, halt_prob, torch.zeros_like(halt_prob))
        )
        
        # Accumulate weighted outputs
        output_accum = state.output_accum + p.unsqueeze(-1) * x
        
        # Update cumulative halt probability
        halt_probs_accum = torch.where(
            still_running & ~new_halted,
            state.halt_probs_accum + halt_prob,
            state.halt_probs_accum
        )
        
        # Track number of updates
        n_updates = state.n_updates + still_running.float()
        
        # Update halted mask
        halted = state.halted | new_halted
        
        new_state = ACTState(
            halted=halted,
            halt_probs_accum=halt_probs_accum,
            remainders=remainders,
            n_updates=n_updates,
            output_accum=output_accum,
        )
        
        return new_state, halted.all().item()
    
    def finalize(
        self, 
        x: torch.Tensor,  # (B, T, C) - final state for non-halted positions
        state: ACTState,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Finalize ACT: handle positions that never halted, compute ponder cost.
        
        Returns:
            (output, ponder_cost) tuple
        """
        still_running = ~state.halted
        remainders = torch.where(
            still_running, 
            1.0 - state.halt_probs_accum, 
            state.remainders
        )
        output = state.output_accum + (still_running.float() * remainders).unsqueeze(-1) * x
        
        ponder_cost = (state.n_updates + remainders).mean()
        
        return output, ponder_cost
