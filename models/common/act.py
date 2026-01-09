# models/common/act.py
"""
Adaptive Computation Time (ACT) controller.

This module isolates the ACT halting mechanism so it can be cleanly
composed with any recurrent transformer. The controller manages:
  - Halting probability accumulation
  - Weighted output accumulation  
  - Ponder cost computation
  - Optional stochastic exploration (forces extra thinking steps)
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ACTState:
    """Tracks ACT state across steps."""
    halted: torch.Tensor          # (B, T) bool - which positions have halted
    halt_probs_accum: torch.Tensor  # (B, T) - cumulative halting probability
    remainders: torch.Tensor      # (B, T) - remainder for final step
    n_updates: torch.Tensor       # (B, T) - number of updates per position
    output_accum: torch.Tensor    # (B, T, C) - weighted state accumulator
    min_steps: Optional[torch.Tensor] = None  # (B, T) - minimum steps for exploration

    @classmethod
    def init(
        cls, 
        B: int, 
        T: int, 
        C: int, 
        device: torch.device,
        exploration_min_steps: Optional[torch.Tensor] = None,
    ) -> "ACTState":
        return cls(
            halted=torch.zeros(B, T, dtype=torch.bool, device=device),
            halt_probs_accum=torch.zeros(B, T, device=device),
            remainders=torch.zeros(B, T, device=device),
            n_updates=torch.zeros(B, T, device=device),
            output_accum=torch.zeros(B, T, C, device=device),
            min_steps=exploration_min_steps,
        )


class ACTController(nn.Module):
    """
    Manages Adaptive Computation Time halting logic.
    
    Usage:
        controller = ACTController(n_embd)
        state = ACTState.init(B, T, C, device)
        
        for step in range(max_steps):
            x = block(x + step_emb)
            state, should_stop = controller.step(x, state, step)
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
        step: int = 0,  # current step index (for exploration)
    ) -> Tuple[ACTState, bool]:
        """
        Process one ACT step: compute halting, update accumulators.
        
        Returns:
            Updated ACTState and bool indicating if all positions halted.
        """
        device = x.device
        
        # Compute halting probability
        halt_logits = self.halt_head(x).squeeze(-1)  # (B, T)
        halt_prob = torch.sigmoid(halt_logits)
        
        still_running = ~state.halted
        
        # Check which positions will halt at this step
        should_halt = (state.halt_probs_accum + halt_prob >= self.halt_threshold)
        
        # Apply exploration constraint if min_steps is set
        if state.min_steps is not None:
            forced_continue = (torch.tensor(step, device=device) < state.min_steps)
            should_halt = should_halt & (~forced_continue)
        
        new_halted = should_halt & still_running
        
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
            min_steps=state.min_steps,
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


def create_exploration_min_steps(
    B: int,
    T: int,
    max_act_steps: int,
    exploration_prob: float,
    device: torch.device,
    training: bool,
) -> Optional[torch.Tensor]:
    """
    Create min_steps tensor for stochastic halt exploration.
    
    During training, randomly forces some positions to think longer,
    preventing premature halting collapse.
    
    Returns:
        (B, T) tensor of minimum steps, or None if not training/no exploration
    """
    if not training or exploration_prob <= 0:
        return None
    
    min_steps = torch.randint(1, max_act_steps, (B, T), device=device)
    do_explore = torch.rand((B, T), device=device) < exploration_prob
    return torch.where(do_explore, min_steps, torch.zeros_like(min_steps))
