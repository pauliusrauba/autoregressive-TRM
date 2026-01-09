# models/common/dual_stream.py
"""
Dual-Stream state management for TRM transformers.

This module isolates the solution/reasoning separation so it can be
cleanly composed with any recurrent model. 

Delta from single-stream (UT):
  - Splits state into 'solution' (answer) and 'reasoning'
  - Two block calls per step: reasoning update, then solution update
"""
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DualStreamState:
    """Holds the two streams: solution and reasoning."""
    solution: torch.Tensor   # (B, T, C) - the "answer" state
    reasoning: torch.Tensor  # (B, T, C) - the "thinking" state
    x_input: torch.Tensor    # (B, T, C) - static input (doesn't change across steps)

    @classmethod
    def init(
        cls, 
        x_input: torch.Tensor,  # (B, T, C) - raw embeddings (tok + pos)
        reasoning_param: nn.Parameter,  # (C,)
    ) -> "DualStreamState":
        """
        Initialize dual-stream state from input embeddings.
        
        - x_input: kept unbiased (static reference to original input)
        - solution: biased with reasoning_param (starting point for answer)
        - reasoning: initialized to reasoning_param
        """
        B, T, C = x_input.shape
        reasoning_bias = reasoning_param.unsqueeze(0).unsqueeze(1)  # (1, 1, C)
        
        return cls(
            solution=x_input + reasoning_bias,  # biased starting point
            reasoning=reasoning_param.view(1, 1, -1).expand(B, T, -1).clone(),
            x_input=x_input,  # UNBIASED - no reasoning_param here
        )


class DualStreamStep(nn.Module):
    """
    Performs one dual-stream update step.
    
    The key insight: we separate "thinking" from "answering":
      1. Reasoning update: z' = block(z + y + x + step_emb)
      2. Solution update:  y' = block(y + z' + step_emb)
    
    Where:
      - x = static input (question)
      - y = solution (answer being refined)
      - z = reasoning (scratch space for thinking)
    """
    
    def __init__(self, block: nn.Module):
        super().__init__()
        self.block = block
    
    def forward(
        self,
        state: DualStreamState,
        step_emb: torch.Tensor,  # (C,) step embedding
    ) -> DualStreamState:
        """
        One step of dual-stream refinement.
        
        Returns new DualStreamState with updated solution and reasoning.
        """
        # Step 1: Update reasoning conditioned on solution + input
        cond_z = state.solution + state.x_input
        u_z = state.reasoning + cond_z + step_emb
        reasoning_new = self.block(u_z)
        
        # Step 2: Update solution conditioned on new reasoning
        cond_y = reasoning_new
        u_y = state.solution + cond_y + step_emb
        solution_new = self.block(u_y)
        
        return DualStreamState(
            solution=solution_new,
            reasoning=reasoning_new,
            x_input=state.x_input,  # static, unchanged
        )
