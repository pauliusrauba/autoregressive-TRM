# models/common/dual_stream.py
"""
Dual-Stream state management for TRM transformers.

The TRM decomposes state into three conceptually distinct streams:
  - x: embedded input (static during recursion) - "the question"
  - y: current solution/answer (evolves) - "the answer canvas"  
  - z: latent reasoning state (evolves) - "scratch space for thinking"

Update pattern (per step):
  1. z' = block(x + y + z + step_emb)   # reasoning uses input
  2. y' = block(y + z' + step_emb)      # answer does NOT use input

This asymmetry is important: x appears in z-update but NOT y-update,
so the model knows whether it's doing "reasoning" vs "answering".
"""
import torch
import torch.nn as nn
from dataclasses import dataclass


@dataclass
class DualStreamState:
    """
    Holds the three streams: x (static), y (solution), z (reasoning).
    
    Attributes:
        solution: (B, T, C) - y, the "answer" state (evolves)
        reasoning: (B, T, C) - z, the "thinking" state (evolves)
        x_input: (B, T, C) - x, static input (doesn't change across steps)
    """
    solution: torch.Tensor   # y - the answer being refined
    reasoning: torch.Tensor  # z - the reasoning/thinking state
    x_input: torch.Tensor    # x - static input (the question)

    @classmethod
    def init(
        cls, 
        x_input: torch.Tensor,           # (B, T, C) - raw embeddings (tok + pos)
        solution_param: nn.Parameter,    # (C,) - learned y_init
        reasoning_param: nn.Parameter,   # (C,) - learned z_init
    ) -> "DualStreamState":
        """
        Initialize dual-stream state with SEPARATE y and z initializations.
        
        - x_input: static reference to embedded input (the question)
        - solution (y): starts from learned solution_param (answer canvas)
        - reasoning (z): starts from learned reasoning_param (thinking init)
        
        Critically, y does NOT start as a copy of x. They are separate streams.
        """
        B, T, C = x_input.shape
        
        return cls(
            solution=solution_param.view(1, 1, -1).expand(B, T, -1).clone(),   # y_init
            reasoning=reasoning_param.view(1, 1, -1).expand(B, T, -1).clone(), # z_init
            x_input=x_input,  # x (static, unmodified)
        )


class DualStreamStep(nn.Module):
    """
    Performs one dual-stream update step following TRM paper.
    
    Update pattern:
      1. z' = block(x + y + z + step_emb)   # reasoning conditioned on input
      2. y' = block(y + z' + step_emb)      # answer NOT conditioned on input
    
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
        # Step 1: z-update - reasoning uses x, y, z
        u_z = state.x_input + state.solution + state.reasoning + step_emb
        reasoning_new = self.block(u_z)
        
        # Step 2: y-update - solution uses y, z' (NOT x!)
        u_y = state.solution + reasoning_new + step_emb
        solution_new = self.block(u_y)
        
        return DualStreamState(
            solution=solution_new,
            reasoning=reasoning_new,
            x_input=state.x_input,  # static, unchanged
        )
