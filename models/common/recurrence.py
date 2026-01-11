# models/common/recurrence.py
"""
Hierarchical Recurrence Engine for fixed-point iteration.

This module isolates the H×L nested loop structure with configurable
gradient detachment. It implements the TRM update pattern:

  z' = block(x + y + z + step_emb)   # reasoning uses input
  y' = block(y + z' + step_emb)      # answer does NOT use input

Delta from flat iteration (UT-Level1):
  - Nested loops: L inner iterations (reasoning), H outer iterations (solution)
  - Partial gradient detachment: only last H-cycle gets gradients
"""
import torch
import torch.nn as nn
from models.common.dual_stream import DualStreamState


class RecurrenceEngine(nn.Module):
    """
    Runs hierarchical H×L fixed-point iteration.
    
    Structure per ACT step:
      - H outer loops (solution updates)
      - L inner loops per outer loop (reasoning refinements)
      - Gradients only flow through the final H-th cycle
    
    Args:
        block: The shared transformer block
        n_inner_loops: L - reasoning refinements per solution update
        n_outer_loops: H - solution update cycles per ACT step
    """
    
    def __init__(
        self,
        block: nn.Module,
        n_inner_loops: int = 4,
        n_outer_loops: int = 4,
    ):
        super().__init__()
        self.block = block
        self.n_inner_loops = n_inner_loops
        self.n_outer_loops = n_outer_loops
    
    def _one_h_cycle(
        self,
        state: DualStreamState,
        step_emb: torch.Tensor,
    ) -> DualStreamState:
        """
        Run one H-cycle: L reasoning updates + 1 solution update.
        
        Update pattern (TRM paper):
          - z-update: z' = block(x + y + z + step)  # uses x
          - y-update: y' = block(y + z' + step)     # does NOT use x
        """
        # L inner loops: refine reasoning (z-updates)
        # Each z-update uses x, y, z
        for l in range(self.n_inner_loops):
            u_z = state.x_input + state.solution + state.reasoning + step_emb
            state = DualStreamState(
                solution=state.solution,      # y unchanged during z-updates
                reasoning=self.block(u_z),    # z updates
                x_input=state.x_input,
            )
        
        # 1 solution update (y-update)
        # y-update uses y, z (not x)
        u_y = state.solution + state.reasoning + step_emb
        state = DualStreamState(
            solution=self.block(u_y),
            reasoning=state.reasoning,
            x_input=state.x_input,
        )
        
        return state
    
    def forward(
        self,
        state: DualStreamState,
        step_emb: torch.Tensor,
    ) -> DualStreamState:
        """
        Run full H×L recurrence with partial gradient detachment.
        
        - First H-1 cycles: no gradients (fixed-point settling)
        - Last cycle: gradients enabled (optimization step)
        """
        # Phase 1: H-1 cycles without gradients
        with torch.no_grad():
            for h in range(self.n_outer_loops - 1):
                state = self._one_h_cycle(state, step_emb)
        
        # Phase 2: Final cycle with gradients
        state = self._one_h_cycle(state, step_emb)
        
        return state


class RecurrenceEngineWithTBPTT(RecurrenceEngine):
    """
    Recurrence engine with full Truncated BPTT between ACT steps.
    
    Delta from RecurrenceEngine:
      - Detaches state at the START of each ACT step
      - This prevents gradients flowing between ACT steps entirely
    """
    
    def forward(
        self,
        state: DualStreamState,
        step_emb: torch.Tensor,
        detach_input: bool = True,  # NEW: control TBPTT
    ) -> DualStreamState:
        """
        Run recurrence with optional input detachment (TBPTT).
        """
        if detach_input:
            state = DualStreamState(
                solution=state.solution.detach(),
                reasoning=state.reasoning.detach(),
                x_input=state.x_input,  # static input stays attached
            )
        
        # Then run standard H×L recurrence
        return super().forward(state, step_emb)
