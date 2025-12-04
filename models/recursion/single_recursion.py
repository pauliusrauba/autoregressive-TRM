# recurrence.py

from dataclasses import dataclass
import torch
import torch.nn as nn

from update_network import ComputationalBlockConfig, UpdateNetwork

@dataclass
class LatentPairState:
    """
    Latent state of the Tiny Recursive Model at one "supervision step".

    solution : current solution embedding (B, L, D)
    reasoning: latent reasoning embedding (B, L, D)
    """
    solution: torch.Tensor
    reasoning: torch.Tensor


class RecurrenceEngine(nn.Module):
    """
    Core TRM recursion over (solution, reasoning) for a single deep-supervision step.

    This wraps the shared tiny network f_θ (UpdateNetwork) and applies the TRM-style update:

        for H_cycles - 1 times (no grad):
            repeat L_cycles times:
                reasoning <- f_θ(reasoning, solution + input_seq)
            solution <- f_θ(solution, reasoning)

        then once with grad:
            repeat L_cycles times:
                reasoning <- f_θ(reasoning, solution + input_seq)
            solution <- f_θ(solution, reasoning)

    Here:
        - input_seq is the embedded "question" (same shape as solution/reasoning: (B, L, D)).
        - solution is interpreted as the current answer representation.
        - reasoning is the latent reasoning state.
    """

    def __init__(
        self,
        block_config: ComputationalBlockConfig,
        num_layers: int,
        H_cycles: int,
        L_cycles: int,
    ):
        super().__init__()
        # f_θ: shared tiny network used to update either reasoning or solution
        self.update_network = UpdateNetwork(block_config, num_layers=num_layers)
        self.H_cycles = H_cycles
        self.L_cycles = L_cycles

        D = block_config.hidden_size

        # Learnable initial states solution_0, reasoning_0 (vectors of size D, broadcast later)
        self.solution_init = nn.Parameter(torch.randn(D) * 0.5)
        self.reasoning_init = nn.Parameter(torch.randn(D) * 0.5)

    # ------------------------------------------------------------------
    # State helpers
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, seq_len: int, device=None) -> LatentPairState:
        """
        Create an initial (solution, reasoning) state for a given batch and sequence length.

        solution_0, reasoning_0 (D,) are broadcast to shape (B, L, D).
        """
        device = device if device is not None else self.solution_init.device
        D = self.solution_init.shape[0]

        solution0 = (
            self.solution_init.to(device).view(1, 1, D).expand(batch_size, seq_len, D)
        )
        reasoning0 = (
            self.reasoning_init.to(device).view(1, 1, D).expand(batch_size, seq_len, D)
        )

        return LatentPairState(solution=solution0.clone(), reasoning=reasoning0.clone())

    def reset_state(self, state: LatentPairState, reset_mask: torch.Tensor) -> LatentPairState:
        """
        Reset solution/reasoning to their initial values where reset_mask is True (shape (B,)).

        Useful when reusing the state across different inputs and
        selectively restarting some batch elements.
        """
        # reset_mask: (B,)
        B, L, D = state.solution.shape
        mask = reset_mask.view(B, 1, 1)  # (B,1,1)

        solution0 = (
            self.solution_init.view(1, 1, D).to(state.solution.device).expand(B, L, D)
        )
        reasoning0 = (
            self.reasoning_init.view(1, 1, D).to(state.reasoning.device).expand(B, L, D)
        )

        solution = torch.where(mask, solution0, state.solution)
        reasoning = torch.where(mask, reasoning0, state.reasoning)
        return LatentPairState(solution=solution, reasoning=reasoning)

    # ------------------------------------------------------------------
    # Forward: one TRM recursion step
    # ------------------------------------------------------------------

    def forward(
        self,
        state: LatentPairState,
        input_seq: torch.Tensor,
        cos_sin=None,
    ) -> LatentPairState:
        """
        Run one full TRM recursion step on the current latent state with input_seq.

        Args:
            state:     LatentPairState(solution, reasoning), each of shape (B, L, D)
            input_seq: embedded "question" sequence, shape (B, L, D)
            cos_sin:   positional info (can be None; passed into the underlying blocks)

        Returns:
            LatentPairState(solution_new, reasoning_new) with the same shapes.

        NOTE: we do NOT detach here; detaching across supervision steps
              (for deep supervision / ACT) is the job of a higher-level wrapper.
        """
        solution = state.solution
        reasoning = state.reasoning

        # --------------------------------------------------------------
        # H_cycles - 1 outer cycles WITHOUT gradients
        # --------------------------------------------------------------
        with torch.no_grad():
            for _ in range(self.H_cycles - 1):
                # L_cycles reasoning updates: reasoning <- f_θ(reasoning, solution + input_seq)
                for _ in range(self.L_cycles):
                    reasoning = self.update_network(
                        current_state=reasoning,
                        conditioning_signal=solution + input_seq,
                        cos_sin=cos_sin,
                    )
                # One solution update: solution <- f_θ(solution, reasoning)
                solution = self.update_network(
                    current_state=solution,
                    conditioning_signal=reasoning,
                    cos_sin=cos_sin,
                )

        # --------------------------------------------------------------
        # Final cycle WITH gradients
        # --------------------------------------------------------------
        for _ in range(self.L_cycles):
            reasoning = self.update_network(
                current_state=reasoning,
                conditioning_signal=solution + input_seq,
                cos_sin=cos_sin,
            )

        solution = self.update_network(
            current_state=solution,
            conditioning_signal=reasoning,
            cos_sin=cos_sin,
        )

        return LatentPairState(solution=solution, reasoning=reasoning)
