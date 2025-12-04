# trm.py

from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn

from single_recursion import RecurrenceEngine, LatentPairState, ComputationalBlockConfig

@dataclass
class TRMConfig:
    """
    Config for the TRM with ACT-style halting.

    hidden_size:            D (dimension of solution, reasoning, input)
    H_cycles:               outer recursion cycles T
    L_cycles:               inner reasoning updates n
    num_layers:             depth of f_θ (UpdateNetwork)
    halt_max_steps:         maximum number of deep-supervision steps
    halt_exploration_prob:  probability of enforcing extra minimum steps (exploration)
    use_q_continue:         if True, model also outputs a Q-continue head (kept for API parity)
    """
    hidden_size: int
    H_cycles: int
    L_cycles: int
    num_layers: int
    halt_max_steps: int
    halt_exploration_prob: float
    use_q_continue: bool = False


@dataclass
class SupervisionLoopState:
    """
    Outer state for the TRM with ACT-style halting.

    latent_state : LatentPairState(solution, reasoning)  (per-sequence recurrent latent)
    steps        : (B,) int32                            supervision steps taken so far
    halted       : (B,) bool                             which sequences are done (halted)
    """
    latent_state: LatentPairState
    steps: torch.Tensor
    halted: torch.Tensor


class TRMModel(nn.Module):
    """
    Full Tiny Recursive Model (TRM) with:

    - RecurrenceEngine over (solution, reasoning)
    - Regression head on solution[:, 0, :] -> scalar prediction per sample
    - Halting head on solution[:, 0, :]   -> 2 logits (halt vs continue)
    - ACT-style outer wrapper:
        * maintains (solution, reasoning) across supervision steps
        * tracks steps and halting
        * can be called repeatedly until all halted or max steps reached.

    Designed for the dataset:
        x_emb in ℝ^{B×D}, y_true in ℝ^{B×1}.
    """

    def __init__(self, cfg: TRMConfig):
        super().__init__()
        self.cfg = cfg

        # f_θ core: sequence length is L=1 for the dataset
        block_cfg = ComputationalBlockConfig(
            seq_len=1,              # L = 1 (tabular / single token)
            hidden_size=cfg.hidden_size,
            num_heads=1,            # 1 head is sufficient for experiments
            expansion=2.0,
            rms_norm_eps=1e-5,
            mlp_t=False,
        )

        self.core = RecurrenceEngine(
            block_config=block_cfg,
            num_layers=cfg.num_layers,
            H_cycles=cfg.H_cycles,
            L_cycles=cfg.L_cycles,
        )

        D = cfg.hidden_size

        # Regression head: solution[:, 0, :] -> scalar
        self.reg_head = nn.Linear(D, 1)

        # Halting head: solution[:, 0, :] -> 2 logits [halt, continue]
        self.q_head = nn.Linear(D, 2)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5.0)  # bias toward "continue" at the start

    # ----------------------------------------------------------------------
    # State helpers
    # ----------------------------------------------------------------------

    def initial_state(self, batch_size: int, device=None) -> SupervisionLoopState:
        """
        Create an initial outer state for a batch of given size.

        All sequences are marked as halted initially; on the first forward call,
        we reset latent_state wherever halted=True to treat them as "fresh".
        """
        device = device if device is not None else self.reg_head.weight.device
        latent_state = self.core.init_state(
            batch_size=batch_size,
            seq_len=1,
            device=device,
        )

        steps = torch.zeros((batch_size,), dtype=torch.int32, device=device)
        halted = torch.ones((batch_size,), dtype=torch.bool, device=device)

        return SupervisionLoopState(
            latent_state=latent_state,
            steps=steps,
            halted=halted,
        )

    # ----------------------------------------------------------------------
    # One deep-supervision + ACT step
    # ----------------------------------------------------------------------

    def forward(
        self,
        state: SupervisionLoopState,
        x_emb: torch.Tensor,
    ) -> Tuple[SupervisionLoopState, Dict[str, torch.Tensor]]:
        """
        Run one *supervision step* of the TRM with ACT-style halting.

        Inputs:
            state : SupervisionLoopState (latent_state, steps, halted)
            x_emb : (B, D) float embeddings (our input)

        Returns:
            new_state: updated SupervisionLoopState
            outputs:   {
                            "y_pred": (B, 1),
                            "q_halt_logits": (B,),
                            "q_continue_logits": (B,),
                        }
        """
        device = x_emb.device
        B, D = x_emb.shape
        assert D == self.cfg.hidden_size

        # Interpret input as sequence of length 1: (B, 1, D)
        input_seq = x_emb.view(B, 1, D)

        # If a sequence is halted, we reset its latent state and steps
        latent_state = self.core.reset_state(
            state.latent_state,
            reset_mask=state.halted.to(device),
        )

        steps = torch.where(
            state.halted.to(device),
            torch.zeros_like(state.steps),
            state.steps,
        )

        # Run one TRM recursion step over (solution, reasoning) with the current input_seq
        latent_state = self.core(latent_state, input_seq, cos_sin=None)

        # Read out from solution[:, 0, :]
        solution_token = latent_state.solution[:, 0, :]  # (B, D)

        # Regression output
        y_pred = self.reg_head(solution_token)   # (B, 1)

        # Halting logits
        q_logits = self.q_head(solution_token)   # (B, 2)
        q_halt_logits = q_logits[:, 0]           # (B,)
        q_continue_logits = q_logits[:, 1]       # (B,)

        outputs = {
            "y_pred": y_pred,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            # Update supervision step counter
            steps = steps + 1

            # Base condition: always halt at max steps
            is_last_step = steps >= self.cfg.halt_max_steps
            halted = is_last_step.clone()

            # ACT halting rule during training:
            # we let the model choose to halt earlier if it wants
            if self.training and self.cfg.halt_max_steps > 1:
                # simple rule: halt when q_halt_logits > 0 (sigmoid > 0.5)
                halted = halted | (q_halt_logits > 0)

                # Exploration: enforce a random minimum number of steps before halting
                if self.cfg.halt_exploration_prob > 0.0:
                    min_halt_steps = (
                        (torch.rand_like(q_halt_logits) < self.cfg.halt_exploration_prob)
                        * torch.randint_like(
                            steps, low=2, high=self.cfg.halt_max_steps + 1
                        )
                    )
                    halted = halted & (steps >= min_halt_steps)

            # Detach latent state between supervision steps
            latent_state_detached = LatentPairState(
                solution=latent_state.solution.detach(),
                reasoning=latent_state.reasoning.detach(),
            )

        new_state = SupervisionLoopState(
            latent_state=latent_state_detached,
            steps=steps,
            halted=halted,
        )
        return new_state, outputs
