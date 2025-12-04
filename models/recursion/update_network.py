# update_network.py

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Config for the block
# -----------------------

@dataclass
class ComputationalBlockConfig:
    """
    Configuration for a single ComputationalBlock.

    seq_len:      length of the sequence (L)
    hidden_size:  model dimension (D)
    num_heads:    number of attention heads
    expansion:    MLP expansion factor
    rms_norm_eps: epsilon for LayerNorm
    mlp_t:        if True, use an MLP along time instead of attention for token mixing
    """

    seq_len: int
    hidden_size: int
    num_heads: int
    expansion: float
    rms_norm_eps: float = 1e-5
    mlp_t: bool = False


class ComputationalBlock(nn.Module):
    """
    One transformer-like / mixer-like layer.

    - If mlp_t = True:
        * apply an MLP along the time dimension (sequence length L) to mix tokens
    - Else:
        * apply self-attention over tokens to mix tokens

    Then always:
        * apply an MLP over the hidden dimension (feature mixing)

    Both stages use residual connections + LayerNorm.

    Input:  hidden_states (B, L, D)
    Output: hidden_states (B, L, D)
    """

    def __init__(self, config: ComputationalBlockConfig) -> None:
        super().__init__()
        self.config = config

        D = config.hidden_size
        L = config.seq_len
        exp_D = int(config.expansion * D)
        exp_L = int(config.expansion * L)

        # --- token mixing: either MLP over time or self-attention over tokens ---
        if self.config.mlp_t:
            # MLP over time dimension (length L), applied per feature channel
            self.mlp_t = nn.Sequential(
                nn.Linear(L, exp_L),
                nn.GELU(),
                nn.Linear(exp_L, L),
            )
            self.self_attn = None
        else:
            # Standard multi-head self-attention over tokens
            self.self_attn = nn.MultiheadAttention(
                embed_dim=D,
                num_heads=config.num_heads,
                batch_first=True,  # inputs: (B, L, D)
            )
            self.mlp_t = None

        # --- feature mixing MLP over hidden dimension ---
        self.mlp = nn.Sequential(
            nn.Linear(D, exp_D),
            nn.GELU(),
            nn.Linear(exp_D, D),
        )

        # layer norms (always over hidden dimension D)
        self.norm1 = nn.LayerNorm(D, eps=config.rms_norm_eps)
        self.norm2 = nn.LayerNorm(D, eps=config.rms_norm_eps)

    def forward(self, cos_sin: Optional[object], hidden_states: torch.Tensor) -> torch.Tensor:
        """
        cos_sin: kept for API compatibility, unused here.
        hidden_states: (B, L, D)
        """
        B, L, D = hidden_states.shape

        # ----- token mixing -----
        if self.config.mlp_t:
            # MLP over time: operate on (B, D, L)
            x = hidden_states.transpose(1, 2)      # (B, D, L)
            x = x + self.mlp_t(x)                  # residual in time space
            x = x.transpose(1, 2)                  # back to (B, L, D)
            hidden_states = self.norm1(x)
        else:
            # self-attention over tokens
            attn_out, _ = self.self_attn(
                hidden_states,  # query: (B, L, D)
                hidden_states,  # key
                hidden_states,  # value
                need_weights=False,
            )
            hidden_states = self.norm1(hidden_states + attn_out)

        # ----- feature mixing -----
        mlp_out = self.mlp(hidden_states)          # (B, L, D)
        hidden_states = self.norm2(hidden_states + mlp_out)

        return hidden_states


class UpdateNetwork(nn.Module):
    """
    Tiny network f_θ used inside TRM.

    Logically equivalent to TinyRecursiveReasoningModel_ACTV1ReasoningModule:

    - Internally: a stack of ComputationalBlock layers.
    - Externally: one call updates a latent tensor given a conditioning signal.

    Forward:
        current_state:       (B, L, D)   current latent (e.g. 'reasoning' z or 'solution' y)
        conditioning_signal: (B, L, D)   context signal (e.g. y + x when updating z, or z when updating y)
        cos_sin:             positional info (kept for API compatibility; ignored here)

    Returns:
        updated_state: (B, L, D)
    """

    def __init__(self, block_config: ComputationalBlockConfig, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [ComputationalBlock(block_config) for _ in range(num_layers)]
        )

    def forward(
        self,
        current_state: torch.Tensor,
        conditioning_signal: torch.Tensor,
        cos_sin=None,
    ) -> torch.Tensor:
        """
        Apply a single f_θ update:

            updated_state = f_θ(current_state, conditioning_signal)

        In TRM this is used as:
            reasoning <- f_θ(reasoning, solution + input)
            solution  <- f_θ(solution, reasoning)
        """
        x = current_state + conditioning_signal  # simple additive conditioning
        for layer in self.layers:
            x = layer(cos_sin=cos_sin, hidden_states=x)
        return x
