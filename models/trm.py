import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block


class TRM(BaseLitGPT):
    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        n_head: int,
        n_layer: int,
        dropout: float,
        lr: float,
        ponder_cost_weight: float = 0.01,
        n_inner_loops: int = 2,
        n_outer_loops: int = 2,
        # NEW: Probability of forcing extra thinking steps
        halt_exploration_prob: float = 0.25,
        max_act_steps: int = None,
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()

        # max_act_steps controls how many ACT steps the model can use
        # If not specified, defaults to n_layer for backward compatibility
        self.max_act_steps = max_act_steps if max_act_steps is not None else n_layer

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(self.max_act_steps, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.reasoning_param = nn.Parameter(torch.randn(n_embd) * 0.02)

        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.halt_head = nn.Linear(n_embd, 1)
        
        # Initialize bias to negative to encourage starting with "continue"
        with torch.no_grad():
            self.halt_head.bias.fill_(-3.0)

        self.max_steps = self.max_act_steps
        self.halt_threshold = 0.5  # Standard sigmoid threshold
        self.act_epsilon = 0.01

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        
        self.n_inner_loops = n_inner_loops
        self.n_outer_loops = n_outer_loops
        self.halt_exploration_prob = halt_exploration_prob
        self.n_layer = n_layer

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        device = idx.device
        assert T <= self.hparams.block_size

        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=device)
        pos_emb = self.position_embedding_table(pos)
        x_input = tok_emb + pos_emb 
        x_input = x_input + self.reasoning_param.unsqueeze(0).unsqueeze(1)

        solution = x_input.clone()
        reasoning = self.reasoning_param.view(1, 1, -1).expand(B, T, -1).clone()

        # ACT Tensors
        halted = torch.zeros(B, T, dtype=torch.bool, device=device)
        halt_probs_accum = torch.zeros(B, T, device=device)
        remainders = torch.zeros(B, T, device=device)
        n_updates = torch.zeros(B, T, device=device)
        output_accum = torch.zeros(B, T, self.hparams.n_embd, device=device)
        
        # To track randomized minimum steps for exploration
        # (B, T) matrix of random integers between 1 and max_act_steps
        if self.training and self.halt_exploration_prob > 0:
            min_steps = torch.randint(1, self.max_act_steps, (B, T), device=device)
            do_explore = torch.rand((B, T), device=device) < self.halt_exploration_prob
            # If exploring, we enforce step >= min_steps
            # If not exploring, min_steps effectively 0
            min_steps = torch.where(do_explore, min_steps, torch.zeros_like(min_steps))
        else:
            min_steps = torch.zeros((B, T), device=device)

        for step in range(self.max_act_steps):
            
            # --- NEW: Gradient Detachment (TBPTT) ---
            # We detach the state from the previous step.
            # This means gradients from step `t` do not flow back to `t-1`.
            # Each step tries to improve the solution locally.
            solution = solution.detach()
            reasoning = reasoning.detach()

            step_emb = self.step_embedding_table(torch.tensor(step, device=device))
            
            curr_solution = solution
            curr_reasoning = reasoning

            # === Recurrence Engine (Same as Level 2) ===
            with torch.no_grad():
                for h in range(self.n_outer_loops - 1):
                    for l in range(self.n_inner_loops):
                        cond_z = curr_solution + x_input
                        u_z = curr_reasoning + cond_z + step_emb
                        curr_reasoning = self.shared_block(u_z)
                    
                    cond_y = curr_reasoning
                    u_y = curr_solution + cond_y + step_emb
                    curr_solution = self.shared_block(u_y)

            for l in range(self.n_inner_loops):
                cond_z = curr_solution + x_input
                u_z = curr_reasoning + cond_z + step_emb
                curr_reasoning = self.shared_block(u_z)

            cond_y = curr_reasoning
            u_y = curr_solution + cond_y + step_emb
            curr_solution = self.shared_block(u_y)

            solution_new = curr_solution
            reasoning_new = curr_reasoning
            # ===========================================

            # ACT Logic
            halt_logits = self.halt_head(solution_new).squeeze(-1)
            halt_prob = torch.sigmoid(halt_logits)

            still_running = ~halted
            
            # Check standard halting condition
            should_halt = (halt_probs_accum + halt_prob >= 1.0) 
            
            # --- NEW: Halting Exploration Logic ---
            # If we are exploring, we force 'should_halt' to False if step < min_steps
            if self.training:
                forced_continue = (torch.tensor(step, device=device) < min_steps)
                should_halt = should_halt & (~forced_continue)

            new_halted = should_halt & still_running
            
            # Standard ACT accumulations
            remainders = torch.where(new_halted, 1.0 - halt_probs_accum, remainders)
            
            p = torch.where(
                new_halted,
                remainders,
                torch.where(still_running, halt_prob, torch.zeros_like(halt_prob))
            )
            
            output_accum = output_accum + p.unsqueeze(-1) * solution_new
            
            halt_probs_accum = torch.where(
                still_running & ~new_halted,
                halt_probs_accum + halt_prob,
                halt_probs_accum
            )
            
            n_updates = n_updates + still_running.float()
            halted = halted | new_halted
            
            solution = solution_new
            reasoning = reasoning_new
            
            if halted.all():
                break

        still_running = ~halted
        remainders = torch.where(still_running, 1.0 - halt_probs_accum, remainders)
        output_accum = output_accum + (still_running.float() * remainders).unsqueeze(-1) * solution

        ponder_cost = (n_updates + remainders).mean()
        self._last_ponder_cost = ponder_cost.item()

        x = output_accum
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            B, T, C = logits.shape
            ce_loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
            loss = ce_loss + self.hparams.ponder_cost_weight * ponder_cost

        return logits, loss