import torch
import torch.nn as nn
import torch.nn.functional as F
from models.common.trainer import BaseLitGPT
from models.common.layers import Block


class UTLevel2(BaseLitGPT):
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
        n_inner_loops: int = 4, # L: How many times to refine reasoning per solution step
        n_outer_loops: int = 4 # H: How many times to update solution per ACT step
    ):
        super().__init__(vocab_size, block_size, n_embd, lr)
        self.save_hyperparameters()

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.step_embedding_table = nn.Embedding(n_layer, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Change: Add a learnable reasoning parameter
        self.reasoning_param = nn.Parameter(torch.randn(n_embd) * 0.02)

        # New for this level: Add inner and outer loop parameters
        self.n_inner_loops = n_inner_loops
        self.n_outer_loops = n_outer_loops

        self.shared_block = Block(n_embd, n_head, block_size, dropout)
        self.halt_head = nn.Linear(n_embd, 1)

        # Store ACT hyperparams
        self.max_steps = n_layer
        self.halt_threshold = 1.0 - 1e-6 # For numerical stabilitt
        self.act_epsilon = 0.01 # for bias initialization

        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)
        self.n_layer = n_layer

        self.apply(self._init_weights)

    def forward(self, idx, targets=None):
            B, T = idx.shape
            device = idx.device
            assert T <= self.hparams.block_size

            # 1. Prepare Static Input
            # In TRM, 'x' is the static "Question" that doesn't change layer-to-layer
            tok_emb = self.token_embedding_table(idx)
            pos = torch.arange(T, device=device)
            pos_emb = self.position_embedding_table(pos)
            x_input = tok_emb + pos_emb 
            # Add reasoning bias to input once
            x_input = x_input + self.reasoning_param.unsqueeze(0).unsqueeze(1)

            # 2. Initialize States
            solution = x_input.clone()
            reasoning = self.reasoning_param.view(1, 1, -1).expand(B, T, -1).clone()

            # ACT Tensors
            halted = torch.zeros(B, T, dtype=torch.bool, device=device)
            halt_probs_accum = torch.zeros(B, T, device=device)
            remainders = torch.zeros(B, T, device=device)
            n_updates = torch.zeros(B, T, device=device)
            output_accum = torch.zeros(B, T, self.hparams.n_embd, device=device)

            # --- The ACT Loop (Supervision Steps) ---
            for step in range(self.hparams.n_layer):
                step_emb = self.step_embedding_table(torch.tensor(step, device=device))
                
                # Temporary holders for the recurrence engine
                curr_solution = solution
                curr_reasoning = reasoning

                # ==================================================
                # RECURRENCE ENGINE (H-Cycles & L-Cycles)
                # ==================================================
                
                # Phase 1: Fixed-Point Iteration (No Gradients)
                # We run H-1 cycles to let the state "settle" without building graph
                with torch.no_grad():
                    for h in range(self.n_outer_loops - 1):
                        # Inner Loop: Refine Reasoning L times
                        for l in range(self.n_inner_loops):
                            cond_z = curr_solution + x_input
                            # We pass step_emb to know which ACT step we are in
                            u_z = curr_reasoning + cond_z + step_emb
                            curr_reasoning = self.shared_block(u_z)
                        
                        # Outer Loop part 2: Update Solution 1 time
                        cond_y = curr_reasoning
                        u_y = curr_solution + cond_y + step_emb
                        curr_solution = self.shared_block(u_y)

                # Phase 2: Final Optimization Step (With Gradients)
                # We run the H-th cycle with gradients enabled
                
                # Inner Loop: Refine Reasoning L times
                for l in range(self.n_inner_loops):
                    cond_z = curr_solution + x_input
                    u_z = curr_reasoning + cond_z + step_emb
                    curr_reasoning = self.shared_block(u_z)

                # Update Solution 1 time
                cond_y = curr_reasoning
                u_y = curr_solution + cond_y + step_emb
                curr_solution = self.shared_block(u_y)

                # Update final states for this ACT step
                solution_new = curr_solution
                reasoning_new = curr_reasoning
                
                # ==================================================
                # End Recurrence Engine
                # ==================================================

                # ACT Logic (Checks if 'solution_new' is ready)
                halt_logits = self.halt_head(solution_new).squeeze(-1)
                halt_prob = torch.sigmoid(halt_logits)

                still_running = ~halted
                new_halted = (halt_probs_accum + halt_prob >= self.halt_threshold) & still_running
                
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
                
                # Pass states to next ACT step
                solution = solution_new
                reasoning = reasoning_new
                
                if halted.all():
                    break

            # Final cleanup for ACT
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

    