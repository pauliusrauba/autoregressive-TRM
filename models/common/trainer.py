# models/trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

class BaseLitGPT(pl.LightningModule):
    def __init__(self, vocab_size, block_size, n_embd, lr, **kwargs):
        super().__init__()
        # We save all arguments (even those passed to children)
        self.save_hyperparameters()
        
        # Flag to force full compute (no early halting) - useful for controlled experiments
        self._force_full_compute = False
        
        # Inference-time compute scaling: allows extrapolating beyond trained steps
        self._inference_steps = None  # None means use training steps (n_layer or max_act_steps)
    
    def set_full_compute(self, force: bool = True):
        """Enable/disable forced full compute (no early halting).
        
        When True, adaptive models (UT, TRM, etc.) will always run max_act_steps.
        When False (default), models use their natural halting behavior.
        """
        self._force_full_compute = force
        return self  # Allow chaining
    
    def set_inference_steps(self, steps: int = None):
        """Set the number of iteration steps to use at inference time.
        
        This enables compute extrapolation: using more (or fewer) steps at inference
        than the model was trained with.
        
        Args:
            steps: Number of steps to use. If None, uses training steps (n_layer or max_act_steps).
                   Can be larger than trained steps for compute extrapolation.
        
        Note:
            - For models with step embeddings, extra steps will reuse the last embedding.
            - For models without step embeddings (or with use_step_embeddings=False),
              extrapolation is clean with no clamping needed.
            - For ACT models, you may also want set_full_compute(True) to prevent early halting.
        """
        self._inference_steps = steps
        return self  # Allow chaining
    
    def get_inference_steps(self, default_steps: int) -> int:
        """Get the number of steps to use for inference.
        
        Args:
            default_steps: The default (training) number of steps.
        
        Returns:
            The number of steps to use (either _inference_steps if set, or default).
        """
        return self._inference_steps if self._inference_steps is not None else default_steps
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def training_step(self, batch, batch_idx):
        x, y = batch
        # Assumes self.forward returns (logits, loss)
        _, loss = self(x, y)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self(x, y)
        self.log('val_loss', loss, prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.hparams.block_size:]
            # Calls the child class forward
            logits, _ = self(idx_cond)       
            logits = logits[:, -1, :]        
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx