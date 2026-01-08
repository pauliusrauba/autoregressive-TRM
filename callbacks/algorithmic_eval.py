# callbacks/algorithmic_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import pytorch_lightning as pl

from data_modules.algorithmic_char import _build_tokenizer, _make_example


@torch.no_grad()
def evaluate_algorithmic(
    model,
    task: str,
    length: int,
    n: int,
    seed: int,
    device: torch.device,
) -> Dict[str, float]:
    """
    Returns detailed metrics for the algorithmic task at a given length.
    
    Metrics returned:
    - char_acc: Overall character accuracy
    - seq_acc: Full sequence accuracy  
    - first_char_acc: Accuracy on first output character
    - last_char_acc: Accuracy on last output character
    - pos_q1/q2/q3/q4_acc: Quartile position accuracies
    """
    tok = _build_tokenizer()

    # Ensure deterministic sample generation
    gen = torch.Generator(device="cpu").manual_seed(seed)

    total_chars = 0
    correct_chars = 0
    correct_seqs = 0
    
    # Position-wise tracking
    out_len = length if task in {"copy", "reverse"} else (length + 1)
    position_correct = [0] * out_len
    position_total = [0] * out_len
    
    # First/last char tracking
    first_correct = 0
    last_correct = 0
    first_total = 0
    last_total = 0

    was_training = model.training
    model.eval()

    for _ in range(n):
        prompt, target, _full = _make_example(task, length, gen)

        # Tokenize prompt (don't assume len(prompt) == #tokens)
        prompt_ids = tok.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        y_ids = model.generate(x, max_new_tokens=out_len)[0].tolist()

        # Slice generated tokens by token length, not character length
        gen_ids = y_ids[len(prompt_ids) : len(prompt_ids) + out_len]
        pred = tok.decode(gen_ids)

        # Accuracy on output only
        if len(pred) != len(target):
            # Length mismatch - count as all wrong
            total_chars += len(target)
            first_total += 1
            last_total += 1
            for i in range(min(len(target), out_len)):
                position_total[i] += 1
            continue

        # Character-level accuracy
        for i, (a, b) in enumerate(zip(pred, target)):
            match = int(a == b)
            correct_chars += match
            total_chars += 1
            if i < out_len:
                position_correct[i] += match
                position_total[i] += 1
        
        # First/last character accuracy
        if len(target) > 0:
            first_correct += int(pred[0] == target[0])
            first_total += 1
            last_correct += int(pred[-1] == target[-1])
            last_total += 1
        
        # Sequence accuracy
        correct_seqs += int(pred == target)

    if was_training:
        model.train()

    # Compute metrics
    metrics = {
        "char_acc": correct_chars / max(1, total_chars),
        "seq_acc": correct_seqs / max(1, n),
        "first_char_acc": first_correct / max(1, first_total),
        "last_char_acc": last_correct / max(1, last_total),
    }
    
    # Add quartile position accuracies (beginning, middle, end of output)
    if out_len >= 4:
        q1_end = out_len // 4
        q2_end = out_len // 2
        q3_end = 3 * out_len // 4
        
        q1_correct = sum(position_correct[:q1_end])
        q1_total = sum(position_total[:q1_end])
        q2_correct = sum(position_correct[q1_end:q2_end])
        q2_total = sum(position_total[q1_end:q2_end])
        q3_correct = sum(position_correct[q2_end:q3_end])
        q3_total = sum(position_total[q2_end:q3_end])
        q4_correct = sum(position_correct[q3_end:])
        q4_total = sum(position_total[q3_end:])
        
        metrics["pos_q1_acc"] = q1_correct / max(1, q1_total)  # First quarter
        metrics["pos_q2_acc"] = q2_correct / max(1, q2_total)  # Second quarter
        metrics["pos_q3_acc"] = q3_correct / max(1, q3_total)  # Third quarter
        metrics["pos_q4_acc"] = q4_correct / max(1, q4_total)  # Last quarter (hardest for addition)
    
    return metrics


@dataclass
class AlgoEvalSpec:
    task: str
    length: int
    n: int


class AlgorithmicEvalCallback(pl.Callback):
    """
    Runs algorithmic generation eval periodically and logs to the active logger (W&B).
    Designed to run at the same cadence as your val_check_interval.
    
    Logs metrics including:
    - char_acc, seq_acc: Overall accuracy
    - first_char_acc, last_char_acc: First/last position accuracy
    - pos_q1/q2/q3/q4_acc: Quartile position accuracy
    - model/param_count: Total model parameters (logged once)
    """

    def __init__(
        self,
        specs: List[AlgoEvalSpec],
        seed: int = 1337,
        log_on_steps: bool = True,
    ):
        super().__init__()
        self.specs = specs
        self.seed = seed
        self.log_on_steps = log_on_steps
        self._logged_params = False

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Avoid running on sanity check
        if trainer.sanity_checking:
            return

        device = pl_module.device
        step = trainer.global_step

        metrics: Dict[str, float] = {}
        
        # Log parameter count once
        if not self._logged_params:
            total_params = sum(p.numel() for p in pl_module.parameters())
            metrics["model/param_count"] = total_params
            metrics["model/param_count_M"] = total_params / 1e6
            self._logged_params = True
        
        # Evaluate at each length
        for spec in self.specs:
            eval_metrics = evaluate_algorithmic(
                model=pl_module,
                task=spec.task,
                length=spec.length,
                n=spec.n,
                seed=self.seed,
                device=device,
            )
            
            prefix = f"TaskEvaluation/{spec.task}/L{spec.length}"
            for key, value in eval_metrics.items():
                metrics[f"{prefix}/{key}"] = value

        # Update trainer's callback_metrics so ModelCheckpoint can see them
        # Convert to tensors as required by Lightning's metric comparison
        tensor_metrics = {k: torch.tensor(v) for k, v in metrics.items()}
        trainer.callback_metrics.update(tensor_metrics)
        
        # Also log to W&B at the current global step for proper alignment
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
