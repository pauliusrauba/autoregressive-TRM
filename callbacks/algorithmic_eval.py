# callbacks/algorithmic_eval.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

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
) -> Tuple[float, float]:
    """
    Returns (char_acc, seq_acc) for the algorithmic task at a given length.
    Uses model.generate() like your standalone script.
    """
    tok = _build_tokenizer()

    # Ensure deterministic sample generation
    gen = torch.Generator(device="cpu").manual_seed(seed)

    total_chars = 0
    correct_chars = 0
    correct_seqs = 0

    # Output length rule from your script
    out_len = length if task in {"copy", "reverse"} else (length + 1)

    was_training = model.training
    model.eval()

    for _ in range(n):
        prompt, target, _full = _make_example(task, length, gen)

        # Tokenize prompt (don’t assume len(prompt) == #tokens)
        prompt_ids = tok.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)

        y_ids = model.generate(x, max_new_tokens=out_len)[0].tolist()

        # Slice generated tokens by token length, not character length
        gen_ids = y_ids[len(prompt_ids) : len(prompt_ids) + out_len]
        pred = tok.decode(gen_ids)

        # Accuracy on output only
        if len(pred) != len(target):
            # If this happens, something about tokenization/generation is off;
            # treat as incorrect sequence but keep going.
            correct_seqs += 0
            total_chars += len(target)
            continue

        char_matches = sum(int(a == b) for a, b in zip(pred, target))
        correct_chars += char_matches
        total_chars += len(target)
        correct_seqs += int(pred == target)

    if was_training:
        model.train()

    char_acc = correct_chars / max(1, total_chars)
    seq_acc = correct_seqs / max(1, n)
    return char_acc, seq_acc


@dataclass
class AlgoEvalSpec:
    task: str
    length: int
    n: int


class AlgorithmicEvalCallback(pl.Callback):
    """
    Runs algorithmic generation eval periodically and logs to the active logger (W&B).
    Designed to run at the same cadence as your val_check_interval.
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

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # Avoid running on sanity check
        if trainer.sanity_checking:
            return

        device = pl_module.device

        metrics: Dict[str, float] = {}
        for spec in self.specs:
            char_acc, seq_acc = evaluate_algorithmic(
                model=pl_module,
                task=spec.task,
                length=spec.length,
                n=spec.n,
                seed=self.seed,
                device=device,
            )
            prefix = f"TaskEvaluation/{spec.task}/L{spec.length}"
            metrics[f"{prefix}/char_acc"] = char_acc
            metrics[f"{prefix}/seq_acc"] = seq_acc

        # Log at the current global step so W&B aligns with training curves
        step = trainer.global_step
        if trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=step)
