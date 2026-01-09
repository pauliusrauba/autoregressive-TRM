# eval_inference_scaling.py
"""
Evaluate trained checkpoints with varying inference compute budgets.

This script allows testing how ACT models perform when given more/fewer
steps at inference time than they were trained with. This is key for
understanding the compute-efficiency frontier.

Usage:
    python eval_inference_scaling.py \
        --ckpt-dir /mnt/pdata/pr501/icml2025/checkpoints \
        --output results_inference_scaling.csv \
        --max-act-steps 8 16 32 64 128

Output: CSV with columns for model, task, length, max_act_steps, accuracy metrics
"""

import argparse
import os
import glob
import csv
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

import torch

from data_modules.algorithmic_char import _build_tokenizer, _make_example
from models import model_registry


@dataclass
class EvalResult:
    model_name: str
    checkpoint_path: str
    task: str
    length: int
    max_act_steps: int
    actual_steps_mean: float  # Average steps actually used
    char_acc: float
    seq_acc: float
    param_count: int


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default="/mnt/pdata/pr501/icml2025/checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--ckpt-pattern",
        type=str,
        default="**/last.ckpt",
        help="Glob pattern for checkpoint files (relative to ckpt-dir)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results_inference_scaling.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--max-act-steps",
        type=int,
        nargs="+",
        default=[8, 16, 32, 64],
        help="List of max_act_steps values to test",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["copy", "reverse", "addition"],
        help="Tasks to evaluate",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[20, 40, 60, 80, 100],
        help="Sequence lengths to evaluate",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=100,
        help="Number of samples per evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )
    parser.add_argument(
        "--filter-model",
        type=str,
        default=None,
        help="Only evaluate checkpoints matching this model name pattern",
    )
    return parser.parse_args()


def get_model_name_from_path(ckpt_path: str) -> str:
    """Extract model name from checkpoint path."""
    # Expected format: .../checkpoints/{dataset}_{model}/...
    parent_dir = os.path.basename(os.path.dirname(ckpt_path))
    # Split by underscore and get model name (after task name)
    parts = parent_dir.split("_")
    if len(parts) >= 2:
        # Handle cases like "addition_char_ut_level2" -> "ut_level2"
        # Find where the model name starts
        for i, part in enumerate(parts):
            if part in model_registry or f"{part}_{parts[i+1]}" in model_registry if i+1 < len(parts) else False:
                return "_".join(parts[i:])
        # Fallback: assume last part(s) is model name
        return "_".join(parts[1:])
    return parent_dir


def is_act_model(model_name: str) -> bool:
    """Check if model uses ACT (Adaptive Computation Time)."""
    return model_name.lower() in ("ut", "ut_level1", "ut_level2", "trm")


def load_model_with_overrides(
    ckpt_path: str,
    max_act_steps: Optional[int] = None,
    device: str = "cuda",
):
    """
    Load a model from checkpoint and optionally override max_act_steps.
    
    For ACT models, this allows testing with more/fewer inference steps
    than the model was trained with.
    """
    # Load checkpoint to inspect hyperparameters
    ckpt = torch.load(ckpt_path, map_location="cpu")
    hparams = ckpt.get("hyper_parameters", {})
    
    # Determine model class
    # Try to infer from checkpoint structure or path
    model_name = get_model_name_from_path(ckpt_path)
    
    # Find the right model class
    ModelClass = None
    for name, cls in model_registry.items():
        if name in model_name.lower():
            ModelClass = cls
            break
    
    if ModelClass is None:
        raise ValueError(f"Could not determine model class for {ckpt_path}")
    
    # Override max_act_steps if specified and model supports it
    if max_act_steps is not None and is_act_model(model_name):
        hparams["max_act_steps"] = max_act_steps
    
    # Load model
    model = ModelClass.load_from_checkpoint(ckpt_path, **hparams)
    
    # Also update the runtime attribute if it exists
    if max_act_steps is not None and hasattr(model, "max_act_steps"):
        model.max_act_steps = max_act_steps
        model.max_steps = max_act_steps
        # Resize step embedding table if needed
        if hasattr(model, "step_embedding_table"):
            current_size = model.step_embedding_table.weight.shape[0]
            if max_act_steps > current_size:
                # Expand step embeddings by cycling
                n_embd = model.step_embedding_table.weight.shape[1]
                new_embeddings = torch.nn.Embedding(max_act_steps, n_embd)
                with torch.no_grad():
                    for i in range(max_act_steps):
                        new_embeddings.weight[i] = model.step_embedding_table.weight[i % current_size]
                model.step_embedding_table = new_embeddings
    
    model = model.to(device)
    model.eval()
    
    return model, model_name


@torch.no_grad()
def evaluate_model(
    model,
    task: str,
    length: int,
    n_samples: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    """Evaluate model on algorithmic task."""
    tok = _build_tokenizer()
    gen = torch.Generator(device="cpu").manual_seed(seed)
    
    total_chars = 0
    correct_chars = 0
    correct_seqs = 0
    total_steps = 0  # Track actual ACT steps used
    
    for _ in range(n_samples):
        prompt, target, _ = _make_example(task, length, gen)
        
        prompt_ids = tok.encode(prompt)
        x = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        
        out_len = length if task in {"copy", "reverse"} else (length + 1)
        
        # Track ponder cost if available
        if hasattr(model, "_last_ponder_cost"):
            model._last_ponder_cost = 0
        
        y_ids = model.generate(x, max_new_tokens=out_len)[0].tolist()
        
        # Accumulate steps
        if hasattr(model, "_last_ponder_cost"):
            total_steps += model._last_ponder_cost
        
        gen_ids = y_ids[len(prompt_ids): len(prompt_ids) + out_len]
        pred = tok.decode(gen_ids)
        
        if len(pred) != len(target):
            total_chars += len(target)
            continue
        
        for a, b in zip(pred, target):
            correct_chars += int(a == b)
            total_chars += 1
        
        correct_seqs += int(pred == target)
    
    return {
        "char_acc": correct_chars / max(1, total_chars),
        "seq_acc": correct_seqs / max(1, n_samples),
        "avg_steps": total_steps / max(1, n_samples),
    }


def find_checkpoints(ckpt_dir: str, pattern: str, filter_model: Optional[str] = None) -> List[str]:
    """Find all checkpoint files matching pattern."""
    full_pattern = os.path.join(ckpt_dir, pattern)
    checkpoints = glob.glob(full_pattern, recursive=True)
    
    if filter_model:
        checkpoints = [c for c in checkpoints if filter_model in c]
    
    return sorted(checkpoints)


def main():
    args = parse_args()
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.ckpt_dir, args.ckpt_pattern, args.filter_model)
    print(f"Found {len(checkpoints)} checkpoints")
    
    if not checkpoints:
        print(f"No checkpoints found in {args.ckpt_dir} matching {args.ckpt_pattern}")
        return
    
    results: List[EvalResult] = []
    
    for ckpt_path in checkpoints:
        print(f"\n{'='*60}")
        print(f"Checkpoint: {ckpt_path}")
        
        # Determine which max_act_steps values to test
        model_name = get_model_name_from_path(ckpt_path)
        
        if is_act_model(model_name):
            steps_to_test = args.max_act_steps
        else:
            # Non-ACT models: just test once (max_act_steps doesn't apply)
            steps_to_test = [None]
        
        for max_steps in steps_to_test:
            try:
                model, model_name = load_model_with_overrides(
                    ckpt_path,
                    max_act_steps=max_steps,
                    device=args.device,
                )
                param_count = sum(p.numel() for p in model.parameters())
                
                steps_str = f"steps={max_steps}" if max_steps else "default"
                print(f"\nModel: {model_name} ({steps_str})")
                
                for task in args.tasks:
                    for length in args.lengths:
                        metrics = evaluate_model(
                            model=model,
                            task=task,
                            length=length,
                            n_samples=args.n_samples,
                            seed=args.seed,
                            device=args.device,
                        )
                        
                        result = EvalResult(
                            model_name=model_name,
                            checkpoint_path=ckpt_path,
                            task=task,
                            length=length,
                            max_act_steps=max_steps if max_steps else 0,
                            actual_steps_mean=metrics["avg_steps"],
                            char_acc=metrics["char_acc"],
                            seq_acc=metrics["seq_acc"],
                            param_count=param_count,
                        )
                        results.append(result)
                        
                        print(f"  {task} L={length}: char_acc={metrics['char_acc']:.3f}, "
                              f"seq_acc={metrics['seq_acc']:.3f}, "
                              f"avg_steps={metrics['avg_steps']:.1f}")
                
                # Free memory
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing {ckpt_path} with max_steps={max_steps}: {e}")
                continue
    
    # Write results to CSV
    if results:
        with open(args.output, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "model_name", "checkpoint_path", "task", "length",
                "max_act_steps", "actual_steps_mean", "char_acc", "seq_acc", "param_count"
            ])
            for r in results:
                writer.writerow([
                    r.model_name, r.checkpoint_path, r.task, r.length,
                    r.max_act_steps, r.actual_steps_mean, r.char_acc, r.seq_acc, r.param_count
                ])
        
        print(f"\n{'='*60}")
        print(f"Results saved to {args.output}")
        print(f"Total evaluations: {len(results)}")


if __name__ == "__main__":
    main()
