# models/common/compute.py
"""
Compute budget utilities for normalizing compute across different model architectures.

Compute is measured in "block passes" - the number of times a transformer Block
is applied during a forward pass. This allows fair comparison across architectures
with different iteration patterns.

Block passes per forward:
- GPTBase:    n_layer (unique blocks)
- GPTLevel1:  n_layer (shared block, reused n_layer times)
- GPTLevel2:  n_layer (shared block with step embeddings)
- UT:         n_layer (max, can halt early with ACT)
- UTLevel1:   2 * n_layer (reasoning + solution per ACT step)
- UTLevel2:   n_layer * n_outer_loops * (n_inner_loops + 1)
- TRM:        n_layer * n_outer_loops * (n_inner_loops + 1)
"""

import math
from typing import Dict, Any, Optional, Tuple

DEFAULT_N_INNER_LOOPS = 2
DEFAULT_N_OUTER_LOOPS = 2

MODEL_LOOP_DEFAULTS = {
    'ut_level2': {'n_inner_loops': 2, 'n_outer_loops': 2},
    'trm': {'n_inner_loops': 2, 'n_outer_loops': 2},
}


def get_loop_defaults(model_name: str) -> Dict[str, int]:
    """Get default loop parameters for a model."""
    model_name = model_name.lower()
    return MODEL_LOOP_DEFAULTS.get(model_name, {'n_inner_loops': 4, 'n_outer_loops': 4})


def calculate_block_passes(
    model_name: str,
    n_layer: int,
    n_inner_loops: Optional[int] = None,
    n_outer_loops: Optional[int] = None,
) -> int:
    """
    Calculate the number of block passes for a given model configuration.
    
    For ACT-based models (UT, UTLevel1, UTLevel2, TRM), this returns the
    MAXIMUM block passes (assuming no early halting).
    
    Args:
        model_name: One of 'gpt', 'gpt_level1', 'gpt_level2', 'ut', 'ut_level1', 'ut_level2', 'trm'
        n_layer: Number of layers/iterations
        n_inner_loops: Inner loop count (for UTLevel2/TRM)
        n_outer_loops: Outer loop count (for UTLevel2/TRM)
    
    Returns:
        Number of block passes per forward pass
    """
    model_name = model_name.lower()
    
    if model_name in ("gpt", "gpt_level1", "gpt_level2"):
        # All GPT variants: one block pass per layer
        return n_layer
    
    elif model_name == "ut":
        # UT: one block pass per ACT step
        return n_layer
    
    elif model_name == "ut_level1":
        # UTLevel1: 2 block passes per ACT step (reasoning + solution)
        return 2 * n_layer
    
    elif model_name in ("ut_level2", "trm"):
        # UTLevel2/TRM: complex nested loops
        # Per ACT step: n_outer_loops * (n_inner_loops + 1) block passes
        # - Each outer loop: n_inner_loops reasoning passes + 1 solution pass
        defaults = get_loop_defaults(model_name)
        n_inner = n_inner_loops if n_inner_loops is not None else defaults['n_inner_loops']
        n_outer = n_outer_loops if n_outer_loops is not None else defaults['n_outer_loops']
        passes_per_act_step = n_outer * (n_inner + 1)
        return n_layer * passes_per_act_step
    
    else:
        raise ValueError(f"Unknown model: {model_name}")


def adjust_params_for_compute_budget(
    model_name: str,
    target_block_passes: int,
    base_n_layer: int = 6,
    base_n_inner_loops: Optional[int] = None,
    base_n_outer_loops: Optional[int] = None,
) -> Dict[str, int]:
    """
    Adjust model parameters to achieve a target compute budget.
    
    The strategy varies by model:
    - GPT variants: Adjust n_layer directly
    - UT: Adjust n_layer directly  
    - UTLevel1: Adjust n_layer (each step = 2 passes)
    - UTLevel2/TRM: Adjust n_layer while keeping loop structure, or adjust loops
    
    Args:
        model_name: Model type
        target_block_passes: Desired number of block passes
        base_n_layer: Base n_layer to use as reference
        base_n_inner_loops: Base inner loops for complex models (None = use model default)
        base_n_outer_loops: Base outer loops for complex models (None = use model default)
    
    Returns:
        Dict with adjusted parameters: {'n_layer': ..., 'n_inner_loops': ..., 'n_outer_loops': ...}
    """
    model_name = model_name.lower()
    
    # Get model-specific defaults for loop params
    defaults = get_loop_defaults(model_name)
    n_inner = base_n_inner_loops if base_n_inner_loops is not None else defaults['n_inner_loops']
    n_outer = base_n_outer_loops if base_n_outer_loops is not None else defaults['n_outer_loops']
    
    result = {
        'n_layer': base_n_layer,
        'n_inner_loops': n_inner,
        'n_outer_loops': n_outer,
    }
    
    if model_name in ("gpt", "gpt_level1", "gpt_level2", "ut"):
        # Simple: n_layer = target_block_passes
        result['n_layer'] = target_block_passes
        
    # For UT level 1, Level2, and TRM, there are multiple forward passes per layer and therefore
    # We need to detach "parameters', 'layers', and 'compute budget'.
    elif model_name == "ut_level1":
        # 2 passes per step, so n_layer = target / 2
        result['n_layer'] = max(1, target_block_passes // 2)
        
    elif model_name in ("ut_level2", "trm"):
        # More complex: passes = n_layer * n_outer * (n_inner + 1)
        # Strategy: Keep loop structure fixed, adjust n_layer
        passes_per_step = n_outer * (n_inner + 1)
        result['n_layer'] = max(1, target_block_passes // passes_per_step)
        result['n_inner_loops'] = n_inner
        result['n_outer_loops'] = n_outer
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return result


def get_compute_summary(
    model_name: str,
    n_layer: int,
    n_inner_loops: Optional[int] = None,
    n_outer_loops: Optional[int] = None,
) -> str:
    """
    Get a human-readable summary of compute for a model configuration.
    """
    block_passes = calculate_block_passes(
        model_name, n_layer, n_inner_loops, n_outer_loops
    )
    
    model_name = model_name.lower()
    
    if model_name in ("gpt", "gpt_level1", "gpt_level2"):
        detail = f"n_layer={n_layer}"
    elif model_name == "ut":
        detail = f"n_layer={n_layer} (max ACT steps)"
    elif model_name == "ut_level1":
        detail = f"n_layer={n_layer} × 2 passes/step"
    elif model_name in ("ut_level2", "trm"):
        n_inner = n_inner_loops if n_inner_loops is not None else DEFAULT_N_INNER_LOOPS
        n_outer = n_outer_loops if n_outer_loops is not None else DEFAULT_N_OUTER_LOOPS
        detail = f"n_layer={n_layer} × {n_outer} outer × ({n_inner}+1) inner"
    else:
        detail = "unknown"
    
    return f"{model_name}: {block_passes} block passes ({detail})"


def normalize_model_kwargs_for_compute(
    model_name: str,
    model_kwargs: Dict[str, Any],
    compute_budget: Optional[int] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Normalize model kwargs to match a compute budget if specified.
    
    Args:
        model_name: Model type
        model_kwargs: Original model kwargs dict
        compute_budget: Target block passes (None = no adjustment)
    
    Returns:
        Tuple of (adjusted_kwargs, summary_string)
    """
    # Work with a copy to avoid mutating the original
    kwargs = model_kwargs.copy()
    
    if compute_budget is None:
        # No budget specified, return as-is with current compute info
        block_passes = calculate_block_passes(
            model_name,
            kwargs.get('n_layer', 6),
            kwargs.get('n_inner_loops'),
            kwargs.get('n_outer_loops'),
        )
        summary = get_compute_summary(
            model_name,
            kwargs.get('n_layer', 6),
            kwargs.get('n_inner_loops'),
            kwargs.get('n_outer_loops'),
        )
        return kwargs, summary
    
    # Adjust parameters for compute budget
    adjusted = adjust_params_for_compute_budget(
        model_name,
        compute_budget,
        base_n_layer=kwargs['n_layer'],
        base_n_inner_loops=kwargs.get('n_inner_loops', DEFAULT_N_INNER_LOOPS),
        base_n_outer_loops=kwargs.get('n_outer_loops', DEFAULT_N_OUTER_LOOPS),
    )
    
    # Update kwargs with adjusted values
    kwargs['n_layer'] = adjusted['n_layer']
    
    # Only add loop params for models that use them
    if model_name.lower() in ("ut_level2", "trm"):
        kwargs['n_inner_loops'] = adjusted['n_inner_loops']
        kwargs['n_outer_loops'] = adjusted['n_outer_loops']
    
    # Calculate actual achieved compute
    actual_passes = calculate_block_passes(
        model_name,
        adjusted['n_layer'],
        adjusted.get('n_inner_loops'),
        adjusted.get('n_outer_loops'),
    )
    
    summary = (
        f"{model_name}: adjusted to {actual_passes} block passes "
        f"(target={compute_budget}, n_layer={adjusted['n_layer']})"
    )
    print(summary)
    
    return kwargs, summary
