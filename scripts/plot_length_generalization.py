#!/usr/bin/env python3
"""
plot_length_generalization.py

Generate the four length generalization plots from W&B experiment data:
    Plot 1: Absolute-length generalization at fixed inference compute
    Plot 2: Relative length ratio (L_eval / L_train) vs performance
    Plot 3: Generalization gap vs relative length
    Plot 4: Training FLOPs vs performance at different lengths

Usage:
    uv run python scripts/plot_length_generalization.py --tag lengthgen_train10_compute24
    
Output:
    - CSV files with all data saved to DATA_DIR/results/
    - PDF plots saved to DATA_DIR/results/plots/
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "/mnt/pdata/pr501/icml2025"
WANDB_PROJECT = "pauliusrauba/icml-recursive-llms"

# Style settings for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 10,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

# Color palette for models (colorblind-friendly)
MODEL_COLORS = {
    'gpt': '#1f77b4',        # Blue
    'gpt_level1': '#ff7f0e',  # Orange
    'gpt_level2': '#2ca02c',  # Green
    'ut': '#d62728',          # Red
    'ut_level1': '#9467bd',   # Purple
    'ut_level2': '#8c564b',   # Brown
    'trm': '#e377c2',         # Pink
}

MODEL_LABELS = {
    'gpt': 'GPT',
    'gpt_level1': 'GPT-L1',
    'gpt_level2': 'GPT-L2',
    'ut': 'UT',
    'ut_level1': 'UT-L1',
    'ut_level2': 'UT-L2',
    'trm': 'TRM',
}

MODEL_ORDER = ['gpt', 'gpt_level1', 'gpt_level2', 'ut', 'ut_level1', 'ut_level2', 'trm']

TASK_LABELS = {
    'addition': 'Addition',
    'copy': 'Copy',
    'reverse': 'Reverse',
}

LINE_STYLES = {
    'gpt': '-',
    'gpt_level1': '-',
    'gpt_level2': '-',
    'ut': '--',
    'ut_level1': '--',
    'ut_level2': '--',
    'trm': '--',
}

MARKERS = {
    'gpt': 'o',
    'gpt_level1': 's',
    'gpt_level2': '^',
    'ut': 'o',
    'ut_level1': 's',
    'ut_level2': '^',
    'trm': 'D',
}


# =============================================================================
# Data Fetching
# =============================================================================

def fetch_runs(tag_filter: str) -> List:
    """Fetch all runs matching the tag filter from W&B."""
    api = wandb.Api()
    runs = api.runs(WANDB_PROJECT)
    filtered_runs = [r for r in runs if tag_filter in r.name and r.state == "finished"]
    print(f"Found {len(filtered_runs)} finished runs matching '{tag_filter}'")
    return filtered_runs


def parse_run_name(run_name: str) -> Tuple[str, str]:
    """Extract model and task from run name like 'gpt_addition_char_lengthgen_train10_compute24'."""
    parts = run_name.split('_')
    
    # Find task (addition, copy, reverse)
    task = None
    task_idx = None
    for i, part in enumerate(parts):
        if part in ('addition', 'copy', 'reverse'):
            task = part
            task_idx = i
            break
    
    if task is None:
        return None, None
    
    # Model is everything before the task
    model = '_'.join(parts[:task_idx])
    
    return model, task


def extract_final_metrics(run) -> Dict:
    """Extract final metrics from a run's summary."""
    config = run.config
    summary = run.summary._json_dict
    
    model, task = parse_run_name(run.name)
    if model is None:
        return None
    
    train_len = config.get('algo_train_len', 10)
    
    # Extract metrics for each evaluation length
    result = {
        'run_name': run.name,
        'model': model,
        'task': task,
        'train_len': train_len,
        'param_count_M': summary.get('model/param_count_M', 0),
        'compute_budget': config.get('compute_budget', 24),
    }
    
    # Find all evaluation lengths from the summary keys
    eval_lengths = set()
    for key in summary.keys():
        if key.startswith(f'TaskEvaluation/{task}/L'):
            # Extract length from key like 'TaskEvaluation/addition/L10/seq_acc'
            parts = key.split('/')
            if len(parts) >= 3:
                length_str = parts[2]  # 'L10'
                if length_str.startswith('L'):
                    try:
                        length = int(length_str[1:])
                        eval_lengths.add(length)
                    except ValueError:
                        pass
    
    # Extract metrics for each length
    for length in sorted(eval_lengths):
        prefix = f'TaskEvaluation/{task}/L{length}'
        result[f'L{length}_seq_acc'] = summary.get(f'{prefix}/seq_acc', None)
        result[f'L{length}_char_acc'] = summary.get(f'{prefix}/char_acc', None)
        result[f'L{length}_ratio'] = length / train_len
    
    result['eval_lengths'] = sorted(eval_lengths)
    
    return result


def extract_training_curves(run) -> pd.DataFrame:
    """Extract training curves with step-by-step metrics for FLOPs analysis."""
    config = run.config
    model, task = parse_run_name(run.name)
    
    if model is None:
        return None
    
    train_len = config.get('algo_train_len', 10)
    batch_size = config.get('batch_size', 64)
    block_size = config.get('block_size', 180)
    compute_budget = config.get('compute_budget', 24)
    
    # Get history
    try:
        history = run.history(pandas=True, samples=10000)
    except Exception as e:
        print(f"  Warning: Could not fetch history for {run.name}: {e}")
        return None
    
    if history.empty:
        return None
    
    # Get param count - prefer from history column, fallback to summary
    param_count_M = None
    if 'model/param_count_M' in history.columns:
        valid_params = history['model/param_count_M'].dropna()
        if not valid_params.empty:
            param_count_M = valid_params.iloc[0]
    if param_count_M is None:
        param_count_M = run.summary._json_dict.get('model/param_count_M', 0)
    
    # Build dataframe with metrics at each step
    records = []
    
    # Find all length-specific metrics in the history
    eval_lengths = set()
    for col in history.columns:
        if col.startswith(f'TaskEvaluation/{task}/L'):
            parts = col.split('/')
            if len(parts) >= 3:
                length_str = parts[2]
                if length_str.startswith('L'):
                    try:
                        length = int(length_str[1:])
                        eval_lengths.add(length)
                    except ValueError:
                        pass
    
    for idx, row in history.iterrows():
        step = row.get('_step', idx)
        if pd.isna(step):
            continue
        
        step = int(step)
        if step == 0:
            continue  # Skip step 0 (sanity check)
        
        # Tokens processed
        tokens_processed = step * batch_size * block_size
        # Line ~192-196
        n_embd = config.get('n_embd', 256)

        # Exact params per Block (from models/common/layers.py):
        # - Attention: 3*n_embd² (QKV, no bias) + n_embd² + n_embd (proj with bias)
        # - FFN: 4*n_embd² + 4*n_embd (fc1) + 4*n_embd² + n_embd (fc2)
        # - LayerNorms: 2 × 2*n_embd
        params_per_block = 12 * n_embd ** 2 + 10 * n_embd

        # Line ~247: Correct FLOPs
        flops = 6 * params_per_block * tokens_processed * compute_budget
        
        # Normalized compute: tokens * block_passes (same for all models at same step)
        # This allows fair comparison since all models have same block_passes
        normalized_compute = tokens_processed * compute_budget
        
        record = {
            'model': model,
            'task': task,
            'step': step,
            'train_len': train_len,
            'param_count_M': param_count_M,
            'compute_budget': compute_budget,
            'tokens_processed': tokens_processed,
            'flops': flops,
            'flops_1e15': flops / 1e15,  # PetaFLOPs
            'normalized_compute': normalized_compute,
            'normalized_compute_1e12': normalized_compute / 1e12,
            'train_loss': row.get('train_loss', None),
            'val_loss': row.get('val_loss', None),
        }
        
        # Add metrics for each evaluation length
        for length in eval_lengths:
            prefix = f'TaskEvaluation/{task}/L{length}'
            record[f'L{length}_seq_acc'] = row.get(f'{prefix}/seq_acc', None)
            record[f'L{length}_char_acc'] = row.get(f'{prefix}/char_acc', None)
        
        records.append(record)
    
    df = pd.DataFrame(records)
    if not df.empty:
        df['eval_lengths'] = [sorted(eval_lengths)] * len(df)
    
    return df


def fetch_all_data(tag_filter: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all data from W&B and return final metrics and training curves."""
    runs = fetch_runs(tag_filter)
    
    final_metrics = []
    training_curves = []
    
    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{len(runs)}: {run.name}")
        
        # Final metrics
        metrics = extract_final_metrics(run)
        if metrics:
            final_metrics.append(metrics)
        
        # Training curves
        curves = extract_training_curves(run)
        if curves is not None and not curves.empty:
            training_curves.append(curves)
    
    final_df = pd.DataFrame(final_metrics)
    curves_df = pd.concat(training_curves, ignore_index=True) if training_curves else pd.DataFrame()
    
    return final_df, curves_df


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_1_absolute_length(df: pd.DataFrame, task: str, metric: str = 'seq_acc', 
                           output_dir: str = None) -> plt.Figure:
    """
    Plot 1: Absolute-length generalization at fixed inference compute.
    X-axis: Evaluation sequence length
    Y-axis: Task performance
    Colors: Model variants
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    task_df = df[df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No data for task {task}")
        return fig
    
    # Get eval lengths from the first row
    eval_lengths = task_df.iloc[0].get('eval_lengths', [])
    if not eval_lengths:
        # Try to infer from columns
        eval_lengths = []
        for col in task_df.columns:
            if col.startswith('L') and col.endswith(f'_{metric}'):
                try:
                    length = int(col.split('_')[0][1:])
                    eval_lengths.append(length)
                except ValueError:
                    pass
        eval_lengths = sorted(set(eval_lengths))
    
    for model in MODEL_ORDER:
        model_df = task_df[task_df['model'] == model]
        if model_df.empty:
            continue
        
        row = model_df.iloc[0]
        lengths = []
        values = []
        
        for length in eval_lengths:
            col = f'L{length}_{metric}'
            if col in row and pd.notna(row[col]):
                lengths.append(length)
                values.append(row[col])
        
        if lengths:
            ax.plot(lengths, values, 
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model))
    
    ax.set_xlabel('Evaluation Sequence Length')
    ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    ax.set_title(f'{TASK_LABELS.get(task, task)}: Length Generalization (Fixed Compute)')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Add vertical line at training length
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else None
    if train_len:
        ax.axvline(x=train_len, color='gray', linestyle=':', alpha=0.5, label=f'Train Length ({train_len})')
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot1_absolute_length_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_2_relative_length(df: pd.DataFrame, task: str, metric: str = 'seq_acc',
                           output_dir: str = None) -> plt.Figure:
    """
    Plot 2: Relative length ratio vs performance.
    X-axis: Length_eval / Length_train (1x, 2x, 3x, etc.)
    Y-axis: Task performance
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    task_df = df[df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No data for task {task}")
        return fig
    
    # Get eval lengths and train length
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else 10
    eval_lengths = task_df.iloc[0].get('eval_lengths', [])
    
    if not eval_lengths:
        for col in task_df.columns:
            if col.startswith('L') and col.endswith(f'_{metric}'):
                try:
                    length = int(col.split('_')[0][1:])
                    eval_lengths.append(length)
                except ValueError:
                    pass
        eval_lengths = sorted(set(eval_lengths))
    
    for model in MODEL_ORDER:
        model_df = task_df[task_df['model'] == model]
        if model_df.empty:
            continue
        
        row = model_df.iloc[0]
        ratios = []
        values = []
        
        for length in eval_lengths:
            col = f'L{length}_{metric}'
            if col in row and pd.notna(row[col]):
                ratios.append(length / train_len)
                values.append(row[col])
        
        if ratios:
            ax.plot(ratios, values,
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model))
    
    ax.set_xlabel('Length Ratio (L_eval / L_train)')
    ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    ax.set_title(f'{TASK_LABELS.get(task, task)}: Relative Length Generalization')
    ax.legend(loc='best', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1x', '2x', '3x', '4x', '5x'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot2_relative_length_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_3_generalization_gap(df: pd.DataFrame, task: str, metric: str = 'seq_acc',
                              output_dir: str = None) -> plt.Figure:
    """
    Plot 3: Generalization gap vs relative length.
    X-axis: r = eval_length / train_length
    Y-axis: Performance(r) - Performance(r=1)
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    task_df = df[df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No data for task {task}")
        return fig
    
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else 10
    eval_lengths = task_df.iloc[0].get('eval_lengths', [])
    
    if not eval_lengths:
        for col in task_df.columns:
            if col.startswith('L') and col.endswith(f'_{metric}'):
                try:
                    length = int(col.split('_')[0][1:])
                    eval_lengths.append(length)
                except ValueError:
                    pass
        eval_lengths = sorted(set(eval_lengths))
    
    for model in MODEL_ORDER:
        model_df = task_df[task_df['model'] == model]
        if model_df.empty:
            continue
        
        row = model_df.iloc[0]
        
        # Get baseline performance at r=1 (train_len)
        baseline_col = f'L{train_len}_{metric}'
        if baseline_col not in row or pd.isna(row[baseline_col]):
            continue
        baseline = row[baseline_col]
        
        ratios = []
        gaps = []
        
        for length in eval_lengths:
            col = f'L{length}_{metric}'
            if col in row and pd.notna(row[col]):
                ratio = length / train_len
                gap = row[col] - baseline
                ratios.append(ratio)
                gaps.append(gap)
        
        if ratios:
            ax.plot(ratios, gaps,
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model))
    
    ax.set_xlabel('Length Ratio (L_eval / L_train)')
    ax.set_ylabel('Generalization Gap (Perf(r) - Perf(1))')
    ax.set_title(f'{TASK_LABELS.get(task, task)}: Generalization Gap')
    ax.legend(loc='best', framealpha=0.9)
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(['1x', '2x', '3x', '4x', '5x'])
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot3_generalization_gap_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_4_flops_performance(curves_df: pd.DataFrame, task: str, metric: str = 'seq_acc',
                             length_ratios: List[int] = [1, 3, 5],
                             output_dir: str = None) -> plt.Figure:
    """
    Plot 4: Training FLOPs vs performance at different generalization lengths.
    X-axis: Training FLOPs (actual compute - different per model due to param count)
    Y-axis: Performance
    Different subplots for 1x, 3x, 5x lengths
        """
    n_ratios = len(length_ratios)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5 * n_ratios, 5), sharey=True)
    if n_ratios == 1:
        axes = [axes]
    
    task_df = curves_df[curves_df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No training curve data for task {task}")
        return fig
    
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else 10
    
    for ax_idx, ratio in enumerate(length_ratios):
        ax = axes[ax_idx]
        eval_length = train_len * ratio
        metric_col = f'L{eval_length}_{metric}'
        
        for model in MODEL_ORDER:
            model_df = task_df[task_df['model'] == model].copy()
            if model_df.empty or metric_col not in model_df.columns:
                continue
            
            # Filter to non-null values
            valid = model_df[['flops_1e15', metric_col]].dropna()
            if valid.empty:
                continue
            
            ax.plot(valid['flops_1e15'], valid[metric_col],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    linewidth=2,
                    alpha=0.8,
                    label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
            
            # Add final point marker
            ax.scatter(valid['flops_1e15'].iloc[-1], valid[metric_col].iloc[-1],
                       color=MODEL_COLORS.get(model, 'gray'),
                       marker=MARKERS.get(model, 'o'),
                       s=60, zorder=5)
        
        ax.set_xlabel('Training FLOPs (×10¹⁵)')
        ax.set_title(f'{ratio}x Length (L={eval_length})')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    
    # Add legend to first subplot
    axes[0].legend(loc='lower right', framealpha=0.9)
    
    fig.suptitle(f'{TASK_LABELS.get(task, task)}: Training FLOPs vs Performance)', y=1.05)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot4_flops_performance_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_4_steps_performance(curves_df: pd.DataFrame, task: str, metric: str = 'seq_acc',
                             length_ratios: List[int] = [1, 3, 5],
                             output_dir: str = None) -> plt.Figure:
    """
    Plot 4 Alternative: Training Steps vs performance at different generalization lengths.
    X-axis: Training Steps (same for all models - normalized compute comparison)
    Y-axis: Performance
    Different subplots for 1x, 3x, 5x lengths
    
    This shows all models at the same number of training iterations, allowing
    direct comparison at matched "inference-equivalent compute" (same block passes).
    """
    n_ratios = len(length_ratios)
    fig, axes = plt.subplots(1, n_ratios, figsize=(5 * n_ratios, 5), sharey=True)
    if n_ratios == 1:
        axes = [axes]
    
    task_df = curves_df[curves_df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No training curve data for task {task}")
        return fig
    
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else 10
    
    for ax_idx, ratio in enumerate(length_ratios):
        ax = axes[ax_idx]
        eval_length = train_len * ratio
        metric_col = f'L{eval_length}_{metric}'
        
        for model in MODEL_ORDER:
            model_df = task_df[task_df['model'] == model].copy()
            if model_df.empty or metric_col not in model_df.columns:
                continue
            
            # Filter to non-null values
            valid = model_df[['step', metric_col]].dropna()
            if valid.empty:
                continue
            
            ax.plot(valid['step'], valid[metric_col],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    linewidth=2,
                    alpha=0.8,
                    label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
            
            # Add final point marker
            ax.scatter(valid['step'].iloc[-1], valid[metric_col].iloc[-1],
                       color=MODEL_COLORS.get(model, 'gray'),
                       marker=MARKERS.get(model, 'o'),
                       s=60, zorder=5)
        
        ax.set_xlabel('Training Steps')
        ax.set_title(f'{ratio}x Length (L={eval_length})')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    
    # Add legend to first subplot
    axes[0].legend(loc='lower right', framealpha=0.9)
    
    fig.suptitle(f'{TASK_LABELS.get(task, task)}: Training Steps vs Performance\n(Normalized compute: all models use {task_df["compute_budget"].iloc[0]} block passes/forward)', y=1.05)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot4_steps_performance_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_4_flops_combined(curves_df: pd.DataFrame, task: str, metric: str = 'seq_acc',
                          length_ratios: List[int] = [1, 3, 5],
                          output_dir: str = None) -> plt.Figure:
    """
    Alternative Plot 4: All models on same plot with different line styles for lengths.
    """
    fig, ax = plt.subplots(figsize=(12, 7))
    
    task_df = curves_df[curves_df['task'] == task].copy()
    
    if task_df.empty:
        print(f"  Warning: No training curve data for task {task}")
        return fig
    
    train_len = task_df['train_len'].iloc[0] if 'train_len' in task_df.columns else 10
    
    length_line_styles = {1: '-', 3: '--', 5: ':'}
    
    for model in MODEL_ORDER:
        model_df = task_df[task_df['model'] == model].copy()
        if model_df.empty:
            continue
        
        for ratio in length_ratios:
            eval_length = train_len * ratio
            metric_col = f'L{eval_length}_{metric}'
            
            if metric_col not in model_df.columns:
                continue
            
            valid = model_df[['flops_1e15', metric_col]].dropna()
            if valid.empty:
                continue
            
            label = f'{MODEL_LABELS.get(model, model)} ({ratio}x)' if ratio == 1 else None
            ax.plot(valid['flops_1e15'], valid[metric_col],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=length_line_styles.get(ratio, '-'),
                    linewidth=2,
                    alpha=0.7,
                    label=label)
    
    ax.set_xlabel('Training FLOPs (×10¹⁵)')
    ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    ax.set_title(f'{TASK_LABELS.get(task, task)}: Training FLOPs vs Performance')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    # Custom legend
    from matplotlib.lines import Line2D
    model_handles = [Line2D([0], [0], color=MODEL_COLORS.get(m, 'gray'), linewidth=2, label=MODEL_LABELS.get(m, m))
                     for m in MODEL_ORDER if m in task_df['model'].values]
    length_handles = [Line2D([0], [0], color='gray', linestyle=length_line_styles[r], linewidth=2, label=f'{r}x Length')
                      for r in length_ratios]
    
    legend1 = ax.legend(handles=model_handles, loc='lower right', title='Models')
    ax.add_artist(legend1)
    ax.legend(handles=length_handles, loc='upper left', title='Eval Length')
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot4_flops_combined_{task}_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Generate length generalization plots from W&B data')
    parser.add_argument('--tag', type=str, required=True,
                        help='W&B run name filter tag (e.g., lengthgen_train10_compute24)')
    parser.add_argument('--metric', type=str, default='seq_acc', choices=['seq_acc', 'char_acc'],
                        help='Performance metric to plot (default: seq_acc)')
    parser.add_argument('--tasks', type=str, nargs='+', default=['addition', 'copy', 'reverse'],
                        help='Tasks to plot (default: all)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help=f'Output directory (default: {DATA_DIR}/results/plots)')
    parser.add_argument('--no-fetch', action='store_true',
                        help='Skip fetching data, use existing CSV files')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directories
    results_dir = os.path.join(DATA_DIR, 'results')
    plots_dir = args.output_dir or os.path.join(results_dir, 'plots')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_final = os.path.join(results_dir, f'final_metrics_{args.tag}.csv')
    csv_curves = os.path.join(results_dir, f'training_curves_{args.tag}.csv')
    
    # Fetch or load data
    if args.no_fetch and os.path.exists(csv_final) and os.path.exists(csv_curves):
        print(f"Loading existing data from {results_dir}")
        final_df = pd.read_csv(csv_final)
        curves_df = pd.read_csv(csv_curves)
        
        # Parse eval_lengths back to list
        if 'eval_lengths' in final_df.columns:
            final_df['eval_lengths'] = final_df['eval_lengths'].apply(eval)
    else:
        print(f"Fetching data from W&B with tag filter: {args.tag}")
        final_df, curves_df = fetch_all_data(args.tag)
        
        if final_df.empty:
            print("ERROR: No data found! Check your tag filter.")
            return
        
        # Save to CSV
        print(f"\nSaving data to {results_dir}")
        final_df.to_csv(csv_final, index=False)
        print(f"  Saved: {csv_final}")
        
        if not curves_df.empty:
            curves_df.to_csv(csv_curves, index=False)
            print(f"  Saved: {csv_curves}")
    
    # Print data summary
    print(f"\nData summary:")
    print(f"  Final metrics: {len(final_df)} runs")
    print(f"  Models: {sorted(final_df['model'].unique())}")
    print(f"  Tasks: {sorted(final_df['task'].unique())}")
    if not curves_df.empty:
        print(f"  Training curves: {len(curves_df)} data points")
    
    # Generate plots
    print(f"\nGenerating plots...")
    print(f"Output directory: {plots_dir}")
    
    for task in args.tasks:
        print(f"\n=== Task: {task} ===")
        
        # Plot 1: Absolute length generalization
        print("  Creating Plot 1: Absolute length generalization...")
        plot_1_absolute_length(final_df, task, args.metric, plots_dir)
        
        # Plot 2: Relative length ratio
        print("  Creating Plot 2: Relative length ratio...")
        plot_2_relative_length(final_df, task, args.metric, plots_dir)
        
        # Plot 3: Generalization gap
        print("  Creating Plot 3: Generalization gap...")
        plot_3_generalization_gap(final_df, task, args.metric, plots_dir)
        
        # Plot 4: FLOPs/Steps vs performance
        if not curves_df.empty:
            print("  Creating Plot 4a: Training FLOPs vs performance...")
            plot_4_flops_performance(curves_df, task, args.metric, [1, 3, 5], plots_dir)
            print("  Creating Plot 4b: Training Steps vs performance (normalized compute)...")
            plot_4_steps_performance(curves_df, task, args.metric, [1, 3, 5], plots_dir)
        else:
            print("  Skipping Plot 4: No training curve data")
    
    print(f"\n{'='*60}")
    print(f"Done! All outputs saved to:")
    print(f"  CSV data: {results_dir}")
    print(f"  Plots:    {plots_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
