#!/usr/bin/env python3
"""
plot_param_inference_scaling.py

Generate four scaling analysis plots from W&B experiment data:
    Plot 1: Parameter-performance frontier (best over inference compute)
    Plot 2: Performance at different inference compute budgets
    Plot 3: Training budget vs testing budget heatmap
    Plot 4: Training budget vs performance per parameter

Usage:
    uv run python scripts/plot_param_inference_scaling.py --tag paraminf_scaling
    
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

# Model families for grouping
MODEL_FAMILIES = {
    'GPT': ['gpt', 'gpt_level1', 'gpt_level2'],
    'UT': ['ut', 'ut_level1', 'ut_level2'],
    'TRM': ['trm'],
}

FAMILY_COLORS = {
    'GPT': '#1f77b4',
    'UT': '#d62728',
    'TRM': '#e377c2',
}

MODEL_ORDER = ['gpt', 'gpt_level1', 'gpt_level2', 'ut', 'ut_level1', 'ut_level2', 'trm']

LINE_STYLES = {
    'gpt': '-',
    'gpt_level1': '--',
    'gpt_level2': ':',
    'ut': '-',
    'ut_level1': '--',
    'ut_level2': ':',
    'trm': '-',
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


def parse_run_name(run_name: str) -> Tuple[str, str, int, int]:
    """
    Extract model, task, n_embd, and compute_budget from run name.
    Format: '{model}_{task}_embd{n_embd}_compute{compute}_paraminf_scaling'
    """
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
        return None, None, None, None
    
    # Model is everything before the task
    model = '_'.join(parts[:task_idx])
    
    # Find embd and compute
    n_embd = None
    compute_budget = None
    for part in parts:
        if part.startswith('embd'):
            try:
                n_embd = int(part[4:])
            except ValueError:
                pass
        elif part.startswith('compute'):
            try:
                compute_budget = int(part[7:])
            except ValueError:
                pass
    
    return model, task, n_embd, compute_budget


def extract_final_metrics(run) -> Dict:
    """Extract final metrics from a run's summary."""
    config = run.config
    summary = run.summary._json_dict
    
    model, task, n_embd, compute_budget = parse_run_name(run.name)
    if model is None:
        return None
    
    train_len = config.get('algo_train_len', 10)
    
    result = {
        'run_name': run.name,
        'model': model,
        'task': task,
        'train_len': train_len,
        'n_embd': n_embd or config.get('n_embd', 256),
        'compute_budget': compute_budget or config.get('compute_budget', 24),
        'param_count_M': summary.get('model/param_count_M', 0),
    }
    
    # Get family
    for family, models in MODEL_FAMILIES.items():
        if model in models:
            result['family'] = family
            break
    else:
        result['family'] = 'Other'
    
    # Extract metrics for evaluation length = train_len (in-distribution)
    prefix = f'TaskEvaluation/{task}/L{train_len}'
    result['seq_acc'] = summary.get(f'{prefix}/seq_acc', None)
    result['char_acc'] = summary.get(f'{prefix}/char_acc', None)
    
    # Also extract for other lengths if available
    for length in [10, 20, 30]:
        prefix = f'TaskEvaluation/{task}/L{length}'
        result[f'L{length}_seq_acc'] = summary.get(f'{prefix}/seq_acc', None)
        result[f'L{length}_char_acc'] = summary.get(f'{prefix}/char_acc', None)
    
    return result


def extract_training_curves(run) -> pd.DataFrame:
    """Extract training curves with step-by-step metrics."""
    config = run.config
    model, task, n_embd, compute_budget = parse_run_name(run.name)
    
    if model is None:
        return None
    
    train_len = config.get('algo_train_len', 10)
    batch_size = config.get('batch_size', 64)
    block_size = config.get('block_size', 150)
    n_embd_val = n_embd or config.get('n_embd', 256)
    compute_budget_val = compute_budget or config.get('compute_budget', 24)
    
    try:
        history = run.history(pandas=True, samples=10000)
    except Exception as e:
        print(f"  Warning: Could not fetch history for {run.name}: {e}")
        return None
    
    if history.empty:
        return None
    
    # Get param count
    param_count_M = None
    if 'model/param_count_M' in history.columns:
        valid_params = history['model/param_count_M'].dropna()
        if not valid_params.empty:
            param_count_M = valid_params.iloc[0]
    if param_count_M is None:
        param_count_M = run.summary._json_dict.get('model/param_count_M', 0)
    
    # Get family
    family = 'Other'
    for fam, models in MODEL_FAMILIES.items():
        if model in models:
            family = fam
            break
    
    records = []
    
    # Compute FLOPs per step
    params_per_block = 12 * n_embd_val ** 2 + 10 * n_embd_val
    
    for idx, row in history.iterrows():
        step = row.get('_step', idx)
        if pd.isna(step) or step == 0:
            continue
        
        step = int(step)
        tokens_processed = step * batch_size * block_size
        flops = 6 * params_per_block * tokens_processed * compute_budget_val
        
        record = {
            'model': model,
            'task': task,
            'family': family,
            'step': step,
            'train_len': train_len,
            'n_embd': n_embd_val,
            'compute_budget': compute_budget_val,
            'param_count_M': param_count_M,
            'tokens_processed': tokens_processed,
            'flops': flops,
            'flops_1e15': flops / 1e15,
            'train_loss': row.get('train_loss', None),
            'val_loss': row.get('val_loss', None),
        }
        
        # Add metrics for each evaluation length
        for length in [10, 20, 30]:
            prefix = f'TaskEvaluation/{task}/L{length}'
            record[f'L{length}_seq_acc'] = row.get(f'{prefix}/seq_acc', None)
            record[f'L{length}_char_acc'] = row.get(f'{prefix}/char_acc', None)
        
        records.append(record)
    
    return pd.DataFrame(records)


def fetch_all_data(tag_filter: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch all data from W&B."""
    runs = fetch_runs(tag_filter)
    
    final_metrics = []
    training_curves = []
    
    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{len(runs)}: {run.name}")
        
        metrics = extract_final_metrics(run)
        if metrics:
            final_metrics.append(metrics)
        
        curves = extract_training_curves(run)
        if curves is not None and not curves.empty:
            training_curves.append(curves)
    
    final_df = pd.DataFrame(final_metrics)
    curves_df = pd.concat(training_curves, ignore_index=True) if training_curves else pd.DataFrame()
    
    return final_df, curves_df


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_1_param_performance_frontier(df: pd.DataFrame, metric: str = 'seq_acc',
                                      output_dir: str = None) -> plt.Figure:
    """
    Plot 1: Parameter-performance frontier (best over inference compute).
    X-axis: Total parameter count
    Y-axis: Performance (best across compute budgets)
    Colors: Model families
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Group by model and n_embd, taking best performance across compute budgets
    grouped = df.groupby(['model', 'n_embd']).agg({
        'param_count_M': 'first',
        metric: 'max',  # Best over compute budgets
        'family': 'first',
    }).reset_index()
    
    for model in MODEL_ORDER:
        model_df = grouped[grouped['model'] == model].sort_values('param_count_M')
        if model_df.empty:
            continue
        
        ax.plot(model_df['param_count_M'], model_df[metric],
                color=MODEL_COLORS.get(model, 'gray'),
                linestyle=LINE_STYLES.get(model, '-'),
                marker=MARKERS.get(model, 'o'),
                markersize=10,
                linewidth=2,
                label=MODEL_LABELS.get(model, model))
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    ax.set_title('Parameter-Performance Frontier\n(Best over inference compute)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot1_param_performance_frontier_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_1b_param_frontier_by_family(df: pd.DataFrame, metric: str = 'seq_acc',
                                      output_dir: str = None) -> plt.Figure:
    """
    Plot 1b: Parameter-performance frontier grouped by family.
    Shows the Pareto frontier for each model family.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for family in ['GPT', 'UT', 'TRM']:
        family_models = MODEL_FAMILIES.get(family, [])
        family_df = df[df['model'].isin(family_models)]
        
        if family_df.empty:
            continue
        
        # Get best performance per param count across all models in family
        grouped = family_df.groupby('param_count_M').agg({
            metric: 'max',
        }).reset_index().sort_values('param_count_M')
        
        ax.plot(grouped['param_count_M'], grouped[metric],
                color=FAMILY_COLORS.get(family, 'gray'),
                marker='o',
                markersize=10,
                linewidth=2.5,
                label=family)
        
        # Add individual model points
        for model in family_models:
            model_df = df[df['model'] == model].groupby('param_count_M').agg({
                metric: 'max',
            }).reset_index()
            ax.scatter(model_df['param_count_M'], model_df[metric],
                      color=MODEL_COLORS.get(model, 'gray'),
                      marker=MARKERS.get(model, 'o'),
                      s=60, alpha=0.6, zorder=5)
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    ax.set_title('Parameter-Performance Frontier by Family\n(Best over all configurations)')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot1b_param_frontier_by_family_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_2_inference_compute_comparison(df: pd.DataFrame, metric: str = 'seq_acc',
                                         output_dir: str = None) -> plt.Figure:
    """
    Plot 2: Performance at different inference compute budgets.
    X-axis: Parameters
    Y-axis: Performance
    Separate lines/panels for different compute budgets
    """
    compute_budgets = sorted(df['compute_budget'].unique())
    n_budgets = len(compute_budgets)
    
    if n_budgets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_budgets, figsize=(5 * n_budgets, 5), sharey=True)
    
    for ax_idx, compute_budget in enumerate(compute_budgets):
        ax = axes[ax_idx]
        budget_df = df[df['compute_budget'] == compute_budget]
        
        for model in MODEL_ORDER:
            model_df = budget_df[budget_df['model'] == model].sort_values('param_count_M')
            if model_df.empty:
                continue
            
            ax.plot(model_df['param_count_M'], model_df[metric],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
        
        ax.set_xlabel('Parameters (M)')
        ax.set_title(f'Compute Budget: {compute_budget}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    
    axes[0].legend(loc='lower right', framealpha=0.9, fontsize=9)
    
    fig.suptitle('Performance vs Parameters at Different Inference Compute Budgets', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot2_inference_compute_comparison_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_2b_compute_scaling_lines(df: pd.DataFrame, metric: str = 'seq_acc',
                                   output_dir: str = None) -> plt.Figure:
    """
    Plot 2b: How performance changes with inference compute for each model.
    X-axis: Inference compute budget
    Y-axis: Performance
    Lines: Different models at fixed n_embd
    """
    n_embd_sizes = sorted(df['n_embd'].unique())
    n_sizes = len(n_embd_sizes)
    
    if n_sizes == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_sizes, figsize=(5 * n_sizes, 5), sharey=True)
    
    for ax_idx, n_embd in enumerate(n_embd_sizes):
        ax = axes[ax_idx]
        embd_df = df[df['n_embd'] == n_embd]
        
        for model in MODEL_ORDER:
            model_df = embd_df[embd_df['model'] == model].sort_values('compute_budget')
            if model_df.empty:
                continue
            
            ax.plot(model_df['compute_budget'], model_df[metric],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    marker=MARKERS.get(model, 'o'),
                    markersize=8,
                    linewidth=2,
                    label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
        
        ax.set_xlabel('Inference Compute (block passes)')
        ax.set_title(f'n_embd = {n_embd}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    
    axes[0].legend(loc='lower right', framealpha=0.9, fontsize=9)
    
    fig.suptitle('Inference Compute Scaling', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot2b_compute_scaling_lines_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_3_training_vs_testing_budget(curves_df: pd.DataFrame, metric: str = 'seq_acc',
                                       length: int = 10, output_dir: str = None) -> plt.Figure:
    """
    Plot 3: Training budget vs testing budget heatmap.
    X-axis: Training FLOPs (binned)
    Y-axis: Inference compute budget
    Color: Performance
    """
    metric_col = f'L{length}_{metric}'
    
    # Get unique compute budgets
    compute_budgets = sorted(curves_df['compute_budget'].unique())
    n_budgets = len(compute_budgets)
    
    # Create subplots for each model family
    families = ['GPT', 'UT', 'TRM']
    fig, axes = plt.subplots(1, len(families), figsize=(5 * len(families), 4))
    
    for ax_idx, family in enumerate(families):
        ax = axes[ax_idx]
        family_models = MODEL_FAMILIES.get(family, [])
        family_df = curves_df[curves_df['model'].isin(family_models)]
        
        if family_df.empty or metric_col not in family_df.columns:
            ax.set_title(f'{family}: No data')
            continue
        
        # Aggregate: best performance per (compute_budget, step bin)
        family_df = family_df.dropna(subset=[metric_col])
        if family_df.empty:
            ax.set_title(f'{family}: No data')
            continue
        
        # Bin training steps
        max_step = family_df['step'].max()
        bins = np.linspace(0, max_step, 11)
        family_df['step_bin'] = pd.cut(family_df['step'], bins=bins, labels=bins[1:].astype(int))
        
        # Create pivot table
        pivot = family_df.groupby(['compute_budget', 'step_bin']).agg({
            metric_col: 'max'
        }).unstack(fill_value=0)
        
        pivot.columns = pivot.columns.droplevel(0)
        
        # Plot heatmap
        im = ax.imshow(pivot.values, aspect='auto', cmap='viridis', vmin=0, vmax=1,
                       origin='lower')
        
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Inference Compute')
        ax.set_title(f'{family}')
        
        # Set tick labels
        ax.set_yticks(range(len(compute_budgets)))
        ax.set_yticklabels(compute_budgets)
        
        n_xticks = min(5, len(pivot.columns))
        xtick_indices = np.linspace(0, len(pivot.columns) - 1, n_xticks, dtype=int)
        ax.set_xticks(xtick_indices)
        ax.set_xticklabels([pivot.columns[i] for i in xtick_indices])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    cbar.set_label('Sequence Accuracy' if metric == 'seq_acc' else 'Character Accuracy')
    
    fig.suptitle(f'Training vs Inference Compute Trade-off (L={length})', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot3_training_vs_testing_budget_{metric}_L{length}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_3b_training_curves_by_compute(curves_df: pd.DataFrame, metric: str = 'seq_acc',
                                        length: int = 10, output_dir: str = None) -> plt.Figure:
    """
    Plot 3b: Training curves showing FLOPs vs performance for different compute budgets.
    """
    metric_col = f'L{length}_{metric}'
    
    compute_budgets = sorted(curves_df['compute_budget'].unique())
    n_budgets = len(compute_budgets)
    
    if n_budgets == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    else:
        fig, axes = plt.subplots(1, n_budgets, figsize=(5 * n_budgets, 5), sharey=True)
    
    for ax_idx, compute_budget in enumerate(compute_budgets):
        ax = axes[ax_idx]
        budget_df = curves_df[curves_df['compute_budget'] == compute_budget]
        
        for model in MODEL_ORDER:
            model_df = budget_df[budget_df['model'] == model]
            if model_df.empty or metric_col not in model_df.columns:
                continue
            
            valid = model_df[['flops_1e15', metric_col]].dropna()
            if valid.empty:
                continue
            
            ax.plot(valid['flops_1e15'], valid[metric_col],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    linewidth=2,
                    alpha=0.8,
                    label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
        
        ax.set_xlabel('Training FLOPs (×10¹⁵)')
        ax.set_title(f'Compute Budget: {compute_budget}')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy')
    
    axes[0].legend(loc='lower right', framealpha=0.9, fontsize=9)
    
    fig.suptitle(f'Training FLOPs vs Performance (L={length})', y=1.02)
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot3b_training_curves_by_compute_{metric}_L{length}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_4_training_budget_vs_efficiency(curves_df: pd.DataFrame, metric: str = 'seq_acc',
                                          length: int = 10, output_dir: str = None) -> plt.Figure:
    """
    Plot 4: Training budget vs performance per parameter.
    X-axis: Training FLOPs
    Y-axis: Performance / Parameter count (efficiency)
    """
    metric_col = f'L{length}_{metric}'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model in MODEL_ORDER:
        model_df = curves_df[curves_df['model'] == model]
        if model_df.empty or metric_col not in model_df.columns:
            continue
        
        # Compute efficiency = performance / param_count
        model_df = model_df.dropna(subset=[metric_col, 'param_count_M'])
        if model_df.empty:
            continue
        
        model_df = model_df.copy()
        model_df['efficiency'] = model_df[metric_col] / model_df['param_count_M']
        
        # Group by n_embd and plot separately
        for n_embd in sorted(model_df['n_embd'].unique()):
            embd_df = model_df[model_df['n_embd'] == n_embd]
            valid = embd_df[['flops_1e15', 'efficiency']].dropna()
            if valid.empty:
                continue
            
            alpha = 0.8 if n_embd == model_df['n_embd'].max() else 0.4
            label = MODEL_LABELS.get(model, model) if n_embd == model_df['n_embd'].min() else None
            
            ax.plot(valid['flops_1e15'], valid['efficiency'],
                    color=MODEL_COLORS.get(model, 'gray'),
                    linestyle=LINE_STYLES.get(model, '-'),
                    linewidth=2,
                    alpha=alpha,
                    label=label)
    
    ax.set_xlabel('Training FLOPs (×10¹⁵)')
    ax.set_ylabel('Performance / Parameters (Accuracy / M params)')
    ax.set_title(f'Training Efficiency: Performance per Parameter (L={length})')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot4_training_efficiency_{metric}_L{length}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


def plot_4b_final_efficiency(df: pd.DataFrame, metric: str = 'seq_acc',
                              output_dir: str = None) -> plt.Figure:
    """
    Plot 4b: Final performance per parameter for each model.
    X-axis: Parameters
    Y-axis: Performance / Parameters
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    df = df.dropna(subset=[metric, 'param_count_M'])
    df = df.copy()
    df['efficiency'] = df[metric] / df['param_count_M']
    
    for model in MODEL_ORDER:
        model_df = df[df['model'] == model].sort_values('param_count_M')
        if model_df.empty:
            continue
        
        ax.plot(model_df['param_count_M'], model_df['efficiency'],
                color=MODEL_COLORS.get(model, 'gray'),
                linestyle=LINE_STYLES.get(model, '-'),
                marker=MARKERS.get(model, 'o'),
                markersize=10,
                linewidth=2,
                label=MODEL_LABELS.get(model, model))
    
    ax.set_xlabel('Parameters (M)')
    ax.set_ylabel('Performance / Parameters (Accuracy / M params)')
    ax.set_title('Parameter Efficiency')
    ax.legend(loc='best', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_dir:
        filename = f'plot4b_final_efficiency_{metric}.pdf'
        fig.savefig(os.path.join(output_dir, filename))
        print(f"  Saved: {filename}")
    
    return fig


# =============================================================================
# Summary Table
# =============================================================================

def generate_summary_table(df: pd.DataFrame, metric: str = 'seq_acc',
                           output_dir: str = None) -> pd.DataFrame:
    """Generate summary table with best configurations per model."""
    
    summary = []
    for model in MODEL_ORDER:
        model_df = df[df['model'] == model]
        if model_df.empty:
            continue
        
        # Best overall
        best_idx = model_df[metric].idxmax()
        best_row = model_df.loc[best_idx]
        
        summary.append({
            'Model': MODEL_LABELS.get(model, model),
            'Best n_embd': best_row['n_embd'],
            'Best Compute': best_row['compute_budget'],
            'Params (M)': f"{best_row['param_count_M']:.2f}",
            'Best Accuracy': f"{best_row[metric]:.3f}",
            'Efficiency': f"{best_row[metric] / best_row['param_count_M']:.3f}",
        })
    
    summary_df = pd.DataFrame(summary)
    
    if output_dir:
        filename = 'summary_table.csv'
        summary_df.to_csv(os.path.join(output_dir, filename), index=False)
        print(f"  Saved: {filename}")
    
    return summary_df


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Generate parameter/inference scaling plots')
    parser.add_argument('--tag', type=str, required=True,
                        help='W&B run name filter tag (e.g., paraminf_scaling)')
    parser.add_argument('--metric', type=str, default='seq_acc', choices=['seq_acc', 'char_acc'],
                        help='Performance metric to plot (default: seq_acc)')
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
    print(f"  n_embd sizes: {sorted(final_df['n_embd'].unique())}")
    print(f"  Compute budgets: {sorted(final_df['compute_budget'].unique())}")
    if not curves_df.empty:
        print(f"  Training curves: {len(curves_df)} data points")
    
    # Generate summary table
    print(f"\nGenerating summary table...")
    summary = generate_summary_table(final_df, args.metric, results_dir)
    print(summary.to_string(index=False))
    
    # Generate plots
    print(f"\nGenerating plots...")
    print(f"Output directory: {plots_dir}")
    
    # Plot 1: Parameter-performance frontier
    print("\n=== Plot 1: Parameter-Performance Frontier ===")
    plot_1_param_performance_frontier(final_df, args.metric, plots_dir)
    plot_1b_param_frontier_by_family(final_df, args.metric, plots_dir)
    
    # Plot 2: Inference compute comparison
    print("\n=== Plot 2: Inference Compute Comparison ===")
    plot_2_inference_compute_comparison(final_df, args.metric, plots_dir)
    plot_2b_compute_scaling_lines(final_df, args.metric, plots_dir)
    
    # Plot 3: Training vs testing budget
    if not curves_df.empty:
        print("\n=== Plot 3: Training vs Testing Budget ===")
        plot_3_training_vs_testing_budget(curves_df, args.metric, 10, plots_dir)
        plot_3b_training_curves_by_compute(curves_df, args.metric, 10, plots_dir)
    
    # Plot 4: Training efficiency
    if not curves_df.empty:
        print("\n=== Plot 4: Training Efficiency ===")
        plot_4_training_budget_vs_efficiency(curves_df, args.metric, 10, plots_dir)
    plot_4b_final_efficiency(final_df, args.metric, plots_dir)
    
    print(f"\n{'='*60}")
    print(f"Done! All outputs saved to:")
    print(f"  CSV data: {results_dir}")
    print(f"  Plots:    {plots_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
