#!/usr/bin/env python3
"""
plot_train_length_scaling.py

Analyze and visualize how training length affects generalization in arithmetic tasks.

This script generates:
    1. Summary tables (CSV) with all metrics
    2. Plot 1: Performance vs Task Difficulty (absolute input lengths)
    3. Plot 2: Generalization Consistency across equivalent relative lengths
    4. Plot 3: Pareto Frontier mapping (training cost vs generalization capability)

Usage:
    uv run python scripts/plot_train_length_scaling.py --tag trainlen_scaling
    
Output:
    - CSV tables in DATA_DIR/results/train_length_scaling/
    - PDF plots in DATA_DIR/results/train_length_scaling/plots/
"""

import argparse
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wandb
from matplotlib.lines import Line2D

# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = "/mnt/pdata/pr501/icml2025"
WANDB_PROJECT = "pauliusrauba/icml-recursive-llms"

# Style settings for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'legend.fontsize': 9,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
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

LINE_STYLES = {
    'gpt': '-',
    'gpt_level1': '-',
    'gpt_level2': '-',
    'ut': '--',
    'ut_level1': '--',
    'ut_level2': '--',
    'trm': '-.',
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

# Training length colors for Pareto plots
TRAIN_LEN_COLORS = {
    2: '#e41a1c',
    5: '#377eb8',
    10: '#4daf4a',
    20: '#984ea3',
    40: '#ff7f00',
}

# Compute budget markers
COMPUTE_MARKERS = {
    6: 'o',
    12: 's',
    18: '^',
    24: 'D',
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


def parse_run_name(run_name: str) -> Tuple[str, int, int]:
    """
    Extract model, train_len, compute_budget from run name.
    Format: '{model}_addition_char_trainlen_scaling_L{train_len}_C{compute}'
    """
    # Extract train_len and compute from tag
    train_len_match = re.search(r'_L(\d+)_', run_name)
    compute_match = re.search(r'_C(\d+)$', run_name)
    
    train_len = int(train_len_match.group(1)) if train_len_match else None
    compute = int(compute_match.group(1)) if compute_match else None
    
    # Extract model name (everything before _addition_char)
    parts = run_name.split('_addition_char_')
    model = parts[0] if parts else None
    
    return model, train_len, compute


def extract_metrics(run) -> Dict:
    """Extract all metrics from a run."""
    config = run.config
    summary = run.summary._json_dict
    
    model, train_len, compute = parse_run_name(run.name)
    if model is None or train_len is None:
        return None
    
    result = {
        'run_name': run.name,
        'model': model,
        'train_len': train_len,
        'compute_budget': compute or config.get('compute_budget', 24),
        'param_count_M': summary.get('model/param_count_M', 0),
    }
    
    # Find all evaluation lengths from summary
    eval_lengths = set()
    for key in summary.keys():
        if key.startswith('TaskEvaluation/addition/L'):
            match = re.search(r'/L(\d+)/', key)
            if match:
                eval_lengths.add(int(match.group(1)))
    
    # Extract metrics for each length
    for length in sorted(eval_lengths):
        prefix = f'TaskEvaluation/addition/L{length}'
        seq_acc = summary.get(f'{prefix}/seq_acc', None)
        char_acc = summary.get(f'{prefix}/char_acc', None)
        
        result[f'L{length}_seq_acc'] = seq_acc
        result[f'L{length}_char_acc'] = char_acc
        result[f'L{length}_ratio'] = length / train_len if train_len > 0 else None
    
    result['eval_lengths'] = sorted(eval_lengths)
    
    return result


def fetch_all_data(tag_filter: str) -> pd.DataFrame:
    """Fetch all data from W&B and return as DataFrame."""
    runs = fetch_runs(tag_filter)
    
    all_metrics = []
    for i, run in enumerate(runs):
        print(f"Processing run {i+1}/{len(runs)}: {run.name}")
        metrics = extract_metrics(run)
        if metrics:
            all_metrics.append(metrics)
    
    return pd.DataFrame(all_metrics)


# =============================================================================
# Table Generation
# =============================================================================

def generate_summary_tables(df: pd.DataFrame, output_dir: str) -> Dict[str, pd.DataFrame]:
    """Generate comprehensive summary tables."""
    tables = {}
    
    # Table 1: Performance at training length (in-distribution)
    print("\n=== Table 1: In-Distribution Performance ===")
    rows = []
    for (model, train_len, compute), group in df.groupby(['model', 'train_len', 'compute_budget']):
        row = group.iloc[0]
        col = f'L{train_len}_seq_acc'
        if col in row:
            rows.append({
                'Model': MODEL_LABELS.get(model, model),
                'Train Length': train_len,
                'Compute': compute,
                'Seq Acc (1x)': row[col],
                'Params (M)': row.get('param_count_M', 0),
            })
    table1 = pd.DataFrame(rows)
    table1 = table1.sort_values(['Train Length', 'Compute', 'Model'])
    tables['in_distribution'] = table1
    print(table1.to_string(index=False))
    table1.to_csv(os.path.join(output_dir, 'table1_in_distribution.csv'), index=False)
    
    # Table 2: Generalization at different ratios
    print("\n=== Table 2: Generalization by Ratio ===")
    ratios_of_interest = [1.0, 1.5, 2.0, 3.0]
    rows = []
    for (model, train_len, compute), group in df.groupby(['model', 'train_len', 'compute_budget']):
        row = group.iloc[0]
        entry = {
            'Model': MODEL_LABELS.get(model, model),
            'Train L': train_len,
            'Compute': compute,
        }
        for target_ratio in ratios_of_interest:
            # Find closest eval length to target ratio
            eval_len = int(round(train_len * target_ratio))
            col = f'L{eval_len}_seq_acc'
            if col in row:
                entry[f'{target_ratio}x'] = row[col]
            else:
                # Try nearby lengths
                for offset in [-1, 1, -2, 2]:
                    col = f'L{eval_len + offset}_seq_acc'
                    if col in row:
                        entry[f'{target_ratio}x'] = row[col]
                        break
        rows.append(entry)
    table2 = pd.DataFrame(rows)
    table2 = table2.sort_values(['Train L', 'Compute', 'Model'])
    tables['generalization_by_ratio'] = table2
    print(table2.to_string(index=False))
    table2.to_csv(os.path.join(output_dir, 'table2_generalization_by_ratio.csv'), index=False)
    
    # Table 3: Best model per configuration
    print("\n=== Table 3: Best Model per Configuration ===")
    rows = []
    for (train_len, compute), group in df.groupby(['train_len', 'compute_budget']):
        # Find best at 1x
        best_1x = None
        best_1x_acc = -1
        # Find best at 2x
        best_2x = None
        best_2x_acc = -1
        # Find best at 3x
        best_3x = None
        best_3x_acc = -1
        
        for _, row in group.iterrows():
            col_1x = f'L{train_len}_seq_acc'
            col_2x = f'L{train_len * 2}_seq_acc'
            col_3x = f'L{train_len * 3}_seq_acc'
            
            if col_1x in row and pd.notna(row[col_1x]) and row[col_1x] > best_1x_acc:
                best_1x_acc = row[col_1x]
                best_1x = row['model']
            if col_2x in row and pd.notna(row[col_2x]) and row[col_2x] > best_2x_acc:
                best_2x_acc = row[col_2x]
                best_2x = row['model']
            if col_3x in row and pd.notna(row[col_3x]) and row[col_3x] > best_3x_acc:
                best_3x_acc = row[col_3x]
                best_3x = row['model']
        
        rows.append({
            'Train L': train_len,
            'Compute': compute,
            'Best @1x': f"{MODEL_LABELS.get(best_1x, best_1x)} ({best_1x_acc:.2f})" if best_1x else '-',
            'Best @2x': f"{MODEL_LABELS.get(best_2x, best_2x)} ({best_2x_acc:.2f})" if best_2x else '-',
            'Best @3x': f"{MODEL_LABELS.get(best_3x, best_3x)} ({best_3x_acc:.2f})" if best_3x else '-',
        })
    table3 = pd.DataFrame(rows)
    table3 = table3.sort_values(['Train L', 'Compute'])
    tables['best_model'] = table3
    print(table3.to_string(index=False))
    table3.to_csv(os.path.join(output_dir, 'table3_best_model.csv'), index=False)
    
    # Table 4: Compute budget impact
    print("\n=== Table 4: Compute Budget Impact ===")
    rows = []
    for (model, train_len), group in df.groupby(['model', 'train_len']):
        entry = {
            'Model': MODEL_LABELS.get(model, model),
            'Train L': train_len,
        }
        for _, row in group.sort_values('compute_budget').iterrows():
            compute = int(row['compute_budget'])
            col = f'L{train_len}_seq_acc'
            if col in row:
                entry[f'C={compute}'] = row[col]
        rows.append(entry)
    table4 = pd.DataFrame(rows)
    table4 = table4.sort_values(['Train L', 'Model'])
    tables['compute_impact'] = table4
    print(table4.to_string(index=False))
    table4.to_csv(os.path.join(output_dir, 'table4_compute_impact.csv'), index=False)
    
    return tables


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_1_task_difficulty(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Plot 1: Performance vs Task Difficulty (absolute input lengths)
    
    Shows how model performance degrades as we increase input length.
    Separate lines for different training lengths, faceted by compute budget.
    """
    compute_budgets = sorted(df['compute_budget'].unique())
    n_budgets = len(compute_budgets)
    
    fig, axes = plt.subplots(2, (n_budgets + 1) // 2, figsize=(5 * ((n_budgets + 1) // 2), 10), 
                              sharey=True, squeeze=False)
    axes = axes.flatten()
    
    for ax_idx, compute in enumerate(compute_budgets):
        ax = axes[ax_idx]
        compute_df = df[df['compute_budget'] == compute]
        
        # For each model, plot performance at absolute lengths
        for model in MODEL_ORDER:
            model_df = compute_df[compute_df['model'] == model]
            if model_df.empty:
                continue
            
            # Collect all (length, accuracy) pairs across all training lengths
            all_points = defaultdict(list)
            for _, row in model_df.iterrows():
                train_len = row['train_len']
                eval_lengths = row.get('eval_lengths', [])
                if isinstance(eval_lengths, str):
                    eval_lengths = eval(eval_lengths)
                
                for eval_len in eval_lengths:
                    col = f'L{eval_len}_{metric}'
                    if col in row and pd.notna(row[col]):
                        all_points[eval_len].append(row[col])
            
            # Average performance at each absolute length
            if all_points:
                lengths = sorted(all_points.keys())
                means = [np.mean(all_points[l]) for l in lengths]
                stds = [np.std(all_points[l]) for l in lengths]
                
                ax.errorbar(lengths, means, yerr=stds,
                           color=MODEL_COLORS.get(model, 'gray'),
                           linestyle=LINE_STYLES.get(model, '-'),
                           marker=MARKERS.get(model, 'o'),
                           markersize=6,
                           linewidth=1.5,
                           capsize=2,
                           label=MODEL_LABELS.get(model, model))
        
        ax.set_xlabel('Absolute Input Length')
        ax.set_title(f'Compute Budget = {compute}')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy')
            ax.legend(loc='lower left', fontsize=8)
    
    # Hide unused axes
    for ax_idx in range(len(compute_budgets), len(axes)):
        axes[ax_idx].set_visible(False)
    
    fig.suptitle('Plot 1: Performance vs Task Difficulty (Absolute Input Length)', fontsize=14, y=1.02)
    plt.tight_layout()
    
    filename = f'plot1_task_difficulty_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")
    
    return fig


def plot_2_generalization_consistency(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Plot 2: Generalization Consistency across equivalent relative lengths
    
    Shows whether the pattern of generalization (e.g., 50% drop at 2x) 
    is consistent across different training lengths.
    """
    train_lengths = sorted(df['train_len'].unique())
    compute_budgets = sorted(df['compute_budget'].unique())
    
    # Create figure: one subplot per compute budget
    n_budgets = len(compute_budgets)
    fig, axes = plt.subplots(1, n_budgets, figsize=(4 * n_budgets, 5), sharey=True)
    if n_budgets == 1:
        axes = [axes]
    
    target_ratios = [1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    
    for ax_idx, compute in enumerate(compute_budgets):
        ax = axes[ax_idx]
        compute_df = df[df['compute_budget'] == compute]
        
        for model in MODEL_ORDER:
            model_df = compute_df[compute_df['model'] == model]
            if model_df.empty:
                continue
            
            # For each ratio, collect normalized performance across training lengths
            ratio_performance = defaultdict(list)
            
            for _, row in model_df.iterrows():
                train_len = row['train_len']
                baseline_col = f'L{train_len}_{metric}'
                if baseline_col not in row or pd.isna(row[baseline_col]):
                    continue
                baseline = row[baseline_col]
                
                for target_ratio in target_ratios:
                    eval_len = int(round(train_len * target_ratio))
                    col = f'L{eval_len}_{metric}'
                    
                    if col in row and pd.notna(row[col]) and baseline > 0:
                        # Normalize by baseline (1.0 = same as training length)
                        normalized = row[col] / baseline if baseline > 0 else 0
                        ratio_performance[target_ratio].append(normalized)
            
            if ratio_performance:
                ratios = sorted(ratio_performance.keys())
                means = [np.mean(ratio_performance[r]) for r in ratios]
                stds = [np.std(ratio_performance[r]) for r in ratios]
                
                ax.errorbar(ratios, means, yerr=stds,
                           color=MODEL_COLORS.get(model, 'gray'),
                           linestyle=LINE_STYLES.get(model, '-'),
                           marker=MARKERS.get(model, 'o'),
                           markersize=6,
                           linewidth=1.5,
                           capsize=2,
                           label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
        
        ax.set_xlabel('Length Ratio (eval / train)')
        ax.set_title(f'Compute = {compute}')
        ax.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
        ax.set_ylim(0, 1.3)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Normalized Performance\n(relative to 1x)')
            ax.legend(loc='lower left', fontsize=7)
    
    fig.suptitle('Plot 2: Generalization Consistency (Normalized by In-Distribution Performance)', 
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    filename = f'plot2_generalization_consistency_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")
    
    return fig


def plot_2b_generalization_by_train_len(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Plot 2b: Generalization patterns for each model, colored by training length.
    
    Shows if models maintain consistent generalization curves regardless of training length.
    """
    compute_budgets = sorted(df['compute_budget'].unique())
    n_models = len(MODEL_ORDER)
    
    fig, axes = plt.subplots(len(compute_budgets), n_models, 
                              figsize=(2.5 * n_models, 3 * len(compute_budgets)), 
                              squeeze=False, sharey=True)
    
    target_ratios = [1.0, 1.1, 1.3, 1.5, 2.0, 2.5, 3.0]
    
    for row_idx, compute in enumerate(compute_budgets):
        compute_df = df[df['compute_budget'] == compute]
        
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            model_df = compute_df[compute_df['model'] == model]
            
            for _, row in model_df.iterrows():
                train_len = row['train_len']
                
                ratios = []
                values = []
                
                for target_ratio in target_ratios:
                    eval_len = int(round(train_len * target_ratio))
                    col = f'L{eval_len}_{metric}'
                    if col in row and pd.notna(row[col]):
                        ratios.append(target_ratio)
                        values.append(row[col])
                
                if ratios:
                    ax.plot(ratios, values,
                           color=TRAIN_LEN_COLORS.get(train_len, 'gray'),
                           marker='o',
                           markersize=4,
                           linewidth=1.5,
                           alpha=0.8,
                           label=f'L={train_len}' if col_idx == 0 and row_idx == 0 else None)
            
            ax.set_ylim(-0.05, 1.05)
            ax.grid(True, alpha=0.3)
            
            if row_idx == len(compute_budgets) - 1:
                ax.set_xlabel('Length Ratio')
            if col_idx == 0:
                ax.set_ylabel(f'C={compute}')
            if row_idx == 0:
                ax.set_title(MODEL_LABELS.get(model, model))
    
    # Add legend
    handles = [Line2D([0], [0], color=TRAIN_LEN_COLORS.get(tl, 'gray'), marker='o', 
                      linewidth=1.5, label=f'Train L={tl}')
               for tl in sorted(TRAIN_LEN_COLORS.keys())]
    fig.legend(handles=handles, loc='upper center', ncol=5, bbox_to_anchor=(0.5, 1.05))
    
    fig.suptitle('Plot 2b: Generalization Curves by Training Length', fontsize=12, y=1.08)
    plt.tight_layout()
    
    filename = f'plot2b_gen_by_train_len_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename), bbox_inches='tight')
    print(f"  Saved: {filename}")
    
    return fig


def plot_3_pareto_frontier(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Plot 3: Pareto Frontier mapping
    
    X-axis: Training cost (steps * train_len, proxy for data seen)
    Y-axis: Generalization capability (performance at 2x or 3x length)
    
    Each point represents a (model, train_len, compute) configuration.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 3a: Performance at 2x vs Training Length
    ax1 = axes[0]
    
    for model in MODEL_ORDER:
        model_df = df[df['model'] == model]
        
        for compute in sorted(df['compute_budget'].unique()):
            compute_df = model_df[model_df['compute_budget'] == compute]
            
            train_lens = []
            perfs_2x = []
            
            for _, row in compute_df.iterrows():
                train_len = row['train_len']
                col_2x = f'L{train_len * 2}_{metric}'
                
                if col_2x in row and pd.notna(row[col_2x]):
                    train_lens.append(train_len)
                    perfs_2x.append(row[col_2x])
            
            if train_lens:
                ax1.scatter(train_lens, perfs_2x,
                           c=[MODEL_COLORS.get(model, 'gray')] * len(train_lens),
                           marker=COMPUTE_MARKERS.get(compute, 'o'),
                           s=80,
                           alpha=0.7,
                           label=f'{MODEL_LABELS.get(model, model)} C={compute}' if compute == sorted(df['compute_budget'].unique())[0] else None)
    
    ax1.set_xlabel('Training Length')
    ax1.set_ylabel(f'Performance at 2x Length')
    ax1.set_title('Pareto Frontier: Training Length vs 2x Generalization')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, alpha=0.3)
    
    # Plot 3b: Efficiency frontier (performance / training_cost)
    ax2 = axes[1]
    
    for model in MODEL_ORDER:
        model_df = df[df['model'] == model]
        
        train_lens = []
        efficiencies = []
        colors = []
        markers = []
        
        for _, row in model_df.iterrows():
            train_len = row['train_len']
            compute = row['compute_budget']
            col_2x = f'L{train_len * 2}_{metric}'
            
            if col_2x in row and pd.notna(row[col_2x]):
                perf = row[col_2x]
                # Training cost proxy: larger training length = more complex task
                cost = train_len
                efficiency = perf  # Could be perf / cost, but let's keep it as raw performance
                
                train_lens.append(train_len)
                efficiencies.append(perf)
                colors.append(TRAIN_LEN_COLORS.get(train_len, 'gray'))
                markers.append(COMPUTE_MARKERS.get(compute, 'o'))
        
        if train_lens:
            for i, (tl, eff, c, m) in enumerate(zip(train_lens, efficiencies, colors, markers)):
                ax2.scatter(tl, eff, c=MODEL_COLORS.get(model, 'gray'), 
                           marker=COMPUTE_MARKERS.get(row['compute_budget'], 'o'),
                           s=80, alpha=0.7)
    
    ax2.set_xlabel('Training Length')
    ax2.set_ylabel('Performance at 2x')
    ax2.set_title('All Configurations: Train Length vs 2x Performance')
    ax2.set_ylim(-0.05, 1.05)
    ax2.grid(True, alpha=0.3)
    
    # Add legends
    model_handles = [Line2D([0], [0], color=MODEL_COLORS.get(m, 'gray'), marker='o',
                            linestyle='None', markersize=8, label=MODEL_LABELS.get(m, m))
                     for m in MODEL_ORDER if m in df['model'].values]
    compute_handles = [Line2D([0], [0], color='gray', marker=COMPUTE_MARKERS.get(c, 'o'),
                              linestyle='None', markersize=8, label=f'C={c}')
                       for c in sorted(df['compute_budget'].unique())]
    
    ax1.legend(handles=model_handles, loc='lower right', fontsize=8, title='Models')
    ax2.legend(handles=compute_handles, loc='lower right', fontsize=8, title='Compute')
    
    plt.tight_layout()
    
    filename = f'plot3_pareto_frontier_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")
    
    return fig


def plot_3b_pareto_detailed(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Plot 3b: Detailed Pareto analysis
    
    Shows the frontier for each model across training lengths,
    with different ratios (2x, 3x) on different subplots.
    """
    ratios_to_plot = [1.5, 2.0, 3.0]
    n_ratios = len(ratios_to_plot)
    
    fig, axes = plt.subplots(1, n_ratios, figsize=(5 * n_ratios, 5), sharey=True)
    
    for ax_idx, target_ratio in enumerate(ratios_to_plot):
        ax = axes[ax_idx]
        
        for model in MODEL_ORDER:
            model_df = df[df['model'] == model]
            if model_df.empty:
                continue
            
            # Group by training length, average across compute budgets
            train_perf = defaultdict(list)
            
            for _, row in model_df.iterrows():
                train_len = row['train_len']
                eval_len = int(round(train_len * target_ratio))
                col = f'L{eval_len}_{metric}'
                
                if col in row and pd.notna(row[col]):
                    train_perf[train_len].append(row[col])
            
            if train_perf:
                train_lens = sorted(train_perf.keys())
                means = [np.mean(train_perf[tl]) for tl in train_lens]
                stds = [np.std(train_perf[tl]) for tl in train_lens]
                
                ax.errorbar(train_lens, means, yerr=stds,
                           color=MODEL_COLORS.get(model, 'gray'),
                           linestyle=LINE_STYLES.get(model, '-'),
                           marker=MARKERS.get(model, 'o'),
                           markersize=8,
                           linewidth=2,
                           capsize=3,
                           label=MODEL_LABELS.get(model, model) if ax_idx == 0 else None)
        
        ax.set_xlabel('Training Length')
        ax.set_title(f'Performance at {target_ratio}x')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy')
            ax.legend(loc='lower left', fontsize=8)
    
    fig.suptitle('Plot 3b: Pareto Analysis - How Training Length Affects OOD Performance',
                 fontsize=12, y=1.02)
    plt.tight_layout()
    
    filename = f'plot3b_pareto_detailed_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")
    
    return fig


def plot_compute_budget_analysis(df: pd.DataFrame, output_dir: str, metric: str = 'seq_acc') -> plt.Figure:
    """
    Supplementary Plot: How compute budget affects performance at each training length.
    """
    train_lengths = sorted(df['train_len'].unique())
    n_lens = len(train_lengths)
    
    fig, axes = plt.subplots(2, (n_lens + 1) // 2, figsize=(5 * ((n_lens + 1) // 2), 10), 
                              sharey=True, squeeze=False)
    axes = axes.flatten()
    
    ratios_to_show = [1.0, 2.0, 3.0]
    ratio_styles = {1.0: '-', 2.0: '--', 3.0: ':'}
    
    for ax_idx, train_len in enumerate(train_lengths):
        ax = axes[ax_idx]
        train_df = df[df['train_len'] == train_len]
        
        for model in MODEL_ORDER:
            model_df = train_df[train_df['model'] == model]
            if model_df.empty:
                continue
            
            for ratio in ratios_to_show:
                eval_len = int(round(train_len * ratio))
                col = f'L{eval_len}_{metric}'
                
                computes = []
                perfs = []
                
                for _, row in model_df.sort_values('compute_budget').iterrows():
                    if col in row and pd.notna(row[col]):
                        computes.append(row['compute_budget'])
                        perfs.append(row[col])
                
                if computes:
                    label = f'{MODEL_LABELS.get(model, model)} @{ratio}x' if ratio == 1.0 else None
                    ax.plot(computes, perfs,
                           color=MODEL_COLORS.get(model, 'gray'),
                           linestyle=ratio_styles[ratio],
                           marker='o',
                           markersize=5,
                           linewidth=1.5,
                           alpha=0.8,
                           label=label)
        
        ax.set_xlabel('Compute Budget')
        ax.set_title(f'Train L = {train_len}')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        
        if ax_idx == 0:
            ax.set_ylabel('Sequence Accuracy')
    
    # Add ratio legend
    ratio_handles = [Line2D([0], [0], color='gray', linestyle=ratio_styles[r], 
                            linewidth=2, label=f'{r}x')
                     for r in ratios_to_show]
    axes[0].legend(handles=ratio_handles, loc='lower right', fontsize=8, title='Eval Ratio')
    
    # Hide unused axes
    for ax_idx in range(len(train_lengths), len(axes)):
        axes[ax_idx].set_visible(False)
    
    fig.suptitle('Compute Budget Impact by Training Length', fontsize=12, y=1.02)
    plt.tight_layout()
    
    filename = f'plot_compute_budget_{metric}.pdf'
    fig.savefig(os.path.join(output_dir, filename))
    print(f"  Saved: {filename}")
    
    return fig


# =============================================================================
# Main
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze training length scaling experiments')
    parser.add_argument('--tag', type=str, required=True,
                        help='W&B run name filter tag (e.g., trainlen_scaling)')
    parser.add_argument('--metric', type=str, default='seq_acc', choices=['seq_acc', 'char_acc'],
                        help='Performance metric to analyze (default: seq_acc)')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory for results')
    parser.add_argument('--no-fetch', action='store_true',
                        help='Skip fetching data, use existing CSV')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Setup output directories
    results_dir = args.output_dir or os.path.join(DATA_DIR, 'results', 'train_length_scaling')
    plots_dir = os.path.join(results_dir, 'plots')
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    csv_path = os.path.join(results_dir, f'metrics_{args.tag}.csv')
    
    # Fetch or load data
    if args.no_fetch and os.path.exists(csv_path):
        print(f"Loading existing data from {csv_path}")
        df = pd.read_csv(csv_path)
        if 'eval_lengths' in df.columns:
            df['eval_lengths'] = df['eval_lengths'].apply(eval)
    else:
        print(f"Fetching data from W&B with tag filter: {args.tag}")
        df = fetch_all_data(args.tag)
        
        if df.empty:
            print("ERROR: No data found! Check your tag filter.")
            return
        
        df.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("Data Summary")
    print(f"{'='*60}")
    print(f"Total runs: {len(df)}")
    print(f"Models: {sorted(df['model'].unique())}")
    print(f"Training lengths: {sorted(df['train_len'].unique())}")
    print(f"Compute budgets: {sorted(df['compute_budget'].unique())}")
    
    # Generate tables
    print(f"\n{'='*60}")
    print("Generating Tables")
    print(f"{'='*60}")
    tables = generate_summary_tables(df, results_dir)
    
    # Generate plots
    print(f"\n{'='*60}")
    print("Generating Plots")
    print(f"{'='*60}")
    
    print("\nPlot 1: Performance vs Task Difficulty...")
    plot_1_task_difficulty(df, plots_dir, args.metric)
    
    print("\nPlot 2: Generalization Consistency...")
    plot_2_generalization_consistency(df, plots_dir, args.metric)
    
    print("\nPlot 2b: Generalization by Training Length...")
    plot_2b_generalization_by_train_len(df, plots_dir, args.metric)
    
    print("\nPlot 3: Pareto Frontier...")
    plot_3_pareto_frontier(df, plots_dir, args.metric)
    
    print("\nPlot 3b: Detailed Pareto Analysis...")
    plot_3b_pareto_detailed(df, plots_dir, args.metric)
    
    print("\nSupplementary: Compute Budget Analysis...")
    plot_compute_budget_analysis(df, plots_dir, args.metric)
    
    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")
    print(f"Tables saved to: {results_dir}")
    print(f"Plots saved to: {plots_dir}")
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(results_dir)):
        print(f"  - {f}")
    print(f"\nPlots:")
    for f in sorted(os.listdir(plots_dir)):
        print(f"  - {f}")


if __name__ == '__main__':
    main()
