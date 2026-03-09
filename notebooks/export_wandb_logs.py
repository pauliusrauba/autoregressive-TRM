"""
Export W&B logs to CSV files.

This module provides functionality to export W&B experiment logs to two CSV files:
1. Summary metrics (final values for each run)
2. Full training history (all intermediate steps)
"""

import os
import pandas as pd
import wandb

# =============================================================================
# Configuration
# =============================================================================
WANDB_PROJECT = "pauliusrauba/icml-recursive-llms"
TAG_FILTER = "lengthgen_train"

MODEL_FAMILIES = {
    'gpt': 'Feedforward', 'gpt_level1': 'Feedforward', 'gpt_level2': 'Feedforward',
    'ut': 'Recurrent', 'ut_level1': 'Recurrent', 'ut_level2': 'Recurrent', 'trm': 'Recurrent',
}


def parse_run_name(run_name):
    """Extract model and task from run name."""
    parts = run_name.split('_')
    task = None
    task_idx = None
    for i, part in enumerate(parts):
        if part in ('addition', 'copy', 'reverse'):
            task = part
            task_idx = i
            break
    if task is None:
        return None, None
    model = '_'.join(parts[:task_idx])
    return model, task


def fetch_all_metrics(project=WANDB_PROJECT, tag_filter=TAG_FILTER):
    """Fetch comprehensive summary metrics from W&B including position-wise accuracy."""
    api = wandb.Api()
    runs = api.runs(project)
    filtered_runs = [r for r in runs if tag_filter in r.name and r.state == "finished"]
    print(f"Found {len(filtered_runs)} finished runs matching '{tag_filter}'")
    
    records = []
    for run in filtered_runs:
        model, task = parse_run_name(run.name)
        if model is None:
            continue
            
        config = run.config
        summary = run.summary._json_dict
        
        record = {
            'run_id': run.id,
            'run_name': run.name,
            'model': model,
            'task': task,
            'family': MODEL_FAMILIES.get(model, 'Unknown'),
            'param_count_M': summary.get('model/param_count_M', 0),
            'compute_budget': config.get('compute_budget', 24),
        }
        
        # Extract ALL metrics for each evaluation length
        for length in [10, 20, 30, 40, 50]:
            prefix = f'TaskEvaluation/{task}/L{length}'
            record[f'L{length}_seq_acc'] = summary.get(f'{prefix}/seq_acc')
            record[f'L{length}_char_acc'] = summary.get(f'{prefix}/char_acc')
            record[f'L{length}_first_char_acc'] = summary.get(f'{prefix}/first_char_acc')
            record[f'L{length}_last_char_acc'] = summary.get(f'{prefix}/last_char_acc')
            record[f'L{length}_pos_q1_acc'] = summary.get(f'{prefix}/pos_q1_acc')
            record[f'L{length}_pos_q2_acc'] = summary.get(f'{prefix}/pos_q2_acc')
            record[f'L{length}_pos_q3_acc'] = summary.get(f'{prefix}/pos_q3_acc')
            record[f'L{length}_pos_q4_acc'] = summary.get(f'{prefix}/pos_q4_acc')
        
        records.append(record)
    
    return pd.DataFrame(records)


def fetch_full_history(project=WANDB_PROJECT, tag_filter=TAG_FILTER):
    """Fetch complete training history from W&B including all intermediate steps."""
    api = wandb.Api()
    runs = api.runs(project)
    filtered_runs = [r for r in runs if tag_filter in r.name and r.state == "finished"]
    print(f"Found {len(filtered_runs)} finished runs matching '{tag_filter}'")
    
    all_records = []
    for i, run in enumerate(filtered_runs):
        model, task = parse_run_name(run.name)
        if model is None:
            continue
        
        config = run.config
        param_count_M = run.summary.get('model/param_count_M', 0)
        
        print(f"[{i+1}/{len(filtered_runs)}] Fetching history for {run.name}...")
        
        # Fetch full history - this can be slow for long runs
        for row in run.scan_history():
            step = row.get('_step', 0)
            
            record = {
                'run_id': run.id,
                'run_name': run.name,
                'model': model,
                'task': task,
                'family': MODEL_FAMILIES.get(model, 'Unknown'),
                'param_count_M': param_count_M,
                'compute_budget': config.get('compute_budget', 24),
                'step': step,
            }
            
            # Extract ALL metrics for each evaluation length
            for length in [10, 20, 30, 40, 50]:
                prefix = f'TaskEvaluation/{task}/L{length}'
                record[f'L{length}_seq_acc'] = row.get(f'{prefix}/seq_acc')
                record[f'L{length}_char_acc'] = row.get(f'{prefix}/char_acc')
                record[f'L{length}_first_char_acc'] = row.get(f'{prefix}/first_char_acc')
                record[f'L{length}_last_char_acc'] = row.get(f'{prefix}/last_char_acc')
                record[f'L{length}_pos_q1_acc'] = row.get(f'{prefix}/pos_q1_acc')
                record[f'L{length}_pos_q2_acc'] = row.get(f'{prefix}/pos_q2_acc')
                record[f'L{length}_pos_q3_acc'] = row.get(f'{prefix}/pos_q3_acc')
                record[f'L{length}_pos_q4_acc'] = row.get(f'{prefix}/pos_q4_acc')
            
            # Only keep rows that have at least one metric logged
            has_metrics = any(record.get(f'L{l}_seq_acc') is not None for l in [10, 20, 30, 40, 50])
            if has_metrics:
                all_records.append(record)
    
    return pd.DataFrame(all_records)


def export_wandb_logs(name, output_dir=".", project=WANDB_PROJECT, tag_filter=TAG_FILTER):
    """
    Export W&B logs to two CSV files.
    
    Parameters
    ----------
    name : str
        Base name for the output files. Will create:
        - {name}_summary.csv : Summary metrics (final values for each run)
        - {name}_history.csv : Full training history (all intermediate steps)
    output_dir : str
        Directory to save the CSV files (default: current directory)
    project : str
        W&B project path (default: pauliusrauba/icml-recursive-llms)
    tag_filter : str
        Tag to filter runs by (default: lengthgen_train)
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_summary, df_history) - The two DataFrames that were exported
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Define output paths
    summary_path = os.path.join(output_dir, f"{name}_summary.csv")
    history_path = os.path.join(output_dir, f"{name}_history.csv")
    
    # Fetch and export summary metrics
    print("=" * 60)
    print("Fetching summary metrics...")
    print("=" * 60)
    df_summary = fetch_all_metrics(project=project, tag_filter=tag_filter)
    df_summary.to_csv(summary_path, index=False)
    print(f"\nSaved summary metrics to: {summary_path}")
    print(f"  - {len(df_summary)} runs")
    print(f"  - Models: {sorted(df_summary['model'].unique())}")
    print(f"  - Tasks: {sorted(df_summary['task'].unique())}")
    
    # Fetch and export full history
    print("\n" + "=" * 60)
    print("Fetching full training history...")
    print("=" * 60)
    df_history = fetch_full_history(project=project, tag_filter=tag_filter)
    df_history.to_csv(history_path, index=False)
    print(f"\nSaved training history to: {history_path}")
    print(f"  - {len(df_history)} history rows across {df_history['run_id'].nunique()} runs")
    
    print("\n" + "=" * 60)
    print("Export complete!")
    print("=" * 60)
    print(f"  Summary: {summary_path}")
    print(f"  History: {history_path}")
    
    return df_summary, df_history


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Export W&B logs to CSV files")
    parser.add_argument("name", help="Base name for output files")
    parser.add_argument("--output-dir", "-o", default=".", help="Output directory")
    parser.add_argument("--project", "-p", default=WANDB_PROJECT, help="W&B project path")
    parser.add_argument("--tag", "-t", default=TAG_FILTER, help="Tag filter for runs")
    
    args = parser.parse_args()
    
    export_wandb_logs(
        name=args.name,
        output_dir=args.output_dir,
        project=args.project,
        tag_filter=args.tag
    )
