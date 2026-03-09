"""
Export W&B logs for all experiment tags to separate CSV files.

Usage:
    python export_all_experiments.py

Output:
    Creates notebooks/data/{tag}_summary.csv and notebooks/data/{tag}_history.csv
    for each experiment tag.
"""

import os
import pandas as pd
import wandb

# =============================================================================
# Configuration
# =============================================================================
WANDB_PROJECT = "pauliusrauba/icml-recursive-llms"
OUTPUT_DIR = "data"

# All experiment tags (from experiments/ scripts)
EXPERIMENT_TAGS = [
    # Active experiments
    "lengthgen_train",           # exp_length_generalization.sh
    "compute_optimal",           # exp_compute_optimal.sh
    "compute_optimal-v2",        # exp_compute_optimal_v2.sh
    "hl_structure",              # exp_hl_structure.sh
    "trainlen_scaling",          # exp_train_length_scaling.sh
    "paraminf_scaling",          # exp_param_inference_scaling.sh
]

MODEL_FAMILIES = {
    'gpt': 'Feedforward', 'gpt_level1': 'Feedforward', 'gpt_level2': 'Feedforward',
    'ut': 'Recurrent', 'ut_level1': 'Recurrent', 'ut_level2': 'Recurrent', 'trm': 'Recurrent',
}


# =============================================================================
# Helper Functions
# =============================================================================
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


def fetch_all_metrics(project, tag_filter):
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


def fetch_full_history(project, tag_filter):
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


def export_wandb_logs(name, output_dir, project, tag_filter):
    """
    Export W&B logs to two CSV files.
    
    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_summary, df_history) - The two DataFrames that were exported
    """
    # Define output paths
    summary_path = os.path.join(output_dir, f"{name}_summary.csv")
    history_path = os.path.join(output_dir, f"{name}_history.csv")
    
    # Fetch and export summary metrics
    print("Fetching summary metrics...")
    df_summary = fetch_all_metrics(project=project, tag_filter=tag_filter)
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved summary metrics to: {summary_path}")
    print(f"  - {len(df_summary)} runs")
    if len(df_summary) > 0:
        print(f"  - Models: {sorted(df_summary['model'].unique())}")
        print(f"  - Tasks: {sorted(df_summary['task'].unique())}")
    
    # Fetch and export full history
    print("\nFetching full training history...")
    df_history = fetch_full_history(project=project, tag_filter=tag_filter)
    df_history.to_csv(history_path, index=False)
    print(f"Saved training history to: {history_path}")
    print(f"  - {len(df_history)} history rows across {df_history['run_id'].nunique() if len(df_history) > 0 else 0} runs")
    
    return df_summary, df_history


# =============================================================================
# Main Export Script
# =============================================================================
def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("Exporting W&B logs for all experiment tags")
    print("=" * 70)
    print(f"Project: {WANDB_PROJECT}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Tags to export: {EXPERIMENT_TAGS}")
    print("=" * 70)
    
    results = {}
    
    for tag in EXPERIMENT_TAGS:
        print(f"\n{'='*70}")
        print(f"EXPORTING TAG: {tag}")
        print(f"{'='*70}")
        
        try:
            df_summary, df_history = export_wandb_logs(
                name=tag,
                output_dir=OUTPUT_DIR,
                project=WANDB_PROJECT,
                tag_filter=tag
            )
            results[tag] = {
                'status': 'success',
                'summary_rows': len(df_summary),
                'history_rows': len(df_history),
            }
            print(f"\n✓ Successfully exported '{tag}'")
        except Exception as e:
            results[tag] = {
                'status': 'failed',
                'error': str(e),
            }
            print(f"\n✗ Failed to export '{tag}': {e}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("EXPORT SUMMARY")
    print("=" * 70)
    for tag, result in results.items():
        if result['status'] == 'success':
            print(f"✓ {tag}: {result['summary_rows']} runs, {result['history_rows']} history rows")
        else:
            print(f"✗ {tag}: FAILED - {result['error']}")
    
    print("\n" + "=" * 70)
    print(f"All exports complete! Files saved to: {OUTPUT_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()