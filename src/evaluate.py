"""
Evaluation script for ERL-SC experiments.
Fetches results from WandB, computes metrics, and generates comparison figures.
"""

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate ERL-SC experiment runs")
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--run_ids",
        type=str,
        required=True,
        help='JSON list of run IDs to evaluate, e.g. \'["run1", "run2"]\''
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=os.environ.get("WANDB_ENTITY", "airas"),
        help="WandB entity (default: from WANDB_ENTITY env or 'airas')"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=os.environ.get("WANDB_PROJECT", "2026-02-14"),
        help="WandB project (default: from WANDB_PROJECT env or '2026-02-14')"
    )
    return parser.parse_args()


def fetch_run_data(entity: str, project: str, run_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch run data from WandB.
    
    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run ID
    
    Returns:
        Dictionary with run data or None if run not found
    """
    try:
        api = wandb.Api()
        run = api.run(f"{entity}/{project}/{run_id}")
        
        # Get summary metrics
        summary = dict(run.summary)
        
        # Get config
        config = dict(run.config)
        
        # Get history (time series)
        history = run.history()
        
        return {
            "run_id": run_id,
            "summary": summary,
            "config": config,
            "history": history,
            "name": run.name,
            "url": run.url,
        }
    except Exception as e:
        print(f"Warning: Could not fetch run {run_id}: {e}")
        return None


def export_per_run_metrics(run_data: Dict[str, Any], output_dir: Path) -> None:
    """
    Export per-run metrics to JSON file.
    
    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_dir / "metrics.json"
    
    metrics = {
        "run_id": run_data["run_id"],
        "accuracy": run_data["summary"].get("accuracy", 0.0),
        "accuracy_std": run_data["summary"].get("accuracy_std", 0.0),
        "avg_samples": run_data["summary"].get("avg_samples", 0.0),
        "avg_tokens": run_data["summary"].get("avg_tokens", 0.0),
        "config": run_data["config"],
    }
    
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    
    print(f"Exported metrics to: {metrics_file}")


def create_per_run_figures(run_data: Dict[str, Any], output_dir: Path) -> None:
    """
    Create per-run figures.
    
    Args:
        run_data: Run data from WandB
        output_dir: Output directory for this run
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    history = run_data["history"]
    
    if history.empty:
        print(f"No history data for {run_data['run_id']}, skipping figures")
        return
    
    # Figure 1: Accuracy over samples
    if "correct" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Compute cumulative accuracy
        cumulative_correct = history["correct"].cumsum()
        cumulative_count = np.arange(1, len(history) + 1)
        cumulative_accuracy = cumulative_correct / cumulative_count
        
        ax.plot(cumulative_count, cumulative_accuracy, linewidth=2)
        ax.set_xlabel("Number of Samples Processed", fontsize=12)
        ax.set_ylabel("Cumulative Accuracy", fontsize=12)
        ax.set_title(f"Cumulative Accuracy: {run_data['run_id']}", fontsize=14)
        ax.grid(True, alpha=0.3)
        
        fig_file = output_dir / "cumulative_accuracy.pdf"
        fig.savefig(fig_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Created figure: {fig_file}")
    
    # Figure 2: Samples per question
    if "num_samples" in history.columns:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.hist(history["num_samples"], bins=20, alpha=0.7, edgecolor="black")
        ax.axvline(history["num_samples"].mean(), color="red", linestyle="--", 
                   label=f"Mean: {history['num_samples'].mean():.2f}")
        ax.set_xlabel("Samples per Question", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title(f"Samples Distribution: {run_data['run_id']}", fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        fig_file = output_dir / "samples_distribution.pdf"
        fig.savefig(fig_file, bbox_inches="tight")
        plt.close(fig)
        print(f"Created figure: {fig_file}")


def create_comparison_figures(
    all_run_data: List[Dict[str, Any]], 
    output_dir: Path
) -> None:
    """
    Create comparison figures across all runs.
    
    Args:
        all_run_data: List of run data dictionaries
        output_dir: Output directory for comparison figures
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    colors = sns.color_palette("husl", len(all_run_data))
    
    # Figure 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    run_ids = []
    accuracies = []
    accuracy_stds = []
    
    for run_data in all_run_data:
        run_ids.append(run_data["run_id"])
        accuracies.append(run_data["summary"].get("accuracy", 0.0))
        accuracy_stds.append(run_data["summary"].get("accuracy_std", 0.0))
    
    x = np.arange(len(run_ids))
    ax.bar(x, accuracies, yerr=accuracy_stds, capsize=5, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison Across Methods", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig_file = output_dir / "comparison_accuracy.pdf"
    fig.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Created figure: {fig_file}")
    
    # Figure 2: Avg samples comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_samples = [run_data["summary"].get("avg_samples", 0.0) for run_data in all_run_data]
    
    ax.bar(x, avg_samples, alpha=0.7, color=colors)
    ax.set_xticks(x)
    ax.set_xticklabels(run_ids, rotation=45, ha="right")
    ax.set_ylabel("Avg Samples per Question", fontsize=12)
    ax.set_title("Sample Efficiency Comparison", fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")
    
    fig_file = output_dir / "comparison_samples.pdf"
    fig.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Created figure: {fig_file}")
    
    # Figure 3: Accuracy vs Cost (Pareto frontier)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    avg_tokens = [run_data["summary"].get("avg_tokens", 0.0) for run_data in all_run_data]
    
    for i, run_data in enumerate(all_run_data):
        ax.scatter(avg_tokens[i], accuracies[i], s=200, alpha=0.7, 
                  color=colors[i], label=run_ids[i], edgecolor="black", linewidth=1.5)
        
        # Add error bars
        if accuracy_stds[i] > 0:
            ax.errorbar(avg_tokens[i], accuracies[i], yerr=accuracy_stds[i], 
                       fmt="none", color=colors[i], capsize=5, alpha=0.5)
    
    ax.set_xlabel("Avg Tokens per Question", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy vs Cost Trade-off", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig_file = output_dir / "comparison_accuracy_vs_cost.pdf"
    fig.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Created figure: {fig_file}")
    
    # Figure 4: Cumulative accuracy overlay (if history available)
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, run_data in enumerate(all_run_data):
        history = run_data["history"]
        if not history.empty and "correct" in history.columns:
            cumulative_correct = history["correct"].cumsum()
            cumulative_count = np.arange(1, len(history) + 1)
            cumulative_accuracy = cumulative_correct / cumulative_count
            
            ax.plot(cumulative_count, cumulative_accuracy, linewidth=2, 
                   color=colors[i], label=run_ids[i], alpha=0.8)
    
    ax.set_xlabel("Number of Samples Processed", fontsize=12)
    ax.set_ylabel("Cumulative Accuracy", fontsize=12)
    ax.set_title("Cumulative Accuracy: All Methods", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    
    fig_file = output_dir / "comparison_cumulative_accuracy.pdf"
    fig.savefig(fig_file, bbox_inches="tight")
    plt.close(fig)
    print(f"Created figure: {fig_file}")


def compute_aggregated_metrics(
    all_run_data: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute aggregated metrics across all runs.
    
    Args:
        all_run_data: List of run data dictionaries
    
    Returns:
        Aggregated metrics dictionary
    """
    metrics_by_run = {}
    
    for run_data in all_run_data:
        run_id = run_data["run_id"]
        metrics_by_run[run_id] = {
            "accuracy": run_data["summary"].get("accuracy", 0.0),
            "accuracy_std": run_data["summary"].get("accuracy_std", 0.0),
            "avg_samples": run_data["summary"].get("avg_samples", 0.0),
            "avg_tokens": run_data["summary"].get("avg_tokens", 0.0),
        }
    
    # Identify proposed vs baseline
    proposed_runs = [rid for rid in metrics_by_run if "proposed" in rid]
    baseline_runs = [rid for rid in metrics_by_run if "comparative" in rid]
    
    best_proposed = None
    best_proposed_acc = 0.0
    if proposed_runs:
        best_proposed = max(proposed_runs, key=lambda r: metrics_by_run[r]["accuracy"])
        best_proposed_acc = metrics_by_run[best_proposed]["accuracy"]
    
    best_baseline = None
    best_baseline_acc = 0.0
    if baseline_runs:
        best_baseline = max(baseline_runs, key=lambda r: metrics_by_run[r]["accuracy"])
        best_baseline_acc = metrics_by_run[best_baseline]["accuracy"]
    
    gap = best_proposed_acc - best_baseline_acc if best_proposed and best_baseline else 0.0
    
    return {
        "primary_metric": "accuracy",
        "metrics_by_run": metrics_by_run,
        "best_proposed": best_proposed,
        "best_baseline": best_baseline,
        "gap": gap,
    }


def main():
    """Main evaluation script."""
    args = parse_args()
    
    # Parse run_ids
    run_ids = json.loads(args.run_ids)
    print(f"Evaluating {len(run_ids)} runs: {run_ids}")
    
    results_dir = Path(args.results_dir)
    
    # Fetch run data from WandB
    all_run_data = []
    for run_id in run_ids:
        print(f"\nFetching data for run: {run_id}")
        run_data = fetch_run_data(args.wandb_entity, args.wandb_project, run_id)
        
        if run_data is None:
            print(f"Warning: Skipping run {run_id} (not found or error)")
            continue
        
        all_run_data.append(run_data)
        
        # Export per-run metrics
        run_dir = results_dir / run_id
        export_per_run_metrics(run_data, run_dir)
        
        # Create per-run figures
        create_per_run_figures(run_data, run_dir)
    
    if not all_run_data:
        print("Error: No valid runs found")
        return
    
    # Create comparison figures
    print("\nCreating comparison figures...")
    comparison_dir = results_dir / "comparison"
    create_comparison_figures(all_run_data, comparison_dir)
    
    # Compute and export aggregated metrics
    print("\nComputing aggregated metrics...")
    aggregated = compute_aggregated_metrics(all_run_data)
    
    aggregated_file = comparison_dir / "aggregated_metrics.json"
    with open(aggregated_file, "w") as f:
        json.dump(aggregated, f, indent=2)
    
    print(f"\nAggregated metrics saved to: {aggregated_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    print(f"Primary metric: {aggregated['primary_metric']}")
    print(f"Best proposed: {aggregated['best_proposed']} " +
          f"(accuracy: {aggregated['metrics_by_run'].get(aggregated['best_proposed'], {}).get('accuracy', 0.0):.4f})")
    print(f"Best baseline: {aggregated['best_baseline']} " +
          f"(accuracy: {aggregated['metrics_by_run'].get(aggregated['best_baseline'], {}).get('accuracy', 0.0):.4f})")
    print(f"Gap: {aggregated['gap']:.4f}")
    print("="*80)


if __name__ == "__main__":
    main()
