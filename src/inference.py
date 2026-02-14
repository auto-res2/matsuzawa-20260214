"""
Inference script for ERL-SC and baseline Self-Consistency experiments.
Implements both methods and handles sanity validation.
"""

import os
import re
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import Counter, defaultdict
from decimal import Decimal
import random

import numpy as np
from scipy import stats
from rapidfuzz import fuzz
import wandb
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm

from src.preprocess import load_dataset_by_name
from src.model import create_llm_interface


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


def extract_final_answer(text: str, dataset: str) -> Optional[str]:
    """
    Extract final answer from model completion.
    
    Args:
        text: Model completion text
        dataset: Dataset name (gsm8k or strategyqa)
    
    Returns:
        Extracted answer string or None if not found
    """
    # Look for "Final answer: <answer>" pattern
    pattern = r"Final answer:\s*(.+?)(?:\n|$)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return match.group(1).strip()
    
    # Fallback: try to extract from last line
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if lines:
        return lines[-1]
    
    return None


def canonicalize_answer(answer: str, dataset: str) -> str:
    """
    Canonicalize answer for equivalence comparison.
    
    Args:
        answer: Raw answer string
        dataset: Dataset name
    
    Returns:
        Canonicalized answer string
    """
    if not answer:
        return ""
    
    # Lowercase and strip whitespace
    canonical = answer.lower().strip()
    
    # Remove punctuation at the end
    canonical = canonical.rstrip(".,!?;:")
    
    if dataset == "gsm8k":
        # Extract numeric value
        # Handle various formats: "42", "$42", "42.0", "4,200", etc.
        canonical = canonical.replace(",", "")
        canonical = canonical.replace("$", "")
        
        # Extract number
        number_pattern = r"[-+]?\d*\.?\d+"
        match = re.search(number_pattern, canonical)
        if match:
            try:
                # Normalize to decimal
                num = Decimal(match.group(0))
                # Quantize to remove trailing zeros
                canonical = str(num.quantize(Decimal("1")) if num == num.to_integral_value() else num.normalize())
            except:
                pass
    
    elif dataset == "strategyqa":
        # Map boolean synonyms
        if canonical in ["yes", "true", "correct", "right", "1"]:
            canonical = "yes"
        elif canonical in ["no", "false", "incorrect", "wrong", "0"]:
            canonical = "no"
    
    return canonical


def compute_aec_clusters(
    answers: List[str], 
    dataset: str, 
    fuzzy_threshold: float = 0.8
) -> Dict[str, List[int]]:
    """
    Cluster answers into Answer Equivalence Classes (AECs).
    
    Args:
        answers: List of canonicalized answers
        dataset: Dataset name
        fuzzy_threshold: String similarity threshold for clustering
    
    Returns:
        Dictionary mapping representative answer to list of indices
    """
    if not answers:
        return {}
    
    clusters = {}
    representatives = []
    
    for i, answer in enumerate(answers):
        # Check if this answer belongs to an existing cluster
        matched = False
        for rep in representatives:
            # Exact match
            if answer == rep:
                clusters[rep].append(i)
                matched = True
                break
            
            # Fuzzy match (for residual text variation)
            if dataset == "strategyqa":
                # For boolean answers, require exact match
                if answer in ["yes", "no"] and rep in ["yes", "no"]:
                    continue
            
            # Check string similarity
            sim = fuzz.ratio(answer, rep) / 100.0
            if sim >= fuzzy_threshold:
                clusters[rep].append(i)
                matched = True
                break
        
        if not matched:
            # Create new cluster
            representatives.append(answer)
            clusters[answer] = [i]
    
    return clusters


def compute_risk_limiting_bound(
    aec_counts: Dict[str, int],
    total_samples: int,
    confidence: float = 0.85
) -> Tuple[str, float, float]:
    """
    Compute risk-limiting lower bound on top AEC probability.
    Uses Wilson score interval for conservative estimate.
    
    Args:
        aec_counts: Dictionary mapping AEC to count
        total_samples: Total number of samples
        confidence: Confidence level (tau parameter)
    
    Returns:
        Tuple of (top_aec, lower_bound, margin)
    """
    if not aec_counts or total_samples == 0:
        return "", 0.0, 0.0
    
    # Sort AECs by count
    sorted_aecs = sorted(aec_counts.items(), key=lambda x: x[1], reverse=True)
    
    if len(sorted_aecs) == 0:
        return "", 0.0, 0.0
    
    top_aec, top_count = sorted_aecs[0]
    
    # Compute Wilson score lower bound for top AEC
    p_hat = top_count / total_samples
    z = stats.norm.ppf(confidence)
    denominator = 1 + z**2 / total_samples
    center = (p_hat + z**2 / (2 * total_samples)) / denominator
    margin_of_error = z * np.sqrt(p_hat * (1 - p_hat) / total_samples + z**2 / (4 * total_samples**2)) / denominator
    lower_bound = center - margin_of_error
    
    # Compute margin vs runner-up
    if len(sorted_aecs) > 1:
        runner_up_count = sorted_aecs[1][1]
        margin = (top_count - runner_up_count) / total_samples
    else:
        margin = p_hat
    
    return top_aec, lower_bound, margin


def run_erl_sc(
    sample: Dict[str, Any],
    llm: Any,
    method_config: Dict[str, Any],
    mode: str = "main"
) -> Dict[str, Any]:
    """
    Run ERL-SC (Equivalence-Aware Risk-Limiting Self-Consistency) on a single sample.
    
    Args:
        sample: Dataset sample with keys: question, answer, prompt, dataset
        llm: LLM interface
        method_config: Method configuration
        mode: Execution mode (main, sanity_check, pilot)
    
    Returns:
        Result dictionary with prediction, metrics, and intermediate data
    """
    prompt = sample["prompt"]
    dataset = sample["dataset"]
    
    m_min = method_config["m_min"]
    m_max = method_config["m_max"]
    tau = method_config["tau"]
    margin_threshold = method_config["margin"]
    fuzzy_threshold = method_config["fuzzy_threshold"]
    temperature = method_config["temperature"]
    top_p = method_config["top_p"]
    
    completions = []
    canonical_answers = []
    tokens_used = 0
    
    # Sequential sampling with early stopping
    for i in range(m_max):
        # Generate completion
        result = llm.generate(prompt, temperature=temperature, top_p=top_p)
        completions.append(result["text"])
        tokens_used += result["tokens"] + result["prompt_tokens"]
        
        # Extract and canonicalize answer
        raw_answer = extract_final_answer(result["text"], dataset)
        if raw_answer:
            canonical = canonicalize_answer(raw_answer, dataset)
            canonical_answers.append(canonical)
        else:
            canonical_answers.append("")
        
        # Check stopping condition after m_min samples
        if i + 1 >= m_min:
            # Compute AEC clusters
            aec_clusters = compute_aec_clusters(canonical_answers, dataset, fuzzy_threshold)
            aec_counts = {aec: len(indices) for aec, indices in aec_clusters.items()}
            
            # Compute risk-limiting bound
            top_aec, lower_bound, margin = compute_risk_limiting_bound(
                aec_counts, len(canonical_answers), tau
            )
            
            # Check early stopping condition
            if lower_bound >= tau and margin >= margin_threshold:
                # Stop early
                break
    
    # Final prediction: top AEC
    aec_clusters = compute_aec_clusters(canonical_answers, dataset, fuzzy_threshold)
    aec_counts = {aec: len(indices) for aec, indices in aec_clusters.items()}
    
    if aec_counts:
        prediction = max(aec_counts.items(), key=lambda x: x[1])[0]
    else:
        prediction = ""
    
    # Check correctness
    gold_canonical = canonicalize_answer(sample["answer"], dataset)
    correct = (prediction == gold_canonical)
    
    return {
        "prediction": prediction,
        "gold": gold_canonical,
        "correct": correct,
        "num_samples": len(completions),
        "tokens_used": tokens_used,
        "completions": completions,
        "canonical_answers": canonical_answers,
        "aec_counts": aec_counts,
        "stopped_early": len(completions) < m_max,
    }


def run_baseline_sc(
    sample: Dict[str, Any],
    llm: Any,
    method_config: Dict[str, Any],
    mode: str = "main"
) -> Dict[str, Any]:
    """
    Run baseline Self-Consistency with fixed budget and exact-match voting.
    
    Args:
        sample: Dataset sample
        llm: LLM interface
        method_config: Method configuration
        mode: Execution mode
    
    Returns:
        Result dictionary
    """
    prompt = sample["prompt"]
    dataset = sample["dataset"]
    
    m_fixed = method_config["m_fixed"]
    temperature = method_config["temperature"]
    top_p = method_config["top_p"]
    
    completions = []
    canonical_answers = []
    tokens_used = 0
    
    # Generate fixed number of samples
    for i in range(m_fixed):
        result = llm.generate(prompt, temperature=temperature, top_p=top_p)
        completions.append(result["text"])
        tokens_used += result["tokens"] + result["prompt_tokens"]
        
        # Extract and canonicalize answer
        raw_answer = extract_final_answer(result["text"], dataset)
        if raw_answer:
            canonical = canonicalize_answer(raw_answer, dataset)
            canonical_answers.append(canonical)
        else:
            canonical_answers.append("")
    
    # Majority vote (exact match)
    if canonical_answers:
        answer_counts = Counter(canonical_answers)
        prediction = answer_counts.most_common(1)[0][0]
    else:
        prediction = ""
    
    # Check correctness
    gold_canonical = canonicalize_answer(sample["answer"], dataset)
    correct = (prediction == gold_canonical)
    
    return {
        "prediction": prediction,
        "gold": gold_canonical,
        "correct": correct,
        "num_samples": len(completions),
        "tokens_used": tokens_used,
        "completions": completions,
        "canonical_answers": canonical_answers,
        "answer_counts": dict(answer_counts),
        "stopped_early": False,
    }


def run_inference(cfg: DictConfig) -> None:
    """
    Main inference loop.
    
    Args:
        cfg: Hydra configuration
    """
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            resume="allow"
        )
        print(f"WandB run URL: {wandb.run.url}")
    
    # Load dataset
    print(f"Loading dataset: {cfg.dataset.name}")
    dataset = load_dataset_by_name(
        cfg.dataset.name,
        split=cfg.dataset.split,
        max_samples=cfg.dataset.max_samples,
        cache_dir=".cache"
    )
    print(f"Loaded {len(dataset)} samples")
    
    # Create LLM interface
    print(f"Initializing LLM: {cfg.model.provider}/{cfg.model.name}")
    llm = create_llm_interface(OmegaConf.to_container(cfg.model, resolve=True))
    
    # Determine method
    method_name = cfg.method.name
    is_erl_sc = (method_name == "erl_sc")
    
    # Run inference across seeds
    all_results = []
    
    for seed_idx in range(cfg.inference.num_seeds):
        seed = 42 + seed_idx
        set_seed(seed)
        
        print(f"\n{'='*80}")
        print(f"Running seed {seed_idx + 1}/{cfg.inference.num_seeds} (seed={seed})")
        print(f"{'='*80}")
        
        seed_results = []
        
        for sample_idx, sample in enumerate(tqdm(dataset, desc=f"Seed {seed}")):
            if is_erl_sc:
                result = run_erl_sc(
                    sample, 
                    llm, 
                    OmegaConf.to_container(cfg.method, resolve=True),
                    cfg.mode
                )
            else:
                result = run_baseline_sc(
                    sample, 
                    llm, 
                    OmegaConf.to_container(cfg.method, resolve=True),
                    cfg.mode
                )
            
            result["seed"] = seed
            result["sample_idx"] = sample_idx
            result["question"] = sample["question"]
            seed_results.append(result)
            
            # Log to WandB
            if cfg.wandb.mode != "disabled":
                wandb.log({
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "correct": int(result["correct"]),
                    "num_samples": result["num_samples"],
                    "tokens_used": result["tokens_used"],
                    "stopped_early": int(result.get("stopped_early", False)),
                })
        
        all_results.extend(seed_results)
        
        # Compute seed-level metrics
        seed_accuracy = np.mean([r["correct"] for r in seed_results])
        seed_avg_samples = np.mean([r["num_samples"] for r in seed_results])
        seed_avg_tokens = np.mean([r["tokens_used"] for r in seed_results])
        
        print(f"\nSeed {seed} results:")
        print(f"  Accuracy: {seed_accuracy:.4f}")
        print(f"  Avg samples per question: {seed_avg_samples:.2f}")
        print(f"  Avg tokens per question: {seed_avg_tokens:.0f}")
    
    # Compute overall metrics
    overall_accuracy = np.mean([r["correct"] for r in all_results])
    overall_avg_samples = np.mean([r["num_samples"] for r in all_results])
    overall_avg_tokens = np.mean([r["tokens_used"] for r in all_results])
    
    # Compute per-seed statistics
    seeds = sorted(set(r["seed"] for r in all_results))
    seed_accuracies = []
    for seed in seeds:
        seed_results = [r for r in all_results if r["seed"] == seed]
        seed_accuracies.append(np.mean([r["correct"] for r in seed_results]))
    
    accuracy_std = np.std(seed_accuracies) if len(seed_accuracies) > 1 else 0.0
    
    print(f"\n{'='*80}")
    print(f"Overall results across {cfg.inference.num_seeds} seeds:")
    print(f"  Accuracy: {overall_accuracy:.4f} ± {accuracy_std:.4f}")
    print(f"  Avg samples per question: {overall_avg_samples:.2f}")
    print(f"  Avg tokens per question: {overall_avg_tokens:.0f}")
    print(f"{'='*80}")
    
    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump({
            "run_id": cfg.run.run_id,
            "config": OmegaConf.to_container(cfg, resolve=True),
            "overall_metrics": {
                "accuracy": float(overall_accuracy),
                "accuracy_std": float(accuracy_std),
                "avg_samples": float(overall_avg_samples),
                "avg_tokens": float(overall_avg_tokens),
            },
            "per_sample_results": all_results,
        }, f, indent=2, default=str)
    
    print(f"\nResults saved to: {results_file}")
    
    # Log summary to WandB
    if cfg.wandb.mode != "disabled":
        wandb.summary["accuracy"] = overall_accuracy
        wandb.summary["accuracy_std"] = accuracy_std
        wandb.summary["avg_samples"] = overall_avg_samples
        wandb.summary["avg_tokens"] = overall_avg_tokens
        
        wandb.finish()
    
    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(all_results, cfg)


def perform_sanity_validation(results: List[Dict[str, Any]], cfg: DictConfig) -> None:
    """
    Perform sanity validation checks.
    
    Args:
        results: List of per-sample results
        cfg: Configuration
    """
    print(f"\n{'='*80}")
    print("SANITY VALIDATION")
    print(f"{'='*80}")
    
    # Check minimum samples processed
    total_samples = len(results)
    
    if total_samples < 5:
        print(f"SANITY_VALIDATION: FAIL reason=insufficient_samples (got {total_samples}, need >=5)")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "status": "fail"}}')
        return
    
    # Check all outputs are valid
    valid_outputs = sum(1 for r in results if r["prediction"])
    
    if valid_outputs == 0:
        print(f"SANITY_VALIDATION: FAIL reason=no_valid_outputs")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "valid_outputs": 0, "status": "fail"}}')
        return
    
    # Check not all predictions are identical (unless dataset is trivial)
    unique_predictions = len(set(r["prediction"] for r in results if r["prediction"]))
    
    if unique_predictions == 1 and total_samples > 5:
        print(f"SANITY_VALIDATION: FAIL reason=all_identical_predictions")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "unique_predictions": 1, "status": "fail"}}')
        return
    
    # Check at least some correct answers (accuracy > 0)
    accuracy = np.mean([r["correct"] for r in results])
    
    if accuracy == 0:
        print(f"SANITY_VALIDATION: FAIL reason=zero_accuracy")
        print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "accuracy": 0.0, "status": "fail"}}')
        return
    
    # All checks passed
    print(f"SANITY_VALIDATION: PASS")
    print(f'SANITY_VALIDATION_SUMMARY: {{"samples": {total_samples}, "valid_outputs": {valid_outputs}, "unique_predictions": {unique_predictions}, "accuracy": {accuracy:.4f}, "status": "pass"}}')


def main():
    """Main entry point when invoked as subprocess."""
    # Load config from environment
    config_file = os.environ.get("HYDRA_CONFIG_FILE")
    if not config_file:
        print("ERROR: HYDRA_CONFIG_FILE environment variable not set", file=sys.stderr)
        sys.exit(1)
    
    cfg = OmegaConf.load(config_file)
    
    # Run inference
    run_inference(cfg)


if __name__ == "__main__":
    main()
