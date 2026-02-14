"""
Dataset loading and preprocessing for ERL-SC experiments.
Supports GSM8K and StrategyQA datasets.
"""

from typing import Dict, List, Any
from datasets import load_dataset


# Few-shot CoT prompts for each dataset
GSM8K_COT_PROMPT = """Answer the following math word problem step by step. Show your reasoning and end with "Final answer: <number>".

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?
A: Let me think step by step. There are originally 15 trees. After planting, there are 21 trees. So the workers planted 21 - 15 = 6 trees. Final answer: 6

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?
A: Let me think step by step. There are originally 3 cars. 2 more cars arrive. So now there are 3 + 2 = 5 cars. Final answer: 5

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?
A: Let me think step by step. Originally, Leah had 32 chocolates and her sister had 42. So in total they had 32 + 42 = 74 chocolates. After eating 35, they have 74 - 35 = 39 chocolates left. Final answer: 39

Q: {question}
A: Let me think step by step."""


STRATEGYQA_COT_PROMPT = """Answer the following yes/no question with step-by-step reasoning. End with "Final answer: yes" or "Final answer: no".

Q: Do hamsters provide food for any animals?
A: Let me think step by step. Hamsters are prey animals. Some predators like snakes, owls, and hawks eat hamsters. So yes, hamsters provide food for these animals. Final answer: yes

Q: Could Brooke Shields succeed at University of Pennsylvania?
A: Let me think step by step. Brooke Shields graduated from Princeton University in 1987 with a bachelor's degree in French literature. Princeton is an Ivy League school comparable to University of Pennsylvania. Since she succeeded at Princeton, she likely could succeed at UPenn. Final answer: yes

Q: Yes or no: Hydrogen's atomic number squared exceeds number of Spice Girls?
A: Let me think step by step. Hydrogen has an atomic number of 1. 1 squared is 1. There are 5 Spice Girls. 1 does not exceed 5. Final answer: no

Q: {question}
A: Let me think step by step."""


def load_gsm8k(split: str = "test", max_samples: int = None, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """
    Load GSM8K dataset.
    
    Args:
        split: Dataset split (train or test)
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Cache directory for downloaded datasets
    
    Returns:
        List of dataset samples with keys: question, answer, prompt
    """
    dataset = load_dataset("gsm8k", "main", split=split, cache_dir=cache_dir)
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        question = item["question"]
        # Extract numeric answer from answer string (format: "#### 42")
        answer_text = item["answer"]
        answer = answer_text.split("####")[-1].strip()
        
        # Create prompt with CoT template
        prompt = GSM8K_COT_PROMPT.format(question=question)
        
        samples.append({
            "question": question,
            "answer": answer,
            "answer_text": answer_text,
            "prompt": prompt,
            "dataset": "gsm8k",
            "index": i,
        })
    
    return samples


def load_strategyqa(split: str = "test", max_samples: int = None, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """
    Load StrategyQA dataset.
    
    Args:
        split: Dataset split (train or test)
        max_samples: Maximum number of samples to load (None for all)
        cache_dir: Cache directory for downloaded datasets
    
    Returns:
        List of dataset samples with keys: question, answer, prompt
    """
    # StrategyQA uses train split for main dataset
    if split == "test":
        split = "train"
    
    dataset = load_dataset("wics/strategy-qa", split=split, cache_dir=cache_dir)
    
    samples = []
    for i, item in enumerate(dataset):
        if max_samples is not None and i >= max_samples:
            break
        
        question = item["question"]
        answer = "yes" if item["answer"] else "no"
        
        # Create prompt with CoT template
        prompt = STRATEGYQA_COT_PROMPT.format(question=question)
        
        samples.append({
            "question": question,
            "answer": answer,
            "prompt": prompt,
            "dataset": "strategyqa",
            "index": i,
        })
    
    return samples


def load_dataset_by_name(name: str, split: str = "test", max_samples: int = None, cache_dir: str = ".cache") -> List[Dict[str, Any]]:
    """
    Load dataset by name.
    
    Args:
        name: Dataset name (gsm8k or strategyqa)
        split: Dataset split
        max_samples: Maximum number of samples to load
        cache_dir: Cache directory for downloaded datasets
    
    Returns:
        List of dataset samples
    """
    if name == "gsm8k":
        return load_gsm8k(split, max_samples, cache_dir)
    elif name == "strategyqa":
        return load_strategyqa(split, max_samples, cache_dir)
    else:
        raise ValueError(f"Unknown dataset: {name}")
