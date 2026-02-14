"""
Main orchestration script for ERL-SC experiments.
Loads Hydra config and invokes the appropriate inference script.
"""

import subprocess
import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main orchestrator for a single run.
    Applies mode overrides and invokes inference.py as a subprocess.
    """
    # Apply mode overrides
    if cfg.mode == "sanity_check":
        # Sanity check: minimal execution
        if "dataset" in cfg and "max_samples" in cfg.dataset:
            cfg.dataset.max_samples = 10
        if "inference" in cfg:
            if "num_seeds" in cfg.inference:
                cfg.inference.num_seeds = 1
        if "method" in cfg:
            if cfg.method.name == "erl_sc":
                cfg.method.m_min = 2
                cfg.method.m_max = 5
            elif cfg.method.name == "baseline_sc":
                cfg.method.m_fixed = 5
        # Set wandb project to sanity namespace
        if "wandb" in cfg and "project" in cfg.wandb:
            cfg.wandb.project = f"{cfg.wandb.project}-sanity"
    elif cfg.mode == "pilot":
        # Pilot mode: reduced scale for quick testing
        if "dataset" in cfg and "max_samples" in cfg.dataset:
            cfg.dataset.max_samples = min(50, cfg.dataset.max_samples)
        if "inference" in cfg:
            if "num_seeds" in cfg.inference:
                cfg.inference.num_seeds = 1
    
    # Print effective config
    print("=" * 80)
    print(f"Starting run: {cfg.run.run_id}")
    print(f"Mode: {cfg.mode}")
    print("=" * 80)
    print("Effective configuration:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)
    
    # Determine task type and invoke appropriate script
    # This is an inference-only task (no training)
    
    # Build command to invoke inference.py as a subprocess
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.inference",
    ]
    
    # Pass config as environment or via hydra overrides
    # We'll pass the resolved config as a temporary file
    import tempfile
    import os
    
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        OmegaConf.save(cfg, f)
        config_file = f.name
    
    env = os.environ.copy()
    env["HYDRA_CONFIG_FILE"] = config_file
    
    try:
        # Run inference as a subprocess
        result = subprocess.run(
            cmd,
            env=env,
            check=False,
            capture_output=False,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"Inference failed with return code {result.returncode}", file=sys.stderr)
            sys.exit(result.returncode)
        
        print("=" * 80)
        print(f"Run {cfg.run.run_id} completed successfully")
        print("=" * 80)
    
    finally:
        # Clean up temp config file
        Path(config_file).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
