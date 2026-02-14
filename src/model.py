"""
LLM interface for ERL-SC experiments.
Supports OpenAI and Anthropic models.
"""

import os
from typing import Dict, Any, Optional
import time


class LLMInterface:
    """Unified interface for LLM providers."""
    
    def __init__(self, provider: str, model_name: str, max_tokens: int = 512, **kwargs):
        """
        Initialize LLM interface.
        
        Args:
            provider: Provider name (openai or anthropic)
            model_name: Model name/identifier
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters
        """
        self.provider = provider.lower()
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.kwargs = kwargs
        
        # Initialize provider client
        if self.provider == "openai":
            from openai import OpenAI
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            self.client = OpenAI(api_key=api_key)
        elif self.provider == "anthropic":
            from anthropic import Anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY environment variable not set")
            self.client = Anthropic(api_key=api_key)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.7, 
        top_p: float = 0.95,
        retry_on_error: bool = True,
        max_retries: int = 3,
    ) -> Dict[str, Any]:
        """
        Generate a single completion from the LLM.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            retry_on_error: Whether to retry on API errors
            max_retries: Maximum number of retries
        
        Returns:
            Dictionary with keys:
                - text: Generated text
                - tokens: Number of tokens generated
                - prompt_tokens: Number of tokens in prompt
                - finish_reason: Reason for completion
        """
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    
                    return {
                        "text": response.choices[0].message.content,
                        "tokens": response.usage.completion_tokens,
                        "prompt_tokens": response.usage.prompt_tokens,
                        "finish_reason": response.choices[0].finish_reason,
                    }
                
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model_name,
                        max_tokens=self.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    
                    return {
                        "text": response.content[0].text,
                        "tokens": response.usage.output_tokens,
                        "prompt_tokens": response.usage.input_tokens,
                        "finish_reason": response.stop_reason,
                    }
            
            except Exception as e:
                if not retry_on_error or attempt == max_retries - 1:
                    raise
                print(f"API error (attempt {attempt + 1}/{max_retries}): {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        raise RuntimeError("Failed to generate completion after max retries")
    
    def batch_generate(
        self, 
        prompts: list[str], 
        temperature: float = 0.7, 
        top_p: float = 0.95,
    ) -> list[Dict[str, Any]]:
        """
        Generate multiple completions (one per prompt).
        
        Args:
            prompts: List of input prompts
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            List of completion dictionaries
        """
        return [self.generate(p, temperature, top_p) for p in prompts]


def create_llm_interface(config: Dict[str, Any]) -> LLMInterface:
    """
    Create LLM interface from config dictionary.
    
    Args:
        config: Configuration dictionary with keys: provider, name, max_tokens
    
    Returns:
        LLMInterface instance
    """
    return LLMInterface(
        provider=config["provider"],
        model_name=config["name"],
        max_tokens=config.get("max_tokens", 512),
    )
