"""LLM client for HuggingFace models via API."""

import os
from typing import List, Dict, Optional
import logging
from openai import OpenAI

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with HuggingFace LLMs via OpenAI-compatible API."""
    
    def __init__(self, model_name: str = None, api_key: str = None):
        """
        Initialize LLM client.
        
        Args:
            model_name: Name of the HuggingFace model
            api_key: HuggingFace API token
        """
        self.model_name = model_name or config.LLM_MODEL
        self.api_key = api_key or config.HF_TOKEN
        
        if not self.api_key:
            raise ValueError("HuggingFace API token not found. Set HF_TOKEN environment variable.")
        
        # Initialize OpenAI client with HuggingFace endpoint
        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key
        )
        
        logger.info(f"Initialized LLM client with model: {self.model_name}")
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        """
        Generate text using the LLM.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False
            )
            
            generated_text = response.choices[0].message.content
            logger.info(f"Generated {len(generated_text)} characters")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """
        Generate with system and user messages.
        
        Args:
            system_prompt: System instruction
            user_message: User query
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.generate(messages, **kwargs)
    
    def chat(
        self,
        conversation: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        Continue a conversation.
        
        Args:
            conversation: List of previous messages
            **kwargs: Additional generation parameters
            
        Returns:
            Generated response
        """
        return self.generate(conversation, **kwargs)


class LocalLLMClient:
    """
    Alternative client for running LLMs locally using transformers.
    Use this if you want to run models locally instead of via API.
    """
    
    def __init__(self, model_name: str = None, device: str = "cpu"):
        """
        Initialize local LLM client.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cpu' or 'cuda')
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        self.model_name = model_name or config.LLM_MODEL
        self.device = device
        
        logger.info(f"Loading local model: {self.model_name}")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            
            # Create pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        """Generate text using local model."""
        try:
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            # Generate
            outputs = self.pipeline(
                prompt,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                return_full_text=False,
                **kwargs
            )
            
            generated_text = outputs[0]['generated_text']
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            raise
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """Generate with system and user messages."""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.generate(messages, **kwargs)


# Example usage
if __name__ == "__main__":
    # Test API client
    try:
        client = LLMClient()
        
        response = client.generate_with_system_prompt(
            system_prompt="You are a helpful assistant.",
            user_message="What is quality management?"
        )
        
        print("Response:", response)
        
    except Exception as e:
        print(f"Error: {e}")

