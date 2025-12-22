"""LLM client for HuggingFace models via API."""

import os
import time
import random
from typing import List, Dict, Optional, Callable, Any
import logging
from functools import wraps
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError

from backend.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Raised when connection to LLM API fails after retries."""
    pass


class LLMRateLimitError(LLMClientError):
    """Raised when rate limit is exceeded and retries exhausted."""
    pass


class LLMTimeoutError(LLMClientError):
    """Raised when request times out after retries."""
    pass


def retry_with_exponential_backoff(
    max_retries: int = None,
    initial_delay: float = None,
    max_delay: float = None,
    exponential_base: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (APIError, APIConnectionError, RateLimitError, APITimeoutError),
):
    max_retries = max_retries if max_retries is not None else getattr(config, 'LLM_MAX_RETRIES', 3)
    initial_delay = initial_delay if initial_delay is not None else getattr(config, 'LLM_INITIAL_RETRY_DELAY', 1.0)
    max_delay = max_delay if max_delay is not None else getattr(config, 'LLM_MAX_RETRY_DELAY', 60.0)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            num_retries = 0
            delay = initial_delay
            last_exception = None

            while True:
                try:
                    return func(*args, **kwargs)
                except retryable_exceptions as e:
                    last_exception = e
                    num_retries += 1

                    if num_retries > max_retries:
                        logger.error(
                            "LLM request failed after %d retries. Last error: %s",
                            max_retries,
                            str(e)
                        )
                        if isinstance(e, RateLimitError):
                            raise LLMRateLimitError(f"Rate limit exceeded after {max_retries} retries: {e}") from e
                        elif isinstance(e, (APIConnectionError, APITimeoutError)):
                            raise LLMConnectionError(f"Connection failed after {max_retries} retries: {e}") from e
                        else:
                            raise LLMClientError(f"LLM request failed after {max_retries} retries: {e}") from e

                    actual_delay = delay
                    if jitter:
                        actual_delay = delay * (0.5 + random.random())

                    logger.warning(
                        "LLM request failed (attempt %d/%d): %s. Retrying in %.2fs...",
                        num_retries,
                        max_retries + 1,
                        str(e),
                        actual_delay
                    )

                    time.sleep(actual_delay)
                    delay = min(delay * exponential_base, max_delay)

        return wrapper
    return decorator


class LLMClient:
    """Client for interacting with HuggingFace LLMs via OpenAI-compatible API."""
    
    def __init__(
        self,
        model_name: str = None,
        api_key: str = None,
        timeout: float = None,
        fallback_models: List[str] = None
    ):
        self.model_name = model_name or config.LLM_MODEL
        self.api_key = api_key or config.HF_TOKEN
        self.timeout = timeout or getattr(config, 'LLM_REQUEST_TIMEOUT', 120.0)
        self.fallback_models = fallback_models or getattr(config, 'LLM_FALLBACK_MODELS', [])
        
        if not self.api_key:
            raise ValueError("HuggingFace API token not found. Set HF_TOKEN environment variable.")

        self.client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=self.api_key,
            timeout=self.timeout
        )
        
        self._healthy = None
        self._last_health_check = 0
        self._health_check_interval = getattr(config, 'LLM_HEALTH_CHECK_INTERVAL', 300)
        
        logger.info("Initialized LLM client with model: %s (timeout: %.1fs)", self.model_name, self.timeout)

    def check_health(self, force: bool = False) -> bool:
        current_time = time.time()
        if not force and self._healthy is not None:
            if current_time - self._last_health_check < self._health_check_interval:
                return self._healthy

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0
            )
            self._healthy = response.choices[0].message.content is not None
            self._last_health_check = current_time
            logger.info("LLM health check passed for model: %s", self.model_name)
            return self._healthy
        except Exception as e:
            self._healthy = False
            self._last_health_check = current_time
            logger.warning("LLM health check failed: %s", str(e))
            return False

    def get_health_status(self) -> Dict[str, Any]:
        return {
            "model": self.model_name,
            "healthy": self._healthy,
            "last_check": self._last_health_check,
            "timeout": self.timeout,
            "fallback_models": self.fallback_models
        }

    @retry_with_exponential_backoff()
    def _make_request(
        self,
        messages: List[Dict[str, str]],
        model: str,
        max_tokens: int,
        temperature: float,
        top_p: float
    ) -> str:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stream=False
        )
        return response.choices[0].message.content
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> str:
        models_to_try = [self.model_name] + self.fallback_models
        last_error = None

        for model in models_to_try:
            try:
                generated_text = self._make_request(
                    messages=messages,
                    model=model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                
                if model != self.model_name:
                    logger.info("Successfully used fallback model: %s", model)
                
                logger.info("Generated %d characters using model: %s", len(generated_text), model)
                return generated_text
                
            except LLMClientError as e:
                last_error = e
                if model != models_to_try[-1]:
                    logger.warning("Model %s failed, trying next fallback: %s", model, str(e))
                continue
            except Exception as e:
                last_error = e
                logger.error("Unexpected error with model %s: %s", model, str(e))
                if model != models_to_try[-1]:
                    continue
                raise

        raise last_error or LLMClientError("All models failed")
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
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
        return self.generate(conversation, **kwargs)


class LocalLLMClient:
    def __init__(self, model_name: str = None, device: str = "cpu"):
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        import torch
        
        self.model_name = model_name or config.LLM_MODEL
        self.device = device
        
        logger.info("Loading local model: %s", self.model_name)
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=device
            )
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error("Error loading model: %s", str(e))
            raise
    
    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 1500,
        temperature: float = 0.7,
        **kwargs
    ) -> str:
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

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
            logger.error("Error generating text: %s", str(e))
            raise
    
    def generate_with_system_prompt(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        return self.generate(messages, **kwargs)


class LLMClientFactory:
    _instances: Dict[str, LLMClient] = {}
    
    @classmethod
    def get_client(cls, model_key: str = None) -> LLMClient:
        model_key = model_key or config.DEFAULT_LLM
        
        if model_key not in cls._instances:
            model_id = config.get_llm_model_id(model_key)
            cls._instances[model_key] = LLMClient(model_name=model_id)
            logger.info("Created new LLM client for model key: %s -> %s", model_key, model_id)
        
        return cls._instances[model_key]
    
    @classmethod
    def get_all_clients(cls) -> Dict[str, LLMClient]:
        for model_info in config.get_available_llms():
            cls.get_client(model_info["key"])
        return cls._instances
    
    @classmethod
    def clear_cache(cls):
        cls._instances.clear()


if __name__ == "__main__":
    try:
        client = LLMClient()
        
        print("Health check:", client.check_health())
        
        response = client.generate_with_system_prompt(
            system_prompt="You are a helpful assistant.",
            user_message="What is quality management?"
        )
        
        print("Response:", response)
        
    except Exception as e:
        print(f"Error: {e}")
