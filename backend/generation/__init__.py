"""Generation module for LLM-based checkpoint generation."""

"""
Avoid importing heavy optional dependencies at package import time.

Some submodules require optional packages (e.g., `openai`) that may not be
installed in all environments (tests, offline runs). Import them directly from
their modules when needed:

  from backend.generation.llm_client import LLMClient
  from backend.generation.checkpoint_generator import CheckpointGenerator
"""

__all__ = [
    "LLMClient",
    "CheckpointGenerator",
    "get_system_prompt",
    "SYSTEM_PROMPT",
    "build_user_prompt",
]

try:  # pragma: no cover
    from .llm_client import LLMClient
    from .checkpoint_generator import CheckpointGenerator
    from .prompts import get_system_prompt, build_user_prompt, SYSTEM_PROMPT
except Exception:
    # Optional dependencies may be missing (e.g., openai in minimal test envs).
    pass

