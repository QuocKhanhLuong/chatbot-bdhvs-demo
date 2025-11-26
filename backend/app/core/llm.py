"""
Core LLM Module with Model Fallback Support for MegaLLM.

This module provides a centralized LLM initialization with automatic
fallback through 12 MegaLLM models ranked from best to weakest.

MegaLLM Models (ranked by capability):
1. openai-gpt-oss-120b - Best, largest OSS model
2. openai-gpt-oss-60b - Very powerful
3. qwen-2.5-72b-instruct - Excellent multilingual
4. llama3.1-70b-instruct - Meta's flagship
5. deepseek-r1-distill-qwen-32b - Strong reasoning
6. qwen-qwq-32b-preview - Good for complex tasks
7. llama3.3-70b-instruct - Latest Llama
8. qwen-2.5-32b-instruct - Fast & capable
9. llama3.1-8b-instruct - Efficient
10. llama3-8b-instruct - Free tier
11. gemma-7b-it - Google's lightweight
12. mistral-7b-instruct - Fallback
"""

import asyncio
from typing import Optional, List, Callable, Any
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel

from app.config import settings


# =============================================================================
# MegaLLM Model Registry (Best to Weakest)
# =============================================================================

MEGALLM_MODELS: List[str] = [
    "openai-gpt-oss-120b",      # 1. Best - Largest OSS model
    "openai-gpt-oss-60b",       # 2. Very powerful
    "qwen-2.5-72b-instruct",    # 3. Excellent multilingual  
    "llama3.1-70b-instruct",    # 4. Meta's flagship
    "deepseek-r1-distill-qwen-32b",  # 5. Strong reasoning
    "qwen-qwq-32b-preview",     # 6. Good for complex tasks
    "llama3.3-70b-instruct",    # 7. Latest Llama
    "qwen-2.5-32b-instruct",    # 8. Fast & capable
    "llama3.1-8b-instruct",     # 9. Efficient
    "llama3-8b-instruct",       # 10. Free tier default
    "gemma-7b-it",              # 11. Google's lightweight
    "mistral-7b-instruct",      # 12. Ultimate fallback
]

# Track which models are currently unavailable (in-memory cache)
_unavailable_models: set = set()


# =============================================================================
# Core LLM Factory with Fallback
# =============================================================================

def get_llm(
    temperature: float = 0.7,
    model: Optional[str] = None,
    fallback: bool = True
) -> BaseChatModel:
    """
    Get LLM based on configuration with optional fallback support.
    
    Args:
        temperature: Model temperature (0.0-1.0)
        model: Specific model to use (overrides config)
        fallback: Enable automatic fallback to next model if unavailable
    
    Returns:
        Configured LLM instance
    """
    google_key = settings.effective_google_api_key
    
    if settings.llm_provider == "megallm" and settings.megallm_api_key:
        # Use specified model or find first available
        target_model = model or settings.megallm_model
        
        if fallback and target_model in _unavailable_models:
            # Find next available model
            target_model = _get_next_available_model(target_model)
        
        return ChatOpenAI(
            model=target_model,
            temperature=temperature,
            api_key=settings.megallm_api_key,  # type: ignore
            base_url=settings.megallm_base_url
        )
    elif settings.openai_api_key:
        return ChatOpenAI(
            model=model or settings.model_name,
            temperature=temperature,
            api_key=settings.openai_api_key  # type: ignore
        )
    elif google_key:
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=temperature,
            google_api_key=google_key  # type: ignore
        )
    else:
        raise ValueError("No LLM API key configured")


def get_llm_with_fallback(
    temperature: float = 0.7,
    preferred_model: Optional[str] = None
) -> BaseChatModel:
    """
    Get LLM with automatic fallback through all available models.
    Starts from preferred_model (or best available) and falls back as needed.
    
    Args:
        temperature: Model temperature
        preferred_model: Start with this model, fallback to next if unavailable
    
    Returns:
        Configured LLM instance with best available model
    """
    if settings.llm_provider != "megallm" or not settings.megallm_api_key:
        return get_llm(temperature=temperature)
    
    # Find starting index
    start_idx = 0
    if preferred_model and preferred_model in MEGALLM_MODELS:
        start_idx = MEGALLM_MODELS.index(preferred_model)
    
    # Find first available model
    for model in MEGALLM_MODELS[start_idx:]:
        if model not in _unavailable_models:
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.megallm_api_key,  # type: ignore
                base_url=settings.megallm_base_url
            )
    
    # All models marked unavailable, clear cache and try first
    _unavailable_models.clear()
    return ChatOpenAI(
        model=MEGALLM_MODELS[0],
        temperature=temperature,
        api_key=settings.megallm_api_key,  # type: ignore
        base_url=settings.megallm_base_url
    )


async def invoke_with_fallback(
    messages: List[Any],
    temperature: float = 0.7,
    preferred_model: Optional[str] = None,
    on_fallback: Optional[Callable[[str, str, str], None]] = None
) -> Any:
    """
    Invoke LLM with automatic fallback on errors.
    
    Args:
        messages: Messages to send to LLM
        temperature: Model temperature
        preferred_model: Start with this model
        on_fallback: Callback(old_model, new_model, error) when fallback occurs
    
    Returns:
        LLM response
    
    Raises:
        ValueError: If all models fail
    """
    if settings.llm_provider != "megallm" or not settings.megallm_api_key:
        llm = get_llm(temperature=temperature)
        return await llm.ainvoke(messages)
    
    # Find starting index
    start_idx = 0
    if preferred_model and preferred_model in MEGALLM_MODELS:
        start_idx = MEGALLM_MODELS.index(preferred_model)
    
    last_error = None
    
    for model in MEGALLM_MODELS[start_idx:]:
        try:
            llm = ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=settings.megallm_api_key,  # type: ignore
                base_url=settings.megallm_base_url
            )
            
            response = await llm.ainvoke(messages)
            
            # Success! Remove from unavailable if it was there
            _unavailable_models.discard(model)
            
            return response
            
        except Exception as e:
            error_str = str(e)
            last_error = e
            
            # Check if model is unavailable
            if "currently unavailable" in error_str or "does not exist" in error_str:
                _unavailable_models.add(model)
                
                # Find next model for callback
                next_idx = MEGALLM_MODELS.index(model) + 1
                next_model = MEGALLM_MODELS[next_idx] if next_idx < len(MEGALLM_MODELS) else None
                
                if on_fallback and next_model:
                    on_fallback(model, next_model, error_str)
                
                continue
            else:
                # Other error, don't mark as unavailable
                raise
    
    raise ValueError(f"All MegaLLM models failed. Last error: {last_error}")


def _get_next_available_model(current_model: str) -> str:
    """Get next available model after the current one."""
    if current_model not in MEGALLM_MODELS:
        return MEGALLM_MODELS[0]
    
    current_idx = MEGALLM_MODELS.index(current_model)
    
    for model in MEGALLM_MODELS[current_idx + 1:]:
        if model not in _unavailable_models:
            return model
    
    # All models after current are unavailable, clear and return first
    _unavailable_models.clear()
    return MEGALLM_MODELS[0]


def mark_model_unavailable(model: str):
    """Mark a model as temporarily unavailable."""
    _unavailable_models.add(model)


def mark_model_available(model: str):
    """Mark a model as available again."""
    _unavailable_models.discard(model)


def get_available_models() -> List[str]:
    """Get list of currently available models."""
    return [m for m in MEGALLM_MODELS if m not in _unavailable_models]


def clear_unavailable_models():
    """Clear the unavailable models cache."""
    _unavailable_models.clear()


# =============================================================================
# Convenience Functions
# =============================================================================

def get_best_llm(temperature: float = 0.7) -> BaseChatModel:
    """Get LLM with the best available model."""
    return get_llm_with_fallback(temperature=temperature, preferred_model=MEGALLM_MODELS[0])


def get_fast_llm(temperature: float = 0.7) -> BaseChatModel:
    """Get LLM optimized for speed (smaller model)."""
    return get_llm_with_fallback(temperature=temperature, preferred_model="llama3.1-8b-instruct")


def get_reasoning_llm(temperature: float = 0.3) -> BaseChatModel:
    """Get LLM optimized for reasoning tasks."""
    return get_llm_with_fallback(temperature=temperature, preferred_model="deepseek-r1-distill-qwen-32b")


def get_multilingual_llm(temperature: float = 0.7) -> BaseChatModel:
    """Get LLM optimized for multilingual tasks."""
    return get_llm_with_fallback(temperature=temperature, preferred_model="qwen-2.5-72b-instruct")
