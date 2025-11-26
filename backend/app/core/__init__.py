"""
Core module for shared utilities, LLM management, and research prompts.
"""

from app.core.llm import (
    get_llm,
    get_llm_with_fallback,
    invoke_with_fallback,
    get_best_llm,
    get_fast_llm,
    get_reasoning_llm,
    get_multilingual_llm,
    mark_model_unavailable,
    mark_model_available,
    get_available_models,
    clear_unavailable_models,
    MEGALLM_MODELS,
)

from app.core.researcher_prompts import (
    AI_RESEARCHER_SYSTEM_PROMPT,
    ACADEMIC_SEARCH_PROMPT,
    EVIDENCE_SYNTHESIS_PROMPT,
    ANSWER_WITH_ACADEMIC_PRIORITY_PROMPT,
    RELATED_QUESTIONS_ACADEMIC_PROMPT,
    classify_source,
    rank_sources_by_quality,
    get_source_tier_emoji,
    format_source_with_quality,
    calculate_answer_confidence,
    SourceType,
    ACADEMIC_DOMAINS,
)

__all__ = [
    # LLM functions
    "get_llm",
    "get_llm_with_fallback",
    "invoke_with_fallback",
    "get_best_llm",
    "get_fast_llm",
    "get_reasoning_llm",
    "get_multilingual_llm",
    "mark_model_unavailable",
    "mark_model_available",
    "get_available_models",
    "clear_unavailable_models",
    "MEGALLM_MODELS",
    # Research prompts
    "AI_RESEARCHER_SYSTEM_PROMPT",
    "ACADEMIC_SEARCH_PROMPT",
    "EVIDENCE_SYNTHESIS_PROMPT",
    "ANSWER_WITH_ACADEMIC_PRIORITY_PROMPT",
    "RELATED_QUESTIONS_ACADEMIC_PROMPT",
    # Source classification
    "classify_source",
    "rank_sources_by_quality",
    "get_source_tier_emoji",
    "format_source_with_quality",
    "calculate_answer_confidence",
    "SourceType",
    "ACADEMIC_DOMAINS",
]
