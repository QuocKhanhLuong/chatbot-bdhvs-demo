"""
Pydantic schemas for the AI Research Assistant API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class ResearchDepth(str, Enum):
    """Research depth levels."""
    QUICK = "quick"  # Fast overview
    STANDARD = "standard"  # Balanced depth
    DEEP = "deep"  # Comprehensive research


class ChatMessage(BaseModel):
    """A single chat message."""
    role: str = Field(..., description="Role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Request for chat endpoint."""
    message: str = Field(..., description="User message")
    thread_id: Optional[str] = Field(default=None, description="Conversation thread ID")


class ChatResponse(BaseModel):
    """Response from chat endpoint."""
    response: str = Field(..., description="Assistant response")
    thread_id: str = Field(..., description="Thread ID for conversation continuity")
    tool_calls: Optional[List[Dict[str, Any]]] = Field(
        default=None, description="Tools that were called"
    )


class DeepResearchRequest(BaseModel):
    """Request for deep research."""
    query: str = Field(..., description="Research topic/question")
    depth: ResearchDepth = Field(
        default=ResearchDepth.STANDARD, 
        description="Research depth"
    )
    breadth: int = Field(
        default=3, 
        ge=1, 
        le=10,
        description="Number of parallel search queries (1-10)"
    )
    max_iterations: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum research iterations"
    )
    include_arxiv: bool = Field(
        default=True,
        description="Include ArXiv paper search"
    )
    language: str = Field(
        default="vi",
        description="Output language (vi/en)"
    )


class ResearchProgress(BaseModel):
    """Progress update during research."""
    stage: str = Field(..., description="Current stage")
    message: str = Field(..., description="Progress message")
    progress: float = Field(..., ge=0, le=100, description="Progress percentage")
    findings_count: int = Field(default=0, description="Number of findings so far")


class ResearchSource(BaseModel):
    """A source from research."""
    title: str
    url: str
    snippet: str
    relevance_score: Optional[float] = None
    source_type: str = Field(default="web", description="web, arxiv, document")


class ResearchFinding(BaseModel):
    """A research finding/learning."""
    content: str
    sources: List[str]
    confidence: float = Field(ge=0, le=1)


class DeepResearchResponse(BaseModel):
    """Response from deep research."""
    query: str
    summary: str = Field(..., description="Executive summary")
    findings: List[ResearchFinding]
    sources: List[ResearchSource]
    follow_up_questions: List[str]
    total_sources_analyzed: int
    research_time_seconds: float


class ArxivSearchRequest(BaseModel):
    """Request for ArXiv paper search."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=50)
    relevance_threshold: float = Field(
        default=0.6, 
        ge=0, 
        le=1,
        description="Minimum relevance score"
    )
    analyze_papers: bool = Field(
        default=True,
        description="Perform detailed analysis on relevant papers"
    )


class ArxivPaper(BaseModel):
    """An ArXiv paper."""
    id: str
    title: str
    authors: List[str]
    abstract: str
    url: str
    published: str
    categories: List[str]
    relevance_score: Optional[float] = None
    analysis: Optional[str] = None


class ArxivSearchResponse(BaseModel):
    """Response from ArXiv search."""
    query: str
    papers: List[ArxivPaper]
    summary: Optional[str] = None
    total_found: int
    analyzed_count: int


class KnowledgeGap(BaseModel):
    """Identified knowledge gap from reflection."""
    topic: str
    description: str
    suggested_queries: List[str]


class ReflectionResult(BaseModel):
    """Result of reflection on research."""
    is_sufficient: bool
    confidence: float
    knowledge_gaps: List[KnowledgeGap]
    suggestions: List[str]
