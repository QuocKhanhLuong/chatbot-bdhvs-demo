"""
Perplexity-Style Answer Engine.

A sophisticated search and answer engine inspired by Consensus and Perplexity:

Features from Perplexity:
- Real-time web search with multiple providers
- Streaming answer generation with citations
- Related questions generation
- Pro Search with multi-step reasoning
- Source quality ranking
- Images and video search

Features from Consensus:
- Academic paper focus (ArXiv, Semantic Scholar)
- Study design detection
- Evidence quality scoring
- Consensus meter (agreement analysis)
- Citation extraction from papers
- Meta-analysis synthesis

Additional Features:
- Multi-provider search (Tavily, Serper, Brave)
- Semantic caching for faster responses
- Rate limiting and quota management
- Structured citations with inline references
- Expert search with query planning
- Follow-up question generation
"""

import asyncio
import json
import os
import re
import hashlib
from typing import List, Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


# =============================================================================
# Pydantic Models
# =============================================================================

class SearchResult(BaseModel):
    """A single search result with metadata."""
    title: str
    url: str
    content: str
    snippet: str = ""
    favicon: str = ""
    source_type: str = "web"  # web, academic, news, video, image
    published_date: Optional[str] = None
    author: Optional[str] = None
    citation_count: Optional[int] = None
    relevance_score: float = 0.0
    
    def __str__(self) -> str:
        return f"Title: {self.title}\nURL: {self.url}\nContent: {self.content}"


class SearchResponse(BaseModel):
    """Response from search operation."""
    results: List[SearchResult] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    videos: List[Dict[str, str]] = Field(default_factory=list)
    total_results: int = 0
    search_time_ms: int = 0


class QueryPlanStep(BaseModel):
    """A step in the query execution plan."""
    id: int
    step: str
    queries: List[str] = Field(default_factory=list)
    dependencies: List[int] = Field(default_factory=list)
    status: str = "pending"  # pending, current, done


class QueryPlan(BaseModel):
    """Multi-step query plan for Pro Search."""
    steps: List[QueryPlanStep]
    estimated_time: int = 30


class RelatedQuestion(BaseModel):
    """A related follow-up question."""
    question: str
    category: str = "general"  # general, deeper, related, clarification


class ConsensusAnalysis(BaseModel):
    """Consensus analysis for academic research."""
    agreement_level: str = "mixed"  # strong_yes, yes, mixed, no, strong_no
    confidence: float = 0.0
    sample_size: int = 0
    key_findings: List[str] = Field(default_factory=list)
    study_designs: Dict[str, int] = Field(default_factory=dict)
    evidence_quality: str = "moderate"  # high, moderate, low


class Citation(BaseModel):
    """Inline citation reference."""
    number: int
    title: str
    url: str
    snippet: str
    source_type: str = "web"


class AnswerResponse(BaseModel):
    """Complete answer response."""
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    related_questions: List[RelatedQuestion] = Field(default_factory=list)
    images: List[str] = Field(default_factory=list)
    consensus: Optional[ConsensusAnalysis] = None
    search_results: List[SearchResult] = Field(default_factory=list)
    query_plan: Optional[QueryPlan] = None


class StreamEvent(str, Enum):
    """Types of streaming events."""
    BEGIN = "begin"
    SEARCH_RESULTS = "search-results"
    QUERY_PLAN = "query-plan"
    STEP_START = "step-start"
    STEP_COMPLETE = "step-complete"
    TEXT_CHUNK = "text-chunk"
    CITATION = "citation"
    RELATED_QUESTIONS = "related-questions"
    CONSENSUS = "consensus"
    IMAGES = "images"
    ERROR = "error"
    DONE = "done"


# =============================================================================
# LLM Setup with Fallback Support
# =============================================================================

# Import from core module
from app.core.llm import (
    get_llm as core_get_llm,
    invoke_with_fallback,
    mark_model_unavailable,
    MEGALLM_MODELS
)

# Import researcher prompts and academic prioritization
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
    SourceType
)


def get_llm(temperature: float = 0.3, model_override: Optional[str] = None):
    """Get LLM with automatic fallback support."""
    return core_get_llm(
        temperature=temperature,
        model=model_override,
        fallback=True
    )


# =============================================================================
# Search Providers
# =============================================================================

class SearchProvider:
    """Base class for search providers."""
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        raise NotImplementedError


class TavilySearchProvider(SearchProvider):
    """Tavily search provider with academic source classification."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        try:
            from tavily import TavilyClient
            
            client = TavilyClient(api_key=self.api_key)
            response = await asyncio.to_thread(
                client.search,
                query=query,
                search_depth="advanced",
                max_results=num_results,
                include_images=True
            )
            
            results = []
            for r in response.get("results", []):
                url = r.get("url", "")
                source_type, quality_score = classify_source(url)
                
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=url,
                    content=r.get("content", ""),
                    snippet=r.get("content", "")[:200],
                    source_type=source_type.value,
                    relevance_score=quality_score  # Use quality score instead of raw score
                ))
            
            # Sort by quality score (academic sources first)
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            return SearchResponse(
                results=results,
                images=response.get("images", [])[:5],
                total_results=len(results)
            )
            
        except Exception as e:
            print(f"Tavily search error: {e}")
            return SearchResponse()


class SerperSearchProvider(SearchProvider):
    """Serper.dev search provider with academic source classification."""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                # Web search
                response = await client.post(
                    "https://google.serper.dev/search",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={"q": query, "num": num_results}
                )
                data = response.json()
                
                results = []
                for r in data.get("organic", []):
                    url = r.get("link", "")
                    source_type, quality_score = classify_source(url)
                    
                    results.append(SearchResult(
                        title=r.get("title", ""),
                        url=url,
                        content=r.get("snippet", ""),
                        snippet=r.get("snippet", ""),
                        favicon=r.get("favicon", ""),
                        source_type=source_type.value,
                        relevance_score=quality_score
                    ))
                
                # Sort by quality score (academic sources first)
                results.sort(key=lambda x: x.relevance_score, reverse=True)
                
                # Image search
                img_response = await client.post(
                    "https://google.serper.dev/images",
                    headers={
                        "X-API-KEY": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={"q": query, "num": 5}
                )
                img_data = img_response.json()
                images = [img.get("imageUrl", "") for img in img_data.get("images", [])[:5]]
                
                return SearchResponse(
                    results=results,
                    images=images,
                    total_results=len(results)
                )
                
        except Exception as e:
            print(f"Serper search error: {e}")
            return SearchResponse()


class AcademicSearchProvider(SearchProvider):
    """Academic paper search using ArXiv, Semantic Scholar, and PubMed."""
    
    async def search(self, query: str, num_results: int = 10) -> SearchResponse:
        results = []
        
        # ArXiv search (highest priority for CS/AI papers)
        try:
            import arxiv
            
            search = arxiv.Search(
                query=query.replace('"', ''),
                max_results=num_results // 2,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            client = arxiv.Client()
            papers = await asyncio.to_thread(lambda: list(client.results(search)))  # type: ignore
            
            for paper in papers:
                results.append(SearchResult(
                    title=paper.title,
                    url=paper.entry_id,
                    content=paper.summary,
                    snippet=paper.summary[:200],
                    source_type=SourceType.ARXIV_PREPRINT.value,
                    published_date=paper.published.strftime("%Y-%m-%d"),
                    author=", ".join(a.name for a in paper.authors[:3]),
                    relevance_score=0.95  # High score for ArXiv
                ))
                
        except Exception as e:
            print(f"ArXiv search error: {e}")
        
        # Semantic Scholar search
        try:
            import httpx
            
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": num_results // 2,
                        "fields": "title,abstract,url,year,authors,citationCount,venue"
                    },
                    timeout=10
                )
                
                if response.status_code == 200:
                    data = response.json()
                    for paper in data.get("data", []):
                        citation_count = paper.get("citationCount") or 0
                        # Higher citation count = higher score
                        citation_boost = min(citation_count / 1000, 0.05)
                        
                        results.append(SearchResult(
                            title=paper.get("title", ""),
                            url=paper.get("url", f"https://www.semanticscholar.org/paper/{paper.get('paperId', '')}"),
                            content=paper.get("abstract", "") or "",
                            snippet=(paper.get("abstract", "") or "")[:200],
                            source_type=SourceType.SEMANTIC_SCHOLAR.value,
                            published_date=str(paper.get("year", "")),
                            author=", ".join(
                                a.get("name", "") for a in (paper.get("authors", []) or [])[:3]
                            ),
                            citation_count=citation_count,
                            relevance_score=0.94 + citation_boost
                        ))
                        
        except Exception as e:
            print(f"Semantic Scholar search error: {e}")
        
        # PubMed search for medical/bio queries
        if any(term in query.lower() for term in ["medical", "health", "disease", "clinical", "therapy", "drug", "patient"]):
            try:
                await self._search_pubmed(query, results, num_results // 3)
            except Exception as e:
                print(f"PubMed search error: {e}")
        
        # Sort by relevance score
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return SearchResponse(results=results, total_results=len(results))
    
    async def _search_pubmed(self, query: str, results: List[SearchResult], max_results: int):
        """Search PubMed for medical/scientific papers."""
        import httpx
        
        async with httpx.AsyncClient() as client:
            # Search for paper IDs
            search_response = await client.get(
                "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
                params={
                    "db": "pubmed",
                    "term": query,
                    "retmax": max_results,
                    "retmode": "json"
                },
                timeout=10
            )
            
            if search_response.status_code == 200:
                search_data = search_response.json()
                ids = search_data.get("esearchresult", {}).get("idlist", [])
                
                if ids:
                    # Fetch paper details
                    fetch_response = await client.get(
                        "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                        params={
                            "db": "pubmed",
                            "id": ",".join(ids),
                            "retmode": "json"
                        },
                        timeout=10
                    )
                    
                    if fetch_response.status_code == 200:
                        fetch_data = fetch_response.json()
                        for pid in ids:
                            paper = fetch_data.get("result", {}).get(pid, {})
                            if paper and isinstance(paper, dict):
                                authors = paper.get("authors", [])
                                author_str = ", ".join(a.get("name", "") for a in authors[:3]) if authors else ""
                                
                                results.append(SearchResult(
                                    title=paper.get("title", ""),
                                    url=f"https://pubmed.ncbi.nlm.nih.gov/{pid}/",
                                    content=paper.get("title", ""),
                                    snippet=paper.get("title", ""),
                                    source_type=SourceType.PUBMED.value,
                                    published_date=paper.get("pubdate", ""),
                                    author=author_str,
                                    relevance_score=0.96  # Very high for PubMed
                                ))


def get_search_provider() -> SearchProvider:
    """Get the configured search provider."""
    if settings.tavily_api_key:
        return TavilySearchProvider(settings.tavily_api_key)
    elif os.getenv("SERPER_API_KEY"):
        return SerperSearchProvider(os.getenv("SERPER_API_KEY") or "")
    else:
        return TavilySearchProvider(settings.tavily_api_key or "")


# =============================================================================
# Query Planning (Pro Search)
# =============================================================================

QUERY_PLAN_PROMPT = """You are an expert at creating search task lists. Break down the given query into logical steps that can be executed using a search engine.

Rules:
1. Use up to 4 steps maximum, fewer if possible
2. Keep steps simple and concise
3. Use dependencies between steps properly
4. Always include a final step to synthesize/compare results

Query: {query}

Return JSON:
{{
    "steps": [
        {{"id": 0, "step": "step description", "dependencies": []}},
        {{"id": 1, "step": "step description", "dependencies": [0]}}
    ]
}}

Only return valid JSON."""


async def generate_query_plan(query: str) -> QueryPlan:
    """Generate multi-step query plan for Pro Search."""
    llm = get_llm(temperature=0.3)
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=QUERY_PLAN_PROMPT.format(query=query))
        ])
        
        content = response.content
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                steps = [
                    QueryPlanStep(
                        id=s["id"],
                        step=s["step"],
                        dependencies=s.get("dependencies", [])
                    )
                    for s in data.get("steps", [])
                ]
                return QueryPlan(steps=steps)
        
        # Fallback single step
        return QueryPlan(steps=[
            QueryPlanStep(id=0, step=f"Research: {query}", dependencies=[])
        ])
        
    except Exception as e:
        print(f"Query plan error: {e}")
        return QueryPlan(steps=[
            QueryPlanStep(id=0, step=f"Research: {query}", dependencies=[])
        ])


SEARCH_QUERY_PROMPT = """Generate concise search queries for the given step.

User Query: {user_query}
Current Step: {current_step}
Previous Context: {prev_context}

Return JSON with 1-3 search queries:
{{
    "queries": ["query 1", "query 2"]
}}

Only return valid JSON."""


async def generate_step_queries(
    user_query: str,
    step: str,
    prev_context: str = ""
) -> List[str]:
    """Generate search queries for a query plan step."""
    llm = get_llm(temperature=0.4)
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=SEARCH_QUERY_PROMPT.format(
                user_query=user_query,
                current_step=step,
                prev_context=prev_context[:2000]
            ))
        ])
        
        content = response.content
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return data.get("queries", [step])
        
        return [step]
        
    except Exception as e:
        print(f"Step query generation error: {e}")
        return [step]


# =============================================================================
# Content Fetching
# =============================================================================

async def fetch_page_content(url: str, timeout: int = 5) -> str:
    """Fetch and extract main content from a URL."""
    try:
        import httpx
        from bs4 import BeautifulSoup
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                url,
                timeout=timeout,
                follow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
            )
            
            if response.status_code != 200:
                return ""
            
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "footer", "iframe", "img", "header"]):
                element.decompose()
            
            # Get text
            text = soup.get_text(separator=" ", strip=True)
            
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text[:10000]  # Limit content size
            
    except Exception as e:
        print(f"Fetch error for {url}: {e}")
        return ""


async def fetch_multiple_contents(
    sources: List[SearchResult],
    max_concurrent: int = 5
) -> Dict[str, str]:
    """Fetch content from multiple URLs concurrently."""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def fetch_with_semaphore(source: SearchResult) -> Tuple[str, str]:
        async with semaphore:
            content = await fetch_page_content(source.url)
            return source.url, content
    
    tasks = [fetch_with_semaphore(s) for s in sources]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    contents = {}
    for result in results:
        if isinstance(result, tuple):
            url, content = result
            if content:
                contents[url] = content
    
    return contents


# =============================================================================
# Answer Generation with Citations (Academic Priority)
# =============================================================================

ANSWER_PROMPT = """You are an AI Research Assistant providing well-sourced answers with academic rigor.

Question: {question}

Search Results (ranked by source credibility):
{context}

Source Quality Indicators:
🏆 = Peer-reviewed/Academic (highest credibility)
🎓 = University/Research Institute
📄 = Conference/Technical Documentation
📰 = Reputable News/Wikipedia
🌐 = General Web Source

Instructions:
1. Prioritize information from 🏆 and 🎓 sources
2. Use inline citations [1], [2], etc.
3. Be accurate, thorough, and academically rigorous
4. Acknowledge limitations and uncertainty
5. Structure your answer clearly
6. For medical/scientific topics, emphasize peer-reviewed findings

Response Language: {language}

Write a comprehensive, well-cited answer:"""


async def generate_answer_with_citations(
    question: str,
    search_results: List[SearchResult],
    contents: Dict[str, str],
    language: str = "vi"
) -> Tuple[str, List[Citation]]:
    """Generate answer with inline citations, prioritizing academic sources."""
    llm = get_llm(temperature=0.3)
    
    # Sort results by quality (academic first)
    sorted_results = sorted(search_results, key=lambda x: x.relevance_score, reverse=True)
    
    # Build context with citation numbers and quality indicators
    context_parts = []
    citations = []
    
    for i, result in enumerate(sorted_results[:10], 1):
        content = contents.get(result.url, result.content)
        if content:
            emoji = get_source_tier_emoji(result.source_type)
            context_parts.append(
                f"[{i}] {emoji} {result.title}\n"
                f"Source Type: {result.source_type} | Quality Score: {result.relevance_score:.2f}\n"
                f"Content: {content[:1500]}"
            )
            citations.append(Citation(
                number=i,
                title=result.title,
                url=result.url,
                snippet=result.snippet or content[:200],
                source_type=result.source_type
            ))
    
    context = "\n\n---\n\n".join(context_parts)
    
    # Calculate confidence based on source quality
    source_dicts = [{"url": r.url, "quality_score": r.relevance_score, "source_type": r.source_type} for r in sorted_results[:10]]
    confidence_info = calculate_answer_confidence(source_dicts)
    
    try:
        # Use the detailed AI Researcher system prompt
        language_instruction = "Trả lời bằng tiếng Việt. Giữ các thuật ngữ kỹ thuật bằng tiếng Anh." if language == "vi" else "Respond in English."
        
        response = await llm.ainvoke([
            SystemMessage(content=AI_RESEARCHER_SYSTEM_PROMPT),
            HumanMessage(content=ANSWER_PROMPT.format(
                question=question,
                context=context,
                language=language_instruction
            ))
        ])
        
        answer = response.content if isinstance(response.content, str) else str(response.content)
        
        # Add confidence note at the end
        confidence_note = f"\n\n---\n📊 **Độ tin cậy:** {confidence_info['level'].upper()} ({confidence_info['confidence']:.0%}) - Dựa trên {confidence_info['academic_sources']} nguồn học thuật trong tổng số {confidence_info['total_sources']} nguồn." if language == "vi" else f"\n\n---\n📊 **Confidence:** {confidence_info['level'].upper()} ({confidence_info['confidence']:.0%}) - Based on {confidence_info['academic_sources']} academic sources out of {confidence_info['total_sources']} total sources."
        
        answer += confidence_note
        
        # Filter citations to only those actually used
        used_citations = []
        for citation in citations:
            if f"[{citation.number}]" in answer:
                used_citations.append(citation)
        
        return answer, used_citations
        
    except Exception as e:
        print(f"Answer generation error: {e}")
        return f"Error generating answer: {e}", []


# =============================================================================
# Related Questions Generation (Academic-focused)
# =============================================================================

RELATED_QUESTIONS_PROMPT = """Based on the research topic and findings, generate 5 academic follow-up questions that would lead to deeper understanding.

Topic: {query}
Current Findings Summary: {context}

Generate questions in these categories:
1. **Deeper**: Explores the topic in more technical depth
2. **Broader**: Connects to related fields or applications
3. **Methodological**: Questions about research methods or evidence quality
4. **Applied**: Practical applications or implementations
5. **Critical**: Questions that challenge assumptions or explore limitations

Return JSON:
{{
    "questions": [
        {{"question": "...", "category": "deeper", "why_relevant": "brief explanation"}},
        {{"question": "...", "category": "broader", "why_relevant": "brief explanation"}},
        {{"question": "...", "category": "methodological", "why_relevant": "brief explanation"}},
        {{"question": "...", "category": "applied", "why_relevant": "brief explanation"}},
        {{"question": "...", "category": "critical", "why_relevant": "brief explanation"}}
    ]
}}

Only return valid JSON."""


async def generate_related_questions(
    query: str,
    search_results: List[SearchResult]
) -> List[RelatedQuestion]:
    """Generate academically-focused related follow-up questions."""
    llm = get_llm(temperature=0.5)
    
    # Prioritize academic sources for context
    academic_results = [r for r in search_results if r.relevance_score >= 0.85]
    if not academic_results:
        academic_results = search_results[:5]
    
    context = "\n".join(f"- {r.title}: {r.snippet}" for r in academic_results[:5])
    
    try:
        response = await llm.ainvoke([
            SystemMessage(content="You are a research assistant helping generate insightful academic follow-up questions."),
            HumanMessage(content=RELATED_QUESTIONS_PROMPT.format(
                query=query,
                context=context[:3000]
            ))
        ])
        
        content = response.content
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return [
                    RelatedQuestion(
                        question=q["question"],
                        category=q.get("category", "general")
                    )
                    for q in data.get("questions", [])
                ]
        
        return []
        
    except Exception as e:
        print(f"Related questions error: {e}")
        return []


# =============================================================================
# Consensus Analysis (for Academic Research)
# =============================================================================

CONSENSUS_PROMPT = """Analyze the academic sources to determine consensus on the research question.

Question: {question}

Academic Sources:
{sources}

Analyze:
1. Agreement level (strong_yes, yes, mixed, no, strong_no)
2. Confidence level (0-1)
3. Sample size (number of studies)
4. Key findings from each study
5. Study designs mentioned
6. Overall evidence quality (high, moderate, low)

Return JSON:
{{
    "agreement_level": "mixed",
    "confidence": 0.7,
    "sample_size": 5,
    "key_findings": ["finding 1", "finding 2"],
    "study_designs": {{"RCT": 2, "Meta-analysis": 1}},
    "evidence_quality": "moderate"
}}

Only return valid JSON."""


async def analyze_consensus(
    question: str,
    academic_results: List[SearchResult]
) -> ConsensusAnalysis:
    """Analyze consensus among academic sources."""
    if not academic_results:
        return ConsensusAnalysis()
    
    llm = get_llm(temperature=0.2)
    
    sources = "\n\n".join(
        f"Study: {r.title}\nAuthors: {r.author or 'N/A'}\nAbstract: {r.content}"
        for r in academic_results
    )
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=CONSENSUS_PROMPT.format(
                question=question,
                sources=sources[:8000]
            ))
        ])
        
        content = response.content
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return ConsensusAnalysis(
                    agreement_level=data.get("agreement_level", "mixed"),
                    confidence=data.get("confidence", 0.5),
                    sample_size=data.get("sample_size", len(academic_results)),
                    key_findings=data.get("key_findings", []),
                    study_designs=data.get("study_designs", {}),
                    evidence_quality=data.get("evidence_quality", "moderate")
                )
        
        return ConsensusAnalysis(sample_size=len(academic_results))
        
    except Exception as e:
        print(f"Consensus analysis error: {e}")
        return ConsensusAnalysis(sample_size=len(academic_results))


# =============================================================================
# Main Answer Engine
# =============================================================================

@dataclass
class AnswerEngineConfig:
    """Configuration for the answer engine."""
    include_images: bool = True
    include_academic: bool = True
    enable_pro_search: bool = False
    max_sources: int = 10
    language: str = "vi"
    enable_consensus: bool = False


async def answer_engine(
    query: str,
    config: Optional[AnswerEngineConfig] = None
) -> AnswerResponse:
    """
    Main answer engine - like Perplexity.
    
    Args:
        query: User's question
        config: Engine configuration
        
    Returns:
        Complete answer with citations, sources, and related questions
    """
    config = config or AnswerEngineConfig()
    
    # Get search provider
    search_provider = get_search_provider()
    
    # Search for sources
    web_response = await search_provider.search(query, num_results=config.max_sources)
    
    # Academic search if enabled
    academic_results = []
    if config.include_academic:
        academic_provider = AcademicSearchProvider()
        academic_response = await academic_provider.search(query, num_results=5)
        academic_results = academic_response.results
    
    # Combine results
    all_results = web_response.results + academic_results
    
    # Fetch content
    contents = await fetch_multiple_contents(all_results[:10])
    
    # Generate answer with citations
    answer, citations = await generate_answer_with_citations(
        query, all_results, contents, config.language
    )
    
    # Generate related questions
    related_questions = await generate_related_questions(query, all_results)
    
    # Consensus analysis for academic queries
    consensus = None
    if config.enable_consensus and academic_results:
        consensus = await analyze_consensus(query, academic_results)
    
    return AnswerResponse(
        answer=answer,
        citations=citations,
        related_questions=related_questions,
        images=web_response.images if config.include_images else [],
        consensus=consensus,
        search_results=all_results
    )


async def answer_engine_stream(
    query: str,
    config: Optional[AnswerEngineConfig] = None
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Streaming answer engine - like Perplexity with real-time updates.
    
    Yields SSE events as the search progresses.
    """
    config = config or AnswerEngineConfig()
    
    # Begin event
    yield {"event": StreamEvent.BEGIN.value, "data": {"query": query}}
    
    try:
        # Query planning for Pro Search
        query_plan = None
        if config.enable_pro_search:
            query_plan = await generate_query_plan(query)
            yield {
                "event": StreamEvent.QUERY_PLAN.value,
                "data": {"steps": [s.dict() for s in query_plan.steps]}
            }
        
        # Search
        search_provider = get_search_provider()
        
        if config.enable_pro_search and query_plan:
            # Pro Search: execute each step
            all_results = []
            step_context = ""
            
            for step in query_plan.steps:
                yield {
                    "event": StreamEvent.STEP_START.value,
                    "data": {"step_id": step.id, "step": step.step}
                }
                
                # Generate queries for this step
                queries = await generate_step_queries(query, step.step, step_context)
                
                # Search for each query
                for q in queries:
                    response = await search_provider.search(q, num_results=5)
                    all_results.extend(response.results)
                
                # Update context
                step_context += "\n".join(r.snippet for r in all_results[-5:])
                
                yield {
                    "event": StreamEvent.STEP_COMPLETE.value,
                    "data": {"step_id": step.id, "results_count": len(all_results)}
                }
            
            web_response = SearchResponse(
                results=all_results[:config.max_sources],
                total_results=len(all_results)
            )
        else:
            # Standard search
            web_response = await search_provider.search(query, num_results=config.max_sources)
        
        # Academic search
        academic_results = []
        if config.include_academic:
            academic_provider = AcademicSearchProvider()
            academic_response = await academic_provider.search(query, num_results=5)
            academic_results = academic_response.results
        
        all_results = web_response.results + academic_results
        
        # Send search results
        yield {
            "event": StreamEvent.SEARCH_RESULTS.value,
            "data": {
                "results": [r.dict() for r in all_results[:10]],
                "images": web_response.images
            }
        }
        
        # Fetch content
        contents = await fetch_multiple_contents(all_results[:10])
        
        # Stream answer generation
        llm = get_llm(temperature=0.3)
        
        # Build context
        context_parts = []
        citations = []
        for i, result in enumerate(all_results[:10], 1):
            content = contents.get(result.url, result.content)
            if content:
                context_parts.append(f"[{i}] {result.title}\n{content[:1500]}")
                citations.append(Citation(
                    number=i,
                    title=result.title,
                    url=result.url,
                    snippet=result.snippet or content[:200],
                    source_type=result.source_type
                ))
        
        context = "\n\n".join(context_parts)
        
        # Stream answer
        system_prompt = "You are a helpful AI assistant. Use [1], [2], etc. to cite sources inline."
        if config.language == "vi":
            system_prompt += " Trả lời bằng tiếng Việt."
        
        full_answer = ""
        async for chunk in llm.astream([
            SystemMessage(content=system_prompt),
            HumanMessage(content=ANSWER_PROMPT.format(
                question=query,
                context=context
            ))
        ]):
            if chunk.content:
                chunk_text = chunk.content if isinstance(chunk.content, str) else str(chunk.content)
                full_answer += chunk_text
                yield {
                    "event": StreamEvent.TEXT_CHUNK.value,
                    "data": {"text": chunk_text}
                }
        
        # Send citations
        used_citations = [c for c in citations if f"[{c.number}]" in full_answer]
        yield {
            "event": StreamEvent.CITATION.value,
            "data": {"citations": [c.dict() for c in used_citations]}
        }
        
        # Related questions
        related_questions = await generate_related_questions(query, all_results)
        yield {
            "event": StreamEvent.RELATED_QUESTIONS.value,
            "data": {"questions": [q.dict() for q in related_questions]}
        }
        
        # Consensus for academic
        if config.enable_consensus and academic_results:
            consensus = await analyze_consensus(query, academic_results)
            yield {
                "event": StreamEvent.CONSENSUS.value,
                "data": consensus.dict()
            }
        
        # Images
        if config.include_images and web_response.images:
            yield {
                "event": StreamEvent.IMAGES.value,
                "data": {"images": web_response.images}
            }
        
        # Done
        yield {
            "event": StreamEvent.DONE.value,
            "data": {
                "total_sources": len(all_results),
                "citations_used": len(used_citations)
            }
        }
        
    except Exception as e:
        yield {
            "event": StreamEvent.ERROR.value,
            "data": {"error": str(e)}
        }


# =============================================================================
# Quick Search (Simple mode with Academic Priority)
# =============================================================================

async def quick_search(
    query: str,
    num_results: int = 5,
    language: str = "vi"
) -> Dict[str, Any]:
    """
    Quick search for simple queries with academic source prioritization.
    
    Returns basic answer with sources ranked by credibility.
    """
    search_provider = get_search_provider()
    response = await search_provider.search(query, num_results=num_results)
    
    # Sort results by quality score (academic first)
    sorted_results = sorted(response.results, key=lambda x: x.relevance_score, reverse=True)
    
    # Build context with quality indicators
    context_parts = []
    for i, r in enumerate(sorted_results, 1):
        emoji = get_source_tier_emoji(r.source_type)
        context_parts.append(f"[{i}] {emoji} {r.title}: {r.content}")
    
    context = "\n\n".join(context_parts)
    
    # Calculate confidence based on source quality
    source_dicts = [{"url": r.url, "quality_score": r.relevance_score, "source_type": r.source_type} for r in sorted_results]
    confidence_info = calculate_answer_confidence(source_dicts)
    
    # Generate quick answer with AI Researcher prompt
    llm = get_llm(temperature=0.3)
    
    language_note = "Trả lời bằng tiếng Việt. Giữ thuật ngữ kỹ thuật bằng tiếng Anh." if language == "vi" else "Respond in English."
    
    prompt = f"""You are an AI Research Assistant. Answer this question using the search results.

Rules:
1. Use inline citations [1], [2] when referencing sources
2. Prioritize information from academic sources (marked with 🏆 or 🎓)
3. Be accurate and concise
4. {language_note}

Question: {query}

Search Results (ranked by credibility):
{context}

Write a brief, well-cited answer:"""
    
    answer_response = await llm.ainvoke([
        SystemMessage(content="You are a helpful AI research assistant that prioritizes accuracy and academic sources."),
        HumanMessage(content=prompt)
    ])
    
    return {
        "answer": answer_response.content,
        "sources": [
            {
                "title": r.title, 
                "url": r.url, 
                "snippet": r.snippet,
                "source_type": r.source_type,
                "quality_score": round(r.relevance_score, 2)
            }
            for r in sorted_results
        ],
        "images": response.images,
        "confidence": confidence_info
    }
