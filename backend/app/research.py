"""
Deep Research Module for AI Research Assistant.

Implements iterative deep research workflow inspired by:
- dzhng/deep-research: Recursive research with breadth/depth
- NVIDIA aiq-research-assistant: Query generation → Research → Reflection

Key Features:
- Iterative query generation based on knowledge gaps
- Accumulating learnings across iterations  
- Reflection to identify missing information
- Multi-source research (web + arxiv + local docs)
"""

import asyncio
import time
from typing import List, Dict, Any, Optional, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import json

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings
from app.tools import search_arxiv_structured


class ResearchDepth(str, Enum):
    """Research depth levels."""
    QUICK = "quick"      # 1 iteration, 2 queries
    STANDARD = "standard"  # 2 iterations, 3 queries
    DEEP = "deep"        # 3 iterations, 5 queries


@dataclass
class ResearchSource:
    """A source from research."""
    title: str
    url: str
    snippet: str
    source_type: str = "web"  # web, arxiv, document
    relevance_score: float = 0.0


@dataclass
class ResearchFinding:
    """A research finding/learning."""
    content: str
    sources: List[str] = field(default_factory=list)
    confidence: float = 0.8


@dataclass
class ResearchState:
    """State for deep research workflow."""
    query: str
    depth: ResearchDepth
    breadth: int
    max_iterations: int
    include_arxiv: bool
    language: str
    
    # Accumulated state
    learnings: List[str] = field(default_factory=list)
    sources: List[ResearchSource] = field(default_factory=list)
    findings: List[ResearchFinding] = field(default_factory=list)
    queries_used: List[str] = field(default_factory=list)
    
    # Progress
    current_iteration: int = 0
    total_sources_analyzed: int = 0
    
    # Final output
    summary: str = ""
    follow_up_questions: List[str] = field(default_factory=list)


def get_llm():
    """Get the LLM based on configuration."""
    google_key = settings.effective_google_api_key
    
    # MegaLLM (OpenAI-compatible API)
    if settings.llm_provider == "megallm" and settings.megallm_api_key:
        return ChatOpenAI(
            model=settings.megallm_model,
            temperature=0.3,
            api_key=settings.megallm_api_key,  # type: ignore
            base_url=settings.megallm_base_url
        )
    elif settings.openai_api_key:
        return ChatOpenAI(
            model=settings.model_name,
            temperature=0.3,
            api_key=settings.openai_api_key  # type: ignore
        )
    elif google_key:
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=0.3,
            google_api_key=google_key  # type: ignore
        )
    else:
        raise ValueError("No LLM API key configured")


# =============================================================================
# Query Generation (from NVIDIA aiq-research-assistant)
# =============================================================================

QUERY_GENERATION_PROMPT = """Bạn là chuyên gia nghiên cứu AI. Nhiệm vụ của bạn là tạo ra các truy vấn tìm kiếm để nghiên cứu một chủ đề.

Chủ đề nghiên cứu: {topic}

Những gì đã học được từ nghiên cứu trước:
{previous_learnings}

Hãy tạo ra {num_queries} truy vấn tìm kiếm MỚI và KHÁC BIỆT để:
1. Lấp đầy khoảng trống kiến thức
2. Khám phá các khía cạnh chưa được nghiên cứu
3. Tìm thông tin chi tiết và cụ thể hơn

Trả về dưới dạng JSON:
{{
    "queries": [
        {{"query": "truy vấn 1", "purpose": "mục đích"}},
        {{"query": "truy vấn 2", "purpose": "mục đích"}}
    ],
    "focus_areas": ["lĩnh vực cần tập trung"]
}}

Chỉ trả về JSON, không có text khác."""


async def generate_research_queries(
    topic: str,
    previous_learnings: List[str],
    num_queries: int = 3
) -> List[Dict[str, str]]:
    """Generate search queries for research topic."""
    llm = get_llm()
    
    learnings_text = "\n".join(f"- {l}" for l in previous_learnings) if previous_learnings else "Chưa có"
    
    prompt = QUERY_GENERATION_PROMPT.format(
        topic=topic,
        previous_learnings=learnings_text,
        num_queries=num_queries
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        # Parse JSON from response
        if isinstance(content, str):
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                return data.get("queries", [])
        
        return [{"query": topic, "purpose": "main query"}]
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [{"query": topic, "purpose": "main query"}]


# =============================================================================
# Web Search with Tavily
# =============================================================================

async def search_web_tavily(query: str, num_results: int = 5) -> List[Dict[str, Any]]:
    """Search web using Tavily API."""
    import os
    
    if not settings.tavily_api_key:
        return []
    
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        
        search_tool = TavilySearchResults(
            max_results=num_results,
            search_depth="advanced",
            include_raw_content=True,
        )
        
        results = await asyncio.to_thread(search_tool.invoke, query)
        return results if isinstance(results, list) else []
        
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []


# =============================================================================
# Process Search Results (from dzhng/deep-research)
# =============================================================================

PROCESS_RESULTS_PROMPT = """Bạn là chuyên gia phân tích nghiên cứu. Hãy phân tích kết quả tìm kiếm và trích xuất thông tin quan trọng.

Chủ đề nghiên cứu: {topic}

Kết quả tìm kiếm:
{search_results}

Những gì đã học được trước đó:
{previous_learnings}

Hãy trích xuất các điểm học được MỚI (không lặp lại những gì đã biết):
1. Tóm tắt thông tin quan trọng
2. Ghi nhận nguồn cho mỗi thông tin
3. Đánh giá độ tin cậy

Trả về JSON:
{{
    "learnings": [
        {{"content": "điểm học được", "sources": ["url1", "url2"], "confidence": 0.8}}
    ],
    "knowledge_gaps": ["những gì còn thiếu"]
}}

Chỉ trả về JSON."""


async def process_search_results(
    topic: str,
    results: List[Dict[str, Any]],
    previous_learnings: List[str]
) -> Dict[str, Any]:
    """Process search results to extract learnings."""
    if not results:
        return {"learnings": [], "knowledge_gaps": []}
    
    llm = get_llm()
    
    # Format results
    results_text = ""
    for i, r in enumerate(results, 1):
        title = r.get('title', 'No title')
        url = r.get('url', '')
        content = r.get('content', '')[:500]
        results_text += f"[{i}] {title}\nURL: {url}\nContent: {content}\n\n"
    
    learnings_text = "\n".join(f"- {l}" for l in previous_learnings) if previous_learnings else "Chưa có"
    
    prompt = PROCESS_RESULTS_PROMPT.format(
        topic=topic,
        search_results=results_text,
        previous_learnings=learnings_text
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        
        return {"learnings": [], "knowledge_gaps": []}
        
    except Exception as e:
        print(f"Error processing results: {e}")
        return {"learnings": [], "knowledge_gaps": []}


# =============================================================================
# Reflection (from NVIDIA aiq-research-assistant)
# =============================================================================

REFLECTION_PROMPT = """Bạn là chuyên gia nghiên cứu AI. Hãy đánh giá tiến độ nghiên cứu.

Chủ đề: {topic}

Những gì đã học được:
{learnings}

Số nguồn đã phân tích: {num_sources}
Số vòng lặp: {iteration}/{max_iterations}

Hãy đánh giá:
1. Nghiên cứu đã đầy đủ chưa?
2. Còn thiếu thông tin gì quan trọng?
3. Có nên tiếp tục nghiên cứu không?

Trả về JSON:
{{
    "is_sufficient": true/false,
    "confidence": 0.0-1.0,
    "missing_aspects": ["khía cạnh còn thiếu"],
    "should_continue": true/false,
    "suggested_queries": ["truy vấn gợi ý nếu cần tiếp tục"]
}}

Chỉ trả về JSON."""


async def reflect_on_research(state: ResearchState) -> Dict[str, Any]:
    """Reflect on research progress and determine if sufficient."""
    llm = get_llm()
    
    learnings_text = "\n".join(f"- {l}" for l in state.learnings) if state.learnings else "Chưa có"
    
    prompt = REFLECTION_PROMPT.format(
        topic=state.query,
        learnings=learnings_text,
        num_sources=state.total_sources_analyzed,
        iteration=state.current_iteration,
        max_iterations=state.max_iterations
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        
        return {
            "is_sufficient": state.current_iteration >= state.max_iterations,
            "confidence": 0.5,
            "should_continue": state.current_iteration < state.max_iterations
        }
        
    except Exception as e:
        print(f"Error in reflection: {e}")
        return {"is_sufficient": True, "confidence": 0.5, "should_continue": False}


# =============================================================================
# Final Report Generation
# =============================================================================

FINAL_REPORT_PROMPT = """Bạn là chuyên gia nghiên cứu AI. Hãy viết báo cáo tổng hợp.

Chủ đề nghiên cứu: {topic}

Những điểm đã học được:
{learnings}

Nguồn tham khảo ({num_sources} nguồn):
{sources}

Ngôn ngữ output: {language}

Hãy viết:
1. **Tóm tắt điều hành** (Executive Summary) - 2-3 câu
2. **Các phát hiện chính** (Key Findings) - bullet points
3. **Phân tích chi tiết** - cho mỗi khía cạnh quan trọng
4. **Kết luận và đề xuất**
5. **Câu hỏi nghiên cứu tiếp theo** - 3-5 câu hỏi

Format output đẹp với Markdown."""


async def generate_final_report(state: ResearchState) -> str:
    """Generate final research report."""
    llm = get_llm()
    
    learnings_text = "\n".join(f"- {l}" for l in state.learnings) if state.learnings else "Không có phát hiện mới"
    
    sources_text = ""
    for i, src in enumerate(state.sources[:20], 1):  # Limit to 20 sources
        sources_text += f"[{i}] {src.title} - {src.url}\n"
    
    language = "Tiếng Việt" if state.language == "vi" else "English"
    
    prompt = FINAL_REPORT_PROMPT.format(
        topic=state.query,
        learnings=learnings_text,
        sources=sources_text or "Không có nguồn",
        num_sources=len(state.sources),
        language=language
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        return response.content if isinstance(response.content, str) else str(response.content)
        
    except Exception as e:
        return f"Lỗi tạo báo cáo: {str(e)}"


FOLLOW_UP_PROMPT = """Dựa trên nghiên cứu về "{topic}", hãy đề xuất 5 câu hỏi nghiên cứu tiếp theo.

Những gì đã tìm hiểu: {learnings}

Trả về JSON:
{{"questions": ["câu hỏi 1", "câu hỏi 2", ...]}}

Chỉ trả về JSON."""


async def generate_follow_up_questions(state: ResearchState) -> List[str]:
    """Generate follow-up research questions."""
    llm = get_llm()
    
    learnings_text = "; ".join(state.learnings[:5]) if state.learnings else state.query
    
    prompt = FOLLOW_UP_PROMPT.format(
        topic=state.query,
        learnings=learnings_text
    )
    
    try:
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        content = response.content
        
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                data = json.loads(content[start:end])
                return data.get("questions", [])[:5]
        
        return []
        
    except Exception:
        return []


# =============================================================================
# Main Deep Research Function
# =============================================================================

async def deep_research(
    query: str,
    depth: ResearchDepth = ResearchDepth.STANDARD,
    breadth: int = 3,
    max_iterations: int = 3,
    include_arxiv: bool = True,
    language: str = "vi"
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Perform deep research on a topic.
    
    Yields progress updates and final report.
    
    Args:
        query: Research topic/question
        depth: Research depth (quick/standard/deep)
        breadth: Number of parallel queries per iteration
        max_iterations: Maximum research iterations
        include_arxiv: Include ArXiv paper search
        language: Output language (vi/en)
    
    Yields:
        Progress updates and final results
    """
    start_time = time.time()
    
    # Adjust parameters based on depth
    if depth == ResearchDepth.QUICK:
        max_iterations = 1
        breadth = min(breadth, 2)
    elif depth == ResearchDepth.DEEP:
        max_iterations = max(max_iterations, 3)
        breadth = max(breadth, 4)
    
    state = ResearchState(
        query=query,
        depth=depth,
        breadth=breadth,
        max_iterations=max_iterations,
        include_arxiv=include_arxiv,
        language=language
    )
    
    yield {
        "type": "progress",
        "stage": "starting",
        "message": f"Bắt đầu nghiên cứu: {query}",
        "progress": 5
    }
    
    # Research iterations
    for iteration in range(max_iterations):
        state.current_iteration = iteration + 1
        
        yield {
            "type": "progress",
            "stage": "generating_queries",
            "message": f"Vòng {iteration + 1}/{max_iterations}: Tạo truy vấn tìm kiếm...",
            "progress": 10 + (iteration * 30)
        }
        
        # Generate search queries
        queries = await generate_research_queries(
            topic=query,
            previous_learnings=state.learnings,
            num_queries=breadth
        )
        
        # Search web
        yield {
            "type": "progress",
            "stage": "searching_web",
            "message": f"Tìm kiếm web với {len(queries)} truy vấn...",
            "progress": 20 + (iteration * 30)
        }
        
        all_results = []
        for q in queries:
            query_text = q.get("query", str(q)) if isinstance(q, dict) else str(q)
            state.queries_used.append(str(query_text))
            
            results = await search_web_tavily(str(query_text), num_results=3)
            all_results.extend(results)
            
            # Add sources
            for r in results:
                state.sources.append(ResearchSource(
                    title=r.get('title', 'No title'),
                    url=r.get('url', ''),
                    snippet=r.get('content', '')[:200],
                    source_type="web"
                ))
        
        state.total_sources_analyzed += len(all_results)
        
        # Search ArXiv if enabled
        if include_arxiv and iteration == 0:  # Only first iteration for ArXiv
            yield {
                "type": "progress",
                "stage": "searching_arxiv",
                "message": "Tìm kiếm papers trên ArXiv...",
                "progress": 30 + (iteration * 30)
            }
            
            arxiv_papers = await search_arxiv_structured(query, max_results=5)
            
            for paper in arxiv_papers:
                state.sources.append(ResearchSource(
                    title=paper.get('title', ''),
                    url=paper.get('url', ''),
                    snippet=paper.get('abstract', '')[:200],
                    source_type="arxiv"
                ))
                
                # Add paper abstract as a "result"
                all_results.append({
                    'title': paper.get('title', ''),
                    'url': paper.get('url', ''),
                    'content': paper.get('abstract', '')
                })
            
            state.total_sources_analyzed += len(arxiv_papers)
        
        # Process results
        yield {
            "type": "progress",
            "stage": "processing",
            "message": f"Phân tích {len(all_results)} kết quả...",
            "progress": 40 + (iteration * 30)
        }
        
        processed = await process_search_results(
            topic=query,
            results=all_results,
            previous_learnings=state.learnings
        )
        
        # Add new learnings
        for learning in processed.get("learnings", []):
            if isinstance(learning, dict):
                content = learning.get("content", "")
                if content and content not in state.learnings:
                    state.learnings.append(content)
                    state.findings.append(ResearchFinding(
                        content=content,
                        sources=learning.get("sources", []),
                        confidence=learning.get("confidence", 0.8)
                    ))
            elif isinstance(learning, str) and learning not in state.learnings:
                state.learnings.append(learning)
        
        # Reflection
        if iteration < max_iterations - 1:
            yield {
                "type": "progress",
                "stage": "reflecting",
                "message": "Đánh giá tiến độ nghiên cứu...",
                "progress": 50 + (iteration * 30)
            }
            
            reflection = await reflect_on_research(state)
            
            if reflection.get("is_sufficient", False) or not reflection.get("should_continue", True):
                yield {
                    "type": "progress",
                    "stage": "sufficient",
                    "message": "Nghiên cứu đã đầy đủ, tạo báo cáo...",
                    "progress": 80
                }
                break
    
    # Generate final report
    yield {
        "type": "progress",
        "stage": "generating_report",
        "message": "Tạo báo cáo tổng hợp...",
        "progress": 85
    }
    
    state.summary = await generate_final_report(state)
    
    # Generate follow-up questions
    yield {
        "type": "progress",
        "stage": "generating_questions",
        "message": "Đề xuất câu hỏi tiếp theo...",
        "progress": 95
    }
    
    state.follow_up_questions = await generate_follow_up_questions(state)
    
    elapsed_time = time.time() - start_time
    
    # Final result
    yield {
        "type": "complete",
        "result": {
            "query": state.query,
            "summary": state.summary,
            "findings": [
                {
                    "content": f.content,
                    "sources": f.sources,
                    "confidence": f.confidence
                }
                for f in state.findings
            ],
            "sources": [
                {
                    "title": s.title,
                    "url": s.url,
                    "snippet": s.snippet,
                    "source_type": s.source_type
                }
                for s in state.sources
            ],
            "follow_up_questions": state.follow_up_questions,
            "total_sources_analyzed": state.total_sources_analyzed,
            "research_time_seconds": round(elapsed_time, 2),
            "iterations_completed": state.current_iteration,
            "queries_used": state.queries_used
        }
    }


async def quick_research(query: str) -> Dict[str, Any]:
    """
    Perform quick research (non-streaming).
    Returns only the final result.
    """
    result = None
    async for update in deep_research(
        query=query,
        depth=ResearchDepth.QUICK,
        breadth=2,
        max_iterations=1,
        include_arxiv=False
    ):
        if update.get("type") == "complete":
            result = update.get("result")
    
    return result or {"error": "Research failed"}
