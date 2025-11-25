"""
Deep Recursive Research Tool.

Ported from dzhng/deep-research (TypeScript) to Python.
Implements iterative deep research with breadth/depth control.

Key Features:
- Generate SERP queries using LLM
- Search using Tavily
- Extract learnings from search results
- Recursively dive deeper if depth > 0
- Return consolidated Markdown report
"""

import asyncio
import os
import json
from typing import List, Dict, Any, Optional, Callable, AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings


# =============================================================================
# Types
# =============================================================================

@dataclass
class ResearchProgress:
    """Progress tracking for deep research."""
    current_depth: int
    total_depth: int
    current_breadth: int
    total_breadth: int
    total_queries: int = 0
    completed_queries: int = 0
    current_query: Optional[str] = None
    stage: str = "initializing"
    message: str = ""


@dataclass
class ResearchResult:
    """Result from deep research."""
    learnings: List[str] = field(default_factory=list)
    visited_urls: List[str] = field(default_factory=list)


@dataclass
class SerpQuery:
    """A search query with research goal."""
    query: str
    research_goal: str


# =============================================================================
# LLM Setup
# =============================================================================

def get_llm(temperature: float = 0.3):
    """Get LLM based on configuration."""
    google_key = settings.effective_google_api_key
    
    if settings.llm_provider == "megallm" and settings.megallm_api_key:
        return ChatOpenAI(
            model=settings.megallm_model,
            temperature=temperature,
            api_key=settings.megallm_api_key,  # type: ignore
            base_url=settings.megallm_base_url
        )
    elif settings.openai_api_key:
        return ChatOpenAI(
            model=settings.model_name,
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


# =============================================================================
# System Prompt
# =============================================================================

def get_system_prompt() -> str:
    """Get the system prompt for research."""
    now = datetime.now().isoformat()
    return f"""You are an expert researcher. Today is {now}. Follow these instructions when responding:
- You may be asked to research subjects after your knowledge cutoff, assume the user is right when presented with news.
- The user is a highly experienced analyst, no need to simplify, be as detailed as possible.
- Be highly organized.
- Suggest solutions that weren't thought about.
- Be proactive and anticipate needs.
- Treat the user as an expert in all subject matter.
- Mistakes erode trust, so be accurate and thorough.
- Provide detailed explanations with lots of detail.
- Value good arguments over authorities, the source is irrelevant.
- Consider new technologies and contrarian ideas, not just conventional wisdom.
- You may use high levels of speculation or prediction, just flag it."""


# =============================================================================
# Web Search with Tavily
# =============================================================================

async def search_tavily(query: str, num_results: int = 5) -> Dict[str, Any]:
    """Search web using Tavily API."""
    if not settings.tavily_api_key:
        return {"data": [], "error": "Tavily API key not configured"}
    
    os.environ["TAVILY_API_KEY"] = settings.tavily_api_key
    
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
        
        search_tool = TavilySearchResults(
            max_results=num_results,
            search_depth="advanced",
            include_raw_content=True,
        )
        
        results = await asyncio.to_thread(search_tool.invoke, query)
        
        # Transform to format similar to Firecrawl
        data = []
        if isinstance(results, list):
            for r in results:
                data.append({
                    "url": r.get("url", ""),
                    "title": r.get("title", ""),
                    "markdown": r.get("content", ""),
                })
        
        return {"data": data}
        
    except Exception as e:
        print(f"Tavily search error: {e}")
        return {"data": [], "error": str(e)}


# =============================================================================
# Generate SERP Queries (from dzhng/deep-research)
# =============================================================================

GENERATE_QUERIES_PROMPT = """Given the following prompt from the user, generate a list of SERP queries to research the topic. Return a maximum of {num_queries} queries, but feel free to return less if the original prompt is clear. Make sure each query is unique and not similar to each other.

<prompt>{query}</prompt>

{learnings_section}

Return a JSON object with format:
{{
    "queries": [
        {{"query": "search query 1", "researchGoal": "goal and how to advance research"}},
        {{"query": "search query 2", "researchGoal": "goal and how to advance research"}}
    ]
}}

Only return JSON, no other text."""


async def generate_serp_queries(
    query: str,
    num_queries: int = 3,
    learnings: Optional[List[str]] = None
) -> List[SerpQuery]:
    """Generate SERP queries for research topic."""
    llm = get_llm()
    
    learnings_section = ""
    if learnings:
        learnings_section = f"Here are some learnings from previous research, use them to generate more specific queries:\n{chr(10).join(learnings)}"
    
    prompt = GENERATE_QUERIES_PROMPT.format(
        query=query,
        num_queries=num_queries,
        learnings_section=learnings_section
    )
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=f"{get_system_prompt()}\n\n{prompt}")
        ])
        content = response.content
        
        if isinstance(content, str):
            # Parse JSON from response
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                json_str = content[start:end]
                data = json.loads(json_str)
                queries = data.get("queries", [])
                return [
                    SerpQuery(
                        query=q.get("query", ""),
                        research_goal=q.get("researchGoal", "")
                    )
                    for q in queries[:num_queries]
                ]
        
        return [SerpQuery(query=query, research_goal="Main research query")]
        
    except Exception as e:
        print(f"Error generating queries: {e}")
        return [SerpQuery(query=query, research_goal="Main research query")]


# =============================================================================
# Process Search Results (from dzhng/deep-research)
# =============================================================================

PROCESS_RESULTS_PROMPT = """Given the following contents from a SERP search for the query <query>{query}</query>, generate a list of learnings from the contents. Return a maximum of {num_learnings} learnings, but feel free to return less if the contents are clear.

Make sure each learning is unique and not similar to each other. The learnings should be concise and to the point, as detailed and information dense as possible. Make sure to include any entities like people, places, companies, products, things, etc in the learnings, as well as any exact metrics, numbers, or dates. The learnings will be used to research the topic further.

<contents>
{contents}
</contents>

Return a JSON object with format:
{{
    "learnings": ["learning 1", "learning 2", ...],
    "followUpQuestions": ["question 1", "question 2", ...]
}}

Only return JSON, no other text."""


async def process_serp_result(
    query: str,
    search_result: Dict[str, Any],
    num_learnings: int = 3,
    num_follow_up_questions: int = 3
) -> Dict[str, Any]:
    """Process search results to extract learnings."""
    data = search_result.get("data", [])
    if not data:
        return {"learnings": [], "followUpQuestions": []}
    
    llm = get_llm()
    
    # Format contents
    contents = []
    for item in data:
        markdown = item.get("markdown", "")
        if markdown:
            # Trim to ~5000 chars per item
            contents.append(f"<content>\n{markdown[:5000]}\n</content>")
    
    if not contents:
        return {"learnings": [], "followUpQuestions": []}
    
    prompt = PROCESS_RESULTS_PROMPT.format(
        query=query,
        num_learnings=num_learnings,
        contents="\n".join(contents)
    )
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=f"{get_system_prompt()}\n\n{prompt}")
        ])
        content = response.content
        
        if isinstance(content, str):
            start = content.find('{')
            end = content.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
        
        return {"learnings": [], "followUpQuestions": []}
        
    except Exception as e:
        print(f"Error processing results: {e}")
        return {"learnings": [], "followUpQuestions": []}


# =============================================================================
# Write Final Report (from dzhng/deep-research)
# =============================================================================

FINAL_REPORT_PROMPT = """Given the following prompt from the user, write a final report on the topic using the learnings from research. Make it as detailed as possible, aim for 3 or more pages, include ALL the learnings from research.

<prompt>{prompt}</prompt>

Here are all the learnings from previous research:

<learnings>
{learnings}
</learnings>

Write a comprehensive report in Markdown format with:
1. Executive Summary
2. Key Findings
3. Detailed Analysis
4. Conclusions and Recommendations
5. Sources

Return the report in Markdown."""


async def write_final_report(
    prompt: str,
    learnings: List[str],
    visited_urls: List[str]
) -> str:
    """Generate final research report."""
    llm = get_llm()
    
    learnings_text = "\n".join(f"<learning>\n{l}\n</learning>" for l in learnings)
    
    report_prompt = FINAL_REPORT_PROMPT.format(
        prompt=prompt,
        learnings=learnings_text
    )
    
    try:
        response = await llm.ainvoke([
            HumanMessage(content=f"{get_system_prompt()}\n\n{report_prompt}")
        ])
        report = response.content if isinstance(response.content, str) else str(response.content)
        
        # Append sources
        if visited_urls:
            urls_section = "\n\n## Sources\n\n" + "\n".join(f"- {url}" for url in visited_urls)
            report += urls_section
        
        return report
        
    except Exception as e:
        return f"Error generating report: {str(e)}"


# =============================================================================
# Deep Research Core Function (from dzhng/deep-research)
# =============================================================================

async def deep_research(
    query: str,
    breadth: int = 4,
    depth: int = 2,
    learnings: Optional[List[str]] = None,
    visited_urls: Optional[List[str]] = None,
    on_progress: Optional[Callable[[ResearchProgress], None]] = None,
    concurrency_limit: int = 2
) -> ResearchResult:
    """
    Perform deep recursive research on a topic.
    
    Args:
        query: Research query/topic
        breadth: Number of parallel queries per level (recommended 2-10)
        depth: How deep to recurse (recommended 1-5)
        learnings: Previous learnings to build upon
        visited_urls: Already visited URLs
        on_progress: Callback for progress updates
        concurrency_limit: Max concurrent searches
    
    Returns:
        ResearchResult with learnings and visited URLs
    """
    if learnings is None:
        learnings = []
    if visited_urls is None:
        visited_urls = []
    
    progress = ResearchProgress(
        current_depth=depth,
        total_depth=depth,
        current_breadth=breadth,
        total_breadth=breadth,
        total_queries=0,
        completed_queries=0,
        stage="generating_queries",
        message=f"Generating {breadth} search queries..."
    )
    
    def report_progress(update: Dict[str, Any]):
        for key, value in update.items():
            if hasattr(progress, key):
                setattr(progress, key, value)
        if on_progress:
            on_progress(progress)
    
    # Generate search queries
    report_progress({
        "stage": "generating_queries",
        "message": f"Generating {breadth} search queries for: {query[:50]}..."
    })
    
    serp_queries = await generate_serp_queries(
        query=query,
        learnings=learnings,
        num_queries=breadth
    )
    
    report_progress({
        "total_queries": len(serp_queries),
        "current_query": serp_queries[0].query if serp_queries else None,
        "stage": "searching",
        "message": f"Starting {len(serp_queries)} searches..."
    })
    
    # Process queries with concurrency limit
    semaphore = asyncio.Semaphore(concurrency_limit)
    
    async def process_query(serp_query: SerpQuery, idx: int) -> ResearchResult:
        async with semaphore:
            try:
                report_progress({
                    "current_query": serp_query.query,
                    "stage": "searching",
                    "message": f"[{idx+1}/{len(serp_queries)}] Searching: {serp_query.query[:40]}..."
                })
                
                # Search
                result = await search_tavily(serp_query.query, num_results=5)
                
                # Collect URLs
                new_urls = [item.get("url", "") for item in result.get("data", []) if item.get("url")]
                
                # Calculate new breadth and depth
                new_breadth = max(1, breadth // 2)
                new_depth = depth - 1
                
                report_progress({
                    "stage": "processing",
                    "message": f"Processing results for: {serp_query.query[:40]}..."
                })
                
                # Process results
                processed = await process_serp_result(
                    query=serp_query.query,
                    search_result=result,
                    num_learnings=3,
                    num_follow_up_questions=new_breadth
                )
                
                new_learnings = processed.get("learnings", [])
                follow_up_questions = processed.get("followUpQuestions", [])
                
                all_learnings = learnings + new_learnings
                all_urls = visited_urls + new_urls
                
                # Recurse if depth > 0
                if new_depth > 0:
                    report_progress({
                        "current_depth": new_depth,
                        "current_breadth": new_breadth,
                        "stage": "recursing",
                        "message": f"Diving deeper (depth={new_depth})..."
                    })
                    
                    # Build next query
                    next_query = f"""
Previous research goal: {serp_query.research_goal}
Follow-up research directions: {', '.join(follow_up_questions[:3]) if follow_up_questions else 'Continue exploring'}
""".strip()
                    
                    return await deep_research(
                        query=next_query,
                        breadth=new_breadth,
                        depth=new_depth,
                        learnings=all_learnings,
                        visited_urls=all_urls,
                        on_progress=on_progress,
                        concurrency_limit=concurrency_limit
                    )
                else:
                    report_progress({
                        "completed_queries": progress.completed_queries + 1,
                        "current_depth": 0,
                        "stage": "completed_query",
                        "message": f"Completed: {serp_query.query[:40]}..."
                    })
                    
                    return ResearchResult(
                        learnings=all_learnings,
                        visited_urls=all_urls
                    )
                    
            except Exception as e:
                print(f"Error processing query '{serp_query.query}': {e}")
                return ResearchResult(
                    learnings=learnings,
                    visited_urls=visited_urls
                )
    
    # Run all queries
    tasks = [process_query(sq, i) for i, sq in enumerate(serp_queries)]
    results = await asyncio.gather(*tasks)
    
    # Merge results
    all_learnings = list(set(
        learning
        for result in results
        for learning in result.learnings
    ))
    all_urls = list(set(
        url
        for result in results
        for url in result.visited_urls
    ))
    
    return ResearchResult(
        learnings=all_learnings,
        visited_urls=all_urls
    )


# =============================================================================
# Deep Research with Streaming (for API)
# =============================================================================

async def deep_research_stream(
    query: str,
    breadth: int = 4,
    depth: int = 2
) -> AsyncGenerator[Dict[str, Any], None]:
    """
    Perform deep research with streaming status updates.
    
    Yields:
        Dict with type (status/result) and data
    """
    progress_updates: List[ResearchProgress] = []
    
    def on_progress(progress: ResearchProgress):
        progress_updates.append(ResearchProgress(
            current_depth=progress.current_depth,
            total_depth=progress.total_depth,
            current_breadth=progress.current_breadth,
            total_breadth=progress.total_breadth,
            total_queries=progress.total_queries,
            completed_queries=progress.completed_queries,
            current_query=progress.current_query,
            stage=progress.stage,
            message=progress.message
        ))
    
    # Start research
    yield {
        "type": "status",
        "stage": "starting",
        "message": f"Starting deep research: {query[:50]}...",
        "progress": 5
    }
    
    # Run research
    result = await deep_research(
        query=query,
        breadth=breadth,
        depth=depth,
        on_progress=on_progress
    )
    
    # Yield progress updates
    for i, prog in enumerate(progress_updates):
        progress_pct = min(80, 10 + (i * 70 // max(1, len(progress_updates))))
        yield {
            "type": "status",
            "stage": prog.stage,
            "message": prog.message,
            "progress": progress_pct,
            "current_query": prog.current_query,
            "depth": prog.current_depth,
            "breadth": prog.current_breadth
        }
    
    # Generate report
    yield {
        "type": "status",
        "stage": "generating_report",
        "message": "Generating final report...",
        "progress": 85
    }
    
    report = await write_final_report(
        prompt=query,
        learnings=result.learnings,
        visited_urls=result.visited_urls
    )
    
    # Final result
    yield {
        "type": "result",
        "report": report,
        "learnings": result.learnings,
        "sources": result.visited_urls,
        "total_sources": len(result.visited_urls),
        "total_learnings": len(result.learnings),
        "progress": 100
    }


# =============================================================================
# LangChain Tool Wrapper
# =============================================================================

@tool
async def deep_research_tool(
    query: str,
    breadth: int = 3,
    depth: int = 2
) -> str:
    """
    Perform deep recursive research on a topic.
    
    Use this tool for complex research queries that require:
    - Multiple search iterations
    - Following up on findings
    - Building comprehensive understanding
    
    Args:
        query: The research query or topic
        breadth: Number of parallel searches (2-10, default 3)
        depth: How deep to research (1-5, default 2)
    
    Returns:
        A comprehensive research report in Markdown
    """
    try:
        result = await deep_research(
            query=query,
            breadth=min(10, max(2, breadth)),
            depth=min(5, max(1, depth))
        )
        
        report = await write_final_report(
            prompt=query,
            learnings=result.learnings,
            visited_urls=result.visited_urls
        )
        
        return report
        
    except Exception as e:
        return f"Research error: {str(e)}"
