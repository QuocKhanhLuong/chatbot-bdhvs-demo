"""
FastAPI Server for AI Research Assistant.

Provides REST API endpoints for:
- Chat interaction (single agent)
- Multi-agent chat (auto-routing)
- Deep research (iterative research workflow)
- ArXiv paper search
"""
import json
import uuid
import os
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.agent import get_agent, reset_agent, SYSTEM_PROMPT
from app.agents import get_multi_agent_runner, get_persistent_runner
from app.research import deep_research, quick_research, ResearchDepth
from app.tools import search_arxiv_structured


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    thread_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    thread_id: str
    tool_calls: Optional[list] = None
    agent_used: Optional[str] = None


class DeepResearchRequest(BaseModel):
    """Request for deep research."""
    query: str = Field(..., description="Research topic/question")
    depth: str = Field(default="standard", description="quick/standard/deep")
    breadth: int = Field(default=3, ge=1, le=10, description="Number of parallel queries")
    max_iterations: int = Field(default=3, ge=1, le=10, description="Max research iterations")
    include_arxiv: bool = Field(default=True, description="Include ArXiv papers")
    language: str = Field(default="vi", description="Output language (vi/en)")


class ArxivSearchRequest(BaseModel):
    """Request for ArXiv search."""
    query: str = Field(..., description="Search query")
    max_results: int = Field(default=10, ge=1, le=50, description="Max results")


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    message: str
    tools: list[str]
    features: list[str]


# ============================================================================
# LIFESPAN MANAGEMENT
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown events."""
    # Startup
    print("🚀 Starting Personal AI Assistant...")
    try:
        agent = get_agent()
        print("✅ Agent initialized successfully!")
    except Exception as e:
        print(f"⚠️ Agent initialization warning: {e}")
    
    yield
    
    # Shutdown
    print("👋 Shutting down Personal AI Assistant...")


# ============================================================================
# FASTAPI APP
# ============================================================================

app = FastAPI(
    title="Personal AI Assistant API",
    description="A Research Assistant powered by LangGraph with web search, code execution, and document retrieval.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins + ["*"],  # Allow all for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static file serving for images and plots
STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
os.makedirs(os.path.join(STATIC_DIR, "images"), exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - health check."""
    features = [
        "Chat with AI assistant",
        "Multi-agent routing (research/coding/document)",
        "Deep research with iterative queries",
        "ArXiv paper search",
        "Web search (Tavily)",
        "Python code execution",
        "Local document retrieval"
    ]
    
    try:
        agent = get_agent()
        tools = ["web_search", "python_repl", "search_arxiv", "document_search"]
        return HealthResponse(
            status="healthy",
            message="AI Research Assistant is running! 🤖🔬",
            tools=tools,
            features=features
        )
    except Exception as e:
        return HealthResponse(
            status="warning",
            message=f"Agent not fully initialized: {str(e)}",
            tools=[],
            features=features
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "ai-research-assistant"}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Chat endpoint - send a message and get a response.
    
    Args:
        request: ChatRequest with message and optional thread_id
        
    Returns:
        ChatResponse with AI response and thread_id
    """
    try:
        agent = get_agent()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not available: {str(e)}"
        )
    
    # Generate thread_id if not provided
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # Prepare input
    input_messages = {"messages": [HumanMessage(content=request.message)]}
    config = {"configurable": {"thread_id": thread_id}}
    
    try:
        # Run the agent
        result = agent.invoke(input_messages, config)  # type: ignore
        
        # Extract the final response
        messages = result.get("messages", [])
        
        # Get the last AI message
        final_response = ""
        tool_calls_info = []
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage):
                if msg.content:
                    # Handle both string and list content
                    if isinstance(msg.content, str):
                        final_response = msg.content
                    elif isinstance(msg.content, list):
                        final_response = " ".join(
                            str(c) if isinstance(c, str) else str(c.get("text", ""))
                            for c in msg.content
                        )
                    break
                # Collect tool call info
                if msg.tool_calls:
                    tool_calls_info.extend([
                        {"name": tc["name"], "args": tc.get("args", {})}
                        for tc in msg.tool_calls
                    ])
        
        return ChatResponse(
            response=final_response or "I processed your request but have no response.",
            thread_id=thread_id,
            tool_calls=tool_calls_info if tool_calls_info else None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Streaming chat endpoint - send a message and get a streamed response.
    
    Returns Server-Sent Events (SSE) stream.
    """
    try:
        agent = get_agent()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail=f"Agent not available: {str(e)}"
        )
    
    # Generate thread_id if not provided
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def generate():
        """Generate SSE events from agent stream."""
        input_messages = {"messages": [HumanMessage(content=request.message)]}
        config = {"configurable": {"thread_id": thread_id}}
        
        try:
            # Stream the response
            for event in agent.stream(input_messages, config, stream_mode="values"):  # type: ignore
                messages = event.get("messages", [])
                if messages:
                    last_msg = messages[-1]
                    
                    if isinstance(last_msg, AIMessage):
                        # Send content
                        if last_msg.content:
                            content = last_msg.content
                            if isinstance(content, list):
                                content = " ".join(
                                    str(c) if isinstance(c, str) else str(c.get("text", ""))
                                    for c in content
                                )
                            data = {
                                "type": "content",
                                "content": content,
                                "thread_id": thread_id
                            }
                            yield f"data: {json.dumps(data)}\n\n"
                        
                        # Send tool calls info
                        if last_msg.tool_calls:
                            for tc in last_msg.tool_calls:
                                data = {
                                    "type": "tool_call",
                                    "name": tc["name"],
                                    "args": tc.get("args", {}),
                                    "thread_id": thread_id
                                }
                                yield f"data: {json.dumps(data)}\n\n"
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
            
        except Exception as e:
            error_data = {"type": "error", "message": str(e)}
            yield f"data: {json.dumps(error_data)}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/reset")
async def reset_conversation():
    """Reset the agent (clears memory)."""
    reset_agent()
    return {"status": "ok", "message": "Agent reset successfully"}


@app.get("/threads/{thread_id}/history")
async def get_thread_history(thread_id: str):
    """
    Get conversation history for a thread.
    
    Note: With MemorySaver, history is in-memory and will be lost on restart.
    """
    try:
        agent = get_agent()
        config = {"configurable": {"thread_id": thread_id}}
        
        # Get state snapshot
        state = agent.get_state(config)  # type: ignore
        
        if not state or not state.values:
            return {"thread_id": thread_id, "messages": []}
        
        messages = state.values.get("messages", [])
        
        # Convert to serializable format
        history = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, list):
                    content = " ".join(
                        str(c) if isinstance(c, str) else str(c.get("text", ""))
                        for c in content
                    )
                history.append({
                    "role": "assistant",
                    "content": content,
                    "tool_calls": msg.tool_calls if msg.tool_calls else None
                })
        
        return {"thread_id": thread_id, "messages": history}
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting history: {str(e)}"
        )


# ============================================================================
# MULTI-AGENT ENDPOINTS
# ============================================================================

@app.post("/agent/chat", response_model=ChatResponse)
async def multi_agent_chat(request: ChatRequest):
    """
    Multi-agent chat endpoint with persistent storage.
    
    Automatically routes to the appropriate agent:
    - Research Agent: Web search, ArXiv papers
    - Coding Agent: Python code execution
    - Document Agent: Local knowledge base
    - General Agent: Direct responses
    
    Chat history is persisted to SQLite database.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    
    try:
        # Use persistent runner with SQLite storage
        async with get_persistent_runner() as runner:
            result = await runner.run(request.message, thread_id)
        
        return ChatResponse(
            response=result.get("response", ""),
            thread_id=thread_id,
            agent_used=result.get("agent_used"),
            tool_calls=None
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )


@app.post("/agent/chat/stream")
async def multi_agent_chat_stream(request: ChatRequest):
    """
    Streaming multi-agent chat endpoint with persistent storage.
    Returns Server-Sent Events (SSE) stream.
    
    Chat history is persisted to SQLite database.
    """
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def generate():
        try:
            # Use persistent runner with SQLite storage
            async with get_persistent_runner() as runner:
                async for event in runner.stream(request.message, thread_id):
                    # Extract agent info
                    agent = event.get("current_agent", "")
                    
                    # Look for AI messages
                    for node_name, node_data in event.items():
                        if isinstance(node_data, dict) and "messages" in node_data:
                            for msg in node_data["messages"]:
                                if isinstance(msg, AIMessage) and msg.content:
                                    data = {
                                        "type": "content",
                                        "content": msg.content,
                                        "agent": agent,
                                        "thread_id": thread_id
                                    }
                                    yield f"data: {json.dumps(data)}\n\n"
            
            yield f"data: {json.dumps({'type': 'done', 'thread_id': thread_id})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# DEEP RESEARCH ENDPOINTS
# ============================================================================

@app.post("/research")
async def research_topic(request: DeepResearchRequest):
    """
    Perform deep research on a topic.
    
    Returns a comprehensive research report with:
    - Executive summary
    - Key findings from web and ArXiv
    - Sources with relevance scores
    - Follow-up research questions
    """
    try:
        depth_map = {
            "quick": ResearchDepth.QUICK,
            "standard": ResearchDepth.STANDARD,
            "deep": ResearchDepth.DEEP
        }
        depth = depth_map.get(request.depth, ResearchDepth.STANDARD)
        
        result = None
        async for update in deep_research(
            query=request.query,
            depth=depth,
            breadth=request.breadth,
            max_iterations=request.max_iterations,
            include_arxiv=request.include_arxiv,
            language=request.language
        ):
            if update.get("type") == "complete":
                result = update.get("result")
        
        if result:
            return result
        else:
            raise HTTPException(status_code=500, detail="Research failed")
            
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Research error: {str(e)}"
        )


@app.post("/research/stream")
async def research_topic_stream(request: DeepResearchRequest):
    """
    Streaming deep research endpoint.
    Returns progress updates and final report via SSE.
    """
    async def generate():
        try:
            depth_map = {
                "quick": ResearchDepth.QUICK,
                "standard": ResearchDepth.STANDARD,
                "deep": ResearchDepth.DEEP
            }
            depth = depth_map.get(request.depth, ResearchDepth.STANDARD)
            
            async for update in deep_research(
                query=request.query,
                depth=depth,
                breadth=request.breadth,
                max_iterations=request.max_iterations,
                include_arxiv=request.include_arxiv,
                language=request.language
            ):
                yield f"data: {json.dumps(update)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/research/quick")
async def quick_research_endpoint(request: DeepResearchRequest):
    """
    Quick research - fast, single-iteration research.
    Good for quick fact-checking or simple queries.
    """
    try:
        result = await quick_research(request.query)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick research error: {str(e)}"
        )


# ============================================================================
# ARXIV ENDPOINTS
# ============================================================================

@app.post("/arxiv/search")
async def search_arxiv_endpoint(request: ArxivSearchRequest):
    """
    Search ArXiv for academic papers.
    
    Returns a list of papers with:
    - Title, authors, abstract
    - Publication date and categories
    - ArXiv URL
    """
    try:
        papers = await search_arxiv_structured(
            query=request.query,
            max_results=request.max_results
        )
        
        return {
            "query": request.query,
            "papers": papers,
            "total": len(papers)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"ArXiv search error: {str(e)}"
        )


# ============================================================================
# SUPERVISOR RESEARCH ENDPOINTS
# ============================================================================

class SupervisorResearchRequest(BaseModel):
    """Request for supervisor-based research."""
    topic: str = Field(..., description="Research topic")
    max_sections: int = Field(default=5, ge=1, le=10, description="Max report sections")
    enable_personas: bool = Field(default=True, description="Use multi-perspective personas")
    num_personas: int = Field(default=3, ge=1, le=5, description="Number of personas")
    enable_review_loop: bool = Field(default=True, description="Enable review/revision cycle")
    max_review_iterations: int = Field(default=3, ge=1, le=5, description="Max review iterations")
    enable_human_feedback: bool = Field(default=False, description="Enable human-in-the-loop")
    export_formats: list[str] = Field(default=["markdown"], description="Export formats")
    language: str = Field(default="vi", description="Output language")


@app.post("/research/supervisor")
async def supervisor_research_endpoint(request: SupervisorResearchRequest):
    """
    Run supervisor-based multi-agent research.
    
    Features:
    - Supervisor-Researcher pattern with parallel research
    - Optional persona-based multi-perspective research
    - Optional review loop with Reviewer → Reviser cycle
    - Multi-format export (markdown, html, pdf, docx)
    - Human-in-the-loop support
    """
    from app.multi_agent_supervisor import (
        run_supervisor_research, 
        SupervisorConfig, 
        ExportFormat
    )
    
    try:
        # Map string export formats to enum
        export_formats = []
        for fmt in request.export_formats:
            try:
                export_formats.append(ExportFormat(fmt.lower()))
            except ValueError:
                export_formats.append(ExportFormat.MARKDOWN)
        
        config = SupervisorConfig(
            max_sections=request.max_sections,
            enable_personas=request.enable_personas,
            num_personas=request.num_personas,
            enable_review_loop=request.enable_review_loop,
            max_review_iterations=request.max_review_iterations,
            enable_human_feedback=request.enable_human_feedback,
            export_formats=export_formats,
            language=request.language
        )
        
        result = await run_supervisor_research(
            topic=request.topic,
            config=config
        )
        
        return {
            "status": "success",
            "topic": request.topic,
            "final_report": result.get("final_report", ""),
            "exports": result.get("exports", {}),
            "personas_used": [p.dict() for p in result.get("personas", [])],
            "sections_count": len(result.get("completed_sections", [])),
            "thinking_log": [t.dict() for t in result.get("thinking_log", [])]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Supervisor research error: {str(e)}"
        )


@app.post("/research/supervisor/stream")
async def supervisor_research_stream(request: SupervisorResearchRequest):
    """
    Streaming supervisor-based research with real-time progress updates.
    """
    from app.multi_agent_supervisor import (
        run_supervisor_research, 
        SupervisorConfig, 
        ExportFormat
    )
    
    async def generate():
        try:
            # Map export formats
            export_formats = []
            for fmt in request.export_formats:
                try:
                    export_formats.append(ExportFormat(fmt.lower()))
                except ValueError:
                    export_formats.append(ExportFormat.MARKDOWN)
            
            config = SupervisorConfig(
                max_sections=request.max_sections,
                enable_personas=request.enable_personas,
                num_personas=request.num_personas,
                enable_review_loop=request.enable_review_loop,
                max_review_iterations=request.max_review_iterations,
                enable_human_feedback=request.enable_human_feedback,
                export_formats=export_formats,
                language=request.language
            )
            
            progress_updates = []
            
            def on_progress(update):
                progress_updates.append(update)
            
            result = await run_supervisor_research(
                topic=request.topic,
                config=config,
                on_progress=on_progress
            )
            
            # Stream progress updates
            for update in progress_updates:
                yield f"data: {json.dumps({'type': 'progress', **update})}\n\n"
            
            # Stream final result
            yield f"data: {json.dumps({'type': 'result', 'report': result.get('final_report', '')})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


# ============================================================================
# PERPLEXITY-STYLE ANSWER ENGINE ENDPOINTS
# ============================================================================

class AnswerRequest(BaseModel):
    """Request for answer engine (Perplexity-style)."""
    query: str = Field(..., description="User's question")
    include_images: bool = Field(default=True, description="Include relevant images")
    include_academic: bool = Field(default=True, description="Include academic sources")
    enable_pro_search: bool = Field(default=False, description="Enable multi-step Pro Search")
    enable_consensus: bool = Field(default=False, description="Enable consensus analysis")
    max_sources: int = Field(default=10, ge=1, le=20, description="Maximum sources to use")
    language: str = Field(default="vi", description="Response language (vi/en)")


class QuickSearchRequest(BaseModel):
    """Request for quick search."""
    query: str = Field(..., description="Search query")
    num_results: int = Field(default=5, ge=1, le=10, description="Number of results")
    language: str = Field(default="vi", description="Response language")


@app.post("/answer")
async def answer_endpoint(request: AnswerRequest):
    """
    Perplexity-style answer engine.
    
    Returns comprehensive answer with:
    - Inline citations [1], [2], etc.
    - Source list with metadata
    - Related follow-up questions
    - Images (optional)
    - Academic consensus analysis (optional)
    
    Example:
    ```
    POST /answer
    {
        "query": "What are the health benefits of intermittent fasting?",
        "include_academic": true,
        "enable_consensus": true
    }
    ```
    """
    from app.tools.perplexity_engine import (
        answer_engine,
        AnswerEngineConfig
    )
    
    try:
        config = AnswerEngineConfig(
            include_images=request.include_images,
            include_academic=request.include_academic,
            enable_pro_search=request.enable_pro_search,
            enable_consensus=request.enable_consensus,
            max_sources=request.max_sources,
            language=request.language
        )
        
        result = await answer_engine(request.query, config)
        
        return {
            "status": "success",
            "answer": result.answer,
            "citations": [c.dict() for c in result.citations],
            "related_questions": [q.dict() for q in result.related_questions],
            "images": result.images,
            "consensus": result.consensus.dict() if result.consensus else None,
            "sources_count": len(result.search_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Answer engine error: {str(e)}"
        )


@app.post("/answer/stream")
async def answer_stream_endpoint(request: AnswerRequest):
    """
    Streaming Perplexity-style answer engine.
    
    Returns SSE events:
    - begin: Query started
    - query-plan: Pro Search steps (if enabled)
    - search-results: Sources found
    - text-chunk: Answer chunks (streamed)
    - citation: Used citations
    - related-questions: Follow-up questions
    - consensus: Academic consensus (if enabled)
    - images: Relevant images
    - done: Complete
    """
    from app.tools.perplexity_engine import (
        answer_engine_stream,
        AnswerEngineConfig
    )
    
    async def generate():
        try:
            config = AnswerEngineConfig(
                include_images=request.include_images,
                include_academic=request.include_academic,
                enable_pro_search=request.enable_pro_search,
                enable_consensus=request.enable_consensus,
                max_sources=request.max_sources,
                language=request.language
            )
            
            async for event in answer_engine_stream(request.query, config):
                yield f"data: {json.dumps(event)}\n\n"
                
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'data': {'error': str(e)}})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.post("/answer/quick")
async def quick_answer_endpoint(request: QuickSearchRequest):
    """
    Quick search for simple queries.
    
    Faster than full answer engine, returns:
    - Brief answer with citations
    - Source list
    - Images
    """
    from app.tools.perplexity_engine import quick_search
    
    try:
        result = await quick_search(
            query=request.query,
            num_results=request.num_results,
            language=request.language
        )
        
        return {
            "status": "success",
            **result
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quick search error: {str(e)}"
        )


@app.post("/answer/pro")
async def pro_search_endpoint(request: AnswerRequest):
    """
    Pro Search with multi-step query planning.
    
    Breaks down complex queries into steps:
    1. Research step A
    2. Research step B (depends on A)
    3. Compare/synthesize results
    
    Best for:
    - Comparison queries ("Compare X vs Y")
    - Multi-faceted research
    - Complex questions requiring multiple angles
    """
    # Force enable pro search
    request.enable_pro_search = True
    
    from app.tools.perplexity_engine import (
        answer_engine,
        AnswerEngineConfig
    )
    
    try:
        config = AnswerEngineConfig(
            include_images=request.include_images,
            include_academic=request.include_academic,
            enable_pro_search=True,
            enable_consensus=request.enable_consensus,
            max_sources=request.max_sources,
            language=request.language
        )
        
        result = await answer_engine(request.query, config)
        
        return {
            "status": "success",
            "answer": result.answer,
            "citations": [c.dict() for c in result.citations],
            "related_questions": [q.dict() for q in result.related_questions],
            "query_plan": result.query_plan.dict() if result.query_plan else None,
            "images": result.images,
            "consensus": result.consensus.dict() if result.consensus else None,
            "sources_count": len(result.search_results)
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pro search error: {str(e)}"
        )


@app.post("/answer/consensus")
async def consensus_search_endpoint(request: AnswerRequest):
    """
    Consensus-style academic search.
    
    Focuses on academic sources and provides:
    - Agreement level (strong_yes, yes, mixed, no, strong_no)
    - Confidence score (0-1)
    - Key findings from studies
    - Study design breakdown (RCT, Meta-analysis, etc.)
    - Evidence quality assessment
    
    Best for:
    - Health/medical questions
    - Scientific research questions
    - Evidence-based inquiries
    """
    # Force enable academic and consensus
    request.include_academic = True
    request.enable_consensus = True
    
    from app.tools.perplexity_engine import (
        answer_engine,
        AnswerEngineConfig
    )
    
    try:
        config = AnswerEngineConfig(
            include_images=False,
            include_academic=True,
            enable_pro_search=False,
            enable_consensus=True,
            max_sources=request.max_sources,
            language=request.language
        )
        
        result = await answer_engine(request.query, config)
        
        return {
            "status": "success",
            "answer": result.answer,
            "consensus": result.consensus.dict() if result.consensus else None,
            "citations": [c.dict() for c in result.citations],
            "academic_sources": [
                s.dict() for s in result.search_results 
                if s.source_type == "academic"
            ],
            "related_questions": [q.dict() for q in result.related_questions]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Consensus search error: {str(e)}"
        )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.server:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug
    )
