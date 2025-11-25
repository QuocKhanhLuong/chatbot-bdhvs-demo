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
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from app.config import settings
from app.agent import get_agent, reset_agent, SYSTEM_PROMPT
from app.agents import get_multi_agent_runner
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
    Multi-agent chat endpoint.
    
    Automatically routes to the appropriate agent:
    - Research Agent: Web search, ArXiv papers
    - Coding Agent: Python code execution
    - Document Agent: Local knowledge base
    - General Agent: Direct responses
    """
    runner = get_multi_agent_runner()
    thread_id = request.thread_id or str(uuid.uuid4())
    
    try:
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
    Streaming multi-agent chat endpoint.
    Returns Server-Sent Events (SSE) stream.
    """
    runner = get_multi_agent_runner()
    thread_id = request.thread_id or str(uuid.uuid4())
    
    async def generate():
        try:
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
