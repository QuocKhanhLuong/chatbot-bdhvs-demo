"""
Multi-Agent System for AI Research Assistant.

Inspired by HKUDS/Auto-Deep-Research:
- Triage Agent: Routes queries to appropriate specialist
- Research Agent: Deep web/arxiv research
- Coding Agent: Python code execution and analysis
- Document Agent: Local knowledge base search

Uses LangGraph for agent orchestration.
"""

from typing import TypedDict, Annotated, List, Literal, Optional, Any, Dict
from dataclasses import dataclass
from contextlib import asynccontextmanager
from pathlib import Path
import operator
import os

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from app.config import settings
from app.tools import (
    get_search_tool, 
    get_python_repl_tool,
    search_arxiv,
    execute_python,
    DocumentRetrieverTool,
    deep_research_tool,
    deep_research,
    write_final_report,
    # Deep Research V2 (enhanced)
    deep_research_v2,
    deep_research_stream_v2,
    write_final_report_v2,
    ResearchConfig,
    ResearchStage,
)


# =============================================================================
# SQLite Checkpointer for Persistent Storage
# =============================================================================

# Path to SQLite database for chat history persistence
DB_PATH = Path(__file__).parent.parent / "data" / "chat_history.db"


@asynccontextmanager
async def get_checkpointer():
    """Get async SQLite checkpointer for persistent chat history.
    
    Yields:
        AsyncSqliteSaver instance for use with LangGraph
    """
    # Ensure data directory exists
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    async with AsyncSqliteSaver.from_conn_string(str(DB_PATH)) as saver:
        yield saver


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict, total=False):
    """State shared across all agents."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    task_type: str  # research, coding, document, general, deep_research
    context: dict
    final_response: Optional[str]
    status_updates: List[Dict[str, Any]]  # For streaming status events


# =============================================================================
# System Prompts
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """You are the **Lab Coordinator (Triage Agent)** of an advanced AI Research facility.

**Your Goal:** Analyze the user's request and route it to the most capable specialist. Do not attempt to answer yourself.

**Routing Logic:**

1.  **🔍 `deep_research` (The Surveyor):**
    * *Trigger:* Requests for comprehensive reports, "state of the art" surveys, comparisons of multiple technologies, or topics requiring iterative searching.
    * *Keywords:* "investigate", "deep dive", "comprehensive report", "compare X and Y", "history of...", "future trends".

2.  **🌐 `research` (The Librarian):**
    * *Trigger:* Quick fact-checks, looking up specific papers (ArXiv), finding latest news, or simple "What is X?" questions.
    * *Keywords:* "find paper", "news", "release date", "who created", "quick search".

3.  **🐍 `coding` (The Engineer):**
    * *Trigger:* Math calculations, data visualization, writing/debugging code, or verifying a hypothesis through simulation.
    * *Keywords:* "plot", "calculate", "script", "debug", "analyze dataset", "verify math".

4.  **📂 `document` (The Archivist):**
    * *Trigger:* Questions explicitly about uploaded files/PDFs.
    * *Keywords:* "summary of this pdf", "in the file", "what does the document say".

5.  **🧠 `general` (The Mentor):**
    * *Trigger:* Conceptual explanations, brainstorming, advice on learning paths, or casual chat.
    * *Keywords:* "explain concept", "give advice", "hello", "help me understand".

**Output:** Return ONLY the agent name: `research`, `coding`, `document`, `deep_research`, or `general`."""


RESEARCH_SYSTEM_PROMPT = """You are the **Literature Review Specialist (Research Agent)**.

**Role:** You are responsible for gathering verifiable facts and academic sources. You do not guess.

**Tool Usage Protocols:**
1.  **`web_search`:** Use for "Novelty Checking" (Is this idea new?) and "Fact Checking" (Is this claim true?).
2.  **`search_arxiv`:** Use immediately if the user mentions "paper", "algorithm", or "model architecture".
3.  **`deep_research_tool`:** DELEGATE to this tool if the query is too broad for a single search (e.g., "Impact of AI on Healthcare").

**Output Standards:**
* **Cite Everything:** Every claim must have a source link.
* **SOTA Awareness:** When discussing AI, always mention the current State-of-the-Art (e.g., "Currently, DeepSeek-V3 and GPT-4o are leading benchmarks...").
* **Structure:** Use Markdown headers. Separate "Academic Sources" from "Industry News"."""


CODING_SYSTEM_PROMPT = """You are the **Lead Data Scientist (Coding Agent)**.

**Role:** You prove truths through execution. You do not just write code; you RUN it to verify results.

**Rigorous Protocols:**
1.  **Execution is Mandatory:** Never write code without running it via `python_repl` to check for errors.
2.  **Visual Proof:** If analyzing data, ALWAYS generate a plot.
    * *Save Path:* `plt.savefig('static/images/filename.png')`
    * *Display:* Return the markdown image syntax: `![Description](/static/images/filename.png)`
3.  **Math Verification:** If the user asks a math question, solve it numerically in Python to double-check their (or your) intuition.
4.  **Self-Correction:** If code fails, analyze the traceback, explain the error to the user, and fix it automatically.

**Tone:** Precise, technical, and results-oriented."""


DOCUMENT_SYSTEM_PROMPT = """You are the **Evidence Analyst (Document Agent)**.

**Role:** You extract ground truth from the user's provided Knowledge Base.

**Protocols:**
1.  **No Hallucinations:** If the answer is not in the documents, say "The provided documents do not contain this information." Do not make it up.
2.  **Citation:** Quote specific sections or page numbers (e.g., "According to the Methodology section (p.4)...").
3.  **Synthesis:** If multiple documents are found, synthesize a coherent answer connecting them, rather than listing them separately."""


GENERAL_SYSTEM_PROMPT = """You are **Dr. AI (The Mentor)**.

**Role:** You are the interface for high-level guidance, brainstorming, and conceptual understanding. You are the "Companion" side of the system.

**Teaching Style:**
* **Socratic:** Ask questions to help the user refine their thinking.
* **First Principles:** Explain *why* things work, not just *how*. (e.g., "Attention works because it creates a content-based addressing system...").
* **Roadmaps:** Provide step-by-step learning paths when asked for advice.
* **Voice:** Encouraging but rigorous. Challenge the user to think deeper.

**Note:** If the user asks for specific external facts, code execution, or file analysis, strictly advise them to ask specifically for those tasks so the Triage agent can route them correctly."""


# =============================================================================
# LLM Setup
# =============================================================================

def get_llm(temperature: float = 0.7):
    """Get LLM based on configuration."""
    google_key = settings.effective_google_api_key
    
    # MegaLLM (OpenAI-compatible API)
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
        raise ValueError("No LLM API key configured. Set MEGALLM_API_KEY, OPENAI_API_KEY or GEMINI_API_KEY.")


# =============================================================================
# Agent Nodes
# =============================================================================

async def triage_node(state: AgentState) -> dict:
    """Triage agent - determines which agent to route to."""
    llm = get_llm(temperature=0.1)
    
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    response = await llm.ainvoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Phân loại yêu cầu sau: {last_message}")
    ])
    
    content = response.content.lower() if isinstance(response.content, str) else ""
    
    # Determine task type - check deep_research first (more specific)
    if "deep_research" in content or "deep research" in content:
        task_type = "deep_research"
    elif "research" in content:
        task_type = "research"
    elif "coding" in content or "code" in content:
        task_type = "coding"
    elif "document" in content:
        task_type = "document"
    else:
        task_type = "general"
    
    return {
        "task_type": task_type,
        "current_agent": "triage",
        "status_updates": [{"type": "status", "stage": "triage", "message": f"Routing to {task_type} agent..."}]
    }


async def research_node(state: AgentState) -> dict:
    """Research agent - web and arxiv search with deep_research option."""
    llm = get_llm()
    
    # Bind tools including deep_research_tool
    tools = []
    search_tool = get_search_tool()
    if search_tool:
        tools.append(search_tool)
    tools.append(search_arxiv)
    tools.append(deep_research_tool)  # Add deep research capability
    
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + state.get("messages", [])
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "research",
        "status_updates": [{"type": "status", "stage": "research", "message": "Researching..."}]
    }


async def coding_node(state: AgentState) -> dict:
    """Coding agent - Python execution."""
    llm = get_llm()
    
    tools = [execute_python]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [SystemMessage(content=CODING_SYSTEM_PROMPT)] + state.get("messages", [])
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "coding",
        "status_updates": [{"type": "status", "stage": "coding", "message": "Executing code..."}]
    }


async def document_node(state: AgentState) -> dict:
    """Document agent - local knowledge base search."""
    llm = get_llm()
    
    # Get document retriever
    doc_retriever = DocumentRetrieverTool()
    
    messages = state.get("messages", [])
    last_message_content = messages[-1].content if messages else ""
    
    # Ensure it's a string
    if isinstance(last_message_content, list):
        last_message_content = " ".join(str(c) for c in last_message_content)
    
    # Search documents
    docs = doc_retriever.search(str(last_message_content), k=3)
    
    context = ""
    if docs:
        context = "\n\n".join([
            f"[Tài liệu {i+1}] {doc.metadata.get('source', 'Unknown')}:\n{doc.page_content[:500]}"
            for i, doc in enumerate(docs)
        ])
    
    prompt = f"""Dựa trên tài liệu tìm được:

{context if context else "Không tìm thấy tài liệu liên quan."}

Hãy trả lời câu hỏi: {last_message_content}"""
    
    all_messages = [SystemMessage(content=DOCUMENT_SYSTEM_PROMPT)] + messages[:-1] + [HumanMessage(content=prompt)]
    
    response = await llm.ainvoke(all_messages)
    
    return {
        "messages": [response],
        "current_agent": "document"
    }


async def general_node(state: AgentState) -> dict:
    """General agent - direct response without tools."""
    llm = get_llm()
    
    messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)] + state.get("messages", [])
    
    response = await llm.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "general"
    }


async def deep_research_node(state: AgentState) -> dict:
    """Deep Research agent - enhanced recursive multi-iteration research.
    
    Uses deep_research_v2 with:
    - Pydantic structured output
    - Concurrent processing with semaphore
    - Learnings accumulation
    - ArXiv paper integration
    """
    messages = state.get("messages", [])
    last_message = messages[-1].content if messages else ""
    
    # Ensure it's a string
    if isinstance(last_message, list):
        last_message = " ".join(str(c) for c in last_message)
    
    status_updates = []
    
    # Progress callback for v2
    def on_progress(progress):
        status_updates.append({
            "type": "status",
            "stage": progress.stage.value if hasattr(progress.stage, 'value') else str(progress.stage),
            "message": progress.message,
            "depth": progress.current_depth,
            "total_depth": progress.total_depth,
            "breadth": progress.current_breadth,
            "total_breadth": progress.total_breadth,
            "current_query": progress.current_query,
            "progress_percent": progress.progress_percent,
            "learnings_count": progress.learnings_count,
            "sources_count": progress.sources_count,
            "completed_queries": progress.completed_queries,
            "total_queries": progress.total_queries
        })
    
    # Configure v2 research
    config = ResearchConfig(
        breadth=4,
        depth=2,
        concurrency_limit=2,
        include_arxiv=True,
        language="vi",
        max_results_per_search=5
    )
    
    # Run deep research v2
    try:
        result = await deep_research_v2(
            query=str(last_message),
            config=config,
            on_progress=on_progress
        )
        
        # Generate report using v2
        report = await write_final_report_v2(
            prompt=str(last_message),
            learnings=result.learnings,
            visited_urls=result.visited_urls,
            language=config.language
        )
        
        # Build metadata for frontend
        metadata_section = f"""
## Research Statistics
- **Total Sources**: {len(result.visited_urls)}
- **Total Learnings**: {len(result.learnings)}
- **Search Iterations**: {result.total_searches}
- **Max Depth Reached**: {result.max_depth_reached}

"""
        
        # Wrap report in artifact tags for frontend display
        artifact_content = f"""Here is the research report:

---REPORT START---
{metadata_section}{report}

### Follow-up Questions
{chr(10).join(f"- {q}" for q in result.follow_up_questions[:5]) if result.follow_up_questions else "- No follow-up questions generated"}

### Sources
{chr(10).join(f"- {url}" for url in result.visited_urls[:20]) if result.visited_urls else "- No sources found"}
---REPORT END---
"""
        
        response_msg = AIMessage(content=artifact_content)
        
        return {
            "messages": [response_msg],
            "current_agent": "deep_research",
            "status_updates": status_updates
        }
        
    except Exception as e:
        error_msg = f"Deep research error: {str(e)}"
        return {
            "messages": [AIMessage(content=error_msg)],
            "current_agent": "deep_research",
            "status_updates": [{"type": "error", "message": error_msg}]
        }


async def tool_executor_node(state: AgentState) -> dict:
    """Execute tools called by agents."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": []}
    
    last_message = messages[-1]
    
    # Check if AIMessage has tool_calls
    if not isinstance(last_message, AIMessage):
        return {"messages": []}
    
    if not last_message.tool_calls:  # type: ignore
        return {"messages": []}
    
    # Build tool node
    tools = [execute_python, search_arxiv]
    search_tool = get_search_tool()
    if search_tool:
        tools.append(search_tool)
    
    tool_node = ToolNode(tools=tools)
    
    result = await tool_node.ainvoke(state)
    
    return result


# =============================================================================
# Router Functions
# =============================================================================

def route_after_triage(state: AgentState) -> str:
    """Route to appropriate agent after triage."""
    task_type = state.get("task_type", "general")
    
    if task_type == "deep_research":
        return "deep_research"
    elif task_type == "research":
        return "research"
    elif task_type == "coding":
        return "coding"
    elif task_type == "document":
        return "document"
    else:
        return "general"


def should_use_tools(state: AgentState) -> str:
    """Check if agent wants to use tools."""
    messages = state.get("messages", [])
    if not messages:
        return END
    
    last_message = messages[-1]
    
    # Check if it's an AIMessage with tool_calls
    if isinstance(last_message, AIMessage) and last_message.tool_calls:  # type: ignore
        return "tools"
    
    return END


# =============================================================================
# Build Multi-Agent Graph
# =============================================================================

def create_multi_agent_graph(checkpointer=None):
    """Create the multi-agent workflow graph.
    
    Args:
        checkpointer: Optional checkpointer for persistent storage.
                      If None, uses in-memory MemorySaver.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("research", research_node)
    workflow.add_node("deep_research", deep_research_node)
    workflow.add_node("coding", coding_node)
    workflow.add_node("document", document_node)
    workflow.add_node("general", general_node)
    workflow.add_node("tools", tool_executor_node)
    
    # Set entry point
    workflow.set_entry_point("triage")
    
    # Add conditional edges from triage
    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "deep_research": "deep_research",
            "research": "research",
            "coding": "coding",
            "document": "document",
            "general": "general"
        }
    )
    
    # Add tool execution edges
    for agent in ["research", "coding"]:
        workflow.add_conditional_edges(
            agent,
            should_use_tools,
            {
                "tools": "tools",
                END: END
            }
        )
    
    # Tools can loop back to calling agent
    workflow.add_conditional_edges(
        "tools",
        lambda s: s.get("current_agent", "general"),
        {
            "research": "research",
            "coding": "coding",
            "general": END  # fallback
        }
    )
    
    # Direct end for document, general, and deep_research
    workflow.add_edge("document", END)
    workflow.add_edge("general", END)
    workflow.add_edge("deep_research", END)
    
    # Use provided checkpointer or fallback to MemorySaver
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return workflow.compile(checkpointer=checkpointer)


# =============================================================================
# Agent Runner
# =============================================================================

class MultiAgentRunner:
    """Runner for multi-agent system with persistent storage support."""
    
    def __init__(self, checkpointer=None):
        """Initialize runner with optional checkpointer.
        
        Args:
            checkpointer: Optional checkpointer for persistent storage.
                          If None, uses in-memory MemorySaver.
        """
        self.graph = create_multi_agent_graph(checkpointer=checkpointer)
        self._checkpointer = checkpointer
    
    async def run(
        self,
        message: str,
        thread_id: str = "default"
    ) -> dict:
        """Run the multi-agent system."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "current_agent": "",
            "task_type": "",
            "context": {},
            "final_response": None,
            "status_updates": []
        }
        
        result = await self.graph.ainvoke(initial_state, config)  # type: ignore
        
        # Extract final response
        messages = result.get("messages", [])
        final_response = ""
        
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                final_response = msg.content
                break
        
        return {
            "response": final_response,
            "agent_used": result.get("current_agent", "unknown"),
            "task_type": result.get("task_type", "unknown"),
            "thread_id": thread_id
        }
    
    async def stream(
        self,
        message: str,
        thread_id: str = "default"
    ):
        """Stream responses from multi-agent system."""
        config = {"configurable": {"thread_id": thread_id}}
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=message)],
            "current_agent": "",
            "task_type": "",
            "context": {},
            "final_response": None,
            "status_updates": []
        }
        
        async for event in self.graph.astream(initial_state, config):  # type: ignore
            yield event


# Global instance
_runner: Optional[MultiAgentRunner] = None


def get_multi_agent_runner() -> MultiAgentRunner:
    """Get or create multi-agent runner (uses in-memory storage).
    
    For persistent storage, use get_persistent_runner() instead.
    """
    global _runner
    if _runner is None:
        _runner = MultiAgentRunner()
    return _runner


@asynccontextmanager
async def get_persistent_runner():
    """Get multi-agent runner with persistent SQLite storage.
    
    This is an async context manager that should be used like:
        async with get_persistent_runner() as runner:
            result = await runner.run(message, thread_id)
    
    Yields:
        MultiAgentRunner with SQLite-backed checkpointer
    """
    async with get_checkpointer() as checkpointer:
        yield MultiAgentRunner(checkpointer=checkpointer)
