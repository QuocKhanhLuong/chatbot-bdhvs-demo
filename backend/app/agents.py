"""
Multi-Agent System for AI Research Assistant.

Inspired by HKUDS/Auto-Deep-Research:
- Triage Agent: Routes queries to appropriate specialist
- Research Agent: Deep web/arxiv research
- Coding Agent: Python code execution and analysis
- Document Agent: Local knowledge base search

Uses LangGraph for agent orchestration.
"""

from typing import TypedDict, Annotated, List, Literal, Optional, Any
from dataclasses import dataclass
import operator

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.tools import (
    get_search_tool, 
    get_python_repl_tool,
    search_arxiv,
    execute_python,
    DocumentRetrieverTool
)


# =============================================================================
# Agent State
# =============================================================================

class AgentState(TypedDict):
    """State shared across all agents."""
    messages: Annotated[List[BaseMessage], operator.add]
    current_agent: str
    task_type: str  # research, coding, document, general
    context: dict
    final_response: Optional[str]


# =============================================================================
# System Prompts
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """Bạn là Triage Agent - agent điều phối của hệ thống AI Research Assistant.

Nhiệm vụ: Phân loại yêu cầu và chuyển đến agent phù hợp.

Các agent có sẵn:
1. **research** - Nghiên cứu web và ArXiv papers
   - Dùng khi: tìm kiếm thông tin, nghiên cứu chủ đề, tìm papers
   - Keywords: "tìm", "nghiên cứu", "papers", "arxiv", "news", "latest"

2. **coding** - Viết và chạy Python code
   - Dùng khi: tính toán, phân tích dữ liệu, viết code
   - Keywords: "tính", "code", "python", "phân tích", "analyze"

3. **document** - Tìm trong knowledge base nội bộ
   - Dùng khi: tìm trong tài liệu đã upload
   - Keywords: "trong tài liệu", "document", "file", "uploaded"

4. **general** - Trả lời trực tiếp không cần tools
   - Dùng khi: câu hỏi đơn giản, chào hỏi, giải thích
   - Keywords: câu hỏi thông thường

Phân tích yêu cầu và trả về tên agent: research, coding, document, hoặc general."""


RESEARCH_SYSTEM_PROMPT = """Bạn là Research Agent - chuyên gia nghiên cứu AI.

Nhiệm vụ:
1. Tìm kiếm web với Tavily để có thông tin mới nhất
2. Tìm papers trên ArXiv cho nghiên cứu học thuật
3. Tổng hợp thông tin và trình bày rõ ràng

Tools có sẵn:
- web_search: Tìm kiếm web
- search_arxiv: Tìm papers trên ArXiv

Luôn cite nguồn và đánh giá độ tin cậy của thông tin.
Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt."""


CODING_SYSTEM_PROMPT = """Bạn là Coding Agent - chuyên gia Python.

Nhiệm vụ:
1. Viết và chạy Python code
2. Phân tích dữ liệu
3. Thực hiện tính toán phức tạp

Tools có sẵn:
- python_repl: Chạy Python code

Guidelines:
- Luôn print() kết quả cần hiển thị
- Xử lý exceptions
- Comment code rõ ràng

Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt."""


DOCUMENT_SYSTEM_PROMPT = """Bạn là Document Agent - chuyên gia tìm kiếm tài liệu.

Nhiệm vụ:
1. Tìm kiếm trong knowledge base nội bộ
2. Trích xuất thông tin từ tài liệu đã upload
3. Tổng hợp và trả lời dựa trên tài liệu

Nếu không tìm thấy thông tin trong tài liệu, hãy thông báo rõ ràng.
Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt."""


GENERAL_SYSTEM_PROMPT = """Bạn là Trợ lý AI cá nhân của Khánh - một AI Research Assistant.

Bạn có thể:
- Trả lời câu hỏi thông thường
- Giải thích khái niệm AI/ML
- Đưa ra lời khuyên về nghiên cứu
- Chat thân thiện

Cá tính: Thông minh, hữu ích, thân thiện nhưng chuyên nghiệp.
Trả lời bằng tiếng Việt nếu người dùng hỏi bằng tiếng Việt."""


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
    
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""
    
    response = await llm.ainvoke([
        SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
        HumanMessage(content=f"Phân loại yêu cầu sau: {last_message}")
    ])
    
    content = response.content.lower() if isinstance(response.content, str) else ""
    
    # Determine task type
    if "research" in content:
        task_type = "research"
    elif "coding" in content or "code" in content:
        task_type = "coding"
    elif "document" in content:
        task_type = "document"
    else:
        task_type = "general"
    
    return {
        "task_type": task_type,
        "current_agent": "triage"
    }


async def research_node(state: AgentState) -> dict:
    """Research agent - web and arxiv search."""
    llm = get_llm()
    
    # Bind tools
    tools = []
    search_tool = get_search_tool()
    if search_tool:
        tools.append(search_tool)
    tools.append(search_arxiv)
    
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    messages = [SystemMessage(content=RESEARCH_SYSTEM_PROMPT)] + state["messages"]
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "research"
    }


async def coding_node(state: AgentState) -> dict:
    """Coding agent - Python execution."""
    llm = get_llm()
    
    tools = [execute_python]
    llm_with_tools = llm.bind_tools(tools)
    
    messages = [SystemMessage(content=CODING_SYSTEM_PROMPT)] + state["messages"]
    
    response = await llm_with_tools.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "coding"
    }


async def document_node(state: AgentState) -> dict:
    """Document agent - local knowledge base search."""
    llm = get_llm()
    
    # Get document retriever
    doc_retriever = DocumentRetrieverTool()
    
    messages = state["messages"]
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
    
    messages = [SystemMessage(content=GENERAL_SYSTEM_PROMPT)] + state["messages"]
    
    response = await llm.ainvoke(messages)
    
    return {
        "messages": [response],
        "current_agent": "general"
    }


async def tool_executor_node(state: AgentState) -> dict:
    """Execute tools called by agents."""
    messages = state["messages"]
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
    
    if task_type == "research":
        return "research"
    elif task_type == "coding":
        return "coding"
    elif task_type == "document":
        return "document"
    else:
        return "general"


def should_use_tools(state: AgentState) -> str:
    """Check if agent wants to use tools."""
    messages = state["messages"]
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

def create_multi_agent_graph():
    """Create the multi-agent workflow graph."""
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("triage", triage_node)
    workflow.add_node("research", research_node)
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
    
    # Direct end for document and general
    workflow.add_edge("document", END)
    workflow.add_edge("general", END)
    
    return workflow.compile(checkpointer=MemorySaver())


# =============================================================================
# Agent Runner
# =============================================================================

class MultiAgentRunner:
    """Runner for multi-agent system."""
    
    def __init__(self):
        self.graph = create_multi_agent_graph()
    
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
            "final_response": None
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
            "final_response": None
        }
        
        async for event in self.graph.astream(initial_state, config):  # type: ignore
            yield event


# Global instance
_runner: Optional[MultiAgentRunner] = None


def get_multi_agent_runner() -> MultiAgentRunner:
    """Get or create multi-agent runner."""
    global _runner
    if _runner is None:
        _runner = MultiAgentRunner()
    return _runner
