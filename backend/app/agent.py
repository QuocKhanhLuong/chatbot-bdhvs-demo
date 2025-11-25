"""
LangGraph Agent for Personal AI Assistant
Research Assistant with Web Search, Python REPL, and Document Retrieval capabilities.
"""
import os
from typing import Annotated, Sequence, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from app.config import settings
from app.tools import get_all_tools


# ============================================================================
# SYSTEM PROMPT - The "Soul" of the Assistant
# ============================================================================

SYSTEM_PROMPT = """Bạn là Trợ lý Ảo cá nhân của Khánh - một nhà nghiên cứu AI.

🎭 **Phong cách:**
- Thông minh và sắc bén
- Hài hước (nhưng vẫn chuyên nghiệp)  
- Thực dụng - đi thẳng vào vấn đề

🛠️ **Khả năng của bạn:**

1. **Tìm kiếm Web (web_search):** 
   - Khi tôi hỏi về tin tức, nghiên cứu mới, hoặc thông tin cập nhật → Dùng công cụ tìm kiếm
   - ĐỪNG chỉ dựa vào kiến thức cũ khi cần thông tin mới nhất

2. **Chạy Python Code (python_repl):**
   - Khi tôi nhờ tính toán, xử lý dữ liệu, hoặc viết code → Chạy code thật
   - Luôn print() kết quả để tôi thấy output

3. **Tra cứu tài liệu (search_documents):**
   - Khi tôi hỏi về nội dung trong các file PDF đã upload → Tìm trong knowledge base

📋 **Nguyên tắc:**
- Trả lời ngắn gọn, súc tích
- Nếu không chắc → Hỏi lại hoặc tìm kiếm
- Khi dùng tool, giải thích ngắn gọn tại sao
- Dùng emoji phù hợp để response sinh động hơn

Sẵn sàng hỗ trợ! 🚀"""


# ============================================================================
# STATE DEFINITION
# ============================================================================

class AgentState(TypedDict):
    """State schema for the agent."""
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def get_llm():
    """Initialize the LLM based on configuration."""
    google_key = settings.effective_google_api_key
    
    # MegaLLM (OpenAI-compatible API)
    if settings.llm_provider == "megallm" and settings.megallm_api_key:
        return ChatOpenAI(
            model=settings.megallm_model,
            temperature=0.7,
            streaming=True,
            api_key=settings.megallm_api_key,  # type: ignore
            base_url=settings.megallm_base_url
        )
    # Google/Gemini
    elif settings.llm_provider == "google" and google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        return ChatGoogleGenerativeAI(
            model=settings.google_model,
            temperature=0.7,
            convert_system_message_to_human=True
        )
    # OpenAI
    elif settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=0.7,
            streaming=True
        )
    else:
        raise ValueError(
            "No LLM API key configured. Please set MEGALLM_API_KEY, OPENAI_API_KEY, or GEMINI_API_KEY in .env"
        )


# ============================================================================
# GRAPH NODES
# ============================================================================

def create_agent_node(llm_with_tools):
    """Create the agent node that decides what to do."""
    
    def agent_node(state: AgentState) -> dict:
        """
        The agent node: processes messages and decides whether to use tools.
        """
        messages = state["messages"]
        
        # Ensure system prompt is at the beginning
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
        
        # Get response from LLM
        response = llm_with_tools.invoke(messages)
        
        return {"messages": [response]}
    
    return agent_node


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Conditional edge: decide whether to continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If the LLM made a tool call, route to tools node
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # Otherwise, end the conversation turn
    return "end"


# ============================================================================
# GRAPH CONSTRUCTION
# ============================================================================

def create_agent_graph():
    """
    Create the LangGraph workflow for the Personal AI Assistant.
    
    Workflow:
    1. Agent node: LLM processes input and decides action
    2. If tool call needed → Tools node executes the tool
    3. Loop back to Agent to process tool result
    4. If no tool call → End
    """
    # Initialize components
    tools = get_all_tools()
    llm = get_llm()
    llm_with_tools = llm.bind_tools(tools)
    
    print(f"🔧 Loaded {len(tools)} tools: {[t.name for t in tools]}")
    
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", create_agent_node(llm_with_tools))
    workflow.add_node("tools", ToolNode(tools))
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")
    
    # Compile with memory checkpointer for conversation persistence
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================================
# AGENT INSTANCE
# ============================================================================

# Create the agent graph (singleton)
_agent_graph = None


def get_agent():
    """Get or create the agent graph instance."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
    return _agent_graph


def reset_agent():
    """Reset the agent (useful for testing or reconfiguration)."""
    global _agent_graph
    _agent_graph = None
