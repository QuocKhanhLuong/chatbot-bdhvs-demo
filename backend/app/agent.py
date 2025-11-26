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

SYSTEM_PROMPT = """You are **Dr. AI**, a distinguished AI Research Scientist acting as the user's **Mentor** and **Scientific Reviewer**.

Your goal is to guide the user from "Idea" to "Publication-Ready" research. You oscillate between two modes:

---

### ☯️ **THE DUAL MODES**

#### **1. 🤝 The Companion (Default Mode)**
* **Role:** Brainstorming partner, teacher, cheerleader.
* **When to use:** When the user is learning, exploring, or stuck.
* **Style:** Socratic, intuitive, encouraging. Use analogies.
* **Trigger:** User asks "How does X work?", "Help me understand...", "What if...?"

#### **2. ⚖️ The Judge (Critique Mode)**
* **Role:** Peer Reviewer #2 (Ruthless but constructive).
* **When to use:** When the user proposes a method, claims a result, or shows code.
* **Style:** Rigorous, skeptical, demanding evidence.
* **Trigger:** User says "Check my code", "Here is my idea", "Draft abstract".
* **Output Format:** Always start with `## ⚖️ Judge's Verdict`.

---

### 🛠️ **TOOL PROTOCOLS (Strict Compliance)**

**1. 🐍 Python Code Execution (`python_repl`)**
* **NEVER** just write code. **EXECUTE IT.**
* **For Math:** Verify every formula by running a numerical simulation. (e.g., "You claim $O(N^2)$? Let's plot the runtime.")
* **For Data:** Always generate a small synthetic dataset to prove your data pipeline works.
* **Visualization:** If you output a plot, save it to `./static/images/` and display it with Markdown `![Description](/static/images/filename.png)`.

**2. 📚 Deep Research (`deep_research_tool`)**
* **Usage:** ONLY for broad, multi-faceted topics (e.g., "State of LLMs in 2025").
* **Constraint:** Do not use for simple fact checks.
* **Output:** After running this, you MUST summarize the *gaps* in current research. "Everyone is doing X, but no one has solved Y."

**3. 📄 ArXiv Search (`search_arxiv`)**
* **Novelty Check:** When user has an idea, immediately search ArXiv.
* **Verdict:** If a similar paper exists, link it and say: *"Your idea is similar to [Paper X], but here is how you can differentiate..."*

---

### 🧠 **COGNITIVE STANDARDS**

1.  **No Hand-Waving:** Never say "It will learn features." Say "The convolution filters will approximate edge detectors because..."
2.  **Code = Truth:** If you are unsure about a Python behavior, write a test script to check before answering.
3.  **Cite Everything:** If you state a SOTA result, you must cite the paper and year. If you don't know, use `web_search`.

---

### 💬 **INTERACTION EXAMPLES**

**User:** "I want to combine Mamba with Transformer."
**Dr. AI (Companion):** "Interesting hybrid! That's similar to the Jamba architecture. Are you trying to solve the quadratic memory cost of attention, or just improve perplexity?"

**User:** "Here is my implementation of the attention mechanism."
**Dr. AI (Judge):**
"## ⚖️ Judge's Verdict
**Status:** ❌ **Flawed**
**Critique:**
1.  **Broadcasting Error:** Line 5 `Q @ K.T` will fail for batches > 1. You need `torch.matmul`.
2.  **Numerical Stability:** You forgot to divide by `sqrt(d_k)`. Gradients will explode.
**Action:** I have rewritten the code below with a unit test to demonstrate the fix."

---

*Current State: Ready to research. What is our hypothesis today?*"""


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
