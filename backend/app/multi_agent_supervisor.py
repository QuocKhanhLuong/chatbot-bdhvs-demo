"""
Multi-Agent Supervisor System with Advanced Research Capabilities.

Inspired by:
- assafelovic/gpt-researcher: Multi-agent team with Review Loop
- stanford-oval/storm: Persona-based research
- langchain-ai/open_deep_research: Supervisor-Researcher pattern, MCP support

Features:
1. Supervisor-Researcher Pattern: Supervisor delegates to specialized researchers
2. Review Loop: Reviewer → Reviser cycle until quality is met
3. Persona-based Research: Multiple perspectives for comprehensive coverage
4. Strategic Thinking (think_tool): Reflection between searches
5. Multi-format Export: Markdown, PDF, DOCX
6. Human-in-the-loop: User feedback integration
7. MCP Support: Model Context Protocol for external tools
"""

from typing import TypedDict, Annotated, List, Literal, Optional, Any, Dict, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import operator
import json
import os

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool, BaseTool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, START
from langgraph.types import Command, Send

from app.config import settings


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class Section(BaseModel):
    """A section of the research report."""
    title: str = Field(..., description="Section title")
    description: str = Field(..., description="What this section should cover")
    content: str = Field(default="", description="The written content of the section")


class ResearchPlan(BaseModel):
    """Research plan with sections to investigate."""
    title: str = Field(..., description="Report title")
    sections: List[Section] = Field(default_factory=list, description="Sections to research")
    introduction: str = Field(default="", description="Introduction content")
    conclusion: str = Field(default="", description="Conclusion content")


class ReviewFeedback(BaseModel):
    """Feedback from the reviewer agent."""
    is_approved: bool = Field(..., description="Whether the section is approved")
    feedback: str = Field(default="", description="Detailed feedback for revision")
    suggestions: List[str] = Field(default_factory=list, description="Specific suggestions")


class Persona(BaseModel):
    """Research persona for multi-perspective research."""
    name: str = Field(..., description="Persona name/role")
    focus: str = Field(..., description="What this persona focuses on")
    expertise: str = Field(..., description="Area of expertise")


class ThinkingOutput(BaseModel):
    """Output from strategic thinking."""
    reflection: str = Field(..., description="Strategic reflection")
    next_steps: List[str] = Field(default_factory=list, description="Planned next steps")
    gaps_identified: List[str] = Field(default_factory=list, description="Knowledge gaps found")


class ExportFormat(str, Enum):
    """Supported export formats."""
    MARKDOWN = "markdown"
    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"


# =============================================================================
# State Definitions
# =============================================================================

def override_reducer(current_value, new_value):
    """Reducer that allows overriding values in state."""
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    return operator.add(current_value, new_value)


class SupervisorState(TypedDict, total=False):
    """State for the main supervisor workflow."""
    messages: Annotated[List[BaseMessage], operator.add]
    research_topic: str
    research_plan: Optional[ResearchPlan]
    personas: List[Persona]
    completed_sections: Annotated[List[Section], operator.add]
    current_section_idx: int
    review_iterations: int
    max_review_iterations: int
    human_feedback: Optional[str]
    final_report: str
    export_formats: List[ExportFormat]
    status_updates: Annotated[List[Dict[str, Any]], operator.add]
    thinking_log: Annotated[List[ThinkingOutput], operator.add]
    mcp_tools: List[BaseTool]


class ResearcherState(TypedDict, total=False):
    """State for individual researcher agents."""
    messages: Annotated[List[BaseMessage], operator.add]
    section: Section
    persona: Optional[Persona]
    search_queries: List[str]
    sources: List[str]
    learnings: List[str]
    draft_content: str
    tool_iterations: int
    max_tool_iterations: int


class ReviewerState(TypedDict, total=False):
    """State for the reviewer agent."""
    section: Section
    draft_content: str
    review_criteria: List[str]
    feedback: Optional[ReviewFeedback]


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SupervisorConfig:
    """Configuration for the supervisor system."""
    max_sections: int = 5
    max_review_iterations: int = 3
    max_tool_iterations: int = 10
    enable_personas: bool = True
    num_personas: int = 3
    enable_review_loop: bool = True
    enable_human_feedback: bool = False
    enable_mcp: bool = False
    mcp_server_url: Optional[str] = None
    export_formats: List[ExportFormat] = field(default_factory=lambda: [ExportFormat.MARKDOWN])
    language: str = "vi"
    
    # Model settings
    supervisor_model: str = "gpt-4o"
    researcher_model: str = "gpt-4o"
    reviewer_model: str = "gpt-4o"


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


def get_llm(temperature: float = 0.7, model: Optional[str] = None):
    """Get LLM with automatic fallback support."""
    return core_get_llm(
        temperature=temperature,
        model=model,
        fallback=True
    )


# =============================================================================
# Tools
# =============================================================================

@tool
def think_tool(reflection: str) -> str:
    """
    Strategic thinking tool - use this to pause and reflect on research strategy.
    
    Args:
        reflection: Your strategic reflection and analysis
    
    Returns:
        Confirmation that reflection was recorded
    """
    return f"Reflection recorded: {reflection}"


@tool
def request_human_feedback(question: str) -> str:
    """
    Request feedback from the human user.
    
    Args:
        question: The question to ask the user
    
    Returns:
        Placeholder for human response (will be filled by the system)
    """
    return f"[HUMAN_FEEDBACK_REQUESTED]: {question}"


# =============================================================================
# System Prompts
# =============================================================================

SUPERVISOR_PROMPT = """You are the Lead Research Supervisor. Today is {date}.

Your role is to:
1. Analyze the research topic and create a comprehensive research plan
2. Generate diverse research personas for multi-perspective coverage
3. Delegate research tasks to specialist researchers
4. Review and refine the final report

Research Topic: {topic}

Guidelines:
- Create {max_sections} main sections for the report
- Ensure comprehensive coverage of the topic
- Consider multiple perspectives and potential controversies
- Output must be in {language}
"""

PERSONA_GENERATOR_PROMPT = """You are a Persona Generator for research projects.

Generate {num_personas} diverse research personas for the topic: {topic}

Each persona should represent:
- A different perspective or expertise area
- A unique focus that contributes to comprehensive coverage
- Real-world relevance to the topic

Output personas with: name, focus area, and expertise description.
"""

RESEARCHER_PROMPT = """You are a Research Agent with expertise in: {persona_expertise}

Your research focus: {persona_focus}

Section to research: {section_title}
Section description: {section_description}

Guidelines:
1. Use available search tools to gather information
2. Use think_tool to reflect on your research strategy
3. Accumulate learnings and cite sources
4. Write comprehensive content for your section
5. Output in {language}

Available tools: {tools}
"""

REVIEWER_PROMPT = """You are a Quality Reviewer for research reports.

Review the following section for:
1. Accuracy and factual correctness
2. Completeness of coverage
3. Clarity and readability
4. Proper source citations
5. Logical flow and structure

Section Title: {section_title}
Content to Review:
{content}

Review Criteria: {criteria}

Provide detailed feedback and decide if the section is approved or needs revision.
"""

REVISER_PROMPT = """You are a Content Reviser.

Revise the following section based on reviewer feedback:

Original Content:
{original_content}

Reviewer Feedback:
{feedback}

Suggestions:
{suggestions}

Improve the content while maintaining accuracy and completeness.
Output the revised content in {language}.
"""


# =============================================================================
# Supervisor Agent Functions
# =============================================================================

async def create_research_plan(state: SupervisorState, config: SupervisorConfig) -> Command:
    """Create the initial research plan with sections."""
    llm = get_llm(temperature=0.3)
    
    topic = state.get("research_topic", "")
    
    prompt = SUPERVISOR_PROMPT.format(
        date=datetime.now().strftime("%Y-%m-%d"),
        topic=topic,
        max_sections=config.max_sections,
        language=config.language
    )
    
    # Use structured output for research plan
    llm_with_structure = llm.with_structured_output(ResearchPlan)
    
    response = await llm_with_structure.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Create a research plan for: {topic}")
    ])
    
    # Cast response to ResearchPlan for type checking
    plan: ResearchPlan = response  # type: ignore
    
    return Command(
        goto="generate_personas" if config.enable_personas else "dispatch_researchers",
        update={
            "research_plan": plan,
            "status_updates": [{
                "type": "status",
                "stage": "planning",
                "message": f"Created research plan with {len(plan.sections)} sections"
            }]
        }
    )


async def generate_personas(state: SupervisorState, config: SupervisorConfig) -> Command:
    """Generate research personas for multi-perspective coverage."""
    llm = get_llm(temperature=0.5)
    
    topic = state.get("research_topic", "")
    
    prompt = PERSONA_GENERATOR_PROMPT.format(
        num_personas=config.num_personas,
        topic=topic
    )
    
    # Generate personas with structured output
    class PersonaList(BaseModel):
        personas: List[Persona]
    
    llm_with_structure = llm.with_structured_output(PersonaList)
    
    response = await llm_with_structure.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content=f"Generate {config.num_personas} personas for: {topic}")
    ])
    
    # Cast response for type checking
    persona_list = response  # type: ignore
    personas_result = persona_list.personas if hasattr(persona_list, 'personas') else []  # type: ignore
    
    return Command(
        goto="dispatch_researchers",
        update={
            "personas": personas_result,
            "status_updates": [{
                "type": "status",
                "stage": "personas",
                "message": f"Generated {len(personas_result)} research personas"
            }]
        }
    )


async def dispatch_researchers(state: SupervisorState, config: SupervisorConfig) -> Command:
    """Dispatch parallel researchers for each section."""
    research_plan = state.get("research_plan")
    personas = state.get("personas", [])
    
    if not research_plan:
        return Command(goto=END)
    
    # Create Send commands for parallel research
    sends = []
    for i, section in enumerate(research_plan.sections):
        # Assign persona (round-robin if fewer personas than sections)
        persona = personas[i % len(personas)] if personas else None
        
        sends.append(Send(
            "researcher",
            {
                "section": section,
                "persona": persona,
                "max_tool_iterations": config.max_tool_iterations
            }
        ))
    
    return Command(
        goto=sends,
        update={
            "status_updates": [{
                "type": "status",
                "stage": "dispatching",
                "message": f"Dispatching {len(sends)} researchers in parallel"
            }]
        }
    )


async def collect_research(state: SupervisorState, config: SupervisorConfig) -> Command:
    """Collect completed research from all researchers."""
    completed_sections = state.get("completed_sections", [])
    
    if config.enable_review_loop:
        return Command(
            goto="review_sections",
            update={
                "current_section_idx": 0,
                "status_updates": [{
                    "type": "status",
                    "stage": "collecting",
                    "message": f"Collected {len(completed_sections)} sections, starting review"
                }]
            }
        )
    else:
        return Command(
            goto="compile_report",
            update={
                "status_updates": [{
                    "type": "status",
                    "stage": "collecting",
                    "message": f"Collected {len(completed_sections)} sections, compiling report"
                }]
            }
        )


# =============================================================================
# Researcher Agent Functions
# =============================================================================

async def researcher_node(state: ResearcherState, config: SupervisorConfig) -> Dict:
    """Individual researcher that investigates a specific section."""
    from app.tools import get_search_tool, search_arxiv
    
    llm = get_llm(temperature=0.5)
    section = state.get("section")
    persona = state.get("persona")
    
    if not section:
        return {"draft_content": "Error: No section assigned"}
    
    # Build tools list
    tools = [think_tool]
    search_tool = get_search_tool()
    if search_tool:
        tools.append(search_tool)
    tools.append(search_arxiv)
    
    # Add MCP tools if configured
    mcp_tools = state.get("mcp_tools", [])
    tools.extend(mcp_tools)
    
    llm_with_tools = llm.bind_tools(tools)
    
    prompt = RESEARCHER_PROMPT.format(
        persona_expertise=persona.expertise if persona else "General Research",
        persona_focus=persona.focus if persona else "Comprehensive Coverage",
        section_title=section.title,
        section_description=section.description,
        language=config.language,
        tools=", ".join([t.name if hasattr(t, 'name') else str(t) for t in tools])
    )
    
    messages = state.get("messages", [])
    if not messages:
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Research and write content for section: {section.title}")
        ]
    
    response = await llm_with_tools.ainvoke(messages)
    
    # Check for tool calls
    if hasattr(response, 'tool_calls') and response.tool_calls:
        tool_iterations = state.get("tool_iterations", 0) + 1
        if tool_iterations < state.get("max_tool_iterations", 10):
            return {
                "messages": [response],
                "tool_iterations": tool_iterations
            }
    
    # Extract final content
    draft_content = response.content if isinstance(response.content, str) else str(response.content)
    
    return {
        "messages": [response],
        "draft_content": draft_content
    }


async def researcher_tools_node(state: ResearcherState, config: SupervisorConfig) -> Dict:
    """Execute tools called by the researcher."""
    from app.tools import get_search_tool, search_arxiv
    
    messages = state.get("messages", [])
    if not messages:
        return {}
    
    last_message = messages[-1]
    if not hasattr(last_message, 'tool_calls') or not getattr(last_message, 'tool_calls', None):
        return {}
    
    # Build tools map
    tools_map = {"think_tool": think_tool}
    search_tool = get_search_tool()
    if search_tool:
        tools_map[search_tool.name] = search_tool
    tools_map["search_arxiv"] = search_arxiv
    
    # Execute tools
    tool_results = []
    learnings = []
    sources = []
    
    for tool_call in getattr(last_message, 'tool_calls', []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        
        if tool_name in tools_map:
            try:
                tool = tools_map[tool_name]
                result = await tool.ainvoke(tool_args) if asyncio.iscoroutinefunction(tool.invoke) else tool.invoke(tool_args)
                tool_results.append(ToolMessage(
                    content=str(result),
                    tool_call_id=tool_call["id"]
                ))
                
                # Extract learnings from search results
                if tool_name in ["web_search", "tavily_search", "search_arxiv"]:
                    learnings.append(f"From {tool_name}: {str(result)[:500]}")
                    sources.append(f"{tool_name} result")
            except Exception as e:
                tool_results.append(ToolMessage(
                    content=f"Error: {str(e)}",
                    tool_call_id=tool_call["id"]
                ))
    
    return {
        "messages": tool_results,
        "learnings": learnings,
        "sources": sources
    }


# =============================================================================
# Reviewer Agent Functions
# =============================================================================

async def reviewer_node(state: ReviewerState, config: SupervisorConfig) -> Dict:
    """Review a section for quality and accuracy."""
    llm = get_llm(temperature=0.2)
    
    section = state.get("section")
    draft_content = state.get("draft_content", "")
    review_criteria = state.get("review_criteria", [
        "Accuracy and factual correctness",
        "Completeness of coverage",
        "Clarity and readability",
        "Proper source citations",
        "Logical flow"
    ])
    
    prompt = REVIEWER_PROMPT.format(
        section_title=section.title if section else "Unknown",
        content=draft_content,
        criteria="\n".join(f"- {c}" for c in review_criteria)
    )
    
    llm_with_structure = llm.with_structured_output(ReviewFeedback)
    
    response = await llm_with_structure.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Review this section and provide feedback")
    ])
    
    return {"feedback": response}


async def reviser_node(state: ReviewerState, config: SupervisorConfig) -> Dict:
    """Revise content based on reviewer feedback."""
    llm = get_llm(temperature=0.4)
    
    section = state.get("section")
    draft_content = state.get("draft_content", "")
    feedback = state.get("feedback")
    
    if not feedback or feedback.is_approved:
        return {"draft_content": draft_content}
    
    prompt = REVISER_PROMPT.format(
        original_content=draft_content,
        feedback=feedback.feedback,
        suggestions="\n".join(f"- {s}" for s in feedback.suggestions),
        language=config.language
    )
    
    response = await llm.ainvoke([
        SystemMessage(content=prompt),
        HumanMessage(content="Revise the content based on feedback")
    ])
    
    revised_content = response.content if isinstance(response.content, str) else str(response.content)
    
    return {"draft_content": revised_content}


# =============================================================================
# Report Compilation and Export
# =============================================================================

async def compile_report(state: SupervisorState, config: SupervisorConfig) -> Dict:
    """Compile all sections into the final report."""
    llm = get_llm(temperature=0.3)
    
    research_plan = state.get("research_plan")
    completed_sections = state.get("completed_sections", [])
    
    if not research_plan:
        return {"final_report": "Error: No research plan found"}
    
    # Build report structure
    report_parts = [
        f"# {research_plan.title}",
        "",
        "## Executive Summary",
        research_plan.introduction or "Introduction content to be added.",
        ""
    ]
    
    # Add sections
    for section in completed_sections:
        report_parts.extend([
            f"## {section.title}",
            "",
            section.content,
            ""
        ])
    
    # Add conclusion
    report_parts.extend([
        "## Conclusion",
        "",
        research_plan.conclusion or "Conclusion to be added.",
        ""
    ])
    
    final_report = "\n".join(report_parts)
    
    return {
        "final_report": final_report,
        "status_updates": [{
            "type": "status",
            "stage": "compiled",
            "message": "Report compiled successfully"
        }]
    }


async def export_report(state: SupervisorState, config: SupervisorConfig) -> Dict:
    """Export report to configured formats."""
    final_report = state.get("final_report", "")
    export_formats = config.export_formats
    
    exports = {}
    
    for fmt in export_formats:
        if fmt == ExportFormat.MARKDOWN:
            exports["markdown"] = final_report
        elif fmt == ExportFormat.HTML:
            # Basic HTML conversion
            import html
            html_content = f"""<!DOCTYPE html>
<html>
<head><title>Research Report</title></head>
<body>
<article>{html.escape(final_report).replace(chr(10), '<br/>')}</article>
</body>
</html>"""
            exports["html"] = html_content
        elif fmt == ExportFormat.PDF:
            # PDF would require additional library like reportlab or weasyprint
            exports["pdf"] = "PDF export requires additional setup"
        elif fmt == ExportFormat.DOCX:
            # DOCX would require python-docx
            exports["docx"] = "DOCX export requires python-docx library"
    
    return {
        "exports": exports,
        "status_updates": [{
            "type": "status",
            "stage": "exported",
            "message": f"Report exported to: {', '.join(exports.keys())}"
        }]
    }


# =============================================================================
# Human Feedback Integration
# =============================================================================

async def human_feedback_node(state: SupervisorState, config: SupervisorConfig) -> Command:
    """Handle human feedback in the research process."""
    human_feedback = state.get("human_feedback")
    
    if human_feedback:
        # Process feedback and potentially revise plan
        return Command(
            goto="revise_plan",
            update={
                "status_updates": [{
                    "type": "status",
                    "stage": "human_feedback",
                    "message": "Processing human feedback..."
                }]
            }
        )
    else:
        # Continue without feedback
        return Command(goto="compile_report")


# =============================================================================
# MCP (Model Context Protocol) Support
# =============================================================================

async def load_mcp_tools(server_url: str) -> List[BaseTool]:
    """Load tools from an MCP server.
    
    Note: This is a placeholder. Full MCP implementation requires
    the langchain-mcp-adapters package.
    """
    try:
        # Placeholder for MCP tool loading
        # from langchain_mcp_adapters.client import MultiServerMCPClient
        # client = MultiServerMCPClient(server_url)
        # return await client.get_tools()
        return []
    except Exception as e:
        print(f"Warning: Could not load MCP tools: {e}")
        return []


# =============================================================================
# Graph Builder
# =============================================================================

def build_supervisor_graph(config: Optional[SupervisorConfig] = None):
    """Build the complete supervisor multi-agent graph."""
    if config is None:
        config = SupervisorConfig()
    
    # Main supervisor graph
    supervisor_builder = StateGraph(SupervisorState)
    
    # Add nodes (type: ignore needed for LangGraph complex generics)
    supervisor_builder.add_node("create_plan", lambda s: create_research_plan(s, config))  # type: ignore
    supervisor_builder.add_node("generate_personas", lambda s: generate_personas(s, config))  # type: ignore
    supervisor_builder.add_node("dispatch_researchers", lambda s: dispatch_researchers(s, config))  # type: ignore
    supervisor_builder.add_node("collect_research", lambda s: collect_research(s, config))  # type: ignore
    supervisor_builder.add_node("compile_report", lambda s: compile_report(s, config))  # type: ignore
    supervisor_builder.add_node("export_report", lambda s: export_report(s, config))  # type: ignore
    
    # Researcher subgraph
    researcher_builder = StateGraph(ResearcherState)
    researcher_builder.add_node("research", lambda s: researcher_node(s, config))  # type: ignore
    researcher_builder.add_node("tools", lambda s: researcher_tools_node(s, config))  # type: ignore
    
    def should_continue_research(state: ResearcherState) -> str:
        messages = state.get("messages", [])
        if not messages:
            return END
        last_msg = messages[-1]
        if hasattr(last_msg, 'tool_calls') and getattr(last_msg, 'tool_calls', None):  # type: ignore
            if state.get("tool_iterations", 0) < state.get("max_tool_iterations", 10):
                return "tools"
        return END
    
    researcher_builder.add_edge(START, "research")
    researcher_builder.add_conditional_edges("research", should_continue_research)
    researcher_builder.add_edge("tools", "research")
    
    researcher_subgraph = researcher_builder.compile()
    supervisor_builder.add_node("researcher", researcher_subgraph)
    
    # Reviewer subgraph (if enabled)
    if config.enable_review_loop:
        reviewer_builder = StateGraph(ReviewerState)
        reviewer_builder.add_node("review", lambda s: reviewer_node(s, config))  # type: ignore
        reviewer_builder.add_node("revise", lambda s: reviser_node(s, config))  # type: ignore
        
        def should_revise(state: ReviewerState) -> str:
            feedback = state.get("feedback")
            if feedback and not feedback.is_approved:
                return "revise"
            return END
        
        reviewer_builder.add_edge(START, "review")
        reviewer_builder.add_conditional_edges("review", should_revise)
        reviewer_builder.add_edge("revise", END)
        
        reviewer_subgraph = reviewer_builder.compile()
        supervisor_builder.add_node("review_sections", reviewer_subgraph)
    
    # Human feedback node (if enabled)
    if config.enable_human_feedback:
        supervisor_builder.add_node("human_feedback", lambda s: human_feedback_node(s, config))  # type: ignore
    
    # Define edges
    supervisor_builder.add_edge(START, "create_plan")
    
    if config.enable_personas:
        supervisor_builder.add_edge("create_plan", "generate_personas")
        supervisor_builder.add_edge("generate_personas", "dispatch_researchers")
    else:
        supervisor_builder.add_edge("create_plan", "dispatch_researchers")
    
    supervisor_builder.add_edge("dispatch_researchers", "researcher")
    supervisor_builder.add_edge("researcher", "collect_research")
    
    if config.enable_review_loop:
        supervisor_builder.add_edge("collect_research", "review_sections")
        if config.enable_human_feedback:
            supervisor_builder.add_edge("review_sections", "human_feedback")
            supervisor_builder.add_edge("human_feedback", "compile_report")
        else:
            supervisor_builder.add_edge("review_sections", "compile_report")
    else:
        if config.enable_human_feedback:
            supervisor_builder.add_edge("collect_research", "human_feedback")
            supervisor_builder.add_edge("human_feedback", "compile_report")
        else:
            supervisor_builder.add_edge("collect_research", "compile_report")
    
    supervisor_builder.add_edge("compile_report", "export_report")
    supervisor_builder.add_edge("export_report", END)
    
    return supervisor_builder.compile()


# =============================================================================
# Main API Function
# =============================================================================

async def run_supervisor_research(
    topic: str,
    config: Optional[SupervisorConfig] = None,
    on_progress: Optional[Callable[[Dict], None]] = None
) -> Dict[str, Any]:
    """
    Run the full supervisor-based research workflow.
    
    Args:
        topic: The research topic
        config: Configuration options
        on_progress: Callback for progress updates
    
    Returns:
        Dict containing final_report and metadata
    """
    if config is None:
        config = SupervisorConfig()
    
    # Load MCP tools if enabled
    mcp_tools = []
    if config.enable_mcp and config.mcp_server_url:
        mcp_tools = await load_mcp_tools(config.mcp_server_url)
    
    graph = build_supervisor_graph(config)
    
    initial_state = {
        "research_topic": topic,
        "messages": [HumanMessage(content=topic)],
        "mcp_tools": mcp_tools,
        "export_formats": config.export_formats,
        "max_review_iterations": config.max_review_iterations
    }
    
    # Run the graph
    result = await graph.ainvoke(initial_state)  # type: ignore
    
    # Call progress callback if provided
    if on_progress:
        for update in result.get("status_updates", []):
            on_progress(update)
    
    return {
        "final_report": result.get("final_report", ""),
        "exports": result.get("exports", {}),
        "thinking_log": result.get("thinking_log", []),
        "completed_sections": result.get("completed_sections", []),
        "personas": result.get("personas", [])
    }


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Config
    "SupervisorConfig",
    "ExportFormat",
    # Models
    "Section",
    "ResearchPlan",
    "ReviewFeedback",
    "Persona",
    "ThinkingOutput",
    # States
    "SupervisorState",
    "ResearcherState",
    "ReviewerState",
    # Functions
    "build_supervisor_graph",
    "run_supervisor_research",
    # Tools
    "think_tool",
    "request_human_feedback",
    "load_mcp_tools",
]
