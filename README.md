# AI Research Assistant

LangGraph-based Multi-Agent AI Research Assistant with Deep Research capabilities, Perplexity-style Answer Engine, Supervisor-Researcher pattern, persistent chat history, and Artifacts UI.

Inspired by top open-source research systems:

- **[Perplexity.ai](https://perplexity.ai)** - Real-time search with citations and Pro Search
- **[Consensus.app](https://consensus.app)** - Academic research with consensus analysis
- **[rashadphz/farfalle](https://github.com/rashadphz/farfalle)** ⭐ ~3k - Perplexity clone
- **[developersdigest/llm-answer-engine](https://github.com/developersdigest/llm-answer-engine)** ⭐ ~4k - Answer engine
- **[assafelovic/gpt-researcher](https://github.com/assafelovic/gpt-researcher)** ⭐ ~20k - Multi-agent team with review loop
- **[stanford-oval/storm](https://github.com/stanford-oval/storm)** ⭐ ~18k - Persona-based research
- **[langchain-ai/open_deep_research](https://github.com/langchain-ai/open_deep_research)** ⭐ ~3k - Supervisor-Researcher pattern
- **[dzhng/deep-research](https://github.com/dzhng/deep-research)** ⭐ ~5k - Recursive deep research

## ✨ Key Features

### Core Features

- **🤖 Multi-Agent System**: Triage → Research/Coding/Document/Deep Research agents
- **🔬 Deep Research V2**: Enhanced recursive research with Pydantic structured output
- **💾 Persistent Storage**: SQLite-backed chat history via LangGraph checkpointer
- **📊 Artifacts UI**: Split-view display for research reports and code
- **🎨 Modern UI**: Dark/Light mode with smooth animations

### 🔍 Perplexity-Style Answer Engine (NEW!)

| Feature | Inspired By | Description |
|---------|-------------|-------------|
| **⚡ Real-time Search** | Perplexity | Multi-provider search (Tavily, Serper, Brave) |
| **📝 Inline Citations** | Perplexity | Answer with [1], [2] citations linked to sources |
| **🎯 Pro Search** | Perplexity | Multi-step query planning for complex questions |
| **❓ Related Questions** | Perplexity | AI-generated follow-up questions |
| **📊 Consensus Meter** | Consensus | Agreement analysis for academic research |
| **🎓 Academic Focus** | Consensus | ArXiv + Semantic Scholar integration |
| **📈 Evidence Quality** | Consensus | Study design detection, citation counts |
| **🖼️ Image Search** | Perplexity | Relevant images alongside answers |
| **⚡ Streaming** | Perplexity | Real-time SSE streaming of answers |

### Advanced Features (Supervisor System)

| Feature | Inspired By | Description |
|---------|-------------|-------------|
| **🎭 Supervisor-Researcher** | open_deep_research | Supervisor delegates to parallel researchers |
| **🔄 Review Loop** | gpt-researcher | Reviewer → Reviser cycle until quality met |
| **👥 Persona-based Research** | STORM | Multiple perspectives for comprehensive coverage |
| **🧠 Strategic Thinking** | open_deep_research | `think_tool` for reflection between searches |
| **📄 Multi-format Export** | gpt-researcher | Markdown, HTML, PDF, DOCX output |
| **👤 Human-in-the-loop** | gpt-researcher | User feedback integration |
| **🔌 MCP Support** | open_deep_research | Model Context Protocol for external tools |

## 🏗️ Project Structure

```
chatbot-sinno/
├── frontend/              # Next.js 15 App Router
│   ├── app/               # App Router pages
│   ├── components/        # React components
│   │   ├── Chat/          # Chat UI components
│   │   ├── Artifacts/     # Artifact panel components
│   │   └── ui/            # Shadcn UI components
│   └── lib/               # Utility libraries
├── backend/               # Python FastAPI + LangGraph
│   ├── app/
│   │   ├── agents.py      # Multi-agent system
│   │   ├── multi_agent_supervisor.py  # Supervisor-Researcher system
│   │   ├── server.py      # FastAPI server
│   │   ├── config.py      # Settings
│   │   ├── tools/         # Agent tools
│   │   │   ├── base.py    # Search, Python REPL, ArXiv
│   │   │   ├── deep_research.py    # Original deep research
│   │   │   ├── deep_research_v2.py # Enhanced v2 with Pydantic
│   │   │   └── perplexity_engine.py # NEW: Perplexity-style engine
│   │   └── api/v1/        # REST API endpoints
│   ├── data/
│   │   ├── chat_history.db # SQLite persistent storage
│   │   ├── faiss_index/    # Vector embeddings
│   │   └── pdf/            # Knowledge base documents
│   └── tests/
├── docker-compose.yml
└── README.md
```

## 🚀 Tech Stack

### Frontend

- **Framework:** Next.js 15.5 (App Router)
- **React:** 19
- **Styling:** Tailwind CSS + Shadcn UI
- **Theme:** Dark/Light mode with next-themes
- **Markdown:** react-markdown + remark-gfm

### Backend

- **Framework:** FastAPI 0.115 + LangGraph 0.2+
- **LLM:** MegaLLM / OpenAI / Google Gemini
- **Embeddings:** FastEmbed (BAAI/bge-small-en-v1.5)
- **Vector Store:** FAISS
- **Search:** Tavily API + Serper + ArXiv + Semantic Scholar
- **Persistence:** SQLite via langgraph-checkpoint-sqlite

## 🤖 Agent System

### Basic Agents

| Agent | Purpose | Tools |
|-------|---------|-------|
| **Triage** | Route queries to appropriate agent | - |
| **Research** | Web search + ArXiv papers | Tavily, ArXiv |
| **Coding** | Python code execution | Python REPL |
| **Document** | Local knowledge base search | FAISS retriever |
| **Deep Research** | Multi-iteration recursive research | Tavily, ArXiv, Pydantic |

### Supervisor System Agents (Advanced)

| Agent | Role | Features |
|-------|------|----------|
| **Supervisor** | Orchestrates research workflow | Plan creation, persona generation, task delegation |
| **Researcher** | Conducts focused research | Web search, ArXiv, think_tool, MCP tools |
| **Reviewer** | Quality control | Accuracy check, completeness, citations |
| **Reviser** | Content improvement | Revise based on feedback |
| **Exporter** | Multi-format output | Markdown, HTML, PDF, DOCX |

### Deep Research V2 Features

- **Pydantic Structured Output**: Validated queries, learnings, reports
- **Follow-up Questions**: Clarify research direction before starting
- **Concurrent Processing**: Asyncio.Semaphore for rate limiting
- **Learnings Accumulation**: Context builds across iterations
- **ArXiv Integration**: Academic paper search
- **Progress Tracking**: Real-time depth, breadth, query stats

### Supervisor Research Features

```python
from app.multi_agent_supervisor import run_supervisor_research, SupervisorConfig

config = SupervisorConfig(
    max_sections=5,           # Report sections
    enable_personas=True,     # Multi-perspective research
    num_personas=3,           # Number of personas
    enable_review_loop=True,  # Reviewer → Reviser cycle
    max_review_iterations=3,  # Max revision rounds
    enable_human_feedback=False,  # Human-in-the-loop
    enable_mcp=False,         # MCP tools
    export_formats=["markdown", "html"],
    language="vi"
)

result = await run_supervisor_research(
    topic="AI trong y tế",
    config=config
)
```

## 🛠️ Development Setup

### Backend

```bash
cd backend
conda create -n chatbot-sinno python=3.11
conda activate chatbot-sinno
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your API keys:
# - MEGALLM_API_KEY or OPENAI_API_KEY
# - TAVILY_API_KEY (for web search)

# Run server
python -m app.main
# Server runs at http://localhost:8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
# UI runs at http://localhost:3000
```

### Docker

```bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up --build
```

## 📡 API Endpoints

### Basic Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Chat with streaming SSE |
| POST | `/api/v1/chat` | Chat (REST) |
| POST | `/api/v1/search` | Similarity search |
| GET | `/api/v1/threads` | List chat threads |
| GET | `/api/v1/threads/{id}/history` | Get thread history |

### 🔍 Answer Engine Endpoints (Perplexity-style)

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/answer` | Full answer with citations, sources, related questions |
| POST | `/answer/stream` | Streaming answer (SSE) |
| POST | `/answer/quick` | Quick search for simple queries |
| POST | `/answer/pro` | Pro Search with multi-step query planning |
| POST | `/answer/consensus` | Academic consensus analysis (Consensus-style) |

#### Answer Engine Request Example

```json
{
  "query": "What are the health benefits of intermittent fasting?",
  "include_images": true,
  "include_academic": true,
  "enable_pro_search": false,
  "enable_consensus": true,
  "max_sources": 10,
  "language": "vi"
}
```

#### Answer Engine Response

```json
{
  "status": "success",
  "answer": "Intermittent fasting has several health benefits [1]. Studies show...",
  "citations": [
    {
      "number": 1,
      "title": "Intermittent Fasting: A Review",
      "url": "https://...",
      "snippet": "A comprehensive review of...",
      "source_type": "academic"
    }
  ],
  "related_questions": [
    {"question": "What is the best intermittent fasting schedule?", "category": "deeper"},
    {"question": "Is intermittent fasting safe for diabetics?", "category": "related"}
  ],
  "images": ["https://..."],
  "consensus": {
    "agreement_level": "yes",
    "confidence": 0.75,
    "sample_size": 12,
    "key_findings": ["Weight loss", "Improved insulin sensitivity"],
    "study_designs": {"RCT": 5, "Meta-analysis": 2},
    "evidence_quality": "high"
  }
}
```

#### Streaming Events (SSE)

```javascript
// Event types from /answer/stream
{event: "begin", data: {query: "..."}}
{event: "search-results", data: {results: [...], images: [...]}}
{event: "query-plan", data: {steps: [...]}}  // Pro Search only
{event: "text-chunk", data: {text: "..."}}   // Streamed answer
{event: "citation", data: {citations: [...]}}
{event: "related-questions", data: {questions: [...]}}
{event: "consensus", data: {...}}            // If enabled
{event: "done", data: {total_sources: 10}}
```

### Research Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/research/deep` | Deep research (recursive) |
| POST | `/research/deep/stream` | Deep research with SSE streaming |
| POST | `/research/quick` | Quick single-iteration research |
| POST | `/research/supervisor` | Supervisor multi-agent research |
| POST | `/research/supervisor/stream` | Supervisor research with streaming |
| POST | `/arxiv/search` | ArXiv paper search |

### Supervisor Research Request

```json
{
  "topic": "AI applications in healthcare",
  "max_sections": 5,
  "enable_personas": true,
  "num_personas": 3,
  "enable_review_loop": true,
  "max_review_iterations": 3,
  "enable_human_feedback": false,
  "export_formats": ["markdown", "html"],
  "language": "vi"
}
```

## 🎨 Artifacts UI

Reports from Deep Research are displayed in a split-view panel:

- **Left**: Chat conversation
- **Right**: Artifact panel with markdown rendering
- Supports copy, download, expand/collapse
- Detects `---REPORT START---` / `---REPORT END---` tags

## 🔌 MCP Support (Model Context Protocol)

Enable MCP for external tool integration:

```python
config = SupervisorConfig(
    enable_mcp=True,
    mcp_server_url="http://localhost:3001"
)
```

MCP allows researchers to access:

- Custom databases
- Internal APIs
- Proprietary data sources
- External services

## 📦 Key Dependencies

```txt
# Backend
fastapi>=0.115.0
langgraph>=0.2.0
langgraph-checkpoint-sqlite>=2.0.0
langchain>=0.3.0
tavily-python>=0.5.0
arxiv>=2.1.0
pydantic>=2.10.0

# Frontend
next@15.5.6
react@19
tailwindcss
@shadcn/ui
```

## 🔗 References

- [gpt-researcher](https://github.com/assafelovic/gpt-researcher) - Multi-agent research
- [STORM](https://github.com/stanford-oval/storm) - Persona-based research
- [open_deep_research](https://github.com/langchain-ai/open_deep_research) - Supervisor pattern
- [deep-research](https://github.com/dzhng/deep-research) - Recursive research
- [LangGraph](https://langchain-ai.github.io/langgraph/) - Agent orchestration

## 📄 License

MIT License
