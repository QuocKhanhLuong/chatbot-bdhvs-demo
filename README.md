# AI Research Assistant

LangGraph-based Multi-Agent AI Research Assistant with Deep Research capabilities, persistent chat history, and Artifacts UI.

## ✨ Key Features

- **🤖 Multi-Agent System**: Triage → Research/Coding/Document/Deep Research agents
- **🔬 Deep Research V2**: Enhanced recursive research with Pydantic structured output
- **💾 Persistent Storage**: SQLite-backed chat history via LangGraph checkpointer
- **📊 Artifacts UI**: Split-view display for research reports and code
- **🎨 Modern UI**: Dark/Light mode with smooth animations

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
│   │   ├── server.py      # FastAPI server
│   │   ├── config.py      # Settings
│   │   ├── tools/         # Agent tools
│   │   │   ├── base.py    # Search, Python REPL, ArXiv
│   │   │   ├── deep_research.py    # Original deep research
│   │   │   └── deep_research_v2.py # Enhanced v2 with Pydantic
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
- **Search:** Tavily API + ArXiv
- **Persistence:** SQLite via langgraph-checkpoint-sqlite

## 🤖 Agent System

| Agent | Purpose | Tools |
|-------|---------|-------|
| **Triage** | Route queries to appropriate agent | - |
| **Research** | Web search + ArXiv papers | Tavily, ArXiv |
| **Coding** | Python code execution | Python REPL |
| **Document** | Local knowledge base search | FAISS retriever |
| **Deep Research** | Multi-iteration recursive research | Tavily, ArXiv, Pydantic |

### Deep Research V2 Features
- **Pydantic Structured Output**: Validated queries, learnings, reports
- **Follow-up Questions**: Clarify research direction before starting
- **Concurrent Processing**: Asyncio.Semaphore for rate limiting
- **Learnings Accumulation**: Context builds across iterations
- **ArXiv Integration**: Academic paper search
- **Progress Tracking**: Real-time depth, breadth, query stats

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

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| POST | `/chat` | Chat with streaming SSE |
| POST | `/api/v1/chat` | Chat (REST) |
| POST | `/api/v1/search` | Similarity search |
| GET | `/api/v1/threads` | List chat threads |
| GET | `/api/v1/threads/{id}/history` | Get thread history |

## 🎨 Artifacts UI

Reports from Deep Research are displayed in a split-view panel:
- **Left**: Chat conversation
- **Right**: Artifact panel with markdown rendering
- Supports copy, download, expand/collapse
- Detects `---REPORT START---` / `---REPORT END---` tags

## 📦 Key Dependencies

```
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

## 📄 License

MIT License
