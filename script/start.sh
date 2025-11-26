#!/bin/bash

# =============================================================================
# Chatbot BDHVS - Start All Services
# =============================================================================
# This script starts both backend (FastAPI) and frontend (Next.js)
# with proper conda environment activation and logging.
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
CONDA_ENV="chatbot-sinno"
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 Starting AI Research Assistant${NC}"
echo -e "${BLUE}   Powered by LangGraph + MegaLLM${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping all services...${NC}"
    
    # Kill processes on ports
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
    
    # Kill by name
    pkill -f "uvicorn app.server:app" 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    
    echo -e "${GREEN}✅ All services stopped${NC}"
    exit
}

trap cleanup SIGINT SIGTERM

# ===========================================
# Pre-flight Checks
# ===========================================
echo -e "${CYAN}🔍 Running pre-flight checks...${NC}\n"

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo -e "${RED}❌ Error: conda not found${NC}"
    echo -e "${YELLOW}Please install Anaconda/Miniconda first${NC}"
    exit 1
fi

# Check if conda environment exists
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo -e "${RED}❌ Error: Conda environment '$CONDA_ENV' not found${NC}"
    echo -e "${YELLOW}Create it with: conda create -n $CONDA_ENV python=3.11${NC}"
    exit 1
fi

# Check if backend/.env exists
if [ ! -f "$PROJECT_ROOT/backend/.env" ]; then
    echo -e "${YELLOW}⚠️  No backend/.env found. Creating template...${NC}"
    cat > "$PROJECT_ROOT/backend/.env" << 'EOF'
# MegaLLM Configuration (Required)
MEGALLM_API_KEY=your_megallm_api_key_here
MEGALLM_BASE_URL=https://ai.megallm.io/v1
MEGALLM_MODEL=openai-gpt-oss-120b

# LLM Provider (megallm, openai, or google)
LLM_PROVIDER=megallm

# Optional: Tavily for web search
TAVILY_API_KEY=your_tavily_key_here

# Optional: OpenAI (if using as provider)
# OPENAI_API_KEY=your_openai_key

# Optional: Google (if using as provider)
# GOOGLE_API_KEY=your_google_key
EOF
    echo -e "${YELLOW}Please update backend/.env with your API keys${NC}"
    echo -e "${YELLOW}At minimum, set MEGALLM_API_KEY${NC}\n"
fi

# Check if frontend has node_modules
if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd "$PROJECT_ROOT/frontend"
    npm install
    cd "$PROJECT_ROOT"
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}\n"

# ===========================================
# Kill existing processes
# ===========================================
echo -e "${CYAN}🧹 Cleaning up existing processes...${NC}"
lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
sleep 1
echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# ===========================================
# Start Backend (Python FastAPI + LangGraph)
# ===========================================
echo -e "${GREEN}🐍 Starting Backend (FastAPI + LangGraph)...${NC}"
echo -e "   Environment: ${CYAN}$CONDA_ENV${NC}"
echo -e "   Port: ${CYAN}$BACKEND_PORT${NC}"

cd "$PROJECT_ROOT/backend"

# Start backend with conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
python -m uvicorn app.server:app --host 0.0.0.0 --port $BACKEND_PORT > "$SCRIPT_DIR/logs/backend.log" 2>&1 &
BACKEND_PID=$!

cd "$PROJECT_ROOT"

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to initialize...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:$BACKEND_PORT/docs > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready!${NC}\n"
        break
    fi
    sleep 1
    echo -ne "   Attempt $i/60...\r"
    if [ $i -eq 60 ]; then
        echo -e "\n${RED}❌ Backend failed to start${NC}"
        echo -e "${YELLOW}Check logs: $SCRIPT_DIR/logs/backend.log${NC}"
        echo -e "\nLast 20 lines of log:"
        tail -20 "$SCRIPT_DIR/logs/backend.log"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# ===========================================
# Start Frontend (Next.js)
# ===========================================
echo -e "${GREEN}⚛️  Starting Frontend (Next.js)...${NC}"
echo -e "   Port: ${CYAN}$FRONTEND_PORT${NC}"

cd "$PROJECT_ROOT/frontend"
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for frontend to be ready
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:$FRONTEND_PORT > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is ready!${NC}\n"
        break
    fi
    sleep 1
    echo -ne "   Attempt $i/60...\r"
    if [ $i -eq 60 ]; then
        echo -e "\n${RED}❌ Frontend failed to start${NC}"
        echo -e "${YELLOW}Check logs: $SCRIPT_DIR/logs/frontend.log${NC}"
        echo -e "\nLast 20 lines of log:"
        tail -20 "$SCRIPT_DIR/logs/frontend.log"
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
        exit 1
    fi
done

# ===========================================
# Success Banner
# ===========================================
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}🎉 AI Research Assistant is running!${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e ""
echo -e "  ${CYAN}📱 Frontend:${NC}     ${GREEN}http://localhost:$FRONTEND_PORT${NC}"
echo -e "  ${CYAN}🔧 Backend API:${NC}  ${GREEN}http://localhost:$BACKEND_PORT${NC}"
echo -e "  ${CYAN}📚 API Docs:${NC}     ${GREEN}http://localhost:$BACKEND_PORT/docs${NC}"
echo -e ""
echo -e "  ${CYAN}🔬 Answer Engine Endpoints:${NC}"
echo -e "     POST /answer        - Full answer with citations"
echo -e "     POST /answer/quick  - Quick search"
echo -e "     POST /answer/pro    - Pro Search (multi-step)"
echo -e "     POST /answer/stream - Streaming SSE"
echo -e "     POST /answer/consensus - Academic consensus"
echo -e ""
echo -e "  ${CYAN}📝 Logs:${NC}"
echo -e "     Backend:  ${YELLOW}$SCRIPT_DIR/logs/backend.log${NC}"
echo -e "     Frontend: ${YELLOW}$SCRIPT_DIR/logs/frontend.log${NC}"
echo -e ""
echo -e "  ${CYAN}🔑 MegaLLM Models (12 with auto-fallback):${NC}"
echo -e "     1. openai-gpt-oss-120b (default)"
echo -e "     2. openai-gpt-oss-60b"
echo -e "     3. qwen-2.5-72b-instruct"
echo -e "     ... and 9 more fallback options"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${BLUE}================================================${NC}"

# Keep script running and wait for child processes
wait
