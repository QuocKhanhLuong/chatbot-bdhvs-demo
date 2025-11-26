#!/bin/bash

# =============================================================================
# Chatbot BDHVS - Start Backend Only
# =============================================================================
# This script starts the FastAPI backend with LangGraph multi-agent system
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

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🐍 Starting Backend (FastAPI + LangGraph)${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping backend...${NC}"
    lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
    pkill -f "uvicorn app.server:app" 2>/dev/null
    echo -e "${GREEN}✅ Backend stopped${NC}"
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

# Initialize conda for the current shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if conda environment exists
if ! conda env list | grep -q "$CONDA_ENV"; then
    echo -e "${RED}❌ Error: Conda environment '$CONDA_ENV' not found${NC}"
    echo -e "${YELLOW}Create it with: conda create -n $CONDA_ENV python=3.11${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}\n"

# ===========================================
# Kill existing processes
# ===========================================
echo -e "${CYAN}🧹 Cleaning up existing processes on port $BACKEND_PORT...${NC}"
lsof -ti:$BACKEND_PORT | xargs kill -9 2>/dev/null
sleep 1
echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# ===========================================
# Start Backend
# ===========================================
echo -e "${GREEN}🚀 Starting Backend...${NC}"
echo -e "   Environment: ${CYAN}$CONDA_ENV${NC}"
echo -e "   Port: ${CYAN}$BACKEND_PORT${NC}\n"

cd "$PROJECT_ROOT/backend"

# Activate conda and start backend
conda activate "$CONDA_ENV"

echo -e "${YELLOW}📦 Python version: $(python --version)${NC}"
echo -e "${YELLOW}📍 Using: $(which python)${NC}\n"

# Start uvicorn with auto-reload for development
python -m uvicorn app.server:app --host 0.0.0.0 --port $BACKEND_PORT --reload
