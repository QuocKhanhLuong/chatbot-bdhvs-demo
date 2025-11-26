#!/bin/bash

# =============================================================================
# Chatbot BDHVS - Start Frontend Only
# =============================================================================
# This script starts the Next.js frontend
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
FRONTEND_PORT=3000

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}⚛️  Starting Frontend (Next.js)${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Get project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Function to cleanup on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping frontend...${NC}"
    lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
    pkill -f "next dev" 2>/dev/null
    echo -e "${GREEN}✅ Frontend stopped${NC}"
    exit
}

trap cleanup SIGINT SIGTERM

# ===========================================
# Pre-flight Checks
# ===========================================
echo -e "${CYAN}🔍 Running pre-flight checks...${NC}\n"

# Check if node is available
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Error: Node.js not found${NC}"
    echo -e "${YELLOW}Please install Node.js first${NC}"
    exit 1
fi

# Check if npm is available
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ Error: npm not found${NC}"
    echo -e "${YELLOW}Please install npm first${NC}"
    exit 1
fi

# Check if frontend directory exists
if [ ! -d "$PROJECT_ROOT/frontend" ]; then
    echo -e "${RED}❌ Error: frontend directory not found${NC}"
    exit 1
fi

# Check if node_modules exists
if [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo -e "${YELLOW}📦 Installing frontend dependencies...${NC}"
    cd "$PROJECT_ROOT/frontend"
    npm install
fi

echo -e "${GREEN}✅ Pre-flight checks passed${NC}\n"

# ===========================================
# Kill existing processes
# ===========================================
echo -e "${CYAN}🧹 Cleaning up existing processes on port $FRONTEND_PORT...${NC}"
lsof -ti:$FRONTEND_PORT | xargs kill -9 2>/dev/null
sleep 1
echo -e "${GREEN}✅ Cleanup complete${NC}\n"

# ===========================================
# Start Frontend
# ===========================================
echo -e "${GREEN}🚀 Starting Frontend...${NC}"
echo -e "   Port: ${CYAN}$FRONTEND_PORT${NC}"
echo -e "   Node: ${CYAN}$(node --version)${NC}\n"

cd "$PROJECT_ROOT/frontend"

# Start Next.js development server
npm run dev
