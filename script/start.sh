#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 Starting Chatbot BDHVS - Monorepo${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Get project root directory (parent of script/)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Create logs directory
mkdir -p "$SCRIPT_DIR/logs"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping all processes...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Check if backend/.env exists
if [ ! -f "$PROJECT_ROOT/backend/.env" ]; then
    echo -e "${RED}❌ Error: backend/.env not found${NC}"
    echo -e "${YELLOW}Please create backend/.env with MEGALLM_API_KEY${NC}"
    echo -e "Run: echo 'MEGALLM_API_KEY=your_key' > backend/.env"
    exit 1
fi

# ===========================================
# Start Backend (Python FastAPI)
# ===========================================
echo -e "${GREEN}🐍 Starting Backend (Python FastAPI)...${NC}"
cd "$PROJECT_ROOT/backend"
python -m app.main > "$SCRIPT_DIR/logs/backend.log" 2>&1 &
BACKEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/api/v1/health > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Backend is ready!${NC}\n"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend failed to start. Check logs/backend.log${NC}"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done

# ===========================================
# Start Frontend (Next.js)
# ===========================================
echo -e "${GREEN}⚛️  Starting Frontend (Next.js)...${NC}"
cd "$PROJECT_ROOT/frontend"
npm run dev > "$SCRIPT_DIR/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!
cd "$PROJECT_ROOT"

# Wait for frontend to be ready
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Frontend is ready!${NC}\n"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Frontend failed to start. Check logs/frontend.log${NC}"
        kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
        exit 1
    fi
done

# ===========================================
# Success
# ===========================================
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}🎉 All services are running!${NC}"
echo -e "${BLUE}================================================${NC}"
echo -e ""
echo -e "  📱 Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "  🔧 Backend:   ${GREEN}http://localhost:8000${NC}"
echo -e "  📚 API Docs:  ${GREEN}http://localhost:8000/docs${NC}"
echo -e ""
echo -e "  📝 Logs:"
echo -e "     - Backend:  logs/backend.log"
echo -e "     - Frontend: logs/frontend.log"
echo -e ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo -e "${BLUE}================================================${NC}"

# Keep script running
wait
