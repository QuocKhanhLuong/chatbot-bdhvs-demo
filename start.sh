#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🚀 Starting Chatbot Application${NC}"
echo -e "${BLUE}================================================${NC}\n"

# Function to cleanup background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping all processes...${NC}"
    kill $(jobs -p) 2>/dev/null
    exit
}

trap cleanup SIGINT SIGTERM

# Check if backend/.env exists
if [ ! -f backend/.env ]; then
    echo -e "${RED}❌ Error: backend/.env not found${NC}"
    echo -e "${YELLOW}Please create backend/.env with MEGALLM_API_KEY${NC}"
    exit 1
fi

# Start Python Backend
echo -e "${GREEN}🐍 Starting Python Backend (FastAPI + RAG)...${NC}"
cd backend/api
python api.py > ../../logs/backend.log 2>&1 &
BACKEND_PID=$!
cd ../..

# Wait for backend to be ready
echo -e "${YELLOW}⏳ Waiting for backend to initialize...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
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

# Start Next.js Frontend
echo -e "${GREEN}⚛️  Starting Next.js Frontend...${NC}"
npm run dev > logs/frontend.log 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo -e "${YELLOW}⏳ Waiting for frontend to start...${NC}"
sleep 5

echo -e "\n${BLUE}================================================${NC}"
echo -e "${GREEN}✅ All services are running!${NC}"
echo -e "${BLUE}================================================${NC}\n"

echo -e "${GREEN}🌐 Frontend:${NC} http://localhost:3000"
echo -e "${GREEN}🔧 Backend:${NC}  http://localhost:8000"
echo -e "${GREEN}📊 Health:${NC}   http://localhost:8000/health\n"

echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "   Backend:  logs/backend.log"
echo -e "   Frontend: logs/frontend.log\n"

echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}\n"

# Keep script running and show live logs
tail -f logs/backend.log logs/frontend.log
