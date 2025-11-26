#!/bin/bash

# =============================================================================
# Chatbot BDHVS - Stop All Services
# =============================================================================
# This script stops both backend and frontend services
# =============================================================================

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
BACKEND_PORT=8000
FRONTEND_PORT=3000

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}🛑 Stopping AI Research Assistant Services${NC}"
echo -e "${BLUE}================================================${NC}\n"

# ===========================================
# Stop Backend
# ===========================================
echo -e "${CYAN}Stopping backend on port $BACKEND_PORT...${NC}"

# Kill by port
BACKEND_PIDS=$(lsof -ti:$BACKEND_PORT 2>/dev/null)
if [ -n "$BACKEND_PIDS" ]; then
    echo "$BACKEND_PIDS" | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✅ Backend stopped (PIDs: $BACKEND_PIDS)${NC}"
else
    echo -e "${YELLOW}ℹ️  No backend process found on port $BACKEND_PORT${NC}"
fi

# Also kill by process name
pkill -f "uvicorn app.server:app" 2>/dev/null
pkill -f "python -m uvicorn" 2>/dev/null

# ===========================================
# Stop Frontend
# ===========================================
echo -e "\n${CYAN}Stopping frontend on port $FRONTEND_PORT...${NC}"

# Kill by port
FRONTEND_PIDS=$(lsof -ti:$FRONTEND_PORT 2>/dev/null)
if [ -n "$FRONTEND_PIDS" ]; then
    echo "$FRONTEND_PIDS" | xargs kill -9 2>/dev/null
    echo -e "${GREEN}✅ Frontend stopped (PIDs: $FRONTEND_PIDS)${NC}"
else
    echo -e "${YELLOW}ℹ️  No frontend process found on port $FRONTEND_PORT${NC}"
fi

# Also kill by process name
pkill -f "next dev" 2>/dev/null
pkill -f "node.*next" 2>/dev/null

# ===========================================
# Additional cleanup
# ===========================================
echo -e "\n${CYAN}Running additional cleanup...${NC}"

# Kill any orphaned Python processes from this project
pkill -f "app.server:app" 2>/dev/null
pkill -f "app.main:app" 2>/dev/null

# Wait a moment
sleep 1

# Verify ports are free
echo -e "\n${CYAN}Verifying ports are free...${NC}"

if lsof -ti:$BACKEND_PORT > /dev/null 2>&1; then
    echo -e "${RED}⚠️  Warning: Port $BACKEND_PORT still in use${NC}"
else
    echo -e "${GREEN}✅ Port $BACKEND_PORT is free${NC}"
fi

if lsof -ti:$FRONTEND_PORT > /dev/null 2>&1; then
    echo -e "${RED}⚠️  Warning: Port $FRONTEND_PORT still in use${NC}"
else
    echo -e "${GREEN}✅ Port $FRONTEND_PORT is free${NC}"
fi

echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}✅ All services stopped${NC}"
echo -e "${GREEN}================================================${NC}"
